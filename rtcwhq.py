import json                         # LIB: JSON parser
import pickle                       # LIB: binary serialization
import sys                          # LIB: System
import os.path, stat                # LIB: Path, File status
import glob                         # LIB: Global
import cv2                          # LIB: OpenCV
import numpy as np                  # LIB: Numpy
import torch                        # LIB: Pytorch
import architecture as arch         # LIB: ERSGAN architecture
import subprocess                   # LIB: Call Subprocess
import pathlib                      # LIB: Pathlib
import time                         # LIB: Time
from PIL import Image               # LIB: PIL
from PIL import ImageEnhance        # LIB: PIL Enhancement
from PIL import ImageFilter         # LIB: PIL Filters
from os.path import splitext        # LIB: extension split

# Changeable flags
powertwo = False                    # check for and correct textures which are not power of two size
rtcwexcludes = False                # exclude defined RTCW/ET folders and use standard settings there
alphaoptimize = True                # use gaussian blur, contrast and brightness (or not, if not needed)
usesharpen = True                   # sharpen the high resolution texture before resize to increase quality
autoconvert = True                  # convert the image to RGB if it is NOT RBG/RGBA!
skiptracemap = True                 # don't resize / include the tracemap
scalelightmaps = True               # resize Lightmaps (could look better, could look strange)
scalelarge = False                  # scale large images too (True = they are initially resized to a lower res)
testmode = False                    # in Testmode, a Lancosz method is used instead of the ESRGAN method
warnings = False                    # ignore (False) or show (True) warnings
cache = True                        # attempt to cache prepared models to speed up future launches (insecure!)

# VRAM limits 8GB
largelimit = 2048*2048              # maximum texture scaling limit (stop scaling if texture is below this size)
vramlimit = 0.99                    # maximum proportion of free VRAM to try to allocate to ESRGAN
									# ESRGAN is ridiculously VRAM intensive, even compared to other GANs,
									# e.g., a 1024px*1024px image would theoretically max out a 8.0GiB video card.
									# Available VRAM is lower due to Desktop Window Manager and other app usage,
									# so free VRAM is queried from 'nvidia-smi.exe' then multiplied by this value.
							
# Predefined Values
target = 'cuda'                     # ESRGAN target device: 'cuda' for nVidia card (fast) or 'cpu' for ATI/CPU (excruciatingly slow!)
modelfactor = 4                     # the scale the selected model has been trained on (default is 4x)
allowed = [".png",".tga",".jpg",".bmp",".dds"]    # allowed image file extensions to process (default: PNG, TGA, JPG)
downscaling = Image.LANCZOS         # scaling method reducing the highres image to the desired resolution
downfactor = 2                      # downscale the final images by this factor in each direction using the downscaling scaling method,
									# because, imho, results aren't that sharp at 4x, so it's a massive waste of disk space (1 = disable)

# Predefined ESRGAN Models
default_model = 'cartoonpainted_400000.pth'       # default model
font_model = 'ReducedColorsAttempt.pth'     # font model

# RTCW exclude files (not implemented yet)
excludes = ["gfx/2d/backtile.jpg"]  # ET: gives a strange background texture in the loading screen, should be black

# create logfile
log=open("convert.log","w+")

# ignore warning
if(warnings==False):
	import warnings
	warnings.filterwarnings("ignore")

# function: write to logfile and output to stdout
def write_log(*args):
	line = ' '.join([str(a) for a in args])
	log.write(line+'\n')
	print(line)

# function: delete a single directory
def remove_empty_dir(path):
	try:
		if(os.rmdir(path)):
			write_log("Removed: " + path)
	except OSError:
		write_log("Not removed: " + path)
		pass

# function: delete a directory tree
def remove_empty_dirs(path):
	for root, dirnames, filenames in os.walk(path, topdown=False):
		for dirname in dirnames:
			remove_empty_dir(os.path.realpath(os.path.join(root, dirname)))

# function: upscale a PNG image
def upscale(im, device, model):
	img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
	img = img * 1.0 / 255
	img = torch.from_numpy(np.transpose(img[:, :, [2, 1, 0]], (2, 0, 1))).float()
	img_LR = img.unsqueeze(0)
	img_LR = img_LR.to(device)
	output = model(img_LR).data.squeeze().float().cpu().clamp_(0, 1).numpy()
	output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0))
	output = (output * 255.0).round()
	output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)	
	return output.astype(np.uint8)

# get the next power of two value of a value (texture size correction)
def poweroftwo(val):
	val=int(val)
	# check range from 2^0=1 to 2^15 = 32768 (should be large enough :-)
	for i in range(0,15):
		# value is below current power of two? found!
		if val<pow(2,i):
			# get previous power of two and next power of two
			mn=pow(2,i-1)
			mx=pow(2,i)
			# get delta between previous and next power (middle)
			delta=(mx-mn)/2
			# value above the middle: use higher power of two else use lower power of two
			if val>=(mn+delta):
				return mx
			else:
				return mn

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60.
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)
	
# "resize" the image until it fits in vram, return the width (aspect is known)
def fitimage(width, height, available_vram_banks):

	# reduce image size by factor 2
	while(width*height>available_vram_banks):
		width=int(width/2)
		height=int(height/2)

	write_log("  - Image resized to "+str(width)+"x"+str(height))
		
	return width

def gpu_stats():
	nvsmi = [
	'C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe',
	'--format=csv,nounits', # header will be used as dict keys
	'--query-gpu=memory.total,memory.used,memory.free', # "nvidia-smi --help-query-gpu" for list of available properties
	'--id=0' # 0 = default GPU; may need changing if multiple GPUs or integrated graphics
	]
	stats = subprocess.run(nvsmi, capture_output=True)
	keys, values = stats.stdout.decode().splitlines() # comma-separated values
	keys = [k.rstrip(' [MiB]') for k in keys.split(', ')]
	values = [int(v) for v in values.split(', ')]
	return dict(zip(keys, values))
	
# commandline input
input = sys.argv[1]
factor = sys.argv[2]
maxsize = sys.argv[3]
blur = sys.argv[4]
contrast = sys.argv[5]
brightness = sys.argv[6]
sharpen = sys.argv[7]
jpegquality = sys.argv[8]

# set data types
input=str(input)
factor=int(factor)
maxsize=int(maxsize)
blur=int(blur)
contrast=float(contrast)
brightness=float(brightness)
sharpen=int(sharpen)
jpegquality=int(jpegquality)

# start
write_log("======================================================================")
write_log("RTCWHQ batch upscaling with ERSGAN started")
write_log("======================================================================")
write_log(f"Model:           {os.path.join('models', default_model)}")
write_log(f"Fontmodel:       {os.path.join('models', font_model)}")
write_log(f"Folder:          {input}")
write_log(f"Maximum scaling: {factor}x")
write_log(f"Maximum size:    {maxsize}px")
write_log(f"Gaussian Blur:   {blur}px")
write_log(f"Contrast:        {contrast*100}%")
write_log(f"Brightness:      {brightness*100}%")
write_log(f"Sharpen:         {sharpen}px")
write_log(f"JPEG Quality:    {jpegquality}%")
write_log("----------------------------------------------------------------------")

# count files to process
file_counters = {}
for ext in allowed:
	file_counters[ext] = sum(1 for f in pathlib.Path(input).glob(f"**/*{ext}"))
write_log(f"Found {sum(file_counters.values())} images:")
for counter in sorted(file_counters.items(), key=lambda i: i[1], reverse=True):
	if counter[1] > 0: write_log(f"  - {counter[1]} {counter[0].lstrip('.').upper()}s")

# init model
if(testmode==False):
	write_log("----------------------------------------------------------------------")
	write_log(f"Preparing the PyTorch models...{'' if cache else ' (this may take a while!)'}")

	device_cuda = torch.device("cuda")
	device_cpu = torch.device("cpu")

	
	# read prepared cuda model from cache if available:
	model_cuda_cache = os.path.join('models', os.path.splitext(default_model)[0]+'.cuda')
	if cache and os.path.isfile(model_cuda_cache):
		with open(model_cuda_cache, 'rb') as c:
			model_cuda = pickle.load(c)
		write_log("CUDA model loaded from disk.")
	else:
		# prepare cuda model
		model_cuda = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
		model_cuda.load_state_dict(torch.load(os.path.join('models', default_model)), strict=True)
		model_cuda.eval()

		for k, v in model_cuda.named_parameters():
			v.requires_grad = False
			model_cuda = model_cuda.to(device_cuda)
		write_log("CUDA model ready.")

		if cache:
			with open(model_cuda_cache, 'wb') as c:
				pickle.dump(model_cuda, c)
			write_log("CUDA model saved to disk.")

	# read prepared cpu model from cache if available:
	model_cpu_cache = os.path.join('models', os.path.splitext(default_model)[0]+'.cpu')
	if cache and os.path.isfile(model_cpu_cache):
		with open(model_cpu_cache, 'rb') as c:
			model_cpu = pickle.load(c)
		write_log("CPU model loaded from disk.")
	else:
		# prepare cpu model
		model_cpu = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
		model_cpu.load_state_dict(torch.load(os.path.join('models', default_model)), strict=True)
		model_cpu.eval()

		for k, v in model_cpu.named_parameters():
			v.requires_grad = False
			model_cpu = model_cpu.to(device_cpu)
		write_log("CPU model ready.")

		if cache:
			with open(model_cpu_cache, 'wb') as c:
				pickle.dump(model_cpu, c)
			write_log("CPU model saved to disk.")

	# # prepare font model
	# fontmodel = arch.RRDB_Net(3, 3, 64, 23, gc=32, upscale=4, norm_type=None, act_type='leakyrelu', mode='CNA', res_scale=1, upsample_mode='upconv')
	# fontmodel.load_state_dict(torch.load(os.path.join('models', font_model)), strict=True)
	# fontmodel.eval()

	# for k, v in fontmodel.named_parameters():
	# 	v.requires_grad = False
	# 	fontmodel_cuda = fontmodel.to(device_cuda)
	# 	fontmodel_cpu = fontmodel.to(device_cpu)

	# write_log("FontModel ready. Let's go!")

starttime=time.time()
cnt=0
dcnt=0
	
# iterate through all subfolders
for dirName, subdirList, fileList in os.walk(input, topdown=False):

	for fname in fileList:

		dirName.replace("/","\\")
		fname.replace("/","\\")
		path=dirName+"/"+fname
		
		delete=False
		delreason=""
		
		# split filename, make it writable and convert extension to lowercase
		os.chmod(path ,stat.S_IWRITE)
		filename,ext=splitext(fname)
		ext=ext.lower()
		
		# only convert allowed file extensions
		if ext in allowed:
			
			# open image, add extension to filename and save it to PNG
			fullname=dirName + "\\" + filename + ext
			
			# try to load the image or throw error if there is a problem
			try:
				im = Image.open(fullname)
				loaded=True
				width, height = im.size
				width=int(width)
				height=int(height)
				imode=im.mode
				cnt+=1

			# image could not open? error!
			except IOError:
				loaded=False
				delete=True
				delreason="could not open image"
				im.close()
			
			# skip tracemap if set
			if(skiptracemap==True and "_tracemap" in filename):
				loaded=False
				delete=True
				delreason="tracemaps not allowed"
				im.close()
						
			# skip lightmaps if set
			if(scalelightmaps==False and "lm_0" in filename):
				loaded=Fals
				delete=True
				delreason="lightmaps not allowed"
				im.close()
						
			# loading successful? check colormode first!
			if(loaded==True):
						
				write_log("----------------------------------------------------------------------")
				write_log(f"IMAGE {cnt} of {sum(file_counters.values())}: {fullname}")
				write_log("----------------------------------------------------------------------")
				stime=time.time()
				
				add="unknown"
				if(imode=="L"):
					add="RGB 8bit Greyscale"
				
				if(imode=="P"):
					add="RGB 8bit Palette"

				if(imode=="RGB"):
					add="RGB 24bit"
					
				if(imode=="RGBA"):
					add="RGB 32bit with Alpha Channel"
				
				write_log("- Colormode: "+add)

				# image is NOT RGB(A)? then try to convert it
				if(autoconvert==True and imode != 'RGBA' and imode != 'RGB'):
				
					# 8bit color palette/greyscale? then convert to 24bit RGB
					if(imode=="P" or imode=="L"):
						im2=Image.new("RGB",im.size)
						im2.paste(im)
						im=im2
						im2=None
						write_log("- NOTICE: 8bit converted to 24bit RGB")
					else:
						# image was not valid, skip
						loaded=False
						delete=True
						delreason="no valid image"
						im.close()
						
			# still a valid image? then process it
			if(loaded==True):
				
				# optional: check and correct textures which are NOT power of two size (can cause errors otherwise)
				if(powertwo):

					twow=poweroftwo(width)
					twoh=poweroftwo(height)
					
					# width/height doesn't match with calculated power of two value? correct it = resize to next power of two
					if (twow != width) or (twoh != height):
						write_log("- NOTICE: Texture size "+str(width)+"x"+str(height)+" corrected - was NOT power of two!")
						width=twow
						height=twoh
						im.resize((width,height),Image.LANCZOS)
				
				# store original width for later use
				ow=width
				oh=height

				# if exists, get alpha channel first before we mess with the original image
				alpha=None
				if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
					alpha=im.split()[-1]
					# convert to RGB to work with ESRGAN
					alpha = alpha.convert("RGB")
					
				scalepass=1
				
				write_log("- Resolution: "+str(width)+"x"+str(height)+" ("+str(factor)+"x = "+str(width*factor)+"x"+str(height*factor)+")")
				
				# initial check if image is already too large: reduce it before enlargement
				process=True
				available_vram_banks = int(gpu_stats()['memory.free'] * vramlimit) << 7 # because width*height*8192 = vram*1024**2 => width*height<<13 = vram<<20 => width*height = vram<<7
				if(scalelarge==True):
					if(width*height > available_vram_banks):
						write_log(f"  - Image too large, must be resized first")
						write_log(f"    - available: {available_vram_banks >> 7} MiB")
						write_log(f"    - required:  {width*height >> 7} MiB")
						aspect=width/height
						width=fitimage(width,height,available_vram_banks)
						height=int(width/aspect)				

						im=im.resize((width,height),downscaling)
						if(alpha):
							alpha=alpha.resize((width,height),Image.LANCZOS)
				# simple check if the image needs processing or not
				else:
					if(width*height>=maxsize*maxsize):
						write_log("- no processing: image is already High Resolution")
						process=False
						
				# assume we're rescaling the image
				rescale=True
				
				# process images if necessary
				if(process==True):
					
					# resize image until it is large enough (change the largelimit var to a lower value if it crashes)
					while(width*height<largelimit and rescale==True):
					
						write_log("- Scale Pass #"+str(scalepass))

						# -------------------------------------------------------------------------------
						# Image is too large to scale? reduce size by factor two until the size is valid
						# according to VRAM limit.
						#
						# Notice: ESRGAN uses a lot of GPU VRAM so there is a limitation for the input
						# depending on the available VRAM. An estimated maximum value for 8GB VRAM is
						# about 1024x512 = 524288 Pixels, so change the available_vram_banks variable if you have
						# more or less than 8GB VRAM. An approximate formula to calculate the size:
						#
						# VRAM needed = width*height*8192 in Bytes
						#
						# You must add to this value the VRAM already assigned by Windows and Apps, which
						# can already take 2-3GB on 8GB VRAM, consider this
						# -------------------------------------------------------------------------------
						if((width*height)>available_vram_banks):
							# write_log("  - Image doesn't fit in VRAM, must be resized")
							
							# aspect=width/height
							# width=fitimage(width,height,available_vram_banks)
							# height=int(width/aspect)
							
							# im=im.resize((width,height),downscaling)
							# if(alpha):
							# 	alpha=alpha.resize((width,height),Image.LANCZOS)
							write_log("  - Image doesn't fit in VRAM; will be processed by CPU")
							write_log(f"    - available: {available_vram_banks >> 7} MiB")
							write_log(f"    - required:  {width*height >> 7} MiB")
							device = device_cpu
							model = model_cpu
						else:
							device = device_cuda
							model = model_cuda


						# scale color image
						width=width*modelfactor
						height=height*modelfactor
						write_log("  - ERSGAN scales Colormap to "+str(width)+"x"+str(height))
						if(testmode==False):
							im=Image.fromarray(upscale(im,device,model))
						else:
							im=im.resize((width*modelfactor,height*modelfactor),downscaling)
						
						# scale alpha channel
						if(alpha):
							write_log("  - ERSGAN scales Alphamap to "+str(width)+"x"+str(height))
							if(testmode==False):
								alpha=Image.fromarray(upscale(alpha,device,model))
							else:
								alpha=alpha.resize((width*modelfactor,height*modelfactor),downscaling)
							
						# image has target size? don't rescale anymore
						if(width==(ow*factor) and height==(oh*factor)):
							rescale=False
						# otherwise do the next scale pass
						else:
							scalepass+=1
												
					# calculate final texture size
					nsw=int(ow*factor)
					nsh=int(oh*factor)
					ms=maxsize
					bl=blur
					co=contrast
					br=brightness
					sh=sharpen
					
					# rtcw/et excludes and default settings for specific folder names
					if(rtcwexcludes):

						# limit general font size to 1024 and use different values for blur/contrast/brightness/sharpen
						if("font" in dirName):
							ms=1024
							sh=4
							bl=1
							co=2.0
							br=-0.5
							write_log("- Font Texture found, limiting size to "+str(ms)+" Pixel")
							
						# limit ET HUD font size to 1024 and use different values for blur/contrast/brightness/sharpen
						if("hudchars" in filename):
							ms=1024
							sh=4
							bl=0
							co=4.0
							br=-0.5
							write_log("- Font Texture found, limiting size to "+str(ms)+" Pixel")
							
						# limit leveshots image size to 1024
						if("levelshots" in dirName):
							ms=512
							# but the survey map can still be large
							if("_cc") in filename:
								ms=maxsize
							write_log("- Levelshot Texture found, limiting size to "+str(ms)+" Pixel")
						
						# limit lightmaps image size to 1024
						if("maps" in dirName):
							ms=1024
							write_log("- Lightmap Texture found, limiting size to "+str(ms)+" Pixel")
						
						# dont' add contrast to the skies and user different blur value
						folders=["skies","sfx","liquids"]
						if(dirName in folders):
						#if(("skies" in dirName) or ("sfx" in dirName) or ("liquids" in dirName)):
							co=0.0
							br=0.0
							bl=2
							write_log("- Blurry Alpha Texture found - no contrast or brightness change!")


					# optional: sharpen filter
					if(usesharpen and sh!=0):
						
						# don't sharpen lightmaps!
						if ("maps" in dirName):				
							write_log("- Lightmap texture found, no sharpen on lightmaps")
						else:
							write_log("- Colormap: Sharpen")
							im=ImageEnhance.Sharpness(im).enhance(sh)
							
					# if texture size is too large? reduce by factor
					if(nsw>ms) or (nsh>ms):
						f=min(ms/nsw,ms/nsh) // downfactor
						sw=int(nsw*f)
						sh=int(nsh*f)
					# or use the calculated values
					else:
						sw=nsw // downfactor
						sh=nsh // downfactor
								
					# scale colormap to desired resolution
					if sw != nsw or sh != nsh:
						write_log("- Colormap: downscaled to "+str(sw)+"x"+str(sh))
						im=im.resize((sw,sh),downscaling)
						
					# scae alphamap, if there is alpha
					if(alpha):
						
						# convert alphamap to 8bit greyscale to increase processing speed (it's greyscale only)
						alpha = alpha.convert("L")
										
						# only perform this if alpha optimizations are desired
						if(alphaoptimize):
						
							# apply gaussian blur
							if(bl!=0):
								write_log("- Alphamap: Gaussian Blur")
								alpha = alpha.filter(ImageFilter.GaussianBlur(bl))
								
							# apply brightness
							if(br!=0.0):
								write_log("- Alphamap: Brightness")
								alpha = ImageEnhance.Brightness(alpha).enhance(1.0+br)

							# apply contrast
							if(co!=0.0):
								write_log("- Alphamap: Contrast")
								alpha = ImageEnhance.Contrast(alpha).enhance(1.0+co)
						
						# scale alpha to desired resolution
						if sw != nsw or sh != nsh:
							write_log("- Alphamap scaled to "+str(sw)+"x"+str(sh))
							alpha=alpha.resize((sw,sh),downscaling)

						# merge alpha channel with RGB
						write_log("- Merging Colormap with Alphamap")
						im.putalpha(alpha.split()[-1])
					
					# save file
					write_log("- Replacing original Texture")
					if ext == ".dds":
						# imblob = io.BytesIO()
						# im.save(imblob, format='PNG')
						# with Wimage(blob=imblob.getvalue()) as im:
						# 	im.format = 'dds'
						# 	im.compression = 'dxt5'
						# 	im.save(filename=fullname)
						im.save(fullname+'.png',quality=jpegquality,optimize=True,progressive=True)
						im.close()
						identify = subprocess.run(['magick', 'convert', fullname, 'json:-'], capture_output=True)
						metadata = json.loads(identify.stdout)[0]['image']
						if not metadata['channelDepth'].get('alpha'):
							compression = 15 # bc1 (dxt1)
						elif metadata['channelDepth']['alpha'] == 1:
							compression = 16 # bc1a (dxt1a)
						elif metadata['compression'] == 'DXT1':
							compression = 15 # bc1 (dxt1)
						elif metadata['compression'] == 'DXT3':
							compression = 17 # bc2 (dxt2, dxt3)
						elif metadata['compression'] == 'DXT5':
							compression = 18 # bc3 (dxt4, dxt5)
						else:
							compression = 18 # bc3 (dxt4, dxt5)
						write_log(f"  - Compressing DDS texture as {['DXT1','DXT1a','DXT3','DXT5'][compression - 15]}")
						nvtt_command = [
							'C:\\Program Files\\NVIDIA Corporation\\NVIDIA Texture Tools Exporter\\nvtt_export.exe',
							f'-f{compression}', # bc1=15, bc1a=16, bc2=17, bc3=18, bc3n=19, bc4=20, bc5=21, bc6=22, bc7=23
							'-q3', # fastest=0, highest=3
							f'--cutout-alpha={compression==16}', # threhold transparency (instead of semi-transparent to fully opaque)
							f'--scale-alpha={compression==16}', # more accurate cutout for mipmaps with cutout alpha
							'-o', fullname, # output filename
							fullname+'.png' # input filename
							]
						subprocess.run(nvtt_command)
						os.remove(fullname+'.png')
					else:
						im.save(fullname,quality=jpegquality,optimize=True,progressive=True)
						im.close()
					write_log("- Conversion completed in "+str(hms_string(time.time()-stime)))
				
		else:
			# remove all other files
			delete = False if fname == '.gitignore' else True

		# delete files if flagged for deletion
		if(delete==True):
			write_log("----------------------------------------------------------------------")
			write_log("- DELETED "+path+" ("+delreason+")")
			os.remove(path)
			dcnt+=1
			
# finish
write_log("----------------------------------------------------------------------")
write_log("Removing empty directories...")
remove_empty_dirs(input)
write_log("Converted " + str(cnt) + " images and deleted "+str(dcnt)+" other files in "+str(hms_string(time.time()-starttime))+". Done.")
log.close()

# wait for input and exit
subprocess.call('timeout /T 5')