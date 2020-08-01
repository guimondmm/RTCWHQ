@ECHO OFF
python.exe unpk3.py input
python.exe rtcwhq.py input 4 4096 2 2.0 0.0 4 90
rem python.exe makepk3.py input