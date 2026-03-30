@echo off
title ESP Detection - Car & Human Tracking + Accident Alert
echo ============================================
echo   ESP Object Detection System
echo   Car + Human Tracking + Accident Alert
echo ============================================
echo.
echo Choose input source:
echo   [1] Use Camera (Webcam)
echo   [2] Choose Video File
echo.
set /p src="Enter choice (1-2): "

echo.
echo Choose window size:
echo   [1] Default (1280x920)
echo   [2] Full HD (1920x1080)
echo   [3] Small (800x600)
echo.
set /p size="Enter choice (1-3): "

set W=1280
set H=920
if "%size%"=="2" set W=1920& set H=1080
if "%size%"=="3" set W=800& set H=600

echo.
if "%src%"=="1" (
    echo Starting with Camera...
    python car+human.py --source 0 --width %W% --height %H%
) else (
    echo Opening file picker...
    python car+human.py --width %W% --height %H%
)

pause
