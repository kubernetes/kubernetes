@echo off

REM Copyright (c) Microsoft Corporation. All rights reserved.
REM Licensed under the MIT License.

setlocal

for /f "usebackq delims=" %%i in (`where pwsh 2^>nul`) do (
    set _cmd=%%i
)

if "%_cmd%"=="" (
    echo Error: PowerShell not found. Please visit https://github.com/powershell/powershell for install instructions.
    exit /b 2
)

call "%_cmd%" -NoLogo -NoProfile -File "%~dpn0.ps1" %*
