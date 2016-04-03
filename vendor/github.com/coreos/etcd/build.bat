@echo off

SET ORG_PATH=github.com\coreos
SET REPO_PATH=%ORG_PATH%\etcd

SET GOPATH=%cd%\gopath

:: Cleanup old builds
IF EXIST "%GOPATH%\src\%REPO_PATH%" rmdir /s /q "%GOPATH%\src\%REPO_PATH%"
IF EXIST "%GOPATH%\src\%ORG_PATH%" rmdir /s /q "%GOPATH%\src\%ORG_PATH%"
IF EXIST "%cd%\bin" rmdir /s /q "%cd%\bin"

md "%GOPATH%\src\%ORG_PATH%"

mklink /d "%GOPATH%\src\%REPO_PATH%" "%cd%"
FOR /f "usebackq tokens=*" %%a IN (`go env`) DO %%a

(FOR /f "tokens=*" %%i IN ('git rev-parse --short HEAD') DO SET GIT_SHA=%%i) 2>NUL
IF NOT DEFINED GIT_SHA SET GIT_SHA=GitNotFound

:: Static compilation is useful when etcd is run in a container
SET CGO_ENABLED=0
go build -a -installsuffix cgo -ldflags "-s -X %REPO_PATH%\version.GitSHA %GIT_SHA%" -o bin\etcd.exe "%REPO_PATH%"
:: TODO: Get the %GIT_SHA% argument to work. Useful for `etcd --version` style commands.
go build -a -installsuffix cgo -ldflags "-s" -o bin\etcdctl.exe "%REPO_PATH%\etcdctl"
