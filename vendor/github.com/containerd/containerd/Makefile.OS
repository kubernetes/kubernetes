#Detect the OS, and return installos value same as GOOS
target_os =
ifeq ($(OS),Windows_NT)
    target_os = windows
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Linux)
        target_os = linux
    endif
    ifeq ($(UNAME_S),Darwin)
        target_os = darwin
    endif
    ifeq ($(UNAME_S),FreeBSD)
        target_os = freebsd
    endif
endif
