package main

import (
	"sync/atomic"

	_ "github.com/docker/docker/autogen/winresources/dockerd"
)

//go:cgo_import_dynamic main.dummy CommandLineToArgvW%2 "shell32.dll"

var dummy uintptr

func init() {
	// Ensure that this import is not removed by the linker. This is used to
	// ensure that shell32.dll is loaded by the system loader, preventing
	// go#15286 from triggering on Nano Server TP5.
	atomic.LoadUintptr(&dummy)
}
