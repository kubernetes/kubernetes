// Package winapi contains various low-level bindings to Windows APIs. It can
// be thought of as an extension to golang.org/x/sys/windows.
package winapi

//go:generate go run ..\..\mksyscall_windows.go -output zsyscall_windows.go system.go net.go path.go thread.go iocp.go jobobject.go logon.go memory.go process.go processor.go devices.go filesystem.go errors.go
