// Package winapi contains various low-level bindings to Windows APIs. It can
// be thought of as an extension to golang.org/x/sys/windows.
package winapi

//go:generate go run ..\..\mksyscall_windows.go -output zsyscall_windows.go net.go iocp.go jobobject.go path.go logon.go memory.go processor.go devices.go filesystem.go errors.go
