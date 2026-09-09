//go:build !windows
// +build !windows

package guid

// GUID represents a GUID/UUID. It has the same structure as
// golang.org/x/sys/windows.GUID so that it can be used with functions expecting
// that type. It is defined as its own type as that is only available to builds
// targeted at `windows`. The representation matches that used by native Windows
// code.
type GUID struct {
	Data1 uint32
	Data2 uint16
	Data3 uint16
	Data4 [8]byte
}
