// +build windows

package term

import (
	"syscall"
	"unsafe"
)

const (
	// Consts for Get/SetConsoleMode function
	// see http://msdn.microsoft.com/en-us/library/windows/desktop/ms683167(v=vs.85).aspx
	ENABLE_ECHO_INPUT      = 0x0004
	ENABLE_INSERT_MODE     = 0x0020
	ENABLE_LINE_INPUT      = 0x0002
	ENABLE_MOUSE_INPUT     = 0x0010
	ENABLE_PROCESSED_INPUT = 0x0001
	ENABLE_QUICK_EDIT_MODE = 0x0040
	ENABLE_WINDOW_INPUT    = 0x0008
	// If parameter is a screen buffer handle, additional values
	ENABLE_PROCESSED_OUTPUT   = 0x0001
	ENABLE_WRAP_AT_EOL_OUTPUT = 0x0002
)

var kernel32DLL = syscall.NewLazyDLL("kernel32.dll")

var (
	setConsoleModeProc             = kernel32DLL.NewProc("SetConsoleMode")
	getConsoleScreenBufferInfoProc = kernel32DLL.NewProc("GetConsoleScreenBufferInfo")
)

func GetConsoleMode(fileDesc uintptr) (uint32, error) {
	var mode uint32
	err := syscall.GetConsoleMode(syscall.Handle(fileDesc), &mode)
	return mode, err
}

func SetConsoleMode(fileDesc uintptr, mode uint32) error {
	r, _, err := setConsoleModeProc.Call(fileDesc, uintptr(mode), 0)
	if r == 0 {
		if err != nil {
			return err
		}
		return syscall.EINVAL
	}
	return nil
}

// types for calling GetConsoleScreenBufferInfo
// see http://msdn.microsoft.com/en-us/library/windows/desktop/ms682093(v=vs.85).aspx
type (
	SHORT int16

	SMALL_RECT struct {
		Left   SHORT
		Top    SHORT
		Right  SHORT
		Bottom SHORT
	}

	COORD struct {
		X SHORT
		Y SHORT
	}

	WORD uint16

	CONSOLE_SCREEN_BUFFER_INFO struct {
		dwSize              COORD
		dwCursorPosition    COORD
		wAttributes         WORD
		srWindow            SMALL_RECT
		dwMaximumWindowSize COORD
	}
)

func GetConsoleScreenBufferInfo(fileDesc uintptr) (*CONSOLE_SCREEN_BUFFER_INFO, error) {
	var info CONSOLE_SCREEN_BUFFER_INFO
	r, _, err := getConsoleScreenBufferInfoProc.Call(uintptr(fileDesc), uintptr(unsafe.Pointer(&info)), 0)
	if r == 0 {
		if err != nil {
			return nil, err
		}
		return nil, syscall.EINVAL
	}
	return &info, nil
}
