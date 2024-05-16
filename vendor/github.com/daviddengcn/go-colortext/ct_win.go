// +build windows

package ct

import (
	"syscall"
	"unsafe"
)

var fg_colors = []uint16{
	0,
	0,
	foreground_red,
	foreground_green,
	foreground_red | foreground_green,
	foreground_blue,
	foreground_red | foreground_blue,
	foreground_green | foreground_blue,
	foreground_red | foreground_green | foreground_blue}

var bg_colors = []uint16{
	0,
	0,
	background_red,
	background_green,
	background_red | background_green,
	background_blue,
	background_red | background_blue,
	background_green | background_blue,
	background_red | background_green | background_blue}

const (
	foreground_blue      = uint16(0x0001)
	foreground_green     = uint16(0x0002)
	foreground_red       = uint16(0x0004)
	foreground_intensity = uint16(0x0008)
	background_blue      = uint16(0x0010)
	background_green     = uint16(0x0020)
	background_red       = uint16(0x0040)
	background_intensity = uint16(0x0080)

	foreground_mask = foreground_blue | foreground_green | foreground_red | foreground_intensity
	background_mask = background_blue | background_green | background_red | background_intensity
)

var (
	kernel32 = syscall.NewLazyDLL("kernel32.dll")

	procGetStdHandle               = kernel32.NewProc("GetStdHandle")
	procSetConsoleTextAttribute    = kernel32.NewProc("SetConsoleTextAttribute")
	procGetConsoleScreenBufferInfo = kernel32.NewProc("GetConsoleScreenBufferInfo")

	hStdout        uintptr
	initScreenInfo *console_screen_buffer_info
)

func setConsoleTextAttribute(hConsoleOutput uintptr, wAttributes uint16) bool {
	ret, _, _ := procSetConsoleTextAttribute.Call(
		hConsoleOutput,
		uintptr(wAttributes))
	return ret != 0
}

type coord struct {
	X, Y int16
}

type small_rect struct {
	Left, Top, Right, Bottom int16
}

type console_screen_buffer_info struct {
	DwSize              coord
	DwCursorPosition    coord
	WAttributes         uint16
	SrWindow            small_rect
	DwMaximumWindowSize coord
}

func getConsoleScreenBufferInfo(hConsoleOutput uintptr) *console_screen_buffer_info {
	var csbi console_screen_buffer_info
	if ret, _, _ := procGetConsoleScreenBufferInfo.Call(hConsoleOutput, uintptr(unsafe.Pointer(&csbi))); ret == 0 {
		return nil
	}
	return &csbi
}

const (
	std_output_handle = uint32(-11 & 0xFFFFFFFF)
)

func init() {
	kernel32 := syscall.NewLazyDLL("kernel32.dll")

	procGetStdHandle = kernel32.NewProc("GetStdHandle")

	hStdout, _, _ = procGetStdHandle.Call(uintptr(std_output_handle))

	initScreenInfo = getConsoleScreenBufferInfo(hStdout)

	syscall.LoadDLL("")
}

func resetColor() {
	if initScreenInfo == nil { // No console info - Ex: stdout redirection
		return
	}
	setConsoleTextAttribute(hStdout, initScreenInfo.WAttributes)
}

func changeColor(fg Color, fgBright bool, bg Color, bgBright bool) {
	attr := uint16(0)
	if fg == None || bg == None {
		cbufinfo := getConsoleScreenBufferInfo(hStdout)
		if cbufinfo == nil { // No console info - Ex: stdout redirection
			return
		}
		attr = cbufinfo.WAttributes
	}
	if fg != None {
		attr = attr & ^foreground_mask | fg_colors[fg]
		if fgBright {
			attr |= foreground_intensity
		}
	}
	if bg != None {
		attr = attr & ^background_mask | bg_colors[bg]
		if bgBright {
			attr |= background_intensity
		}
	}
	setConsoleTextAttribute(hStdout, attr)
}
