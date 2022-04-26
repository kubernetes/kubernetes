// +build windows

package dotwriter

import (
	"fmt"
	"io"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/windows"
)

var kernel32 = syscall.NewLazyDLL("kernel32.dll")

var (
	procSetConsoleCursorPosition   = kernel32.NewProc("SetConsoleCursorPosition")
	procFillConsoleOutputCharacter = kernel32.NewProc("FillConsoleOutputCharacterW")
)

// clear the line and move the cursor up
var clear = fmt.Sprintf("%c[%dA%c[2K\r", ESC, 0, ESC)

type dword uint32

type coord struct {
	x int16
	y int16
}

type fdWriter interface {
	io.Writer
	Fd() uintptr
}

func (w *Writer) clearLines(count int) {
	f, ok := w.out.(fdWriter)
	if ok && !isConsole(f.Fd()) {
		ok = false
	}
	if !ok {
		_, _ = fmt.Fprint(w.out, strings.Repeat(clear, count))
		return
	}
	fd := f.Fd()

	var csbi windows.ConsoleScreenBufferInfo
	if err := windows.GetConsoleScreenBufferInfo(windows.Handle(fd), &csbi); err != nil {
		return
	}

	for i := 0; i < count; i++ {
		// move the cursor up
		csbi.CursorPosition.Y--
		_, _, _ = procSetConsoleCursorPosition.Call(fd, uintptr(*(*int32)(unsafe.Pointer(&csbi.CursorPosition))))
		// clear the line
		cursor := coord{
			x: csbi.Window.Left,
			y: csbi.Window.Top + csbi.CursorPosition.Y,
		}
		var count, w dword
		count = dword(csbi.Size.X)
		_, _, _ = procFillConsoleOutputCharacter.Call(fd, uintptr(' '), uintptr(count), *(*uintptr)(unsafe.Pointer(&cursor)), uintptr(unsafe.Pointer(&w)))
	}
}

func isConsole(fd uintptr) bool {
	var mode uint32
	err := windows.GetConsoleMode(windows.Handle(fd), &mode)
	return err == nil
}
