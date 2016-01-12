// +build windows

package windows

import (
	"io"
	"os"
	"syscall"

	"github.com/Azure/go-ansiterm/winterm"
)

// ConsoleStreams returns a wrapped version for each standard stream referencing a console,
// that handles ANSI character sequences.
func ConsoleStreams() (stdIn io.ReadCloser, stdOut, stdErr io.Writer) {
	if IsConsole(os.Stdin.Fd()) {
		stdIn = newAnsiReader(syscall.STD_INPUT_HANDLE)
	} else {
		stdIn = os.Stdin
	}

	if IsConsole(os.Stdout.Fd()) {
		stdOut = newAnsiWriter(syscall.STD_OUTPUT_HANDLE)
	} else {
		stdOut = os.Stdout
	}

	if IsConsole(os.Stderr.Fd()) {
		stdErr = newAnsiWriter(syscall.STD_ERROR_HANDLE)
	} else {
		stdErr = os.Stderr
	}

	return stdIn, stdOut, stdErr
}

// GetHandleInfo returns file descriptor and bool indicating whether the file is a console.
func GetHandleInfo(in interface{}) (uintptr, bool) {
	switch t := in.(type) {
	case *ansiReader:
		return t.Fd(), true
	case *ansiWriter:
		return t.Fd(), true
	}

	var inFd uintptr
	var isTerminal bool

	if file, ok := in.(*os.File); ok {
		inFd = file.Fd()
		isTerminal = IsConsole(inFd)
	}
	return inFd, isTerminal
}

// IsConsole returns true if the given file descriptor is a Windows Console.
// The code assumes that GetConsoleMode will return an error for file descriptors that are not a console.
func IsConsole(fd uintptr) bool {
	_, e := winterm.GetConsoleMode(fd)
	return e == nil
}
