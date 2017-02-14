// +build windows

package windows

import (
	"io"
	"os"
	"syscall"

	"github.com/Azure/go-ansiterm/winterm"

	ansiterm "github.com/Azure/go-ansiterm"
	"github.com/Sirupsen/logrus"
	"io/ioutil"
)

// ConEmuStreams returns prepared versions of console streams,
// for proper use in ConEmu terminal.
// The ConEmu terminal emulates ANSI on output streams well by default.
func ConEmuStreams() (stdIn io.ReadCloser, stdOut, stdErr io.Writer) {
	if IsConsole(os.Stdin.Fd()) {
		stdIn = newAnsiReader(syscall.STD_INPUT_HANDLE)
	} else {
		stdIn = os.Stdin
	}

	stdOut = os.Stdout
	stdErr = os.Stderr

	// WARNING (BEGIN): sourced from newAnsiWriter

	logFile := ioutil.Discard

	if isDebugEnv := os.Getenv(ansiterm.LogEnv); isDebugEnv == "1" {
		logFile, _ = os.Create("ansiReaderWriter.log")
	}

	logger = &logrus.Logger{
		Out:       logFile,
		Formatter: new(logrus.TextFormatter),
		Level:     logrus.DebugLevel,
	}

	// WARNING (END): sourced from newAnsiWriter

	return stdIn, stdOut, stdErr
}

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
