/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

package console

import (
	"fmt"
	"os"

	"github.com/pkg/errors"
	"golang.org/x/sys/windows"
)

var (
	vtInputSupported  bool
	ErrNotImplemented = errors.New("not implemented")
)

func (m *master) initStdios() {
	m.in = windows.Handle(os.Stdin.Fd())
	if err := windows.GetConsoleMode(m.in, &m.inMode); err == nil {
		// Validate that windows.ENABLE_VIRTUAL_TERMINAL_INPUT is supported, but do not set it.
		if err = windows.SetConsoleMode(m.in, m.inMode|windows.ENABLE_VIRTUAL_TERMINAL_INPUT); err == nil {
			vtInputSupported = true
		}
		// Unconditionally set the console mode back even on failure because SetConsoleMode
		// remembers invalid bits on input handles.
		windows.SetConsoleMode(m.in, m.inMode)
	} else {
		fmt.Printf("failed to get console mode for stdin: %v\n", err)
	}

	m.out = windows.Handle(os.Stdout.Fd())
	if err := windows.GetConsoleMode(m.out, &m.outMode); err == nil {
		if err := windows.SetConsoleMode(m.out, m.outMode|windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING); err == nil {
			m.outMode |= windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING
		} else {
			windows.SetConsoleMode(m.out, m.outMode)
		}
	} else {
		fmt.Printf("failed to get console mode for stdout: %v\n", err)
	}

	m.err = windows.Handle(os.Stderr.Fd())
	if err := windows.GetConsoleMode(m.err, &m.errMode); err == nil {
		if err := windows.SetConsoleMode(m.err, m.errMode|windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING); err == nil {
			m.errMode |= windows.ENABLE_VIRTUAL_TERMINAL_PROCESSING
		} else {
			windows.SetConsoleMode(m.err, m.errMode)
		}
	} else {
		fmt.Printf("failed to get console mode for stderr: %v\n", err)
	}
}

type master struct {
	in     windows.Handle
	inMode uint32

	out     windows.Handle
	outMode uint32

	err     windows.Handle
	errMode uint32
}

func (m *master) SetRaw() error {
	if err := makeInputRaw(m.in, m.inMode); err != nil {
		return err
	}

	// Set StdOut and StdErr to raw mode, we ignore failures since
	// windows.DISABLE_NEWLINE_AUTO_RETURN might not be supported on this version of
	// Windows.

	windows.SetConsoleMode(m.out, m.outMode|windows.DISABLE_NEWLINE_AUTO_RETURN)

	windows.SetConsoleMode(m.err, m.errMode|windows.DISABLE_NEWLINE_AUTO_RETURN)

	return nil
}

func (m *master) Reset() error {
	for _, s := range []struct {
		fd   windows.Handle
		mode uint32
	}{
		{m.in, m.inMode},
		{m.out, m.outMode},
		{m.err, m.errMode},
	} {
		if err := windows.SetConsoleMode(s.fd, s.mode); err != nil {
			return errors.Wrap(err, "unable to restore console mode")
		}
	}

	return nil
}

func (m *master) Size() (WinSize, error) {
	var info windows.ConsoleScreenBufferInfo
	err := windows.GetConsoleScreenBufferInfo(m.out, &info)
	if err != nil {
		return WinSize{}, errors.Wrap(err, "unable to get console info")
	}

	winsize := WinSize{
		Width:  uint16(info.Window.Right - info.Window.Left + 1),
		Height: uint16(info.Window.Bottom - info.Window.Top + 1),
	}

	return winsize, nil
}

func (m *master) Resize(ws WinSize) error {
	return ErrNotImplemented
}

func (m *master) ResizeFrom(c Console) error {
	return ErrNotImplemented
}

func (m *master) DisableEcho() error {
	mode := m.inMode &^ windows.ENABLE_ECHO_INPUT
	mode |= windows.ENABLE_PROCESSED_INPUT
	mode |= windows.ENABLE_LINE_INPUT

	if err := windows.SetConsoleMode(m.in, mode); err != nil {
		return errors.Wrap(err, "unable to set console to disable echo")
	}

	return nil
}

func (m *master) Close() error {
	return nil
}

func (m *master) Read(b []byte) (int, error) {
	return os.Stdin.Read(b)
}

func (m *master) Write(b []byte) (int, error) {
	return os.Stdout.Write(b)
}

func (m *master) Fd() uintptr {
	return uintptr(m.in)
}

// on windows, console can only be made from os.Std{in,out,err}, hence there
// isnt a single name here we can use. Return a dummy "console" value in this
// case should be sufficient.
func (m *master) Name() string {
	return "console"
}

// makeInputRaw puts the terminal (Windows Console) connected to the given
// file descriptor into raw mode
func makeInputRaw(fd windows.Handle, mode uint32) error {
	// See
	// -- https://msdn.microsoft.com/en-us/library/windows/desktop/ms686033(v=vs.85).aspx
	// -- https://msdn.microsoft.com/en-us/library/windows/desktop/ms683462(v=vs.85).aspx

	// Disable these modes
	mode &^= windows.ENABLE_ECHO_INPUT
	mode &^= windows.ENABLE_LINE_INPUT
	mode &^= windows.ENABLE_MOUSE_INPUT
	mode &^= windows.ENABLE_WINDOW_INPUT
	mode &^= windows.ENABLE_PROCESSED_INPUT

	// Enable these modes
	mode |= windows.ENABLE_EXTENDED_FLAGS
	mode |= windows.ENABLE_INSERT_MODE
	mode |= windows.ENABLE_QUICK_EDIT_MODE

	if vtInputSupported {
		mode |= windows.ENABLE_VIRTUAL_TERMINAL_INPUT
	}

	if err := windows.SetConsoleMode(fd, mode); err != nil {
		return errors.Wrap(err, "unable to set console to raw mode")
	}

	return nil
}

func checkConsole(f *os.File) error {
	var mode uint32
	if err := windows.GetConsoleMode(windows.Handle(f.Fd()), &mode); err != nil {
		return err
	}
	return nil
}

func newMaster(f *os.File) (Console, error) {
	if f != os.Stdin && f != os.Stdout && f != os.Stderr {
		return nil, errors.New("creating a console from a file is not supported on windows")
	}
	m := &master{}
	m.initStdios()
	return m, nil
}
