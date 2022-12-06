// +build zos

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

	"golang.org/x/sys/unix"
)

// NewPty creates a new pty pair
// The master is returned as the first console and a string
// with the path to the pty slave is returned as the second
func NewPty() (Console, string, error) {
	var f File
	var err error
	var slave string
	for i := 0;; i++ {
		ptyp := fmt.Sprintf("/dev/ptyp%04d", i)
		f, err = os.OpenFile(ptyp, os.O_RDWR, 0600)
		if err == nil {
			slave = fmt.Sprintf("/dev/ttyp%04d", i)
			break
		}
		if os.IsNotExist(err) {
			return nil, "", err
		}
		// else probably Resource Busy
	}
	m, err := newMaster(f)
	if err != nil {
		return nil, "", err
	}
	return m, slave, nil
}

type master struct {
	f        File
	original *unix.Termios
}

func (m *master) Read(b []byte) (int, error) {
	return m.f.Read(b)
}

func (m *master) Write(b []byte) (int, error) {
	return m.f.Write(b)
}

func (m *master) Close() error {
	return m.f.Close()
}

func (m *master) Resize(ws WinSize) error {
	return tcswinsz(m.f.Fd(), ws)
}

func (m *master) ResizeFrom(c Console) error {
	ws, err := c.Size()
	if err != nil {
		return err
	}
	return m.Resize(ws)
}

func (m *master) Reset() error {
	if m.original == nil {
		return nil
	}
	return tcset(m.f.Fd(), m.original)
}

func (m *master) getCurrent() (unix.Termios, error) {
	var termios unix.Termios
	if err := tcget(m.f.Fd(), &termios); err != nil {
		return unix.Termios{}, err
	}
	return termios, nil
}

func (m *master) SetRaw() error {
	rawState, err := m.getCurrent()
	if err != nil {
		return err
	}
	rawState = cfmakeraw(rawState)
	rawState.Oflag = rawState.Oflag | unix.OPOST
	return tcset(m.f.Fd(), &rawState)
}

func (m *master) DisableEcho() error {
	rawState, err := m.getCurrent()
	if err != nil {
		return err
	}
	rawState.Lflag = rawState.Lflag &^ unix.ECHO
	return tcset(m.f.Fd(), &rawState)
}

func (m *master) Size() (WinSize, error) {
	return tcgwinsz(m.f.Fd())
}

func (m *master) Fd() uintptr {
	return m.f.Fd()
}

func (m *master) Name() string {
	return m.f.Name()
}

// checkConsole checks if the provided file is a console
func checkConsole(f File) error {
	var termios unix.Termios
	if tcget(f.Fd(), &termios) != nil {
		return ErrNotAConsole
	}
	return nil
}

func newMaster(f File) (Console, error) {
	m := &master{
		f: f,
	}
	t, err := m.getCurrent()
	if err != nil {
		return nil, err
	}
	m.original = &t
	return m, nil
}

// ClearONLCR sets the necessary tty_ioctl(4)s to ensure that a pty pair
// created by us acts normally. In particular, a not-very-well-known default of
// Linux unix98 ptys is that they have +onlcr by default. While this isn't a
// problem for terminal emulators, because we relay data from the terminal we
// also relay that funky line discipline.
func ClearONLCR(fd uintptr) error {
	return setONLCR(fd, false)
}

// SetONLCR sets the necessary tty_ioctl(4)s to ensure that a pty pair
// created by us acts as intended for a terminal emulator.
func SetONLCR(fd uintptr) error {
	return setONLCR(fd, true)
}
