package libcontainer

import (
	"fmt"
	"os"
	"syscall"
	"unsafe"
)

// newConsole returns an initialized console that can be used within a container by copying bytes
// from the master side to the slave that is attached as the tty for the container's init process.
func newConsole() (Console, error) {
	master, err := os.OpenFile("/dev/ptmx", syscall.O_RDWR|syscall.O_NOCTTY|syscall.O_CLOEXEC, 0)
	if err != nil {
		return nil, err
	}
	if err := saneTerminal(master); err != nil {
		return nil, err
	}
	console, err := ptsname(master)
	if err != nil {
		return nil, err
	}
	if err := unlockpt(master); err != nil {
		return nil, err
	}
	return &linuxConsole{
		slavePath: console,
		master:    master,
	}, nil
}

// linuxConsole is a linux pseudo TTY for use within a container.
type linuxConsole struct {
	master    *os.File
	slavePath string
}

func (c *linuxConsole) File() *os.File {
	return c.master
}

func (c *linuxConsole) Path() string {
	return c.slavePath
}

func (c *linuxConsole) Read(b []byte) (int, error) {
	return c.master.Read(b)
}

func (c *linuxConsole) Write(b []byte) (int, error) {
	return c.master.Write(b)
}

func (c *linuxConsole) Close() error {
	if m := c.master; m != nil {
		return m.Close()
	}
	return nil
}

// mount initializes the console inside the rootfs mounting with the specified mount label
// and applying the correct ownership of the console.
func (c *linuxConsole) mount() error {
	oldMask := syscall.Umask(0000)
	defer syscall.Umask(oldMask)
	f, err := os.Create("/dev/console")
	if err != nil && !os.IsExist(err) {
		return err
	}
	if f != nil {
		f.Close()
	}
	return syscall.Mount(c.slavePath, "/dev/console", "bind", syscall.MS_BIND, "")
}

// dupStdio opens the slavePath for the console and dups the fds to the current
// processes stdio, fd 0,1,2.
func (c *linuxConsole) dupStdio() error {
	slave, err := c.open(syscall.O_RDWR)
	if err != nil {
		return err
	}
	fd := int(slave.Fd())
	for _, i := range []int{0, 1, 2} {
		if err := syscall.Dup3(fd, i, 0); err != nil {
			return err
		}
	}
	return nil
}

// open is a clone of os.OpenFile without the O_CLOEXEC used to open the pty slave.
func (c *linuxConsole) open(flag int) (*os.File, error) {
	r, e := syscall.Open(c.slavePath, flag, 0)
	if e != nil {
		return nil, &os.PathError{
			Op:   "open",
			Path: c.slavePath,
			Err:  e,
		}
	}
	return os.NewFile(uintptr(r), c.slavePath), nil
}

func ioctl(fd uintptr, flag, data uintptr) error {
	if _, _, err := syscall.Syscall(syscall.SYS_IOCTL, fd, flag, data); err != 0 {
		return err
	}
	return nil
}

// unlockpt unlocks the slave pseudoterminal device corresponding to the master pseudoterminal referred to by f.
// unlockpt should be called before opening the slave side of a pty.
func unlockpt(f *os.File) error {
	var u int32
	return ioctl(f.Fd(), syscall.TIOCSPTLCK, uintptr(unsafe.Pointer(&u)))
}

// ptsname retrieves the name of the first available pts for the given master.
func ptsname(f *os.File) (string, error) {
	var n int32
	if err := ioctl(f.Fd(), syscall.TIOCGPTN, uintptr(unsafe.Pointer(&n))); err != nil {
		return "", err
	}
	return fmt.Sprintf("/dev/pts/%d", n), nil
}

// saneTerminal sets the necessary tty_ioctl(4)s to ensure that a pty pair
// created by us acts normally. In particular, a not-very-well-known default of
// Linux unix98 ptys is that they have +onlcr by default. While this isn't a
// problem for terminal emulators, because we relay data from the terminal we
// also relay that funky line discipline.
func saneTerminal(terminal *os.File) error {
	// Go doesn't have a wrapper for any of the termios ioctls.
	var termios syscall.Termios

	if err := ioctl(terminal.Fd(), syscall.TCGETS, uintptr(unsafe.Pointer(&termios))); err != nil {
		return fmt.Errorf("ioctl(tty, tcgets): %s", err.Error())
	}

	// Set -onlcr so we don't have to deal with \r.
	termios.Oflag &^= syscall.ONLCR

	if err := ioctl(terminal.Fd(), syscall.TCSETS, uintptr(unsafe.Pointer(&termios))); err != nil {
		return fmt.Errorf("ioctl(tty, tcsets): %s", err.Error())
	}

	return nil
}
