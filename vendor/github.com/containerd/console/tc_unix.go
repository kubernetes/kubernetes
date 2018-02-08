// +build darwin freebsd linux solaris

package console

import (
	"golang.org/x/sys/unix"
)

func tcget(fd uintptr, p *unix.Termios) error {
	termios, err := unix.IoctlGetTermios(int(fd), cmdTcGet)
	if err != nil {
		return err
	}
	*p = *termios
	return nil
}

func tcset(fd uintptr, p *unix.Termios) error {
	return unix.IoctlSetTermios(int(fd), cmdTcSet, p)
}

func tcgwinsz(fd uintptr) (WinSize, error) {
	var ws WinSize

	uws, err := unix.IoctlGetWinsize(int(fd), unix.TIOCGWINSZ)
	if err != nil {
		return ws, err
	}

	// Translate from unix.Winsize to console.WinSize
	ws.Height = uws.Row
	ws.Width = uws.Col
	ws.x = uws.Xpixel
	ws.y = uws.Ypixel
	return ws, nil
}

func tcswinsz(fd uintptr, ws WinSize) error {
	// Translate from console.WinSize to unix.Winsize

	var uws unix.Winsize
	uws.Row = ws.Height
	uws.Col = ws.Width
	uws.Xpixel = ws.x
	uws.Ypixel = ws.y

	return unix.IoctlSetWinsize(int(fd), unix.TIOCSWINSZ, &uws)
}

func setONLCR(fd uintptr, enable bool) error {
	var termios unix.Termios
	if err := tcget(fd, &termios); err != nil {
		return err
	}
	if enable {
		// Set +onlcr so we can act like a real terminal
		termios.Oflag |= unix.ONLCR
	} else {
		// Set -onlcr so we don't have to deal with \r.
		termios.Oflag &^= unix.ONLCR
	}
	return tcset(fd, &termios)
}

func cfmakeraw(t unix.Termios) unix.Termios {
	t.Iflag &^= (unix.IGNBRK | unix.BRKINT | unix.PARMRK | unix.ISTRIP | unix.INLCR | unix.IGNCR | unix.ICRNL | unix.IXON)
	t.Oflag &^= unix.OPOST
	t.Lflag &^= (unix.ECHO | unix.ECHONL | unix.ICANON | unix.ISIG | unix.IEXTEN)
	t.Cflag &^= (unix.CSIZE | unix.PARENB)
	t.Cflag &^= unix.CS8
	t.Cc[unix.VMIN] = 1
	t.Cc[unix.VTIME] = 0

	return t
}
