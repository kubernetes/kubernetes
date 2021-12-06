//go:build !windows
// +build !windows

package term

import (
	"golang.org/x/sys/unix"
)

// GetWinsize returns the window size based on the specified file descriptor.
func GetWinsize(fd uintptr) (*Winsize, error) {
	uws, err := unix.IoctlGetWinsize(int(fd), unix.TIOCGWINSZ)
	ws := &Winsize{Height: uws.Row, Width: uws.Col, x: uws.Xpixel, y: uws.Ypixel}
	return ws, err
}

// SetWinsize tries to set the specified window size for the specified file descriptor.
func SetWinsize(fd uintptr, ws *Winsize) error {
	uws := &unix.Winsize{Row: ws.Height, Col: ws.Width, Xpixel: ws.x, Ypixel: ws.y}
	return unix.IoctlSetWinsize(int(fd), unix.TIOCSWINSZ, uws)
}
