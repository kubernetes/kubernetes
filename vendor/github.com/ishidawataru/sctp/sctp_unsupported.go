// +build !linux linux,386

package sctp

import (
	"errors"
	"net"
	"runtime"
)

var ErrUnsupported = errors.New("SCTP is unsupported on " + runtime.GOOS + "/" + runtime.GOARCH)

func setsockopt(fd int, optname, optval, optlen uintptr) (uintptr, uintptr, error) {
	return 0, 0, ErrUnsupported
}

func getsockopt(fd int, optname, optval, optlen uintptr) (uintptr, uintptr, error) {
	return 0, 0, ErrUnsupported
}

func (c *SCTPConn) SCTPWrite(b []byte, info *SndRcvInfo) (int, error) {
	return 0, ErrUnsupported
}

func (c *SCTPConn) SCTPRead(b []byte) (int, *SndRcvInfo, error) {
	return 0, nil, ErrUnsupported
}

func (c *SCTPConn) Close() error {
	return ErrUnsupported
}

func (c *SCTPConn) SetWriteBuffer(bytes int) error {
	return ErrUnsupported
}

func (c *SCTPConn) GetWriteBuffer() (int, error) {
	return 0, ErrUnsupported
}

func (c *SCTPConn) SetReadBuffer(bytes int) error {
	return ErrUnsupported
}

func (c *SCTPConn) GetReadBuffer() (int, error) {
	return 0, ErrUnsupported
}

func ListenSCTP(net string, laddr *SCTPAddr) (*SCTPListener, error) {
	return nil, ErrUnsupported
}

func ListenSCTPExt(net string, laddr *SCTPAddr, options InitMsg) (*SCTPListener, error) {
	return nil, ErrUnsupported
}

func (ln *SCTPListener) Accept() (net.Conn, error) {
	return nil, ErrUnsupported
}

func (ln *SCTPListener) AcceptSCTP() (*SCTPConn, error) {
	return nil, ErrUnsupported
}

func (ln *SCTPListener) Close() error {
	return ErrUnsupported
}

func DialSCTP(net string, laddr, raddr *SCTPAddr) (*SCTPConn, error) {
	return nil, ErrUnsupported
}

func DialSCTPExt(network string, laddr, raddr *SCTPAddr, options InitMsg) (*SCTPConn, error) {
	return nil, ErrUnsupported
}
