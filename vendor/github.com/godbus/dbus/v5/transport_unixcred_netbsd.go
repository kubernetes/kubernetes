package dbus

import "io"

func (t *unixTransport) SendNullByte() error {
	n, _, err := t.UnixConn.WriteMsgUnix([]byte{0}, nil, nil)
	if err != nil {
		return err
	}
	if n != 1 {
		return io.ErrShortWrite
	}
	return nil
}
