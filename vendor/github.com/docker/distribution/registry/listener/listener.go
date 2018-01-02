package listener

import (
	"fmt"
	"net"
	"os"
	"time"
)

// tcpKeepAliveListener sets TCP keep-alive timeouts on accepted
// connections. It's used by ListenAndServe and ListenAndServeTLS so
// dead TCP connections (e.g. closing laptop mid-download) eventually
// go away.
// it is a plain copy-paste from net/http/server.go
type tcpKeepAliveListener struct {
	*net.TCPListener
}

func (ln tcpKeepAliveListener) Accept() (c net.Conn, err error) {
	tc, err := ln.AcceptTCP()
	if err != nil {
		return
	}
	tc.SetKeepAlive(true)
	tc.SetKeepAlivePeriod(3 * time.Minute)
	return tc, nil
}

// NewListener announces on laddr and net. Accepted values of the net are
// 'unix' and 'tcp'
func NewListener(net, laddr string) (net.Listener, error) {
	switch net {
	case "unix":
		return newUnixListener(laddr)
	case "tcp", "": // an empty net means tcp
		return newTCPListener(laddr)
	default:
		return nil, fmt.Errorf("unknown address type %s", net)
	}
}

func newUnixListener(laddr string) (net.Listener, error) {
	fi, err := os.Stat(laddr)
	if err == nil {
		// the file exists.
		// try to remove it if it's a socket
		if !isSocket(fi.Mode()) {
			return nil, fmt.Errorf("file %s exists and is not a socket", laddr)
		}

		if err := os.Remove(laddr); err != nil {
			return nil, err
		}
	} else if !os.IsNotExist(err) {
		// we can't do stat on the file.
		// it means we can not remove it
		return nil, err
	}

	return net.Listen("unix", laddr)
}

func isSocket(m os.FileMode) bool {
	return m&os.ModeSocket != 0
}

func newTCPListener(laddr string) (net.Listener, error) {
	ln, err := net.Listen("tcp", laddr)
	if err != nil {
		return nil, err
	}

	return tcpKeepAliveListener{ln.(*net.TCPListener)}, nil
}
