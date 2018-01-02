// +build !windows

package hack

import "net"

// MalformedHostHeaderOverride is a wrapper to be able
// to overcome the 400 Bad request coming from old docker
// clients that send an invalid Host header.
type MalformedHostHeaderOverride struct {
	net.Listener
}

// MalformedHostHeaderOverrideConn wraps the underlying unix
// connection and keeps track of the first read from http.Server
// which just reads the headers.
type MalformedHostHeaderOverrideConn struct {
	net.Conn
	first bool
}

var closeConnHeader = []byte("\r\nConnection: close\r")

// Read reads the first *read* request from http.Server to inspect
// the Host header. If the Host starts with / then we're talking to
// an old docker client which send an invalid Host header. To not
// error out in http.Server we rewrite the first bytes of the request
// to sanitize the Host header itself.
// In case we're not dealing with old docker clients the data is just passed
// to the server w/o modification.
func (l *MalformedHostHeaderOverrideConn) Read(b []byte) (n int, err error) {
	// http.Server uses a 4k buffer
	if l.first && len(b) == 4096 {
		// This keeps track of the first read from http.Server which just reads
		// the headers
		l.first = false
		// The first read of the connection by http.Server is done limited to
		// DefaultMaxHeaderBytes (usually 1 << 20) + 4096.
		// Here we do the first read which gets us all the http headers to
		// be inspected and modified below.
		c, err := l.Conn.Read(b)
		if err != nil {
			return c, err
		}

		var (
			start, end    int
			firstLineFeed = -1
			buf           []byte
		)
		for i := 0; i <= c-1-7; i++ {
			if b[i] == '\n' && firstLineFeed == -1 {
				firstLineFeed = i
			}
			if b[i] != '\n' {
				continue
			}

			if b[i+1] == '\r' && b[i+2] == '\n' {
				return c, nil
			}

			if b[i+1] != 'H' {
				continue
			}
			if b[i+2] != 'o' {
				continue
			}
			if b[i+3] != 's' {
				continue
			}
			if b[i+4] != 't' {
				continue
			}
			if b[i+5] != ':' {
				continue
			}
			if b[i+6] != ' ' {
				continue
			}
			if b[i+7] != '/' {
				continue
			}
			// ensure clients other than the docker clients do not get this hack
			if i != firstLineFeed {
				return c, nil
			}
			start = i + 7
			// now find where the value ends
			for ii, bbb := range b[start:c] {
				if bbb == '\n' {
					end = start + ii
					break
				}
			}
			buf = make([]byte, 0, c+len(closeConnHeader)-(end-start))
			// strip the value of the host header and
			// inject `Connection: close` to ensure we don't reuse this connection
			buf = append(buf, b[:start]...)
			buf = append(buf, closeConnHeader...)
			buf = append(buf, b[end:c]...)
			copy(b, buf)
			break
		}
		if len(buf) == 0 {
			return c, nil
		}
		return len(buf), nil
	}
	return l.Conn.Read(b)
}

// Accept makes the listener accepts connections and wraps the connection
// in a MalformedHostHeaderOverrideConn initializing first to true.
func (l *MalformedHostHeaderOverride) Accept() (net.Conn, error) {
	c, err := l.Listener.Accept()
	if err != nil {
		return c, err
	}
	return &MalformedHostHeaderOverrideConn{c, true}, nil
}
