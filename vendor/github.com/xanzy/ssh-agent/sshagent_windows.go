//
// Copyright (c) 2014 David Mzareulyan
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge, publish, distribute,
// sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or substantial
// portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
//

// +build windows

package sshagent

import (
	"errors"
	"io"
	"net"
	"sync"

	"golang.org/x/crypto/ssh/agent"
)

// New returns a new agent.Agent and the (custom) connection it uses
// to communicate with a running pagent.exe instance (see README.md)
func New() (agent.Agent, net.Conn, error) {
	if !Available() {
		return nil, nil, errors.New("SSH agent requested but Pageant not running")
	}

	return agent.NewClient(&conn{}), nil, nil
}

type conn struct {
	sync.Mutex
	buf []byte
}

func (c *conn) Close() {
	c.Lock()
	defer c.Unlock()
	c.buf = nil
}

func (c *conn) Write(p []byte) (int, error) {
	c.Lock()
	defer c.Unlock()

	resp, err := query(p)
	if err != nil {
		return 0, err
	}

	c.buf = append(c.buf, resp...)

	return len(p), nil
}

func (c *conn) Read(p []byte) (int, error) {
	c.Lock()
	defer c.Unlock()

	if len(c.buf) == 0 {
		return 0, io.EOF
	}

	n := copy(p, c.buf)
	c.buf = c.buf[n:]

	return n, nil
}
