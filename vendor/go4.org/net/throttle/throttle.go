/*
Copyright 2012 Google Inc.

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

// Package throttle provides a net.Listener that returns
// artificially-delayed connections for testing real-world
// connectivity.
package throttle // import "go4.org/net/throttle"

import (
	"fmt"
	"net"
	"sync"
	"time"
)

const unitSize = 1400 // read/write chunk size. ~MTU size.

type Rate struct {
	KBps    int // or 0, to not rate-limit bandwidth
	Latency time.Duration
}

// byteTime returns the time required for n bytes.
func (r Rate) byteTime(n int) time.Duration {
	if r.KBps == 0 {
		return 0
	}
	return time.Duration(float64(n)/1024/float64(r.KBps)) * time.Second
}

type Listener struct {
	net.Listener
	Down Rate // server Writes to Client
	Up   Rate // server Reads from client
}

func (ln *Listener) Accept() (net.Conn, error) {
	c, err := ln.Listener.Accept()
	time.Sleep(ln.Up.Latency)
	if err != nil {
		return nil, err
	}
	tc := &conn{Conn: c, Down: ln.Down, Up: ln.Up}
	tc.start()
	return tc, nil
}

type nErr struct {
	n   int
	err error
}

type writeReq struct {
	writeAt time.Time
	p       []byte
	resc    chan nErr
}

type conn struct {
	net.Conn
	Down Rate // for reads
	Up   Rate // for writes

	wchan     chan writeReq
	closeOnce sync.Once
	closeErr  error
}

func (c *conn) start() {
	c.wchan = make(chan writeReq, 1024)
	go c.writeLoop()
}

func (c *conn) writeLoop() {
	for req := range c.wchan {
		time.Sleep(req.writeAt.Sub(time.Now()))
		var res nErr
		for len(req.p) > 0 && res.err == nil {
			writep := req.p
			if len(writep) > unitSize {
				writep = writep[:unitSize]
			}
			n, err := c.Conn.Write(writep)
			time.Sleep(c.Up.byteTime(len(writep)))
			res.n += n
			res.err = err
			req.p = req.p[n:]
		}
		req.resc <- res
	}
}

func (c *conn) Close() error {
	c.closeOnce.Do(func() {
		err := c.Conn.Close()
		close(c.wchan)
		c.closeErr = err
	})
	return c.closeErr
}

func (c *conn) Write(p []byte) (n int, err error) {
	defer func() {
		if e := recover(); e != nil {
			n = 0
			err = fmt.Errorf("%v", err)
			return
		}
	}()
	resc := make(chan nErr, 1)
	c.wchan <- writeReq{time.Now().Add(c.Up.Latency), p, resc}
	res := <-resc
	return res.n, res.err
}

func (c *conn) Read(p []byte) (n int, err error) {
	const max = 1024
	if len(p) > max {
		p = p[:max]
	}
	n, err = c.Conn.Read(p)
	time.Sleep(c.Down.byteTime(n))
	return
}
