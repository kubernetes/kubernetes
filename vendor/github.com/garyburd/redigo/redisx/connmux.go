// Copyright 2014 Gary Burd
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package redisx

import (
	"errors"
	"sync"

	"github.com/garyburd/redigo/internal"
	"github.com/garyburd/redigo/redis"
)

// ConnMux multiplexes one or more connections to a single underlying
// connection. The ConnMux connections do not support concurrency, commands
// that associate server side state with the connection or commands that put
// the connection in a special mode.
type ConnMux struct {
	c redis.Conn

	sendMu sync.Mutex
	sendID uint

	recvMu   sync.Mutex
	recvID   uint
	recvWait map[uint]chan struct{}
}

func NewConnMux(c redis.Conn) *ConnMux {
	return &ConnMux{c: c, recvWait: make(map[uint]chan struct{})}
}

// Get gets a connection. The application must close the returned connection.
func (p *ConnMux) Get() redis.Conn {
	c := &muxConn{p: p}
	c.ids = c.buf[:0]
	return c
}

// Close closes the underlying connection.
func (p *ConnMux) Close() error {
	return p.c.Close()
}

type muxConn struct {
	p   *ConnMux
	ids []uint
	buf [8]uint
}

func (c *muxConn) send(flush bool, cmd string, args ...interface{}) error {
	if internal.LookupCommandInfo(cmd).Set != 0 {
		return errors.New("command not supported by mux pool")
	}
	p := c.p
	p.sendMu.Lock()
	id := p.sendID
	c.ids = append(c.ids, id)
	p.sendID++
	err := p.c.Send(cmd, args...)
	if flush {
		err = p.c.Flush()
	}
	p.sendMu.Unlock()
	return err
}

func (c *muxConn) Send(cmd string, args ...interface{}) error {
	return c.send(false, cmd, args...)
}

func (c *muxConn) Flush() error {
	p := c.p
	p.sendMu.Lock()
	err := p.c.Flush()
	p.sendMu.Unlock()
	return err
}

func (c *muxConn) Receive() (interface{}, error) {
	if len(c.ids) == 0 {
		return nil, errors.New("mux pool underflow")
	}

	id := c.ids[0]
	c.ids = c.ids[1:]
	if len(c.ids) == 0 {
		c.ids = c.buf[:0]
	}

	p := c.p
	p.recvMu.Lock()
	if p.recvID != id {
		ch := make(chan struct{})
		p.recvWait[id] = ch
		p.recvMu.Unlock()
		<-ch
		p.recvMu.Lock()
		if p.recvID != id {
			panic("out of sync")
		}
	}

	v, err := p.c.Receive()

	id++
	p.recvID = id
	ch, ok := p.recvWait[id]
	if ok {
		delete(p.recvWait, id)
	}
	p.recvMu.Unlock()
	if ok {
		ch <- struct{}{}
	}

	return v, err
}

func (c *muxConn) Close() error {
	var err error
	if len(c.ids) == 0 {
		return nil
	}
	c.Flush()
	for _ = range c.ids {
		_, err = c.Receive()
	}
	return err
}

func (c *muxConn) Do(cmd string, args ...interface{}) (interface{}, error) {
	if err := c.send(true, cmd, args...); err != nil {
		return nil, err
	}
	return c.Receive()
}

func (c *muxConn) Err() error {
	return c.p.c.Err()
}
