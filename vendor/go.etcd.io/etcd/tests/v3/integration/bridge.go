// Copyright 2016 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package integration

import (
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"sync"

	"go.etcd.io/etcd/client/pkg/v3/transport"
)

// bridge creates a unix socket bridge to another unix socket, making it possible
// to disconnect grpc network connections without closing the logical grpc connection.
type bridge struct {
	inaddr  string
	outaddr string
	l       net.Listener
	conns   map[*bridgeConn]struct{}

	stopc      chan struct{}
	pausec     chan struct{}
	blackholec chan struct{}
	wg         sync.WaitGroup

	mu sync.Mutex
}

func newBridge(addr string) (*bridge, error) {
	b := &bridge{
		// bridge "port" is ("%05d%05d0", port, pid) since go1.8 expects the port to be a number
		inaddr:     addr + "0",
		outaddr:    addr,
		conns:      make(map[*bridgeConn]struct{}),
		stopc:      make(chan struct{}),
		pausec:     make(chan struct{}),
		blackholec: make(chan struct{}),
	}
	close(b.pausec)

	l, err := transport.NewUnixListener(b.inaddr)
	if err != nil {
		return nil, fmt.Errorf("listen failed on socket %s (%v)", addr, err)
	}
	b.l = l
	b.wg.Add(1)
	go b.serveListen()
	return b, nil
}

func (b *bridge) URL() string { return "unix://" + b.inaddr }

func (b *bridge) Close() {
	b.l.Close()
	b.mu.Lock()
	select {
	case <-b.stopc:
	default:
		close(b.stopc)
	}
	b.mu.Unlock()
	b.wg.Wait()
}

func (b *bridge) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()
	for bc := range b.conns {
		bc.Close()
	}
	b.conns = make(map[*bridgeConn]struct{})
}

func (b *bridge) Pause() {
	b.mu.Lock()
	b.pausec = make(chan struct{})
	b.mu.Unlock()
}

func (b *bridge) Unpause() {
	b.mu.Lock()
	select {
	case <-b.pausec:
	default:
		close(b.pausec)
	}
	b.mu.Unlock()
}

func (b *bridge) serveListen() {
	defer func() {
		b.l.Close()
		b.mu.Lock()
		for bc := range b.conns {
			bc.Close()
		}
		b.mu.Unlock()
		b.wg.Done()
	}()

	for {
		inc, ierr := b.l.Accept()
		if ierr != nil {
			return
		}
		b.mu.Lock()
		pausec := b.pausec
		b.mu.Unlock()
		select {
		case <-b.stopc:
			inc.Close()
			return
		case <-pausec:
		}

		outc, oerr := net.Dial("unix", b.outaddr)
		if oerr != nil {
			inc.Close()
			return
		}

		bc := &bridgeConn{inc, outc, make(chan struct{})}
		b.wg.Add(1)
		b.mu.Lock()
		b.conns[bc] = struct{}{}
		go b.serveConn(bc)
		b.mu.Unlock()
	}
}

func (b *bridge) serveConn(bc *bridgeConn) {
	defer func() {
		close(bc.donec)
		bc.Close()
		b.mu.Lock()
		delete(b.conns, bc)
		b.mu.Unlock()
		b.wg.Done()
	}()

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		b.ioCopy(bc.out, bc.in)
		bc.close()
		wg.Done()
	}()
	go func() {
		b.ioCopy(bc.in, bc.out)
		bc.close()
		wg.Done()
	}()
	wg.Wait()
}

type bridgeConn struct {
	in    net.Conn
	out   net.Conn
	donec chan struct{}
}

func (bc *bridgeConn) Close() {
	bc.close()
	<-bc.donec
}

func (bc *bridgeConn) close() {
	bc.in.Close()
	bc.out.Close()
}

func (b *bridge) Blackhole() {
	b.mu.Lock()
	close(b.blackholec)
	b.mu.Unlock()
}

func (b *bridge) Unblackhole() {
	b.mu.Lock()
	for bc := range b.conns {
		bc.Close()
	}
	b.conns = make(map[*bridgeConn]struct{})
	b.blackholec = make(chan struct{})
	b.mu.Unlock()
}

// ref. https://github.com/golang/go/blob/master/src/io/io.go copyBuffer
func (b *bridge) ioCopy(dst io.Writer, src io.Reader) (err error) {
	buf := make([]byte, 32*1024)
	for {
		select {
		case <-b.blackholec:
			io.Copy(ioutil.Discard, src)
			return nil
		default:
		}
		nr, er := src.Read(buf)
		if nr > 0 {
			nw, ew := dst.Write(buf[0:nr])
			if ew != nil {
				return ew
			}
			if nr != nw {
				return io.ErrShortWrite
			}
		}
		if er != nil {
			err = er
			break
		}
	}
	return err
}
