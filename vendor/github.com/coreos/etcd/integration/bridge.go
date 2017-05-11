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
	"net"
	"sync"

	"github.com/coreos/etcd/pkg/transport"
)

// bridge creates a unix socket bridge to another unix socket, making it possible
// to disconnect grpc network connections without closing the logical grpc connection.
type bridge struct {
	inaddr  string
	outaddr string
	l       net.Listener
	conns   map[*bridgeConn]struct{}

	stopc chan struct{}
	wg    sync.WaitGroup

	mu sync.Mutex
}

func newBridge(addr string) (*bridge, error) {
	b := &bridge{
		// bridge "port" is ("%05d%05d0", port, pid) since go1.8 expects the port to be a number
		inaddr:  addr + "0",
		outaddr: addr,
		conns:   make(map[*bridgeConn]struct{}),
		stopc:   make(chan struct{}, 1),
	}
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
	select {
	case b.stopc <- struct{}{}:
	default:
	}
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
		outc, oerr := net.Dial("unix", b.outaddr)
		if oerr != nil {
			inc.Close()
			return
		}

		bc := &bridgeConn{inc, outc}
		b.wg.Add(1)
		b.mu.Lock()
		b.conns[bc] = struct{}{}
		go b.serveConn(bc)
		b.mu.Unlock()
	}
}

func (b *bridge) serveConn(bc *bridgeConn) {
	defer func() {
		bc.Close()
		b.mu.Lock()
		delete(b.conns, bc)
		b.mu.Unlock()
		b.wg.Done()
	}()

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		io.Copy(bc.out, bc.in)
		wg.Done()
	}()
	go func() {
		io.Copy(bc.in, bc.out)
		wg.Done()
	}()
	wg.Wait()
}

type bridgeConn struct {
	in  net.Conn
	out net.Conn
}

func (bc *bridgeConn) Close() {
	bc.in.Close()
	bc.out.Close()
}
