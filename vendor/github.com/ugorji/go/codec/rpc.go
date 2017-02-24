// Copyright (c) 2012-2015 Ugorji Nwoke. All rights reserved.
// Use of this source code is governed by a MIT license found in the LICENSE file.

package codec

import (
	"bufio"
	"io"
	"net/rpc"
	"sync"
)

// rpcEncodeTerminator allows a handler specify a []byte terminator to send after each Encode.
//
// Some codecs like json need to put a space after each encoded value, to serve as a
// delimiter for things like numbers (else json codec will continue reading till EOF).
type rpcEncodeTerminator interface {
	rpcEncodeTerminate() []byte
}

// Rpc provides a rpc Server or Client Codec for rpc communication.
type Rpc interface {
	ServerCodec(conn io.ReadWriteCloser, h Handle) rpc.ServerCodec
	ClientCodec(conn io.ReadWriteCloser, h Handle) rpc.ClientCodec
}

// RpcCodecBuffered allows access to the underlying bufio.Reader/Writer
// used by the rpc connection. It accommodates use-cases where the connection
// should be used by rpc and non-rpc functions, e.g. streaming a file after
// sending an rpc response.
type RpcCodecBuffered interface {
	BufferedReader() *bufio.Reader
	BufferedWriter() *bufio.Writer
}

// -------------------------------------

// rpcCodec defines the struct members and common methods.
type rpcCodec struct {
	rwc io.ReadWriteCloser
	dec *Decoder
	enc *Encoder
	bw  *bufio.Writer
	br  *bufio.Reader
	mu  sync.Mutex
	h   Handle

	cls   bool
	clsmu sync.RWMutex
}

func newRPCCodec(conn io.ReadWriteCloser, h Handle) rpcCodec {
	bw := bufio.NewWriter(conn)
	br := bufio.NewReader(conn)
	return rpcCodec{
		rwc: conn,
		bw:  bw,
		br:  br,
		enc: NewEncoder(bw, h),
		dec: NewDecoder(br, h),
		h:   h,
	}
}

func (c *rpcCodec) BufferedReader() *bufio.Reader {
	return c.br
}

func (c *rpcCodec) BufferedWriter() *bufio.Writer {
	return c.bw
}

func (c *rpcCodec) write(obj1, obj2 interface{}, writeObj2, doFlush bool) (err error) {
	if c.isClosed() {
		return io.EOF
	}
	if err = c.enc.Encode(obj1); err != nil {
		return
	}
	t, tOk := c.h.(rpcEncodeTerminator)
	if tOk {
		c.bw.Write(t.rpcEncodeTerminate())
	}
	if writeObj2 {
		if err = c.enc.Encode(obj2); err != nil {
			return
		}
		if tOk {
			c.bw.Write(t.rpcEncodeTerminate())
		}
	}
	if doFlush {
		return c.bw.Flush()
	}
	return
}

func (c *rpcCodec) read(obj interface{}) (err error) {
	if c.isClosed() {
		return io.EOF
	}
	//If nil is passed in, we should still attempt to read content to nowhere.
	if obj == nil {
		var obj2 interface{}
		return c.dec.Decode(&obj2)
	}
	return c.dec.Decode(obj)
}

func (c *rpcCodec) isClosed() bool {
	c.clsmu.RLock()
	x := c.cls
	c.clsmu.RUnlock()
	return x
}

func (c *rpcCodec) Close() error {
	if c.isClosed() {
		return io.EOF
	}
	c.clsmu.Lock()
	c.cls = true
	c.clsmu.Unlock()
	return c.rwc.Close()
}

func (c *rpcCodec) ReadResponseBody(body interface{}) error {
	return c.read(body)
}

// -------------------------------------

type goRpcCodec struct {
	rpcCodec
}

func (c *goRpcCodec) WriteRequest(r *rpc.Request, body interface{}) error {
	// Must protect for concurrent access as per API
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.write(r, body, true, true)
}

func (c *goRpcCodec) WriteResponse(r *rpc.Response, body interface{}) error {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.write(r, body, true, true)
}

func (c *goRpcCodec) ReadResponseHeader(r *rpc.Response) error {
	return c.read(r)
}

func (c *goRpcCodec) ReadRequestHeader(r *rpc.Request) error {
	return c.read(r)
}

func (c *goRpcCodec) ReadRequestBody(body interface{}) error {
	return c.read(body)
}

// -------------------------------------

// goRpc is the implementation of Rpc that uses the communication protocol
// as defined in net/rpc package.
type goRpc struct{}

// GoRpc implements Rpc using the communication protocol defined in net/rpc package.
// Its methods (ServerCodec and ClientCodec) return values that implement RpcCodecBuffered.
var GoRpc goRpc

func (x goRpc) ServerCodec(conn io.ReadWriteCloser, h Handle) rpc.ServerCodec {
	return &goRpcCodec{newRPCCodec(conn, h)}
}

func (x goRpc) ClientCodec(conn io.ReadWriteCloser, h Handle) rpc.ClientCodec {
	return &goRpcCodec{newRPCCodec(conn, h)}
}

var _ RpcCodecBuffered = (*rpcCodec)(nil) // ensure *rpcCodec implements RpcCodecBuffered
