package sftp

import (
	"encoding"
	"io"
	"sync"

	"github.com/pkg/errors"
)

// conn implements a bidirectional channel on which client and server
// connections are multiplexed.
type conn struct {
	io.Reader
	io.WriteCloser
	sync.Mutex // used to serialise writes to sendPacket
}

func (c *conn) recvPacket() (uint8, []byte, error) {
	return recvPacket(c)
}

func (c *conn) sendPacket(m encoding.BinaryMarshaler) error {
	c.Lock()
	defer c.Unlock()
	return sendPacket(c, m)
}

type clientConn struct {
	conn
	wg         sync.WaitGroup
	sync.Mutex                          // protects inflight
	inflight   map[uint32]chan<- result // outstanding requests
}

// Close closes the SFTP session.
func (c *clientConn) Close() error {
	defer c.wg.Wait()
	return c.conn.Close()
}

func (c *clientConn) loop() {
	defer c.wg.Done()
	err := c.recv()
	if err != nil {
		c.broadcastErr(err)
	}
}

// recv continuously reads from the server and forwards responses to the
// appropriate channel.
func (c *clientConn) recv() error {
	defer c.conn.Close()
	for {
		typ, data, err := c.recvPacket()
		if err != nil {
			return err
		}
		sid, _ := unmarshalUint32(data)
		c.Lock()
		ch, ok := c.inflight[sid]
		delete(c.inflight, sid)
		c.Unlock()
		if !ok {
			// This is an unexpected occurrence. Send the error
			// back to all listeners so that they terminate
			// gracefully.
			return errors.Errorf("sid: %v not fond", sid)
		}
		ch <- result{typ: typ, data: data}
	}
}

// result captures the result of receiving the a packet from the server
type result struct {
	typ  byte
	data []byte
	err  error
}

type idmarshaler interface {
	id() uint32
	encoding.BinaryMarshaler
}

func (c *clientConn) sendPacket(p idmarshaler) (byte, []byte, error) {
	ch := make(chan result, 1)
	c.dispatchRequest(ch, p)
	s := <-ch
	return s.typ, s.data, s.err
}

func (c *clientConn) dispatchRequest(ch chan<- result, p idmarshaler) {
	c.Lock()
	c.inflight[p.id()] = ch
	if err := c.conn.sendPacket(p); err != nil {
		delete(c.inflight, p.id())
		ch <- result{err: err}
	}
	c.Unlock()
}

// broadcastErr sends an error to all goroutines waiting for a response.
func (c *clientConn) broadcastErr(err error) {
	c.Lock()
	listeners := make([]chan<- result, 0, len(c.inflight))
	for _, ch := range c.inflight {
		listeners = append(listeners, ch)
	}
	c.Unlock()
	for _, ch := range listeners {
		ch <- result{err: err}
	}
}

type serverConn struct {
	conn
}

func (s *serverConn) sendError(p id, err error) error {
	return s.sendPacket(statusFromError(p, err))
}
