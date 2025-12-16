package websocket

import (
	"context"
	"fmt"
	"io"
	"math"
	"net"
	"sync/atomic"
	"time"
)

// NetConn converts a *websocket.Conn into a net.Conn.
//
// It's for tunneling arbitrary protocols over WebSockets.
// Few users of the library will need this but it's tricky to implement
// correctly and so provided in the library.
// See https://github.com/nhooyr/websocket/issues/100.
//
// Every Write to the net.Conn will correspond to a message write of
// the given type on *websocket.Conn.
//
// The passed ctx bounds the lifetime of the net.Conn. If cancelled,
// all reads and writes on the net.Conn will be cancelled.
//
// If a message is read that is not of the correct type, the connection
// will be closed with StatusUnsupportedData and an error will be returned.
//
// Close will close the *websocket.Conn with StatusNormalClosure.
//
// When a deadline is hit and there is an active read or write goroutine, the
// connection will be closed. This is different from most net.Conn implementations
// where only the reading/writing goroutines are interrupted but the connection
// is kept alive.
//
// The Addr methods will return the real addresses for connections obtained
// from websocket.Accept. But for connections obtained from websocket.Dial, a mock net.Addr
// will be returned that gives "websocket" for Network() and "websocket/unknown-addr" for
// String(). This is because websocket.Dial only exposes a io.ReadWriteCloser instead of the
// full net.Conn to us.
//
// When running as WASM, the Addr methods will always return the mock address described above.
//
// A received StatusNormalClosure or StatusGoingAway close frame will be translated to
// io.EOF when reading.
//
// Furthermore, the ReadLimit is set to -1 to disable it.
func NetConn(ctx context.Context, c *Conn, msgType MessageType) net.Conn {
	c.SetReadLimit(-1)

	nc := &netConn{
		c:       c,
		msgType: msgType,
		readMu:  newMu(c),
		writeMu: newMu(c),
	}

	nc.writeCtx, nc.writeCancel = context.WithCancel(ctx)
	nc.readCtx, nc.readCancel = context.WithCancel(ctx)

	nc.writeTimer = time.AfterFunc(math.MaxInt64, func() {
		if !nc.writeMu.tryLock() {
			// If the lock cannot be acquired, then there is an
			// active write goroutine and so we should cancel the context.
			nc.writeCancel()
			return
		}
		defer nc.writeMu.unlock()

		// Prevents future writes from writing until the deadline is reset.
		nc.writeExpired.Store(1)
	})
	if !nc.writeTimer.Stop() {
		<-nc.writeTimer.C
	}

	nc.readTimer = time.AfterFunc(math.MaxInt64, func() {
		if !nc.readMu.tryLock() {
			// If the lock cannot be acquired, then there is an
			// active read goroutine and so we should cancel the context.
			nc.readCancel()
			return
		}
		defer nc.readMu.unlock()

		// Prevents future reads from reading until the deadline is reset.
		nc.readExpired.Store(1)
	})
	if !nc.readTimer.Stop() {
		<-nc.readTimer.C
	}

	return nc
}

type netConn struct {
	c       *Conn
	msgType MessageType

	writeTimer   *time.Timer
	writeMu      *mu
	writeExpired atomic.Int64
	writeCtx     context.Context
	writeCancel  context.CancelFunc

	readTimer   *time.Timer
	readMu      *mu
	readExpired atomic.Int64
	readCtx     context.Context
	readCancel  context.CancelFunc
	readEOFed   bool
	reader      io.Reader
}

var _ net.Conn = &netConn{}

func (nc *netConn) Close() error {
	nc.writeTimer.Stop()
	nc.writeCancel()
	nc.readTimer.Stop()
	nc.readCancel()
	return nc.c.Close(StatusNormalClosure, "")
}

func (nc *netConn) Write(p []byte) (int, error) {
	nc.writeMu.forceLock()
	defer nc.writeMu.unlock()

	if nc.writeExpired.Load() == 1 {
		return 0, fmt.Errorf("failed to write: %w", context.DeadlineExceeded)
	}

	err := nc.c.Write(nc.writeCtx, nc.msgType, p)
	if err != nil {
		return 0, err
	}
	return len(p), nil
}

func (nc *netConn) Read(p []byte) (int, error) {
	nc.readMu.forceLock()
	defer nc.readMu.unlock()

	for {
		n, err := nc.read(p)
		if err != nil {
			return n, err
		}
		if n == 0 {
			continue
		}
		return n, nil
	}
}

func (nc *netConn) read(p []byte) (int, error) {
	if nc.readExpired.Load() == 1 {
		return 0, fmt.Errorf("failed to read: %w", context.DeadlineExceeded)
	}

	if nc.readEOFed {
		return 0, io.EOF
	}

	if nc.reader == nil {
		typ, r, err := nc.c.Reader(nc.readCtx)
		if err != nil {
			switch CloseStatus(err) {
			case StatusNormalClosure, StatusGoingAway:
				nc.readEOFed = true
				return 0, io.EOF
			}
			return 0, err
		}
		if typ != nc.msgType {
			err := fmt.Errorf("unexpected frame type read (expected %v): %v", nc.msgType, typ)
			nc.c.Close(StatusUnsupportedData, err.Error())
			return 0, err
		}
		nc.reader = r
	}

	n, err := nc.reader.Read(p)
	if err == io.EOF {
		nc.reader = nil
		err = nil
	}
	return n, err
}

type websocketAddr struct{}

func (a websocketAddr) Network() string {
	return "websocket"
}

func (a websocketAddr) String() string {
	return "websocket/unknown-addr"
}

func (nc *netConn) SetDeadline(t time.Time) error {
	nc.SetWriteDeadline(t)
	nc.SetReadDeadline(t)
	return nil
}

func (nc *netConn) SetWriteDeadline(t time.Time) error {
	nc.writeExpired.Store(0)
	if t.IsZero() {
		nc.writeTimer.Stop()
	} else {
		dur := time.Until(t)
		if dur <= 0 {
			dur = 1
		}
		nc.writeTimer.Reset(dur)
	}
	return nil
}

func (nc *netConn) SetReadDeadline(t time.Time) error {
	nc.readExpired.Store(0)
	if t.IsZero() {
		nc.readTimer.Stop()
	} else {
		dur := time.Until(t)
		if dur <= 0 {
			dur = 1
		}
		nc.readTimer.Reset(dur)
	}
	return nil
}
