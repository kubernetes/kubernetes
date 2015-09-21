package ws

import (
	"github.com/gorilla/websocket"
	"io"
	"log"
	"time"
)

// Wrap an HTTP2 connection over WebSockets and
// use the underlying WebSocket framing for proxy
// compatibility.
type Conn struct {
	*websocket.Conn
	reader io.Reader
}

func NewConnection(w *websocket.Conn) *Conn {
	return &Conn{Conn: w}
}

func (c Conn) Write(b []byte) (int, error) {
	err := c.WriteMessage(websocket.BinaryMessage, b)
	if err != nil {
		return 0, err
	}
	return len(b), nil
}

func (c Conn) Read(b []byte) (int, error) {
	if c.reader == nil {
		t, r, err := c.NextReader()
		if err != nil {
			return 0, err
		}
		if t != websocket.BinaryMessage {
			log.Printf("ws: ignored non-binary message in stream")
			return 0, nil
		}
		c.reader = r
	}
	n, err := c.reader.Read(b)
	if err != nil {
		if err == io.EOF {
			c.reader = nil
		}
		return n, err
	}
	return n, nil
}

func (c Conn) SetDeadline(t time.Time) error {
	if err := c.Conn.SetReadDeadline(t); err != nil {
		return err
	}
	if err := c.Conn.SetWriteDeadline(t); err != nil {
		return err
	}
	return nil
}

func (c Conn) Close() error {
	err := c.Conn.Close()
	return err
}
