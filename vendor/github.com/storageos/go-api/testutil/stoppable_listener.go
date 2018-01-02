package testutil

import (
	"net"
	"sync"
)

// StoppableListener - stoppable listener for testing purposes
type StoppableListener struct {
	net.Listener
	Stop      chan bool // Any message to this channel will gracefully stop the server
	Stopped   bool      // True if the server was stopped gracefully
	ConnCount counter   // Number of active client connections
}

type watchedConn struct {
	net.Conn
	connCount *counter
}

type counter struct {
	sync.Mutex
	c int
}

// Handle - stoppable listener
func Handle(l net.Listener) (sl *StoppableListener) {
	sl = &StoppableListener{Listener: l, Stop: make(chan bool, 1)}

	// Start a goroutine that will wait until the stop channel
	// receives a message then close the Listener to stop accepting
	// new connections (but continue to server the established ones)
	go func() {
		<-sl.Stop
		sl.Stopped = true
		sl.Listener.Close()
	}()
	return
}

func (sl *StoppableListener) Accept() (c net.Conn, err error) {
	c, err = sl.Listener.Accept()
	if err != nil {
		return
	}

	// Wrap the returned connection so we're able to observe
	// when it is closed
	c = watchedConn{Conn: c, connCount: &sl.ConnCount}

	// Count it
	sl.ConnCount.Lock()
	sl.ConnCount.c++
	sl.ConnCount.Unlock()
	return
}

func (c *counter) Get() int {
	c.Lock()
	defer c.Unlock()
	return c.c
}

func (w watchedConn) Close() error {
	w.connCount.Lock()
	w.connCount.c--
	w.connCount.Unlock()
	return w.Conn.Close()
}
