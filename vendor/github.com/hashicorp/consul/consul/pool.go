package consul

import (
	"container/list"
	"fmt"
	"io"
	"net"
	"net/rpc"
	"sync"
	"sync/atomic"
	"time"

	"github.com/hashicorp/consul/consul/agent"
	"github.com/hashicorp/consul/tlsutil"
	"github.com/hashicorp/net-rpc-msgpackrpc"
	"github.com/hashicorp/yamux"
)

// muxSession is used to provide an interface for a stream multiplexer.
type muxSession interface {
	Open() (net.Conn, error)
	Close() error
}

// streamClient is used to wrap a stream with an RPC client
type StreamClient struct {
	stream net.Conn
	codec  rpc.ClientCodec
}

func (sc *StreamClient) Close() {
	sc.stream.Close()
	sc.codec.Close()
}

// Conn is a pooled connection to a Consul server
type Conn struct {
	refCount    int32
	shouldClose int32

	addr     net.Addr
	session  muxSession
	lastUsed time.Time
	version  int

	pool *ConnPool

	clients    *list.List
	clientLock sync.Mutex
}

func (c *Conn) Close() error {
	return c.session.Close()
}

// getClient is used to get a cached or new client
func (c *Conn) getClient() (*StreamClient, error) {
	// Check for cached client
	c.clientLock.Lock()
	front := c.clients.Front()
	if front != nil {
		c.clients.Remove(front)
	}
	c.clientLock.Unlock()
	if front != nil {
		return front.Value.(*StreamClient), nil
	}

	// Open a new session
	stream, err := c.session.Open()
	if err != nil {
		return nil, err
	}

	// Create the RPC client
	codec := msgpackrpc.NewClientCodec(stream)

	// Return a new stream client
	sc := &StreamClient{
		stream: stream,
		codec:  codec,
	}
	return sc, nil
}

// returnStream is used when done with a stream
// to allow re-use by a future RPC
func (c *Conn) returnClient(client *StreamClient) {
	didSave := false
	c.clientLock.Lock()
	if c.clients.Len() < c.pool.maxStreams && atomic.LoadInt32(&c.shouldClose) == 0 {
		c.clients.PushFront(client)
		didSave = true

		// If this is a Yamux stream, shrink the internal buffers so that
		// we can GC the idle memory
		if ys, ok := client.stream.(*yamux.Stream); ok {
			ys.Shrink()
		}
	}
	c.clientLock.Unlock()
	if !didSave {
		client.Close()
	}
}

// markForUse does all the bookkeeping required to ready a connection for use.
func (c *Conn) markForUse() {
	c.lastUsed = time.Now()
	atomic.AddInt32(&c.refCount, 1)
}

// ConnPool is used to maintain a connection pool to other
// Consul servers. This is used to reduce the latency of
// RPC requests between servers. It is only used to pool
// connections in the rpcConsul mode. Raft connections
// are pooled separately.
type ConnPool struct {
	sync.Mutex

	// LogOutput is used to control logging
	logOutput io.Writer

	// The maximum time to keep a connection open
	maxTime time.Duration

	// The maximum number of open streams to keep
	maxStreams int

	// Pool maps an address to a open connection
	pool map[string]*Conn

	// limiter is used to throttle the number of connect attempts
	// to a given address. The first thread will attempt a connection
	// and put a channel in here, which all other threads will wait
	// on to close.
	limiter map[string]chan struct{}

	// TLS wrapper
	tlsWrap tlsutil.DCWrapper

	// Used to indicate the pool is shutdown
	shutdown   bool
	shutdownCh chan struct{}
}

// NewPool is used to make a new connection pool
// Maintain at most one connection per host, for up to maxTime.
// Set maxTime to 0 to disable reaping. maxStreams is used to control
// the number of idle streams allowed.
// If TLS settings are provided outgoing connections use TLS.
func NewPool(logOutput io.Writer, maxTime time.Duration, maxStreams int, tlsWrap tlsutil.DCWrapper) *ConnPool {
	pool := &ConnPool{
		logOutput:  logOutput,
		maxTime:    maxTime,
		maxStreams: maxStreams,
		pool:       make(map[string]*Conn),
		limiter:    make(map[string]chan struct{}),
		tlsWrap:    tlsWrap,
		shutdownCh: make(chan struct{}),
	}
	if maxTime > 0 {
		go pool.reap()
	}
	return pool
}

// Shutdown is used to close the connection pool
func (p *ConnPool) Shutdown() error {
	p.Lock()
	defer p.Unlock()

	for _, conn := range p.pool {
		conn.Close()
	}
	p.pool = make(map[string]*Conn)

	if p.shutdown {
		return nil
	}
	p.shutdown = true
	close(p.shutdownCh)
	return nil
}

// acquire will return a pooled connection, if available. Otherwise it will
// wait for an existing connection attempt to finish, if one if in progress,
// and will return that one if it succeeds. If all else fails, it will return a
// newly-created connection and add it to the pool.
func (p *ConnPool) acquire(dc string, addr net.Addr, version int) (*Conn, error) {
	// Check to see if there's a pooled connection available. This is up
	// here since it should the the vastly more common case than the rest
	// of the code here.
	p.Lock()
	c := p.pool[addr.String()]
	if c != nil {
		c.markForUse()
		p.Unlock()
		return c, nil
	}

	// If not (while we are still locked), set up the throttling structure
	// for this address, which will make everyone else wait until our
	// attempt is done.
	var wait chan struct{}
	var ok bool
	if wait, ok = p.limiter[addr.String()]; !ok {
		wait = make(chan struct{})
		p.limiter[addr.String()] = wait
	}
	isLeadThread := !ok
	p.Unlock()

	// If we are the lead thread, make the new connection and then wake
	// everybody else up to see if we got it.
	if isLeadThread {
		c, err := p.getNewConn(dc, addr, version)
		p.Lock()
		delete(p.limiter, addr.String())
		close(wait)
		if err != nil {
			p.Unlock()
			return nil, err
		}

		p.pool[addr.String()] = c
		p.Unlock()
		return c, nil
	}

	// Otherwise, wait for the lead thread to attempt the connection
	// and use what's in the pool at that point.
	select {
	case <-p.shutdownCh:
		return nil, fmt.Errorf("rpc error: shutdown")
	case <-wait:
	}

	// See if the lead thread was able to get us a connection.
	p.Lock()
	if c := p.pool[addr.String()]; c != nil {
		c.markForUse()
		p.Unlock()
		return c, nil
	}

	p.Unlock()
	return nil, fmt.Errorf("rpc error: lead thread didn't get connection")
}

// getNewConn is used to return a new connection
func (p *ConnPool) getNewConn(dc string, addr net.Addr, version int) (*Conn, error) {
	// Try to dial the conn
	conn, err := net.DialTimeout("tcp", addr.String(), 10*time.Second)
	if err != nil {
		return nil, err
	}

	// Cast to TCPConn
	if tcp, ok := conn.(*net.TCPConn); ok {
		tcp.SetKeepAlive(true)
		tcp.SetNoDelay(true)
	}

	// Check if TLS is enabled
	if p.tlsWrap != nil {
		// Switch the connection into TLS mode
		if _, err := conn.Write([]byte{byte(rpcTLS)}); err != nil {
			conn.Close()
			return nil, err
		}

		// Wrap the connection in a TLS client
		tlsConn, err := p.tlsWrap(dc, conn)
		if err != nil {
			conn.Close()
			return nil, err
		}
		conn = tlsConn
	}

	// Switch the multiplexing based on version
	var session muxSession
	if version < 2 {
		conn.Close()
		return nil, fmt.Errorf("cannot make client connection, unsupported protocol version %d", version)
	} else {
		// Write the Consul multiplex byte to set the mode
		if _, err := conn.Write([]byte{byte(rpcMultiplexV2)}); err != nil {
			conn.Close()
			return nil, err
		}

		// Setup the logger
		conf := yamux.DefaultConfig()
		conf.LogOutput = p.logOutput

		// Create a multiplexed session
		session, _ = yamux.Client(conn, conf)
	}

	// Wrap the connection
	c := &Conn{
		refCount: 1,
		addr:     addr,
		session:  session,
		clients:  list.New(),
		lastUsed: time.Now(),
		version:  version,
		pool:     p,
	}
	return c, nil
}

// clearConn is used to clear any cached connection, potentially in response to an error
func (p *ConnPool) clearConn(conn *Conn) {
	// Ensure returned streams are closed
	atomic.StoreInt32(&conn.shouldClose, 1)

	// Clear from the cache
	p.Lock()
	if c, ok := p.pool[conn.addr.String()]; ok && c == conn {
		delete(p.pool, conn.addr.String())
	}
	p.Unlock()

	// Close down immediately if idle
	if refCount := atomic.LoadInt32(&conn.refCount); refCount == 0 {
		conn.Close()
	}
}

// releaseConn is invoked when we are done with a conn to reduce the ref count
func (p *ConnPool) releaseConn(conn *Conn) {
	refCount := atomic.AddInt32(&conn.refCount, -1)
	if refCount == 0 && atomic.LoadInt32(&conn.shouldClose) == 1 {
		conn.Close()
	}
}

// getClient is used to get a usable client for an address and protocol version
func (p *ConnPool) getClient(dc string, addr net.Addr, version int) (*Conn, *StreamClient, error) {
	retries := 0
START:
	// Try to get a conn first
	conn, err := p.acquire(dc, addr, version)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to get conn: %v", err)
	}

	// Get a client
	client, err := conn.getClient()
	if err != nil {
		p.clearConn(conn)
		p.releaseConn(conn)

		// Try to redial, possible that the TCP session closed due to timeout
		if retries == 0 {
			retries++
			goto START
		}
		return nil, nil, fmt.Errorf("failed to start stream: %v", err)
	}
	return conn, client, nil
}

// RPC is used to make an RPC call to a remote host
func (p *ConnPool) RPC(dc string, addr net.Addr, version int, method string, args interface{}, reply interface{}) error {
	// Get a usable client
	conn, sc, err := p.getClient(dc, addr, version)
	if err != nil {
		return fmt.Errorf("rpc error: %v", err)
	}

	// Make the RPC call
	err = msgpackrpc.CallWithCodec(sc.codec, method, args, reply)
	if err != nil {
		sc.Close()
		p.releaseConn(conn)
		return fmt.Errorf("rpc error: %v", err)
	}

	// Done with the connection
	conn.returnClient(sc)
	p.releaseConn(conn)
	return nil
}

// PingConsulServer sends a Status.Ping message to the specified server and
// returns true if healthy, false if an error occurred
func (p *ConnPool) PingConsulServer(s *agent.Server) (bool, error) {
	// Get a usable client
	conn, sc, err := p.getClient(s.Datacenter, s.Addr, s.Version)
	if err != nil {
		return false, err
	}

	// Make the RPC call
	var out struct{}
	err = msgpackrpc.CallWithCodec(sc.codec, "Status.Ping", struct{}{}, &out)
	if err != nil {
		sc.Close()
		p.releaseConn(conn)
		return false, err
	}

	// Done with the connection
	conn.returnClient(sc)
	p.releaseConn(conn)
	return true, nil
}

// Reap is used to close conns open over maxTime
func (p *ConnPool) reap() {
	for {
		// Sleep for a while
		select {
		case <-p.shutdownCh:
			return
		case <-time.After(time.Second):
		}

		// Reap all old conns
		p.Lock()
		var removed []string
		now := time.Now()
		for host, conn := range p.pool {
			// Skip recently used connections
			if now.Sub(conn.lastUsed) < p.maxTime {
				continue
			}

			// Skip connections with active streams
			if atomic.LoadInt32(&conn.refCount) > 0 {
				continue
			}

			// Close the conn
			conn.Close()

			// Remove from pool
			removed = append(removed, host)
		}
		for _, host := range removed {
			delete(p.pool, host)
		}
		p.Unlock()
	}
}
