package client

import (
	"crypto/tls"
	"fmt"
	"io"
	"net"
	"sync"
	"time"

	"github.com/hashicorp/net-rpc-msgpackrpc"
	"github.com/hashicorp/yamux"
)

const (
	// clientPreamble is the preamble to send before upgrading
	// the connection into a SCADA version 1 connection.
	clientPreamble = "SCADA 1\n"

	// rpcTimeout is how long of a read deadline we provide
	rpcTimeout = 10 * time.Second
)

// Opts is used to parameterize a Dial
type Opts struct {
	// Addr is the dial address
	Addr string

	// TLS controls if TLS is used
	TLS bool

	// TLSConfig or nil for default
	TLSConfig *tls.Config

	// Modifies the log output
	LogOutput io.Writer
}

// Client is a SCADA compatible client. This is a bare bones client that
// only handles the framing and RPC protocol. Higher-level clients should
// be prefered.
type Client struct {
	conn   net.Conn
	client *yamux.Session

	closed     bool
	closedLock sync.Mutex
}

// Dial is used to establish a new connection over TCP
func Dial(addr string) (*Client, error) {
	opts := Opts{Addr: addr, TLS: false}
	return DialOpts(&opts)
}

// DialTLS is used to establish a new connection using TLS/TCP
func DialTLS(addr string, tlsConf *tls.Config) (*Client, error) {
	opts := Opts{Addr: addr, TLS: true, TLSConfig: tlsConf}
	return DialOpts(&opts)
}

// DialOpts is a parameterized Dial
func DialOpts(opts *Opts) (*Client, error) {
	var conn net.Conn
	var err error
	if opts.TLS {
		conn, err = tls.Dial("tcp", opts.Addr, opts.TLSConfig)
	} else {
		conn, err = net.DialTimeout("tcp", opts.Addr, 10*time.Second)
	}
	if err != nil {
		return nil, err
	}
	return initClient(conn, opts)
}

// initClient does the common initialization
func initClient(conn net.Conn, opts *Opts) (*Client, error) {
	// Send the preamble
	_, err := conn.Write([]byte(clientPreamble))
	if err != nil {
		return nil, fmt.Errorf("preamble write failed: %v", err)
	}

	// Wrap the connection in yamux for multiplexing
	ymConf := yamux.DefaultConfig()
	if opts.LogOutput != nil {
		ymConf.LogOutput = opts.LogOutput
	}
	client, _ := yamux.Client(conn, ymConf)

	// Create the client
	c := &Client{
		conn:   conn,
		client: client,
	}
	return c, nil
}

// Close is used to terminate the client connection
func (c *Client) Close() error {
	c.closedLock.Lock()
	defer c.closedLock.Unlock()

	if c.closed {
		return nil
	}
	c.closed = true
	c.client.GoAway() // Notify the other side of the close
	return c.client.Close()
}

// RPC is used to perform an RPC
func (c *Client) RPC(method string, args interface{}, resp interface{}) error {
	// Get a stream
	stream, err := c.Open()
	if err != nil {
		return fmt.Errorf("failed to open stream: %v", err)
	}
	defer stream.Close()
	stream.SetDeadline(time.Now().Add(rpcTimeout))

	// Create the RPC client
	cc := msgpackrpc.NewCodec(true, true, stream)
	return msgpackrpc.CallWithCodec(cc, method, args, resp)
}

// Accept is used to accept an incoming connection
func (c *Client) Accept() (net.Conn, error) {
	return c.client.Accept()
}

// Open is used to open an outgoing connection
func (c *Client) Open() (net.Conn, error) {
	return c.client.Open()
}

// Addr is so that client can act like a net.Listener
func (c *Client) Addr() net.Addr {
	return c.client.LocalAddr()
}

// NumStreams returns the number of open streams on the client
func (c *Client) NumStreams() int {
	return c.client.NumStreams()
}
