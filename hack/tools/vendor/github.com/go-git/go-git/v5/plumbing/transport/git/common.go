// Package git implements the git transport protocol.
package git

import (
	"fmt"
	"io"
	"net"

	"github.com/go-git/go-git/v5/plumbing/format/pktline"
	"github.com/go-git/go-git/v5/plumbing/transport"
	"github.com/go-git/go-git/v5/plumbing/transport/internal/common"
	"github.com/go-git/go-git/v5/utils/ioutil"
)

// DefaultClient is the default git client.
var DefaultClient = common.NewClient(&runner{})

const DefaultPort = 9418

type runner struct{}

// Command returns a new Command for the given cmd in the given Endpoint
func (r *runner) Command(cmd string, ep *transport.Endpoint, auth transport.AuthMethod) (common.Command, error) {
	// auth not allowed since git protocol doesn't support authentication
	if auth != nil {
		return nil, transport.ErrInvalidAuthMethod
	}
	c := &command{command: cmd, endpoint: ep}
	if err := c.connect(); err != nil {
		return nil, err
	}
	return c, nil
}

type command struct {
	conn      net.Conn
	connected bool
	command   string
	endpoint  *transport.Endpoint
}

// Start executes the command sending the required message to the TCP connection
func (c *command) Start() error {
	cmd := endpointToCommand(c.command, c.endpoint)

	e := pktline.NewEncoder(c.conn)
	return e.Encode([]byte(cmd))
}

func (c *command) connect() error {
	if c.connected {
		return transport.ErrAlreadyConnected
	}

	var err error
	c.conn, err = net.Dial("tcp", c.getHostWithPort())
	if err != nil {
		return err
	}

	c.connected = true
	return nil
}

func (c *command) getHostWithPort() string {
	host := c.endpoint.Host
	port := c.endpoint.Port
	if port <= 0 {
		port = DefaultPort
	}

	return fmt.Sprintf("%s:%d", host, port)
}

// StderrPipe git protocol doesn't have any dedicated error channel
func (c *command) StderrPipe() (io.Reader, error) {
	return nil, nil
}

// StdinPipe return the underlying connection as WriteCloser, wrapped to prevent
// call to the Close function from the connection, a command execution in git
// protocol can't be closed or killed
func (c *command) StdinPipe() (io.WriteCloser, error) {
	return ioutil.WriteNopCloser(c.conn), nil
}

// StdoutPipe return the underlying connection as Reader
func (c *command) StdoutPipe() (io.Reader, error) {
	return c.conn, nil
}

func endpointToCommand(cmd string, ep *transport.Endpoint) string {
	host := ep.Host
	if ep.Port != DefaultPort {
		host = fmt.Sprintf("%s:%d", ep.Host, ep.Port)
	}

	return fmt.Sprintf("%s %s%chost=%s%c", cmd, ep.Path, 0, host, 0)
}

// Close closes the TCP connection and connection.
func (c *command) Close() error {
	if !c.connected {
		return nil
	}

	c.connected = false
	return c.conn.Close()
}
