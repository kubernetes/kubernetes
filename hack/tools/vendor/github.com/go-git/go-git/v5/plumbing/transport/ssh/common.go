// Package ssh implements the SSH transport protocol.
package ssh

import (
	"context"
	"fmt"
	"reflect"
	"strconv"

	"github.com/go-git/go-git/v5/plumbing/transport"
	"github.com/go-git/go-git/v5/plumbing/transport/internal/common"

	"github.com/kevinburke/ssh_config"
	"golang.org/x/crypto/ssh"
	"golang.org/x/net/proxy"
)

// DefaultClient is the default SSH client.
var DefaultClient = NewClient(nil)

// DefaultSSHConfig is the reader used to access parameters stored in the
// system's ssh_config files. If nil all the ssh_config are ignored.
var DefaultSSHConfig sshConfig = ssh_config.DefaultUserSettings

type sshConfig interface {
	Get(alias, key string) string
}

// NewClient creates a new SSH client with an optional *ssh.ClientConfig.
func NewClient(config *ssh.ClientConfig) transport.Transport {
	return common.NewClient(&runner{config: config})
}

// DefaultAuthBuilder is the function used to create a default AuthMethod, when
// the user doesn't provide any.
var DefaultAuthBuilder = func(user string) (AuthMethod, error) {
	return NewSSHAgentAuth(user)
}

const DefaultPort = 22

type runner struct {
	config *ssh.ClientConfig
}

func (r *runner) Command(cmd string, ep *transport.Endpoint, auth transport.AuthMethod) (common.Command, error) {
	c := &command{command: cmd, endpoint: ep, config: r.config}
	if auth != nil {
		c.setAuth(auth)
	}

	if err := c.connect(); err != nil {
		return nil, err
	}
	return c, nil
}

type command struct {
	*ssh.Session
	connected bool
	command   string
	endpoint  *transport.Endpoint
	client    *ssh.Client
	auth      AuthMethod
	config    *ssh.ClientConfig
}

func (c *command) setAuth(auth transport.AuthMethod) error {
	a, ok := auth.(AuthMethod)
	if !ok {
		return transport.ErrInvalidAuthMethod
	}

	c.auth = a
	return nil
}

func (c *command) Start() error {
	return c.Session.Start(endpointToCommand(c.command, c.endpoint))
}

// Close closes the SSH session and connection.
func (c *command) Close() error {
	if !c.connected {
		return nil
	}

	c.connected = false

	//XXX: If did read the full packfile, then the session might be already
	//     closed.
	_ = c.Session.Close()

	return c.client.Close()
}

// connect connects to the SSH server, unless a AuthMethod was set with
// SetAuth method, by default uses an auth method based on PublicKeysCallback,
// it connects to a SSH agent, using the address stored in the SSH_AUTH_SOCK
// environment var.
func (c *command) connect() error {
	if c.connected {
		return transport.ErrAlreadyConnected
	}

	if c.auth == nil {
		if err := c.setAuthFromEndpoint(); err != nil {
			return err
		}
	}

	var err error
	config, err := c.auth.ClientConfig()
	if err != nil {
		return err
	}

	overrideConfig(c.config, config)

	c.client, err = dial("tcp", c.getHostWithPort(), config)
	if err != nil {
		return err
	}

	c.Session, err = c.client.NewSession()
	if err != nil {
		_ = c.client.Close()
		return err
	}

	c.connected = true
	return nil
}

func dial(network, addr string, config *ssh.ClientConfig) (*ssh.Client, error) {
	var (
		ctx    = context.Background()
		cancel context.CancelFunc
	)
	if config.Timeout > 0 {
		ctx, cancel = context.WithTimeout(ctx, config.Timeout)
	} else {
		ctx, cancel = context.WithCancel(ctx)
	}
	defer cancel()

	conn, err := proxy.Dial(ctx, network, addr)
	if err != nil {
		return nil, err
	}
	c, chans, reqs, err := ssh.NewClientConn(conn, addr, config)
	if err != nil {
		return nil, err
	}
	return ssh.NewClient(c, chans, reqs), nil
}

func (c *command) getHostWithPort() string {
	if addr, found := c.doGetHostWithPortFromSSHConfig(); found {
		return addr
	}

	host := c.endpoint.Host
	port := c.endpoint.Port
	if port <= 0 {
		port = DefaultPort
	}

	return fmt.Sprintf("%s:%d", host, port)
}

func (c *command) doGetHostWithPortFromSSHConfig() (addr string, found bool) {
	if DefaultSSHConfig == nil {
		return
	}

	host := c.endpoint.Host
	port := c.endpoint.Port

	configHost := DefaultSSHConfig.Get(c.endpoint.Host, "Hostname")
	if configHost != "" {
		host = configHost
		found = true
	}

	if !found {
		return
	}

	configPort := DefaultSSHConfig.Get(c.endpoint.Host, "Port")
	if configPort != "" {
		if i, err := strconv.Atoi(configPort); err == nil {
			port = i
		}
	}

	addr = fmt.Sprintf("%s:%d", host, port)
	return
}

func (c *command) setAuthFromEndpoint() error {
	var err error
	c.auth, err = DefaultAuthBuilder(c.endpoint.User)
	return err
}

func endpointToCommand(cmd string, ep *transport.Endpoint) string {
	return fmt.Sprintf("%s '%s'", cmd, ep.Path)
}

func overrideConfig(overrides *ssh.ClientConfig, c *ssh.ClientConfig) {
	if overrides == nil {
		return
	}

	t := reflect.TypeOf(*c)
	vc := reflect.ValueOf(c).Elem()
	vo := reflect.ValueOf(overrides).Elem()

	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		vcf := vc.FieldByName(f.Name)
		vof := vo.FieldByName(f.Name)
		vcf.Set(vof)
	}

	*c = vc.Interface().(ssh.ClientConfig)
}
