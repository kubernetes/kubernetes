package agent

import (
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/hashicorp/scada-client"
)

const (
	// providerService is the service name we use
	providerService = "consul"

	// resourceType is the type of resource we represent
	// when connecting to SCADA
	resourceType = "infrastructures"
)

// ProviderService returns the service information for the provider
func ProviderService(c *Config) *client.ProviderService {
	return &client.ProviderService{
		Service:        providerService,
		ServiceVersion: fmt.Sprintf("%s%s", c.Version, c.VersionPrerelease),
		Capabilities: map[string]int{
			"http": 1,
		},
		Meta: map[string]string{
			"auto-join":  strconv.FormatBool(c.AtlasJoin),
			"datacenter": c.Datacenter,
			"server":     strconv.FormatBool(c.Server),
		},
		ResourceType: resourceType,
	}
}

// ProviderConfig returns the configuration for the SCADA provider
func ProviderConfig(c *Config) *client.ProviderConfig {
	return &client.ProviderConfig{
		Service: ProviderService(c),
		Handlers: map[string]client.CapabilityProvider{
			"http": nil,
		},
		Endpoint:      c.AtlasEndpoint,
		ResourceGroup: c.AtlasInfrastructure,
		Token:         c.AtlasToken,
	}
}

// NewProvider creates a new SCADA provider using the
// given configuration. Requests for the HTTP capability
// are passed off to the listener that is returned.
func NewProvider(c *Config, logOutput io.Writer) (*client.Provider, net.Listener, error) {
	// Get the configuration of the provider
	config := ProviderConfig(c)
	config.LogOutput = logOutput

	// SCADA_INSECURE env variable is used for testing to disable
	// TLS certificate verification.
	if os.Getenv("SCADA_INSECURE") != "" {
		config.TLSConfig = &tls.Config{
			InsecureSkipVerify: true,
		}
	}

	// Create an HTTP listener and handler
	list := newScadaListener(c.AtlasInfrastructure)
	config.Handlers["http"] = func(capability string, meta map[string]string,
		conn io.ReadWriteCloser) error {
		return list.PushRWC(conn)
	}

	// Create the provider
	provider, err := client.NewProvider(config)
	if err != nil {
		list.Close()
		return nil, nil, err
	}
	return provider, list, nil
}

// scadaListener is used to return a net.Listener for
// incoming SCADA connections
type scadaListener struct {
	addr    *scadaAddr
	pending chan net.Conn

	closed   bool
	closedCh chan struct{}
	l        sync.Mutex
}

// newScadaListener returns a new listener
func newScadaListener(infra string) *scadaListener {
	l := &scadaListener{
		addr:     &scadaAddr{infra},
		pending:  make(chan net.Conn),
		closedCh: make(chan struct{}),
	}
	return l
}

// PushRWC is used to push a io.ReadWriteCloser as a net.Conn
func (s *scadaListener) PushRWC(conn io.ReadWriteCloser) error {
	// Check if this already implements net.Conn
	if nc, ok := conn.(net.Conn); ok {
		return s.Push(nc)
	}

	// Wrap to implement the interface
	wrapped := &scadaRWC{conn, s.addr}
	return s.Push(wrapped)
}

// Push is used to add a connection to the queue
func (s *scadaListener) Push(conn net.Conn) error {
	select {
	case s.pending <- conn:
		return nil
	case <-time.After(time.Second):
		return fmt.Errorf("accept timed out")
	case <-s.closedCh:
		return fmt.Errorf("scada listener closed")
	}
}

func (s *scadaListener) Accept() (net.Conn, error) {
	select {
	case conn := <-s.pending:
		return conn, nil
	case <-s.closedCh:
		return nil, fmt.Errorf("scada listener closed")
	}
}

func (s *scadaListener) Close() error {
	s.l.Lock()
	defer s.l.Unlock()
	if s.closed {
		return nil
	}
	s.closed = true
	close(s.closedCh)
	return nil
}

func (s *scadaListener) Addr() net.Addr {
	return s.addr
}

// scadaAddr is used to return a net.Addr for SCADA
type scadaAddr struct {
	infra string
}

func (s *scadaAddr) Network() string {
	return "SCADA"
}

func (s *scadaAddr) String() string {
	return fmt.Sprintf("SCADA::Atlas::%s", s.infra)
}

type scadaRWC struct {
	io.ReadWriteCloser
	addr *scadaAddr
}

func (s *scadaRWC) LocalAddr() net.Addr {
	return s.addr
}

func (s *scadaRWC) RemoteAddr() net.Addr {
	return s.addr
}

func (s *scadaRWC) SetDeadline(t time.Time) error {
	return errors.New("SCADA.Conn does not support deadlines")
}

func (s *scadaRWC) SetReadDeadline(t time.Time) error {
	return errors.New("SCADA.Conn does not support deadlines")
}

func (s *scadaRWC) SetWriteDeadline(t time.Time) error {
	return errors.New("SCADA.Conn does not support deadlines")
}
