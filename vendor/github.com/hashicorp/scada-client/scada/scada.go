package scada

import (
	"crypto/tls"
	"errors"
	"fmt"
	"io"
	"net"
	"os"
	"sync"
	"time"

	sc "github.com/hashicorp/scada-client"
)

// Provider wraps scada-client.Provider to allow most applications to only pull
// in this package
type Provider struct {
	*sc.Provider
}

type AtlasConfig struct {
	// Endpoint is the SCADA endpoint used for Atlas integration. If empty, the
	// defaults from the provider are used.
	Endpoint string `mapstructure:"endpoint"`

	// The name of the infrastructure we belong to, e.g. "hashicorp/prod"
	Infrastructure string `mapstructure:"infrastructure"`

	// The Atlas authentication token
	Token string `mapstructure:"token" json:"-"`
}

// Config holds the high-level information used to instantiate a SCADA provider
// and listener
type Config struct {
	// The service name to use
	Service string

	// The version of the service
	Version string

	// The type of resource we represent
	ResourceType string

	// Metadata to send to along with the service information
	Meta map[string]string

	// If set, TLS certificate verification will be skipped. The value of the
	// SCADA_INSECURE environment variable will be considered if this is false.
	// If using SCADA_INSECURE, any non-empty value will trigger insecure mode.
	Insecure bool

	// Holds Atlas configuration
	Atlas AtlasConfig
}

// ProviderService returns the service information for the provider
func providerService(c *Config) *sc.ProviderService {
	ret := &sc.ProviderService{
		Service:        c.Service,
		ServiceVersion: c.Version,
		Capabilities:   map[string]int{},
		Meta:           c.Meta,
		ResourceType:   c.ResourceType,
	}

	return ret
}

// providerConfig returns the configuration for the SCADA provider
func providerConfig(c *Config) *sc.ProviderConfig {
	ret := &sc.ProviderConfig{
		Service:       providerService(c),
		Handlers:      map[string]sc.CapabilityProvider{},
		Endpoint:      c.Atlas.Endpoint,
		ResourceGroup: c.Atlas.Infrastructure,
		Token:         c.Atlas.Token,
	}

	// SCADA_INSECURE env variable is used for testing to disable TLS
	// certificate verification.
	insecure := c.Insecure
	if !insecure {
		if os.Getenv("SCADA_INSECURE") != "" {
			insecure = true
		}
	}
	if insecure {
		ret.TLSConfig = &tls.Config{
			InsecureSkipVerify: true,
		}
	}

	return ret
}

// NewProvider creates a new SCADA provider using the given configuration.
// Requests for the HTTP capability are passed off to the listener that is
// returned.
func NewHTTPProvider(c *Config, logOutput io.Writer) (*Provider, net.Listener, error) {
	// Get the configuration of the provider
	config := providerConfig(c)
	config.LogOutput = logOutput

	// Set the HTTP capability
	config.Service.Capabilities["http"] = 1

	// Create an HTTP listener and handler
	list := newScadaListener(c.Atlas.Infrastructure)
	config.Handlers["http"] = func(capability string, meta map[string]string,
		conn io.ReadWriteCloser) error {
		return list.PushRWC(conn)
	}

	// Create the provider
	provider, err := sc.NewProvider(config)
	if err != nil {
		list.Close()
		return nil, nil, err
	}

	return &Provider{provider}, list, nil
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

// Push is used to add a connection to the queu
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
