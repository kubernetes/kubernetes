package client

import (
	"crypto/tls"
	"fmt"
	"io"
	"log"
	"math/rand"
	"net"
	"net/rpc"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/armon/go-metrics"
	"github.com/hashicorp/net-rpc-msgpackrpc"
)

const (
	// DefaultEndpoint is the endpoint used if none is provided
	DefaultEndpoint = "scada.hashicorp.com:7223"

	// DefaultBackoff is the amount of time we back off if we encounter
	// and error, and no specific backoff is available.
	DefaultBackoff = 120 * time.Second

	// DisconnectDelay is how long we delay the disconnect to allow
	// the RPC to complete.
	DisconnectDelay = time.Second
)

// CapabilityProvider is used to provide a given capability
// when requested remotely. They must return a connection
// that is bridged or an error.
type CapabilityProvider func(capability string, meta map[string]string, conn io.ReadWriteCloser) error

// ProviderService is the service being exposed
type ProviderService struct {
	Service        string
	ServiceVersion string
	Capabilities   map[string]int
	Meta           map[string]string
	ResourceType   string
}

// ProviderConfig is used to parameterize a provider
type ProviderConfig struct {
	// Endpoint is the SCADA endpoint, defaults to DefaultEndpoint
	Endpoint string

	// Service is the service to expose
	Service *ProviderService

	// Handlers are invoked to provide the named capability
	Handlers map[string]CapabilityProvider

	// ResourceGroup is the named group e.g. "hashicorp/prod"
	ResourceGroup string

	// Token is the Atlas authentication token
	Token string

	// Optional TLS configuration, defaults used otherwise
	TLSConfig *tls.Config

	// LogOutput is to control the log output
	LogOutput io.Writer
}

// Provider is a high-level interface to SCADA by which
// clients declare themselves as a service providing capabilities.
// Provider manages the client/server interactions required,
// making it simpler to integrate.
type Provider struct {
	config *ProviderConfig
	logger *log.Logger

	client     *Client
	clientLock sync.Mutex

	noRetry     bool          // set when the server instructs us to not retry
	backoff     time.Duration // set when the server provides a longer backoff
	backoffLock sync.Mutex

	sessionID   string
	sessionAuth bool
	sessionLock sync.RWMutex

	shutdown     bool
	shutdownCh   chan struct{}
	shutdownLock sync.Mutex
}

// validateConfig is used to sanity check the configuration
func validateConfig(config *ProviderConfig) error {
	// Validate the inputs
	if config == nil {
		return fmt.Errorf("missing config")
	}
	if config.Service == nil {
		return fmt.Errorf("missing service")
	}
	if config.Service.Service == "" {
		return fmt.Errorf("missing service name")
	}
	if config.Service.ServiceVersion == "" {
		return fmt.Errorf("missing service version")
	}
	if config.Service.ResourceType == "" {
		return fmt.Errorf("missing service resource type")
	}
	if config.Handlers == nil && len(config.Service.Capabilities) != 0 {
		return fmt.Errorf("missing handlers")
	}
	for c := range config.Service.Capabilities {
		if _, ok := config.Handlers[c]; !ok {
			return fmt.Errorf("missing handler for '%s' capability", c)
		}
	}
	if config.ResourceGroup == "" {
		return fmt.Errorf("missing resource group")
	}
	if config.Token == "" {
		config.Token = os.Getenv("ATLAS_TOKEN")
	}
	if config.Token == "" {
		return fmt.Errorf("missing token")
	}

	// Default the endpoint
	if config.Endpoint == "" {
		config.Endpoint = DefaultEndpoint
		if end := os.Getenv("SCADA_ENDPOINT"); end != "" {
			config.Endpoint = end
		}
	}
	return nil
}

// NewProvider is used to create a new provider
func NewProvider(config *ProviderConfig) (*Provider, error) {
	if err := validateConfig(config); err != nil {
		return nil, err
	}

	// Create logger
	if config.LogOutput == nil {
		config.LogOutput = os.Stderr
	}
	logger := log.New(config.LogOutput, "", log.LstdFlags)

	p := &Provider{
		config:     config,
		logger:     logger,
		shutdownCh: make(chan struct{}),
	}
	go p.run()
	return p, nil
}

// Shutdown is used to close the provider
func (p *Provider) Shutdown() {
	p.shutdownLock.Lock()
	p.shutdownLock.Unlock()
	if p.shutdown {
		return
	}
	p.shutdown = true
	close(p.shutdownCh)
}

// IsShutdown checks if we have been shutdown
func (p *Provider) IsShutdown() bool {
	select {
	case <-p.shutdownCh:
		return true
	default:
		return false
	}
}

// backoffDuration is used to compute the next backoff duration
func (p *Provider) backoffDuration() time.Duration {
	// Use the default backoff
	backoff := DefaultBackoff

	// Check for a server specified backoff
	p.backoffLock.Lock()
	if p.backoff != 0 {
		backoff = p.backoff
	}
	if p.noRetry {
		backoff = 0
	}
	p.backoffLock.Unlock()

	return backoff
}

// wait is used to delay dialing on an error
func (p *Provider) wait() {
	// Compute the backoff time
	backoff := p.backoffDuration()

	// Setup a wait timer
	var wait <-chan time.Time
	if backoff > 0 {
		jitter := time.Duration(rand.Uint32()) % backoff
		wait = time.After(backoff + jitter)
	}

	// Wait until timer or shutdown
	select {
	case <-wait:
	case <-p.shutdownCh:
	}
}

// run is a long running routine to manage the provider
func (p *Provider) run() {
	for !p.IsShutdown() {
		// Setup a new connection
		client, err := p.clientSetup()
		if err != nil {
			p.wait()
			continue
		}

		// Handle the session
		doneCh := make(chan struct{})
		go p.handleSession(client, doneCh)

		// Wait for session termination or shutdown
		select {
		case <-doneCh:
			p.wait()
		case <-p.shutdownCh:
			p.clientLock.Lock()
			client.Close()
			p.clientLock.Unlock()
			return
		}
	}
}

// handleSession is used to handle an established session
func (p *Provider) handleSession(list net.Listener, doneCh chan struct{}) {
	defer close(doneCh)
	defer list.Close()
	// Accept new connections
	for !p.IsShutdown() {
		conn, err := list.Accept()
		if err != nil {
			p.logger.Printf("[ERR] scada-client: failed to accept connection: %v", err)
			return
		}
		p.logger.Printf("[DEBUG] scada-client: accepted connection")
		go p.handleConnection(conn)
	}
}

// handleConnection handles an incoming connection
func (p *Provider) handleConnection(conn net.Conn) {
	// Create an RPC server to handle inbound
	pe := &providerEndpoint{p: p}
	rpcServer := rpc.NewServer()
	rpcServer.RegisterName("Client", pe)
	rpcCodec := msgpackrpc.NewCodec(false, false, conn)

	defer func() {
		if !pe.hijacked() {
			conn.Close()
		}
	}()

	for !p.IsShutdown() {
		if err := rpcServer.ServeRequest(rpcCodec); err != nil {
			if err != io.EOF && !strings.Contains(err.Error(), "closed") {
				p.logger.Printf("[ERR] scada-client: RPC error: %v", err)
			}
			return
		}

		// Handle potential hijack in Client.Connect
		if pe.hijacked() {
			cb := pe.getHijack()
			cb(conn)
			return
		}
	}
}

// clientSetup is used to setup a new connection
func (p *Provider) clientSetup() (*Client, error) {
	defer metrics.MeasureSince([]string{"scada", "setup"}, time.Now())

	// Reset the previous backoff
	p.backoffLock.Lock()
	p.noRetry = false
	p.backoff = 0
	p.backoffLock.Unlock()

	// Dial a new connection
	opts := Opts{
		Addr:      p.config.Endpoint,
		TLS:       true,
		TLSConfig: p.config.TLSConfig,
		LogOutput: p.config.LogOutput,
	}
	client, err := DialOpts(&opts)
	if err != nil {
		p.logger.Printf("[ERR] scada-client: failed to dial: %v", err)
		return nil, err
	}

	// Perform a handshake
	resp, err := p.handshake(client)
	if err != nil {
		p.logger.Printf("[ERR] scada-client: failed to handshake: %v", err)
		client.Close()
		return nil, err
	}
	if resp != nil && resp.SessionID != "" {
		p.logger.Printf("[DEBUG] scada-client: assigned session '%s'", resp.SessionID)
	}
	if resp != nil && !resp.Authenticated {
		p.logger.Printf("[WARN] scada-client: authentication failed: %v", resp.Reason)
	}

	// Set the new client
	p.clientLock.Lock()
	if p.client != nil {
		p.client.Close()
	}
	p.client = client
	p.clientLock.Unlock()

	p.sessionLock.Lock()
	p.sessionID = resp.SessionID
	p.sessionAuth = resp.Authenticated
	p.sessionLock.Unlock()

	return client, nil
}

// SessionID provides the current session ID
func (p *Provider) SessionID() string {
	p.sessionLock.RLock()
	defer p.sessionLock.RUnlock()
	return p.sessionID
}

// SessionAuth checks if the current session is authenticated
func (p *Provider) SessionAuthenticated() bool {
	p.sessionLock.RLock()
	defer p.sessionLock.RUnlock()
	return p.sessionAuth
}

// handshake does the initial handshake
func (p *Provider) handshake(client *Client) (*HandshakeResponse, error) {
	defer metrics.MeasureSince([]string{"scada", "handshake"}, time.Now())
	req := HandshakeRequest{
		Service:        p.config.Service.Service,
		ServiceVersion: p.config.Service.ServiceVersion,
		Capabilities:   p.config.Service.Capabilities,
		Meta:           p.config.Service.Meta,
		ResourceType:   p.config.Service.ResourceType,
		ResourceGroup:  p.config.ResourceGroup,
		Token:          p.config.Token,
	}
	resp := new(HandshakeResponse)
	if err := client.RPC("Session.Handshake", &req, resp); err != nil {
		return nil, err
	}
	return resp, nil
}

type HijackFunc func(io.ReadWriteCloser)

// providerEndpoint is used to implement the Client.* RPC endpoints
// as part of the provider.
type providerEndpoint struct {
	p      *Provider
	hijack HijackFunc
}

// Hijacked is used to check if the connection has been hijacked
func (pe *providerEndpoint) hijacked() bool {
	return pe.hijack != nil
}

// GetHijack returns the hijack function
func (pe *providerEndpoint) getHijack() HijackFunc {
	return pe.hijack
}

// Hijack is used to take over the yamux stream for Client.Connect
func (pe *providerEndpoint) setHijack(cb HijackFunc) {
	pe.hijack = cb
}

// Connect is invoked by the broker to connect to a capability
func (pe *providerEndpoint) Connect(args *ConnectRequest, resp *ConnectResponse) error {
	defer metrics.IncrCounter([]string{"scada", "connect", args.Capability}, 1)
	pe.p.logger.Printf("[INFO] scada-client: connect requested (capability: %s)",
		args.Capability)

	// Handle potential flash
	if args.Severity != "" && args.Message != "" {
		pe.p.logger.Printf("[%s] scada-client: %s", args.Severity, args.Message)
	}

	// Look for the handler
	handler := pe.p.config.Handlers[args.Capability]
	if handler == nil {
		pe.p.logger.Printf("[WARN] scada-client: requested capability '%s' not available",
			args.Capability)
		return fmt.Errorf("invalid capability")
	}

	// Hijack the connection
	pe.setHijack(func(a io.ReadWriteCloser) {
		if err := handler(args.Capability, args.Meta, a); err != nil {
			pe.p.logger.Printf("[ERR] scada-client: '%s' handler error: %v",
				args.Capability, err)
		}
	})
	resp.Success = true
	return nil
}

// Disconnect is invoked by the broker to ask us to backoff
func (pe *providerEndpoint) Disconnect(args *DisconnectRequest, resp *DisconnectResponse) error {
	defer metrics.IncrCounter([]string{"scada", "disconnect"}, 1)
	if args.Reason == "" {
		args.Reason = "<no reason provided>"
	}
	pe.p.logger.Printf("[INFO] scada-client: disconnect requested (retry: %v, backoff: %v): %v",
		!args.NoRetry, args.Backoff, args.Reason)

	// Use the backoff information
	pe.p.backoffLock.Lock()
	pe.p.noRetry = args.NoRetry
	pe.p.backoff = args.Backoff
	pe.p.backoffLock.Unlock()

	// Clear the session information
	pe.p.sessionLock.Lock()
	pe.p.sessionID = ""
	pe.p.sessionAuth = false
	pe.p.sessionLock.Unlock()

	// Force the disconnect
	time.AfterFunc(DisconnectDelay, func() {
		pe.p.clientLock.Lock()
		if pe.p.client != nil {
			pe.p.client.Close()
		}
		pe.p.clientLock.Unlock()
	})
	return nil
}

// Flash is invoked by the broker log a message
func (pe *providerEndpoint) Flash(args *FlashRequest, resp *FlashResponse) error {
	defer metrics.IncrCounter([]string{"scada", "flash"}, 1)
	if args.Severity != "" && args.Message != "" {
		pe.p.logger.Printf("[%s] scada-client: %s", args.Severity, args.Message)
	}
	return nil
}
