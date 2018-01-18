package libtrust

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net"
	"os"
	"path"
	"sync"
)

// ClientKeyManager manages client keys on the filesystem
type ClientKeyManager struct {
	key        PrivateKey
	clientFile string
	clientDir  string

	clientLock sync.RWMutex
	clients    []PublicKey

	configLock sync.Mutex
	configs    []*tls.Config
}

// NewClientKeyManager loads a new manager from a set of key files
// and managed by the given private key.
func NewClientKeyManager(trustKey PrivateKey, clientFile, clientDir string) (*ClientKeyManager, error) {
	m := &ClientKeyManager{
		key:        trustKey,
		clientFile: clientFile,
		clientDir:  clientDir,
	}
	if err := m.loadKeys(); err != nil {
		return nil, err
	}
	// TODO Start watching file and directory

	return m, nil
}

func (c *ClientKeyManager) loadKeys() (err error) {
	// Load authorized keys file
	var clients []PublicKey
	if c.clientFile != "" {
		clients, err = LoadKeySetFile(c.clientFile)
		if err != nil {
			return fmt.Errorf("unable to load authorized keys: %s", err)
		}
	}

	// Add clients from authorized keys directory
	files, err := ioutil.ReadDir(c.clientDir)
	if err != nil && !os.IsNotExist(err) {
		return fmt.Errorf("unable to open authorized keys directory: %s", err)
	}
	for _, f := range files {
		if !f.IsDir() {
			publicKey, err := LoadPublicKeyFile(path.Join(c.clientDir, f.Name()))
			if err != nil {
				return fmt.Errorf("unable to load authorized key file: %s", err)
			}
			clients = append(clients, publicKey)
		}
	}

	c.clientLock.Lock()
	c.clients = clients
	c.clientLock.Unlock()

	return nil
}

// RegisterTLSConfig registers a tls configuration to manager
// such that any changes to the keys may be reflected in
// the tls client CA pool
func (c *ClientKeyManager) RegisterTLSConfig(tlsConfig *tls.Config) error {
	c.clientLock.RLock()
	certPool, err := GenerateCACertPool(c.key, c.clients)
	if err != nil {
		return fmt.Errorf("CA pool generation error: %s", err)
	}
	c.clientLock.RUnlock()

	tlsConfig.ClientCAs = certPool

	c.configLock.Lock()
	c.configs = append(c.configs, tlsConfig)
	c.configLock.Unlock()

	return nil
}

// NewIdentityAuthTLSConfig creates a tls.Config for the server to use for
// libtrust identity authentication for the domain specified
func NewIdentityAuthTLSConfig(trustKey PrivateKey, clients *ClientKeyManager, addr string, domain string) (*tls.Config, error) {
	tlsConfig := newTLSConfig()

	tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
	if err := clients.RegisterTLSConfig(tlsConfig); err != nil {
		return nil, err
	}

	// Generate cert
	ips, domains, err := parseAddr(addr)
	if err != nil {
		return nil, err
	}
	// add domain that it expects clients to use
	domains = append(domains, domain)
	x509Cert, err := GenerateSelfSignedServerCert(trustKey, domains, ips)
	if err != nil {
		return nil, fmt.Errorf("certificate generation error: %s", err)
	}
	tlsConfig.Certificates = []tls.Certificate{{
		Certificate: [][]byte{x509Cert.Raw},
		PrivateKey:  trustKey.CryptoPrivateKey(),
		Leaf:        x509Cert,
	}}

	return tlsConfig, nil
}

// NewCertAuthTLSConfig creates a tls.Config for the server to use for
// certificate authentication
func NewCertAuthTLSConfig(caPath, certPath, keyPath string) (*tls.Config, error) {
	tlsConfig := newTLSConfig()

	cert, err := tls.LoadX509KeyPair(certPath, keyPath)
	if err != nil {
		return nil, fmt.Errorf("Couldn't load X509 key pair (%s, %s): %s. Key encrypted?", certPath, keyPath, err)
	}
	tlsConfig.Certificates = []tls.Certificate{cert}

	// Verify client certificates against a CA?
	if caPath != "" {
		certPool := x509.NewCertPool()
		file, err := ioutil.ReadFile(caPath)
		if err != nil {
			return nil, fmt.Errorf("Couldn't read CA certificate: %s", err)
		}
		certPool.AppendCertsFromPEM(file)

		tlsConfig.ClientAuth = tls.RequireAndVerifyClientCert
		tlsConfig.ClientCAs = certPool
	}

	return tlsConfig, nil
}

func newTLSConfig() *tls.Config {
	return &tls.Config{
		NextProtos: []string{"http/1.1"},
		// Avoid fallback on insecure SSL protocols
		MinVersion: tls.VersionTLS10,
	}
}

// parseAddr parses an address into an array of IPs and domains
func parseAddr(addr string) ([]net.IP, []string, error) {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return nil, nil, err
	}
	var domains []string
	var ips []net.IP
	ip := net.ParseIP(host)
	if ip != nil {
		ips = []net.IP{ip}
	} else {
		domains = []string{host}
	}
	return ips, domains, nil
}
