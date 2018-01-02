// Package registry contains client primitives to interact with a remote Docker registry.
package registry

import (
	"crypto/tls"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/docker/distribution/registry/client/transport"
	"github.com/docker/go-connections/sockets"
	"github.com/docker/go-connections/tlsconfig"
	"github.com/sirupsen/logrus"
)

var (
	// ErrAlreadyExists is an error returned if an image being pushed
	// already exists on the remote side
	ErrAlreadyExists = errors.New("Image already exists")
)

func newTLSConfig(hostname string, isSecure bool) (*tls.Config, error) {
	// PreferredServerCipherSuites should have no effect
	tlsConfig := tlsconfig.ServerDefault()

	tlsConfig.InsecureSkipVerify = !isSecure

	if isSecure && CertsDir != "" {
		hostDir := filepath.Join(CertsDir, cleanPath(hostname))
		logrus.Debugf("hostDir: %s", hostDir)
		if err := ReadCertsDirectory(tlsConfig, hostDir); err != nil {
			return nil, err
		}
	}

	return tlsConfig, nil
}

func hasFile(files []os.FileInfo, name string) bool {
	for _, f := range files {
		if f.Name() == name {
			return true
		}
	}
	return false
}

// ReadCertsDirectory reads the directory for TLS certificates
// including roots and certificate pairs and updates the
// provided TLS configuration.
func ReadCertsDirectory(tlsConfig *tls.Config, directory string) error {
	fs, err := ioutil.ReadDir(directory)
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	for _, f := range fs {
		if strings.HasSuffix(f.Name(), ".crt") {
			if tlsConfig.RootCAs == nil {
				systemPool, err := tlsconfig.SystemCertPool()
				if err != nil {
					return fmt.Errorf("unable to get system cert pool: %v", err)
				}
				tlsConfig.RootCAs = systemPool
			}
			logrus.Debugf("crt: %s", filepath.Join(directory, f.Name()))
			data, err := ioutil.ReadFile(filepath.Join(directory, f.Name()))
			if err != nil {
				return err
			}
			tlsConfig.RootCAs.AppendCertsFromPEM(data)
		}
		if strings.HasSuffix(f.Name(), ".cert") {
			certName := f.Name()
			keyName := certName[:len(certName)-5] + ".key"
			logrus.Debugf("cert: %s", filepath.Join(directory, f.Name()))
			if !hasFile(fs, keyName) {
				return fmt.Errorf("Missing key %s for client certificate %s. Note that CA certificates should use the extension .crt.", keyName, certName)
			}
			cert, err := tls.LoadX509KeyPair(filepath.Join(directory, certName), filepath.Join(directory, keyName))
			if err != nil {
				return err
			}
			tlsConfig.Certificates = append(tlsConfig.Certificates, cert)
		}
		if strings.HasSuffix(f.Name(), ".key") {
			keyName := f.Name()
			certName := keyName[:len(keyName)-4] + ".cert"
			logrus.Debugf("key: %s", filepath.Join(directory, f.Name()))
			if !hasFile(fs, certName) {
				return fmt.Errorf("Missing client certificate %s for key %s", certName, keyName)
			}
		}
	}

	return nil
}

// DockerHeaders returns request modifiers with a User-Agent and metaHeaders
func DockerHeaders(userAgent string, metaHeaders http.Header) []transport.RequestModifier {
	modifiers := []transport.RequestModifier{}
	if userAgent != "" {
		modifiers = append(modifiers, transport.NewHeaderRequestModifier(http.Header{
			"User-Agent": []string{userAgent},
		}))
	}
	if metaHeaders != nil {
		modifiers = append(modifiers, transport.NewHeaderRequestModifier(metaHeaders))
	}
	return modifiers
}

// HTTPClient returns an HTTP client structure which uses the given transport
// and contains the necessary headers for redirected requests
func HTTPClient(transport http.RoundTripper) *http.Client {
	return &http.Client{
		Transport:     transport,
		CheckRedirect: addRequiredHeadersToRedirectedRequests,
	}
}

func trustedLocation(req *http.Request) bool {
	var (
		trusteds = []string{"docker.com", "docker.io"}
		hostname = strings.SplitN(req.Host, ":", 2)[0]
	)
	if req.URL.Scheme != "https" {
		return false
	}

	for _, trusted := range trusteds {
		if hostname == trusted || strings.HasSuffix(hostname, "."+trusted) {
			return true
		}
	}
	return false
}

// addRequiredHeadersToRedirectedRequests adds the necessary redirection headers
// for redirected requests
func addRequiredHeadersToRedirectedRequests(req *http.Request, via []*http.Request) error {
	if via != nil && via[0] != nil {
		if trustedLocation(req) && trustedLocation(via[0]) {
			req.Header = via[0].Header
			return nil
		}
		for k, v := range via[0].Header {
			if k != "Authorization" {
				for _, vv := range v {
					req.Header.Add(k, vv)
				}
			}
		}
	}
	return nil
}

// NewTransport returns a new HTTP transport. If tlsConfig is nil, it uses the
// default TLS configuration.
func NewTransport(tlsConfig *tls.Config) *http.Transport {
	if tlsConfig == nil {
		tlsConfig = tlsconfig.ServerDefault()
	}

	direct := &net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
		DualStack: true,
	}

	base := &http.Transport{
		Proxy:               http.ProxyFromEnvironment,
		Dial:                direct.Dial,
		TLSHandshakeTimeout: 10 * time.Second,
		TLSClientConfig:     tlsConfig,
		// TODO(dmcgowan): Call close idle connections when complete and use keep alive
		DisableKeepAlives: true,
	}

	proxyDialer, err := sockets.DialerFromEnvironment(direct)
	if err == nil {
		base.Dial = proxyDialer.Dial
	}
	return base
}
