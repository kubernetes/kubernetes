package registry

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"

	"github.com/docker/distribution/registry/client/transport"
	registrytypes "github.com/docker/docker/api/types/registry"
	"github.com/sirupsen/logrus"
)

// V1Endpoint stores basic information about a V1 registry endpoint.
type V1Endpoint struct {
	client   *http.Client
	URL      *url.URL
	IsSecure bool
}

// NewV1Endpoint parses the given address to return a registry endpoint.
func NewV1Endpoint(index *registrytypes.IndexInfo, userAgent string, metaHeaders http.Header) (*V1Endpoint, error) {
	tlsConfig, err := newTLSConfig(index.Name, index.Secure)
	if err != nil {
		return nil, err
	}

	endpoint, err := newV1EndpointFromStr(GetAuthConfigKey(index), tlsConfig, userAgent, metaHeaders)
	if err != nil {
		return nil, err
	}

	if err := validateEndpoint(endpoint); err != nil {
		return nil, err
	}

	return endpoint, nil
}

func validateEndpoint(endpoint *V1Endpoint) error {
	logrus.Debugf("pinging registry endpoint %s", endpoint)

	// Try HTTPS ping to registry
	endpoint.URL.Scheme = "https"
	if _, err := endpoint.Ping(); err != nil {
		if endpoint.IsSecure {
			// If registry is secure and HTTPS failed, show user the error and tell them about `--insecure-registry`
			// in case that's what they need. DO NOT accept unknown CA certificates, and DO NOT fallback to HTTP.
			return fmt.Errorf("invalid registry endpoint %s: %v. If this private registry supports only HTTP or HTTPS with an unknown CA certificate, please add `--insecure-registry %s` to the daemon's arguments. In the case of HTTPS, if you have access to the registry's CA certificate, no need for the flag; simply place the CA certificate at /etc/docker/certs.d/%s/ca.crt", endpoint, err, endpoint.URL.Host, endpoint.URL.Host)
		}

		// If registry is insecure and HTTPS failed, fallback to HTTP.
		logrus.Debugf("Error from registry %q marked as insecure: %v. Insecurely falling back to HTTP", endpoint, err)
		endpoint.URL.Scheme = "http"

		var err2 error
		if _, err2 = endpoint.Ping(); err2 == nil {
			return nil
		}

		return fmt.Errorf("invalid registry endpoint %q. HTTPS attempt: %v. HTTP attempt: %v", endpoint, err, err2)
	}

	return nil
}

func newV1Endpoint(address url.URL, tlsConfig *tls.Config, userAgent string, metaHeaders http.Header) (*V1Endpoint, error) {
	endpoint := &V1Endpoint{
		IsSecure: (tlsConfig == nil || !tlsConfig.InsecureSkipVerify),
		URL:      new(url.URL),
	}

	*endpoint.URL = address

	// TODO(tiborvass): make sure a ConnectTimeout transport is used
	tr := NewTransport(tlsConfig)
	endpoint.client = HTTPClient(transport.NewTransport(tr, DockerHeaders(userAgent, metaHeaders)...))
	return endpoint, nil
}

// trimV1Address trims the version off the address and returns the
// trimmed address or an error if there is a non-V1 version.
func trimV1Address(address string) (string, error) {
	var (
		chunks        []string
		apiVersionStr string
	)

	if strings.HasSuffix(address, "/") {
		address = address[:len(address)-1]
	}

	chunks = strings.Split(address, "/")
	apiVersionStr = chunks[len(chunks)-1]
	if apiVersionStr == "v1" {
		return strings.Join(chunks[:len(chunks)-1], "/"), nil
	}

	for k, v := range apiVersions {
		if k != APIVersion1 && apiVersionStr == v {
			return "", fmt.Errorf("unsupported V1 version path %s", apiVersionStr)
		}
	}

	return address, nil
}

func newV1EndpointFromStr(address string, tlsConfig *tls.Config, userAgent string, metaHeaders http.Header) (*V1Endpoint, error) {
	if !strings.HasPrefix(address, "http://") && !strings.HasPrefix(address, "https://") {
		address = "https://" + address
	}

	address, err := trimV1Address(address)
	if err != nil {
		return nil, err
	}

	uri, err := url.Parse(address)
	if err != nil {
		return nil, err
	}

	endpoint, err := newV1Endpoint(*uri, tlsConfig, userAgent, metaHeaders)
	if err != nil {
		return nil, err
	}

	return endpoint, nil
}

// Get the formatted URL for the root of this registry Endpoint
func (e *V1Endpoint) String() string {
	return e.URL.String() + "/v1/"
}

// Path returns a formatted string for the URL
// of this endpoint with the given path appended.
func (e *V1Endpoint) Path(path string) string {
	return e.URL.String() + "/v1/" + path
}

// Ping returns a PingResult which indicates whether the registry is standalone or not.
func (e *V1Endpoint) Ping() (PingResult, error) {
	logrus.Debugf("attempting v1 ping for registry endpoint %s", e)

	if e.String() == IndexServer {
		// Skip the check, we know this one is valid
		// (and we never want to fallback to http in case of error)
		return PingResult{Standalone: false}, nil
	}

	req, err := http.NewRequest("GET", e.Path("_ping"), nil)
	if err != nil {
		return PingResult{Standalone: false}, err
	}

	resp, err := e.client.Do(req)
	if err != nil {
		return PingResult{Standalone: false}, err
	}

	defer resp.Body.Close()

	jsonString, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return PingResult{Standalone: false}, fmt.Errorf("error while reading the http response: %s", err)
	}

	// If the header is absent, we assume true for compatibility with earlier
	// versions of the registry. default to true
	info := PingResult{
		Standalone: true,
	}
	if err := json.Unmarshal(jsonString, &info); err != nil {
		logrus.Debugf("Error unmarshaling the _ping PingResult: %s", err)
		// don't stop here. Just assume sane defaults
	}
	if hdr := resp.Header.Get("X-Docker-Registry-Version"); hdr != "" {
		logrus.Debugf("Registry version header: '%s'", hdr)
		info.Version = hdr
	}
	logrus.Debugf("PingResult.Version: %q", info.Version)

	standalone := resp.Header.Get("X-Docker-Registry-Standalone")
	logrus.Debugf("Registry standalone header: '%s'", standalone)
	// Accepted values are "true" (case-insensitive) and "1".
	if strings.EqualFold(standalone, "true") || standalone == "1" {
		info.Standalone = true
	} else if len(standalone) > 0 {
		// there is a header set, and it is not "true" or "1", so assume fails
		info.Standalone = false
	}
	logrus.Debugf("PingResult.Standalone: %t", info.Standalone)
	return info, nil
}
