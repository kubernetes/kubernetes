// Package client implements a Go client for CFSSL API commands.
package client

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	stderr "errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/auth"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/info"
	"github.com/cloudflare/cfssl/log"
)

// A server points to a single remote CFSSL instance.
type server struct {
	URL            string
	TLSConfig      *tls.Config
	reqModifier    func(*http.Request, []byte)
	RequestTimeout time.Duration
	proxy          func(*http.Request) (*url.URL, error)
}

// A Remote points to at least one (but possibly multiple) remote
// CFSSL instances. It must be able to perform a authenticated and
// unauthenticated certificate signing requests, return information
// about the CA on the other end, and return a list of the hosts that
// are used by the remote.
type Remote interface {
	AuthSign(req, id []byte, provider auth.Provider) ([]byte, error)
	Sign(jsonData []byte) ([]byte, error)
	Info(jsonData []byte) (*info.Resp, error)
	Hosts() []string
	SetReqModifier(func(*http.Request, []byte))
	SetRequestTimeout(d time.Duration)
	SetProxy(func(*http.Request) (*url.URL, error))
}

// NewServer sets up a new server target. The address should be of
// The format [protocol:]name[:port] of the remote CFSSL instance.
// If no protocol is given http is default. If no port
// is specified, the CFSSL default port (8888) is used. If the name is
// a comma-separated list of hosts, an ordered group will be returned.
func NewServer(addr string) Remote {
	return NewServerTLS(addr, nil)
}

// NewServerTLS is the TLS version of NewServer
func NewServerTLS(addr string, tlsConfig *tls.Config) Remote {
	addrs := strings.Split(addr, ",")

	var remote Remote

	if len(addrs) > 1 {
		remote, _ = NewGroup(addrs, tlsConfig, StrategyOrderedList)
	} else {
		u, err := normalizeURL(addrs[0])
		if err != nil {
			log.Errorf("bad url: %v", err)
			return nil
		}
		srv := newServer(u, tlsConfig)
		if srv != nil {
			remote = srv
		}
	}
	return remote
}

func (srv *server) Hosts() []string {
	return []string{srv.URL}
}

func (srv *server) SetReqModifier(mod func(*http.Request, []byte)) {
	srv.reqModifier = mod
}

func (srv *server) SetRequestTimeout(timeout time.Duration) {
	srv.RequestTimeout = timeout
}

func (srv *server) SetProxy(proxy func(*http.Request) (*url.URL, error)) {
	srv.proxy = proxy
}

func newServer(u *url.URL, tlsConfig *tls.Config) *server {
	URL := u.String()
	return &server{
		URL:       URL,
		TLSConfig: tlsConfig,
	}
}

func (srv *server) getURL(endpoint string) string {
	return fmt.Sprintf("%s/api/v1/cfssl/%s", srv.URL, endpoint)
}

func (srv *server) createTransport() (transport *http.Transport) {
	transport = new(http.Transport)
	// Setup HTTPS client
	tlsConfig := srv.TLSConfig
	tlsConfig.BuildNameToCertificate()
	transport.TLSClientConfig = tlsConfig
	// Setup Proxy
	transport.Proxy = srv.proxy
	return transport
}

// post connects to the remote server and returns a Response struct
func (srv *server) post(url string, jsonData []byte) (*api.Response, error) {
	var resp *http.Response
	var err error
	client := &http.Client{}
	if srv.TLSConfig != nil {
		client.Transport = srv.createTransport()
	}
	if srv.RequestTimeout != 0 {
		client.Timeout = srv.RequestTimeout
	}
	req, err := http.NewRequest("POST", url, bytes.NewReader(jsonData))
	if err != nil {
		err = fmt.Errorf("failed POST to %s: %v", url, err)
		return nil, errors.Wrap(errors.APIClientError, errors.ClientHTTPError, err)
	}
	req.Close = true
	req.Header.Set("content-type", "application/json")
	if srv.reqModifier != nil {
		srv.reqModifier(req, jsonData)
	}
	resp, err = client.Do(req)
	if err != nil {
		err = fmt.Errorf("failed POST to %s: %v", url, err)
		return nil, errors.Wrap(errors.APIClientError, errors.ClientHTTPError, err)
	}
	defer req.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(errors.APIClientError, errors.IOError, err)
	}

	if resp.StatusCode != http.StatusOK {
		log.Errorf("http error with %s", url)
		return nil, errors.Wrap(errors.APIClientError, errors.ClientHTTPError, stderr.New(string(body)))
	}

	var response api.Response
	err = json.Unmarshal(body, &response)
	if err != nil {
		log.Debug("Unable to parse response body:", string(body))
		return nil, errors.Wrap(errors.APIClientError, errors.JSONError, err)
	}

	if !response.Success || response.Result == nil {
		if len(response.Errors) > 0 {
			return nil, errors.Wrap(errors.APIClientError, errors.ServerRequestFailed, stderr.New(response.Errors[0].Message))
		}
		return nil, errors.New(errors.APIClientError, errors.ServerRequestFailed)
	}

	return &response, nil
}

// AuthSign fills out an authenticated signing request to the server,
// receiving a certificate or error in response.
// It takes the serialized JSON request to send, remote address and
// authentication provider.
func (srv *server) AuthSign(req, id []byte, provider auth.Provider) ([]byte, error) {
	return srv.authReq(req, id, provider, "sign")
}

// AuthInfo fills out an authenticated info request to the server,
// receiving a certificate or error in response.
// It takes the serialized JSON request to send, remote address and
// authentication provider.
func (srv *server) AuthInfo(req, id []byte, provider auth.Provider) ([]byte, error) {
	return srv.authReq(req, id, provider, "info")
}

// authReq is the common logic for AuthSign and AuthInfo -- perform the given
// request, and return the resultant certificate.
// The target is either 'sign' or 'info'.
func (srv *server) authReq(req, ID []byte, provider auth.Provider, target string) ([]byte, error) {
	url := srv.getURL("auth" + target)

	token, err := provider.Token(req)
	if err != nil {
		return nil, errors.Wrap(errors.APIClientError, errors.AuthenticationFailure, err)
	}

	aReq := &auth.AuthenticatedRequest{
		Timestamp:     time.Now().Unix(),
		RemoteAddress: ID,
		Token:         token,
		Request:       req,
	}

	jsonData, err := json.Marshal(aReq)
	if err != nil {
		return nil, errors.Wrap(errors.APIClientError, errors.JSONError, err)
	}

	response, err := srv.post(url, jsonData)
	if err != nil {
		return nil, err
	}

	result, ok := response.Result.(map[string]interface{})
	if !ok {
		return nil, errors.New(errors.APIClientError, errors.JSONError)
	}

	cert, ok := result["certificate"].(string)
	if !ok {
		return nil, errors.New(errors.APIClientError, errors.JSONError)
	}

	return []byte(cert), nil
}

// Sign sends a signature request to the remote CFSSL server,
// receiving a signed certificate or an error in response.
// It takes the serialized JSON request to send.
func (srv *server) Sign(jsonData []byte) ([]byte, error) {
	return srv.request(jsonData, "sign")
}

// Info sends an info request to the remote CFSSL server, receiving a
// response or an error in response.
// It takes the serialized JSON request to send.
func (srv *server) Info(jsonData []byte) (*info.Resp, error) {
	res, err := srv.getResultMap(jsonData, "info")
	if err != nil {
		return nil, err
	}

	info := new(info.Resp)

	if val, ok := res["certificate"]; ok {
		info.Certificate = val.(string)
	}
	var usages []interface{}
	if val, ok := res["usages"]; ok && val != nil {
		usages = val.([]interface{})
	}
	if val, ok := res["expiry"]; ok && val != nil {
		info.ExpiryString = val.(string)
	}

	info.Usage = make([]string, len(usages))
	for i, s := range usages {
		info.Usage[i] = s.(string)
	}

	return info, nil
}

func (srv *server) getResultMap(jsonData []byte, target string) (result map[string]interface{}, err error) {
	url := srv.getURL(target)
	response, err := srv.post(url, jsonData)
	if err != nil {
		return
	}
	result, ok := response.Result.(map[string]interface{})
	if !ok {
		err = errors.Wrap(errors.APIClientError, errors.ClientHTTPError, stderr.New("response is formatted improperly"))
		return
	}
	return
}

// request performs the common logic for Sign and Info, performing the actual
// request and returning the resultant certificate.
func (srv *server) request(jsonData []byte, target string) ([]byte, error) {
	result, err := srv.getResultMap(jsonData, target)
	if err != nil {
		return nil, err
	}
	cert := result["certificate"].(string)
	if cert != "" {
		return []byte(cert), nil
	}

	return nil, errors.Wrap(errors.APIClientError, errors.ClientHTTPError, stderr.New("response doesn't contain certificate."))
}

// AuthRemote acts as a Remote with a default Provider for AuthSign.
type AuthRemote struct {
	Remote
	provider auth.Provider
}

// NewAuthServer sets up a new auth server target with an addr
// in the same format at NewServer and a default authentication provider to
// use for Sign requests.
func NewAuthServer(addr string, tlsConfig *tls.Config, provider auth.Provider) *AuthRemote {
	return &AuthRemote{
		Remote:   NewServerTLS(addr, tlsConfig),
		provider: provider,
	}
}

// Sign is overloaded to perform an AuthSign request using the default auth provider.
func (ar *AuthRemote) Sign(req []byte) ([]byte, error) {
	return ar.AuthSign(req, nil, ar.provider)
}

// nomalizeURL checks for http/https protocol, appends "http" as default protocol if not defiend in url
func normalizeURL(addr string) (*url.URL, error) {
	addr = strings.TrimSpace(addr)

	u, err := url.Parse(addr)
	if err != nil {
		return nil, err
	}

	if u.Opaque != "" {
		u.Host = net.JoinHostPort(u.Scheme, u.Opaque)
		u.Opaque = ""
	} else if u.Path != "" && !strings.Contains(u.Path, ":") {
		u.Host = net.JoinHostPort(u.Path, "8888")
		u.Path = ""
	} else if u.Scheme == "" {
		u.Host = u.Path
		u.Path = ""
	}

	if u.Scheme != "https" {
		u.Scheme = "http"
	}

	_, port, err := net.SplitHostPort(u.Host)
	if err != nil {
		_, port, err = net.SplitHostPort(u.Host + ":8888")
		if err != nil {
			return nil, err
		}
	}

	if port != "" {
		_, err = strconv.Atoi(port)
		if err != nil {
			return nil, err
		}
	}
	return u, nil
}
