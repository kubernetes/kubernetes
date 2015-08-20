// Package client implements the a Go client for CFSSL API commands.
package client

import (
	"bytes"
	"encoding/json"
	stderr "errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/cloudflare/cfssl/api"
	"github.com/cloudflare/cfssl/auth"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/info"
)

// A server points to a single remote CFSSL instance.
type server struct {
	Address string
	Port    int
}

// A Remote points to at least one (but possibly multiple) remote CFSSL instances.
type Remote interface {
	AuthSign(req, id []byte, provider auth.Provider) ([]byte, error)
	Sign(jsonData []byte) ([]byte, error)
	Info(jsonData []byte) (*info.Resp, error)
}

// NewServer sets up a new server target. The address should be the
// DNS name (or "name:port") of the remote CFSSL instance. If no port
// is specified, the CFSSL default port (8888) is used. If the name is
// a comma-separated list of hosts, an ordered group will be returned.
func NewServer(addr string) Remote {
	addrs := strings.Split(addr, ",")

	var remote Remote

	if len(addrs) > 1 {
		remote, _ = NewGroup(addrs, StrategyOrderedList)
	} else {
		srv := newServer(addr)
		if srv != nil {
			remote = srv
		}
	}
	return remote
}

func newServer(addr string) *server {
	addr = strings.TrimSpace(addr)
	host, port, err := net.SplitHostPort(addr)
	if err != nil {
		host, port, err = net.SplitHostPort(addr + ":8888")
		if err != nil {
			return nil
		}
	}

	var portno int
	if port == "" {
		portno = 8888
	} else {
		portno, err = strconv.Atoi(port)
		if err != nil {
			return nil
		}
	}

	return &server{host, portno}
}

func (srv *server) getURL(endpoint string) string {
	return fmt.Sprintf("http://%s:%d/api/v1/cfssl/%s", srv.Address, srv.Port, endpoint)
}

// post connects to the remote server and returns a Response struct
func (srv *server) post(url string, jsonData []byte) (*api.Response, error) {
	buf := bytes.NewBuffer(jsonData)
	resp, err := http.Post(url, "application/json", buf)
	if err != nil {
		return nil, errors.Wrap(errors.APIClientError, errors.ClientHTTPError, err)
	}
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, errors.Wrap(errors.APIClientError, errors.IOError, err)
	}
	resp.Body.Close()

	var response api.Response
	err = json.Unmarshal(body, &response)
	if err != nil {
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
	if val, ok := res["usages"]; ok {
		usages = val.([]interface{})
	}
	if val, ok := res["expiry"]; ok {
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
