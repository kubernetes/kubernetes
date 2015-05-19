// Copyright 2015 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package docker provides a client for the Docker remote API.
//
// See http://goo.gl/G3plxW for more details on the remote API.
package docker

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/http/httputil"
	"net/url"
	"reflect"
	"strconv"
	"strings"

	"github.com/fsouza/go-dockerclient/vendor/github.com/docker/docker/pkg/stdcopy"
)

const userAgent = "go-dockerclient"

var (
	// ErrInvalidEndpoint is returned when the endpoint is not a valid HTTP URL.
	ErrInvalidEndpoint = errors.New("invalid endpoint")

	// ErrConnectionRefused is returned when the client cannot connect to the given endpoint.
	ErrConnectionRefused = errors.New("cannot connect to Docker endpoint")

	apiVersion112, _ = NewAPIVersion("1.12")
)

// APIVersion is an internal representation of a version of the Remote API.
type APIVersion []int

// NewAPIVersion returns an instance of APIVersion for the given string.
//
// The given string must be in the form <major>.<minor>.<patch>, where <major>,
// <minor> and <patch> are integer numbers.
func NewAPIVersion(input string) (APIVersion, error) {
	if !strings.Contains(input, ".") {
		return nil, fmt.Errorf("Unable to parse version %q", input)
	}
	arr := strings.Split(input, ".")
	ret := make(APIVersion, len(arr))
	var err error
	for i, val := range arr {
		ret[i], err = strconv.Atoi(val)
		if err != nil {
			return nil, fmt.Errorf("Unable to parse version %q: %q is not an integer", input, val)
		}
	}
	return ret, nil
}

func (version APIVersion) String() string {
	var str string
	for i, val := range version {
		str += strconv.Itoa(val)
		if i < len(version)-1 {
			str += "."
		}
	}
	return str
}

// LessThan is a function for comparing APIVersion structs
func (version APIVersion) LessThan(other APIVersion) bool {
	return version.compare(other) < 0
}

// LessThanOrEqualTo is a function for comparing APIVersion structs
func (version APIVersion) LessThanOrEqualTo(other APIVersion) bool {
	return version.compare(other) <= 0
}

// GreaterThan is a function for comparing APIVersion structs
func (version APIVersion) GreaterThan(other APIVersion) bool {
	return version.compare(other) > 0
}

// GreaterThanOrEqualTo is a function for comparing APIVersion structs
func (version APIVersion) GreaterThanOrEqualTo(other APIVersion) bool {
	return version.compare(other) >= 0
}

func (version APIVersion) compare(other APIVersion) int {
	for i, v := range version {
		if i <= len(other)-1 {
			otherVersion := other[i]

			if v < otherVersion {
				return -1
			} else if v > otherVersion {
				return 1
			}
		}
	}
	if len(version) > len(other) {
		return 1
	}
	if len(version) < len(other) {
		return -1
	}
	return 0
}

// Client is the basic type of this package. It provides methods for
// interaction with the API.
type Client struct {
	SkipServerVersionCheck bool
	HTTPClient             *http.Client
	TLSConfig              *tls.Config

	endpoint            string
	endpointURL         *url.URL
	eventMonitor        *eventMonitoringState
	requestedAPIVersion APIVersion
	serverAPIVersion    APIVersion
	expectedAPIVersion  APIVersion
}

// NewClient returns a Client instance ready for communication with the given
// server endpoint. It will use the latest remote API version available in the
// server.
func NewClient(endpoint string) (*Client, error) {
	client, err := NewVersionedClient(endpoint, "")
	if err != nil {
		return nil, err
	}
	client.SkipServerVersionCheck = true
	return client, nil
}

// NewTLSClient returns a Client instance ready for TLS communications with the givens
// server endpoint, key and certificates . It will use the latest remote API version
// available in the server.
func NewTLSClient(endpoint string, cert, key, ca string) (*Client, error) {
	client, err := NewVersionedTLSClient(endpoint, cert, key, ca, "")
	if err != nil {
		return nil, err
	}
	client.SkipServerVersionCheck = true
	return client, nil
}

// NewVersionedClient returns a Client instance ready for communication with
// the given server endpoint, using a specific remote API version.
func NewVersionedClient(endpoint string, apiVersionString string) (*Client, error) {
	u, err := parseEndpoint(endpoint, false)
	if err != nil {
		return nil, err
	}
	var requestedAPIVersion APIVersion
	if strings.Contains(apiVersionString, ".") {
		requestedAPIVersion, err = NewAPIVersion(apiVersionString)
		if err != nil {
			return nil, err
		}
	}
	return &Client{
		HTTPClient:          http.DefaultClient,
		endpoint:            endpoint,
		endpointURL:         u,
		eventMonitor:        new(eventMonitoringState),
		requestedAPIVersion: requestedAPIVersion,
	}, nil
}

// NewVersionnedTLSClient has been DEPRECATED, please use NewVersionedTLSClient.
func NewVersionnedTLSClient(endpoint string, cert, key, ca, apiVersionString string) (*Client, error) {
	return NewVersionedTLSClient(endpoint, cert, key, ca, apiVersionString)
}

// NewVersionedTLSClient returns a Client instance ready for TLS communications with the givens
// server endpoint, key and certificates, using a specific remote API version.
func NewVersionedTLSClient(endpoint string, cert, key, ca, apiVersionString string) (*Client, error) {
	u, err := parseEndpoint(endpoint, true)
	if err != nil {
		return nil, err
	}
	var requestedAPIVersion APIVersion
	if strings.Contains(apiVersionString, ".") {
		requestedAPIVersion, err = NewAPIVersion(apiVersionString)
		if err != nil {
			return nil, err
		}
	}
	if cert == "" || key == "" {
		return nil, errors.New("Both cert and key path are required")
	}
	tlsCert, err := tls.LoadX509KeyPair(cert, key)
	if err != nil {
		return nil, err
	}
	tlsConfig := &tls.Config{Certificates: []tls.Certificate{tlsCert}}
	if ca == "" {
		tlsConfig.InsecureSkipVerify = true
	} else {
		cert, err := ioutil.ReadFile(ca)
		if err != nil {
			return nil, err
		}
		caPool := x509.NewCertPool()
		if !caPool.AppendCertsFromPEM(cert) {
			return nil, errors.New("Could not add RootCA pem")
		}
		tlsConfig.RootCAs = caPool
	}
	tr := &http.Transport{
		TLSClientConfig: tlsConfig,
	}
	if err != nil {
		return nil, err
	}
	return &Client{
		HTTPClient:          &http.Client{Transport: tr},
		TLSConfig:           tlsConfig,
		endpoint:            endpoint,
		endpointURL:         u,
		eventMonitor:        new(eventMonitoringState),
		requestedAPIVersion: requestedAPIVersion,
	}, nil
}

func (c *Client) checkAPIVersion() error {
	serverAPIVersionString, err := c.getServerAPIVersionString()
	if err != nil {
		return err
	}
	c.serverAPIVersion, err = NewAPIVersion(serverAPIVersionString)
	if err != nil {
		return err
	}
	if c.requestedAPIVersion == nil {
		c.expectedAPIVersion = c.serverAPIVersion
	} else {
		c.expectedAPIVersion = c.requestedAPIVersion
	}
	return nil
}

// Ping pings the docker server
//
// See http://goo.gl/stJENm for more details.
func (c *Client) Ping() error {
	path := "/_ping"
	body, status, err := c.do("GET", path, doOptions{})
	if err != nil {
		return err
	}
	if status != http.StatusOK {
		return newError(status, body)
	}
	return nil
}

func (c *Client) getServerAPIVersionString() (version string, err error) {
	body, status, err := c.do("GET", "/version", doOptions{})
	if err != nil {
		return "", err
	}
	if status != http.StatusOK {
		return "", fmt.Errorf("Received unexpected status %d while trying to retrieve the server version", status)
	}
	var versionResponse map[string]string
	err = json.Unmarshal(body, &versionResponse)
	if err != nil {
		return "", err
	}
	version = versionResponse["ApiVersion"]
	return version, nil
}

type doOptions struct {
	data      interface{}
	forceJSON bool
}

func (c *Client) do(method, path string, doOptions doOptions) ([]byte, int, error) {
	var params io.Reader
	if doOptions.data != nil || doOptions.forceJSON {
		buf, err := json.Marshal(doOptions.data)
		if err != nil {
			return nil, -1, err
		}
		params = bytes.NewBuffer(buf)
	}
	if path != "/version" && !c.SkipServerVersionCheck && c.expectedAPIVersion == nil {
		err := c.checkAPIVersion()
		if err != nil {
			return nil, -1, err
		}
	}
	req, err := http.NewRequest(method, c.getURL(path), params)
	if err != nil {
		return nil, -1, err
	}
	req.Header.Set("User-Agent", userAgent)
	if doOptions.data != nil {
		req.Header.Set("Content-Type", "application/json")
	} else if method == "POST" {
		req.Header.Set("Content-Type", "plain/text")
	}
	var resp *http.Response
	protocol := c.endpointURL.Scheme
	address := c.endpointURL.Path
	if protocol == "unix" {
		dial, err := net.Dial(protocol, address)
		if err != nil {
			return nil, -1, err
		}
		defer dial.Close()
		clientconn := httputil.NewClientConn(dial, nil)
		resp, err = clientconn.Do(req)
		if err != nil {
			return nil, -1, err
		}
		defer clientconn.Close()
	} else {
		resp, err = c.HTTPClient.Do(req)
	}
	if err != nil {
		if strings.Contains(err.Error(), "connection refused") {
			return nil, -1, ErrConnectionRefused
		}
		return nil, -1, err
	}
	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, -1, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		return nil, resp.StatusCode, newError(resp.StatusCode, body)
	}
	return body, resp.StatusCode, nil
}

type streamOptions struct {
	setRawTerminal bool
	rawJSONStream  bool
	headers        map[string]string
	in             io.Reader
	stdout         io.Writer
	stderr         io.Writer
}

func (c *Client) stream(method, path string, streamOptions streamOptions) error {
	if (method == "POST" || method == "PUT") && streamOptions.in == nil {
		streamOptions.in = bytes.NewReader(nil)
	}
	if path != "/version" && !c.SkipServerVersionCheck && c.expectedAPIVersion == nil {
		err := c.checkAPIVersion()
		if err != nil {
			return err
		}
	}
	req, err := http.NewRequest(method, c.getURL(path), streamOptions.in)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", userAgent)
	if method == "POST" {
		req.Header.Set("Content-Type", "plain/text")
	}
	for key, val := range streamOptions.headers {
		req.Header.Set(key, val)
	}
	var resp *http.Response
	protocol := c.endpointURL.Scheme
	address := c.endpointURL.Path
	if streamOptions.stdout == nil {
		streamOptions.stdout = ioutil.Discard
	}
	if streamOptions.stderr == nil {
		streamOptions.stderr = ioutil.Discard
	}
	if protocol == "unix" {
		dial, err := net.Dial(protocol, address)
		if err != nil {
			return err
		}
		clientconn := httputil.NewClientConn(dial, nil)
		resp, err = clientconn.Do(req)
		defer clientconn.Close()
	} else {
		resp, err = c.HTTPClient.Do(req)
	}
	if err != nil {
		if strings.Contains(err.Error(), "connection refused") {
			return ErrConnectionRefused
		}
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			return err
		}
		return newError(resp.StatusCode, body)
	}
	if resp.Header.Get("Content-Type") == "application/json" {
		// if we want to get raw json stream, just copy it back to output
		// without decoding it
		if streamOptions.rawJSONStream {
			_, err = io.Copy(streamOptions.stdout, resp.Body)
			return err
		}
		dec := json.NewDecoder(resp.Body)
		for {
			var m jsonMessage
			if err := dec.Decode(&m); err == io.EOF {
				break
			} else if err != nil {
				return err
			}
			if m.Stream != "" {
				fmt.Fprint(streamOptions.stdout, m.Stream)
			} else if m.Progress != "" {
				fmt.Fprintf(streamOptions.stdout, "%s %s\r", m.Status, m.Progress)
			} else if m.Error != "" {
				return errors.New(m.Error)
			}
			if m.Status != "" {
				fmt.Fprintln(streamOptions.stdout, m.Status)
			}
		}
	} else {
		if streamOptions.setRawTerminal {
			_, err = io.Copy(streamOptions.stdout, resp.Body)
		} else {
			_, err = stdcopy.StdCopy(streamOptions.stdout, streamOptions.stderr, resp.Body)
		}
		return err
	}
	return nil
}

type hijackOptions struct {
	success        chan struct{}
	setRawTerminal bool
	in             io.Reader
	stdout         io.Writer
	stderr         io.Writer
	data           interface{}
}

func (c *Client) hijack(method, path string, hijackOptions hijackOptions) error {
	if path != "/version" && !c.SkipServerVersionCheck && c.expectedAPIVersion == nil {
		err := c.checkAPIVersion()
		if err != nil {
			return err
		}
	}

	var params io.Reader
	if hijackOptions.data != nil {
		buf, err := json.Marshal(hijackOptions.data)
		if err != nil {
			return err
		}
		params = bytes.NewBuffer(buf)
	}

	if hijackOptions.stdout == nil {
		hijackOptions.stdout = ioutil.Discard
	}
	if hijackOptions.stderr == nil {
		hijackOptions.stderr = ioutil.Discard
	}
	req, err := http.NewRequest(method, c.getURL(path), params)
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "plain/text")
	protocol := c.endpointURL.Scheme
	address := c.endpointURL.Path
	if protocol != "unix" {
		protocol = "tcp"
		address = c.endpointURL.Host
	}
	var dial net.Conn
	if c.TLSConfig != nil && protocol != "unix" {
		dial, err = tlsDial(protocol, address, c.TLSConfig)
		if err != nil {
			return err
		}
	} else {
		dial, err = net.Dial(protocol, address)
		if err != nil {
			return err
		}
	}
	clientconn := httputil.NewClientConn(dial, nil)
	defer clientconn.Close()
	clientconn.Do(req)
	if hijackOptions.success != nil {
		hijackOptions.success <- struct{}{}
		<-hijackOptions.success
	}
	rwc, br := clientconn.Hijack()
	defer rwc.Close()
	errChanOut := make(chan error, 1)
	errChanIn := make(chan error, 1)
	exit := make(chan bool)
	go func() {
		defer close(exit)
		defer close(errChanOut)
		var err error
		if hijackOptions.setRawTerminal {
			// When TTY is ON, use regular copy
			_, err = io.Copy(hijackOptions.stdout, br)
		} else {
			_, err = stdcopy.StdCopy(hijackOptions.stdout, hijackOptions.stderr, br)
		}
		errChanOut <- err
	}()
	go func() {
		if hijackOptions.in != nil {
			_, err := io.Copy(rwc, hijackOptions.in)
			errChanIn <- err
		}
		rwc.(interface {
			CloseWrite() error
		}).CloseWrite()
	}()
	<-exit
	select {
	case err = <-errChanIn:
		return err
	case err = <-errChanOut:
		return err
	}
}

func (c *Client) getURL(path string) string {
	urlStr := strings.TrimRight(c.endpointURL.String(), "/")
	if c.endpointURL.Scheme == "unix" {
		urlStr = ""
	}

	if c.requestedAPIVersion != nil {
		return fmt.Sprintf("%s/v%s%s", urlStr, c.requestedAPIVersion, path)
	}
	return fmt.Sprintf("%s%s", urlStr, path)
}

type jsonMessage struct {
	Status   string `json:"status,omitempty"`
	Progress string `json:"progress,omitempty"`
	Error    string `json:"error,omitempty"`
	Stream   string `json:"stream,omitempty"`
}

func queryString(opts interface{}) string {
	if opts == nil {
		return ""
	}
	value := reflect.ValueOf(opts)
	if value.Kind() == reflect.Ptr {
		value = value.Elem()
	}
	if value.Kind() != reflect.Struct {
		return ""
	}
	items := url.Values(map[string][]string{})
	for i := 0; i < value.NumField(); i++ {
		field := value.Type().Field(i)
		if field.PkgPath != "" {
			continue
		}
		key := field.Tag.Get("qs")
		if key == "" {
			key = strings.ToLower(field.Name)
		} else if key == "-" {
			continue
		}
		addQueryStringValue(items, key, value.Field(i))
	}
	return items.Encode()
}

func addQueryStringValue(items url.Values, key string, v reflect.Value) {
	switch v.Kind() {
	case reflect.Bool:
		if v.Bool() {
			items.Add(key, "1")
		}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		if v.Int() > 0 {
			items.Add(key, strconv.FormatInt(v.Int(), 10))
		}
	case reflect.Float32, reflect.Float64:
		if v.Float() > 0 {
			items.Add(key, strconv.FormatFloat(v.Float(), 'f', -1, 64))
		}
	case reflect.String:
		if v.String() != "" {
			items.Add(key, v.String())
		}
	case reflect.Ptr:
		if !v.IsNil() {
			if b, err := json.Marshal(v.Interface()); err == nil {
				items.Add(key, string(b))
			}
		}
	case reflect.Map:
		if len(v.MapKeys()) > 0 {
			if b, err := json.Marshal(v.Interface()); err == nil {
				items.Add(key, string(b))
			}
		}
	case reflect.Array, reflect.Slice:
		vLen := v.Len()
		if vLen > 0 {
			for i := 0; i < vLen; i++ {
				addQueryStringValue(items, key, v.Index(i))
			}
		}
	}
}

// Error represents failures in the API. It represents a failure from the API.
type Error struct {
	Status  int
	Message string
}

func newError(status int, body []byte) *Error {
	return &Error{Status: status, Message: string(body)}
}

func (e *Error) Error() string {
	return fmt.Sprintf("API error (%d): %s", e.Status, e.Message)
}

func parseEndpoint(endpoint string, tls bool) (*url.URL, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return nil, ErrInvalidEndpoint
	}
	if tls {
		u.Scheme = "https"
	}
	if u.Scheme == "tcp" {
		_, port, err := net.SplitHostPort(u.Host)
		if err != nil {
			if e, ok := err.(*net.AddrError); ok {
				if e.Err == "missing port in address" {
					return u, nil
				}
			}
			return nil, ErrInvalidEndpoint
		}

		number, err := strconv.ParseInt(port, 10, 64)
		if err == nil && number == 2376 {
			u.Scheme = "https"
		} else {
			u.Scheme = "http"
		}
	}
	if u.Scheme != "http" && u.Scheme != "https" && u.Scheme != "unix" {
		return nil, ErrInvalidEndpoint
	}
	if u.Scheme != "unix" {
		_, port, err := net.SplitHostPort(u.Host)
		if err != nil {
			if e, ok := err.(*net.AddrError); ok {
				if e.Err == "missing port in address" {
					return u, nil
				}
			}
			return nil, ErrInvalidEndpoint
		}
		number, err := strconv.ParseInt(port, 10, 64)
		if err == nil && number > 0 && number < 65536 {
			return u, nil
		}
	} else {
		return u, nil // we don't need port when using a unix socket
	}
	return nil, ErrInvalidEndpoint
}
