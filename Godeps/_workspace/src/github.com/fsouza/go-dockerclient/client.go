// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package docker provides a client for the Docker remote API.
//
// See http://goo.gl/mxyql for more details on the remote API.
package docker

import (
	"bytes"
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
)

const userAgent = "go-dockerclient"

var (
	// ErrInvalidEndpoint is returned when the endpoint is not a valid HTTP URL.
	ErrInvalidEndpoint = errors.New("invalid endpoint")

	// ErrConnectionRefused is returned when the client cannot connect to the given endpoint.
	ErrConnectionRefused = errors.New("cannot connect to Docker endpoint")

	apiVersion_1_12, _ = NewApiVersion("1.12")
)

// ApiVersion is an internal representation of a version of the Remote API.
type ApiVersion []int

// NewApiVersion returns an instance of ApiVersion for the given string.
//
// The given string must be in the form <major>.<minor>.<patch>, where <major>,
// <minor> and <patch> are integer numbers.
func NewApiVersion(input string) (ApiVersion, error) {
	if !strings.Contains(input, ".") {
		return nil, fmt.Errorf("Unable to parse version %q", input)
	}
	arr := strings.Split(input, ".")
	ret := make(ApiVersion, len(arr))
	var err error
	for i, val := range arr {
		ret[i], err = strconv.Atoi(val)
		if err != nil {
			return nil, fmt.Errorf("Unable to parse version %q: %q is not an integer", input, val)
		}
	}
	return ret, nil
}

func (version ApiVersion) String() string {
	var str string
	for i, val := range version {
		str += strconv.Itoa(val)
		if i < len(version)-1 {
			str += "."
		}
	}
	return str
}

func (version ApiVersion) LessThan(other ApiVersion) bool {
	return version.compare(other) < 0
}

func (version ApiVersion) LessThanOrEqualTo(other ApiVersion) bool {
	return version.compare(other) <= 0
}

func (version ApiVersion) GreaterThan(other ApiVersion) bool {
	return version.compare(other) > 0
}

func (version ApiVersion) GreaterThanOrEqualTo(other ApiVersion) bool {
	return version.compare(other) >= 0
}

func (version ApiVersion) compare(other ApiVersion) int {
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

	endpoint            string
	endpointURL         *url.URL
	eventMonitor        *eventMonitoringState
	requestedApiVersion ApiVersion
	serverApiVersion    ApiVersion
	expectedApiVersion  ApiVersion
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

// NewVersionedClient returns a Client instance ready for communication with
// the given server endpoint, using a specific remote API version.
func NewVersionedClient(endpoint string, apiVersionString string) (*Client, error) {
	u, err := parseEndpoint(endpoint)
	if err != nil {
		return nil, err
	}
	var requestedApiVersion ApiVersion
	if strings.Contains(apiVersionString, ".") {
		requestedApiVersion, err = NewApiVersion(apiVersionString)
		if err != nil {
			return nil, err
		}
	}
	return &Client{
		HTTPClient:          http.DefaultClient,
		endpoint:            endpoint,
		endpointURL:         u,
		eventMonitor:        new(eventMonitoringState),
		requestedApiVersion: requestedApiVersion,
	}, nil
}

func (c *Client) checkApiVersion() error {
	serverApiVersionString, err := c.getServerApiVersionString()
	if err != nil {
		return err
	}
	c.serverApiVersion, err = NewApiVersion(serverApiVersionString)
	if err != nil {
		return err
	}
	if c.requestedApiVersion == nil {
		c.expectedApiVersion = c.serverApiVersion
	} else {
		c.expectedApiVersion = c.requestedApiVersion
	}
	return nil
}

// Ping pings the docker server
//
// See http://goo.gl/stJENm for more details.
func (c *Client) Ping() error {
	path := "/_ping"
	body, status, err := c.do("GET", path, nil)
	if err != nil {
		return err
	}
	if status != http.StatusOK {
		return newError(status, body)
	}
	return nil
}

func (c *Client) getServerApiVersionString() (version string, err error) {
	body, status, err := c.do("GET", "/version", nil)
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

func (c *Client) do(method, path string, data interface{}) ([]byte, int, error) {
	var params io.Reader
	if data != nil {
		buf, err := json.Marshal(data)
		if err != nil {
			return nil, -1, err
		}
		params = bytes.NewBuffer(buf)
	}
	if path != "/version" && !c.SkipServerVersionCheck && c.expectedApiVersion == nil {
		err := c.checkApiVersion()
		if err != nil {
			return nil, -1, err
		}
	}
	req, err := http.NewRequest(method, c.getURL(path), params)
	if err != nil {
		return nil, -1, err
	}
	req.Header.Set("User-Agent", userAgent)
	if data != nil {
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

func (c *Client) stream(method, path string, setRawTerminal bool, headers map[string]string, in io.Reader, stdout, stderr io.Writer) error {
	if (method == "POST" || method == "PUT") && in == nil {
		in = bytes.NewReader(nil)
	}
	if path != "/version" && !c.SkipServerVersionCheck && c.expectedApiVersion == nil {
		err := c.checkApiVersion()
		if err != nil {
			return err
		}
	}
	req, err := http.NewRequest(method, c.getURL(path), in)
	if err != nil {
		return err
	}
	req.Header.Set("User-Agent", userAgent)
	if method == "POST" {
		req.Header.Set("Content-Type", "plain/text")
	}
	for key, val := range headers {
		req.Header.Set(key, val)
	}
	var resp *http.Response
	protocol := c.endpointURL.Scheme
	address := c.endpointURL.Path
	if stdout == nil {
		stdout = ioutil.Discard
	}
	if stderr == nil {
		stderr = ioutil.Discard
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
		dec := json.NewDecoder(resp.Body)
		for {
			var m jsonMessage
			if err := dec.Decode(&m); err == io.EOF {
				break
			} else if err != nil {
				return err
			}
			if m.Stream != "" {
				fmt.Fprint(stdout, m.Stream)
			} else if m.Progress != "" {
				fmt.Fprintf(stdout, "%s %s\r", m.Status, m.Progress)
			} else if m.Error != "" {
				return errors.New(m.Error)
			}
			if m.Status != "" {
				fmt.Fprintln(stdout, m.Status)
			}
		}
	} else {
		if setRawTerminal {
			_, err = io.Copy(stdout, resp.Body)
		} else {
			_, err = stdCopy(stdout, stderr, resp.Body)
		}
		return err
	}
	return nil
}

func (c *Client) hijack(method, path string, success chan struct{}, setRawTerminal bool, in io.Reader, stderr, stdout io.Writer) error {
	if path != "/version" && !c.SkipServerVersionCheck && c.expectedApiVersion == nil {
		err := c.checkApiVersion()
		if err != nil {
			return err
		}
	}
	if stdout == nil {
		stdout = ioutil.Discard
	}
	if stderr == nil {
		stderr = ioutil.Discard
	}
	req, err := http.NewRequest(method, c.getURL(path), nil)
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
	dial, err := net.Dial(protocol, address)
	if err != nil {
		return err
	}
	defer dial.Close()
	clientconn := httputil.NewClientConn(dial, nil)
	clientconn.Do(req)
	if success != nil {
		success <- struct{}{}
		<-success
	}
	rwc, br := clientconn.Hijack()
	errs := make(chan error, 2)
	exit := make(chan bool)
	go func() {
		defer close(exit)
		var err error
		if setRawTerminal {
			_, err = io.Copy(stdout, br)
		} else {
			_, err = stdCopy(stdout, stderr, br)
		}
		errs <- err
	}()
	go func() {
		var err error
		if in != nil {
			_, err = io.Copy(rwc, in)
		}
		rwc.(interface {
			CloseWrite() error
		}).CloseWrite()
		errs <- err
	}()
	<-exit
	return <-errs
}

func (c *Client) getURL(path string) string {
	urlStr := strings.TrimRight(c.endpointURL.String(), "/")
	if c.endpointURL.Scheme == "unix" {
		urlStr = ""
	}

	if c.requestedApiVersion != nil {
		return fmt.Sprintf("%s/v%s%s", urlStr, c.requestedApiVersion, path)
	} else {
		return fmt.Sprintf("%s%s", urlStr, path)
	}
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
		v := value.Field(i)
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
		}
	}
	return items.Encode()
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

func parseEndpoint(endpoint string) (*url.URL, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return nil, ErrInvalidEndpoint
	}
	if u.Scheme == "tcp" {
		u.Scheme = "http"
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
