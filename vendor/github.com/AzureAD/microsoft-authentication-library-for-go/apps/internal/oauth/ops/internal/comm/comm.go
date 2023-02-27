// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// Package comm provides helpers for communicating with HTTP backends.
package comm

import (
	"bytes"
	"context"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"reflect"
	"runtime"
	"strings"
	"time"

	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/errors"
	customJSON "github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/json"
	"github.com/AzureAD/microsoft-authentication-library-for-go/apps/internal/version"
	"github.com/google/uuid"
)

// HTTPClient represents an HTTP client.
// It's usually an *http.Client from the standard library.
type HTTPClient interface {
	// Do sends an HTTP request and returns an HTTP response.
	Do(req *http.Request) (*http.Response, error)

	// CloseIdleConnections closes any idle connections in a "keep-alive" state.
	CloseIdleConnections()
}

// Client provides a wrapper to our *http.Client that handles compression and serialization needs.
type Client struct {
	client HTTPClient
}

// New returns a new Client object.
func New(httpClient HTTPClient) *Client {
	if httpClient == nil {
		panic("http.Client cannot == nil")
	}

	return &Client{client: httpClient}
}

// JSONCall connects to the REST endpoint passing the HTTP query values, headers and JSON conversion
// of body in the HTTP body. It automatically handles compression and decompression with gzip. The response is JSON
// unmarshalled into resp. resp must be a pointer to a struct. If the body struct contains a field called
// "AdditionalFields" we use a custom marshal/unmarshal engine.
func (c *Client) JSONCall(ctx context.Context, endpoint string, headers http.Header, qv url.Values, body, resp interface{}) error {
	if qv == nil {
		qv = url.Values{}
	}

	v := reflect.ValueOf(resp)
	if err := c.checkResp(v); err != nil {
		return err
	}

	// Choose a JSON marshal/unmarshal depending on if we have AdditionalFields attribute.
	var marshal = json.Marshal
	var unmarshal = json.Unmarshal
	if _, ok := v.Elem().Type().FieldByName("AdditionalFields"); ok {
		marshal = customJSON.Marshal
		unmarshal = customJSON.Unmarshal
	}

	u, err := url.Parse(endpoint)
	if err != nil {
		return fmt.Errorf("could not parse path URL(%s): %w", endpoint, err)
	}
	u.RawQuery = qv.Encode()

	addStdHeaders(headers)

	req := &http.Request{Method: http.MethodGet, URL: u, Header: headers}

	if body != nil {
		// Note: In case your wondering why we are not gzip encoding....
		// I'm not sure if these various services support gzip on send.
		headers.Add("Content-Type", "application/json; charset=utf-8")
		data, err := marshal(body)
		if err != nil {
			return fmt.Errorf("bug: conn.Call(): could not marshal the body object: %w", err)
		}
		req.Body = io.NopCloser(bytes.NewBuffer(data))
		req.Method = http.MethodPost
	}

	data, err := c.do(ctx, req)
	if err != nil {
		return err
	}

	if resp != nil {
		if err := unmarshal(data, resp); err != nil {
			return fmt.Errorf("json decode error: %w\njson message bytes were: %s", err, string(data))
		}
	}
	return nil
}

// XMLCall connects to an endpoint and decodes the XML response into resp. This is used when
// sending application/xml . If sending XML via SOAP, use SOAPCall().
func (c *Client) XMLCall(ctx context.Context, endpoint string, headers http.Header, qv url.Values, resp interface{}) error {
	if err := c.checkResp(reflect.ValueOf(resp)); err != nil {
		return err
	}

	if qv == nil {
		qv = url.Values{}
	}

	u, err := url.Parse(endpoint)
	if err != nil {
		return fmt.Errorf("could not parse path URL(%s): %w", endpoint, err)
	}
	u.RawQuery = qv.Encode()

	headers.Set("Content-Type", "application/xml; charset=utf-8") // This was not set in he original Mex(), but...
	addStdHeaders(headers)

	return c.xmlCall(ctx, u, headers, "", resp)
}

// SOAPCall returns the SOAP message given an endpoint, action, body of the request and the response object to marshal into.
func (c *Client) SOAPCall(ctx context.Context, endpoint, action string, headers http.Header, qv url.Values, body string, resp interface{}) error {
	if body == "" {
		return fmt.Errorf("cannot make a SOAP call with body set to empty string")
	}

	if err := c.checkResp(reflect.ValueOf(resp)); err != nil {
		return err
	}

	if qv == nil {
		qv = url.Values{}
	}

	u, err := url.Parse(endpoint)
	if err != nil {
		return fmt.Errorf("could not parse path URL(%s): %w", endpoint, err)
	}
	u.RawQuery = qv.Encode()

	headers.Set("Content-Type", "application/soap+xml; charset=utf-8")
	headers.Set("SOAPAction", action)
	addStdHeaders(headers)

	return c.xmlCall(ctx, u, headers, body, resp)
}

// xmlCall sends an XML in body and decodes into resp. This simply does the transport and relies on
// an upper level call to set things such as SOAP parameters and Content-Type, if required.
func (c *Client) xmlCall(ctx context.Context, u *url.URL, headers http.Header, body string, resp interface{}) error {
	req := &http.Request{Method: http.MethodGet, URL: u, Header: headers}

	if len(body) > 0 {
		req.Method = http.MethodPost
		req.Body = io.NopCloser(strings.NewReader(body))
	}

	data, err := c.do(ctx, req)
	if err != nil {
		return err
	}

	return xml.Unmarshal(data, resp)
}

// URLFormCall is used to make a call where we need to send application/x-www-form-urlencoded data
// to the backend and receive JSON back. qv will be encoded into the request body.
func (c *Client) URLFormCall(ctx context.Context, endpoint string, qv url.Values, resp interface{}) error {
	if len(qv) == 0 {
		return fmt.Errorf("URLFormCall() requires qv to have non-zero length")
	}

	if err := c.checkResp(reflect.ValueOf(resp)); err != nil {
		return err
	}

	u, err := url.Parse(endpoint)
	if err != nil {
		return fmt.Errorf("could not parse path URL(%s): %w", endpoint, err)
	}

	headers := http.Header{}
	headers.Set("Content-Type", "application/x-www-form-urlencoded; charset=utf-8")
	addStdHeaders(headers)

	enc := qv.Encode()

	req := &http.Request{
		Method:        http.MethodPost,
		URL:           u,
		Header:        headers,
		ContentLength: int64(len(enc)),
		Body:          io.NopCloser(strings.NewReader(enc)),
		GetBody: func() (io.ReadCloser, error) {
			return io.NopCloser(strings.NewReader(enc)), nil
		},
	}

	data, err := c.do(ctx, req)
	if err != nil {
		return err
	}

	v := reflect.ValueOf(resp)
	if err := c.checkResp(v); err != nil {
		return err
	}

	var unmarshal = json.Unmarshal
	if _, ok := v.Elem().Type().FieldByName("AdditionalFields"); ok {
		unmarshal = customJSON.Unmarshal
	}
	if resp != nil {
		if err := unmarshal(data, resp); err != nil {
			return fmt.Errorf("json decode error: %w\nraw message was: %s", err, string(data))
		}
	}
	return nil
}

// do makes the HTTP call to the server and returns the contents of the body.
func (c *Client) do(ctx context.Context, req *http.Request) ([]byte, error) {
	if _, ok := ctx.Deadline(); !ok {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, 30*time.Second)
		defer cancel()
	}
	req = req.WithContext(ctx)

	reply, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("server response error:\n %w", err)
	}
	defer reply.Body.Close()

	data, err := c.readBody(reply)
	if err != nil {
		return nil, fmt.Errorf("could not read the body of an HTTP Response: %w", err)
	}
	reply.Body = io.NopCloser(bytes.NewBuffer(data))

	// NOTE: This doesn't happen immediately after the call so that we can get an error message
	// from the server and include it in our error.
	switch reply.StatusCode {
	case 200, 201:
	default:
		sd := strings.TrimSpace(string(data))
		if sd != "" {
			// We probably have the error in the body.
			return nil, errors.CallErr{
				Req:  req,
				Resp: reply,
				Err:  fmt.Errorf("http call(%s)(%s) error: reply status code was %d:\n%s", req.URL.String(), req.Method, reply.StatusCode, sd),
			}
		}
		return nil, errors.CallErr{
			Req:  req,
			Resp: reply,
			Err:  fmt.Errorf("http call(%s)(%s) error: reply status code was %d", req.URL.String(), req.Method, reply.StatusCode),
		}
	}

	return data, nil
}

// checkResp checks a response object o make sure it is a pointer to a struct.
func (c *Client) checkResp(v reflect.Value) error {
	if v.Kind() != reflect.Ptr {
		return fmt.Errorf("bug: resp argument must a *struct, was %T", v.Interface())
	}
	v = v.Elem()
	if v.Kind() != reflect.Struct {
		return fmt.Errorf("bug: resp argument must be a *struct, was %T", v.Interface())
	}
	return nil
}

// readBody reads the body out of an *http.Response. It supports gzip encoded responses.
func (c *Client) readBody(resp *http.Response) ([]byte, error) {
	var reader io.Reader = resp.Body
	switch resp.Header.Get("Content-Encoding") {
	case "":
		// Do nothing
	case "gzip":
		reader = gzipDecompress(resp.Body)
	default:
		return nil, fmt.Errorf("bug: comm.Client.JSONCall(): content was send with unsupported content-encoding %s", resp.Header.Get("Content-Encoding"))
	}
	return io.ReadAll(reader)
}

var testID string

// addStdHeaders adds the standard headers we use on all calls.
func addStdHeaders(headers http.Header) http.Header {
	headers.Set("Accept-Encoding", "gzip")
	// So that I can have a static id for tests.
	if testID != "" {
		headers.Set("client-request-id", testID)
		headers.Set("Return-Client-Request-Id", "false")
	} else {
		headers.Set("client-request-id", uuid.New().String())
		headers.Set("Return-Client-Request-Id", "false")
	}
	headers.Set("x-client-sku", "MSAL.Go")
	headers.Set("x-client-os", runtime.GOOS)
	headers.Set("x-client-cpu", runtime.GOARCH)
	headers.Set("x-client-ver", version.Version)
	return headers
}
