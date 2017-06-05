// Copyright 2015 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package prometheus provides bindings to the Prometheus HTTP API:
// http://prometheus.io/docs/querying/api/
package prometheus

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/prometheus/common/model"
	"golang.org/x/net/context"
	"golang.org/x/net/context/ctxhttp"
)

const (
	statusAPIError = 422
	apiPrefix      = "/api/v1"

	epQuery       = "/query"
	epQueryRange  = "/query_range"
	epLabelValues = "/label/:name/values"
	epSeries      = "/series"
)

type ErrorType string

const (
	// The different API error types.
	ErrBadData     ErrorType = "bad_data"
	ErrTimeout               = "timeout"
	ErrCanceled              = "canceled"
	ErrExec                  = "execution"
	ErrBadResponse           = "bad_response"
)

// Error is an error returned by the API.
type Error struct {
	Type ErrorType
	Msg  string
}

func (e *Error) Error() string {
	return fmt.Sprintf("%s: %s", e.Type, e.Msg)
}

// CancelableTransport is like net.Transport but provides
// per-request cancelation functionality.
type CancelableTransport interface {
	http.RoundTripper
	CancelRequest(req *http.Request)
}

var DefaultTransport CancelableTransport = &http.Transport{
	Proxy: http.ProxyFromEnvironment,
	Dial: (&net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}).Dial,
	TLSHandshakeTimeout: 10 * time.Second,
}

// Config defines configuration parameters for a new client.
type Config struct {
	// The address of the Prometheus to connect to.
	Address string

	// Transport is used by the Client to drive HTTP requests. If not
	// provided, DefaultTransport will be used.
	Transport CancelableTransport
}

func (cfg *Config) transport() CancelableTransport {
	if cfg.Transport == nil {
		return DefaultTransport
	}
	return cfg.Transport
}

type Client interface {
	url(ep string, args map[string]string) *url.URL
	do(context.Context, *http.Request) (*http.Response, []byte, error)
}

// New returns a new Client.
func New(cfg Config) (Client, error) {
	u, err := url.Parse(cfg.Address)
	if err != nil {
		return nil, err
	}
	u.Path = apiPrefix

	return &httpClient{
		endpoint:  u,
		transport: cfg.transport(),
	}, nil
}

type httpClient struct {
	endpoint  *url.URL
	transport CancelableTransport
}

func (c *httpClient) url(ep string, args map[string]string) *url.URL {
	p := path.Join(c.endpoint.Path, ep)

	for arg, val := range args {
		arg = ":" + arg
		p = strings.Replace(p, arg, val, -1)
	}

	u := *c.endpoint
	u.Path = p

	return &u
}

func (c *httpClient) do(ctx context.Context, req *http.Request) (*http.Response, []byte, error) {
	resp, err := ctxhttp.Do(ctx, &http.Client{Transport: c.transport}, req)

	defer func() {
		if resp != nil {
			resp.Body.Close()
		}
	}()

	if err != nil {
		return nil, nil, err
	}

	var body []byte
	done := make(chan struct{})
	go func() {
		body, err = ioutil.ReadAll(resp.Body)
		close(done)
	}()

	select {
	case <-ctx.Done():
		err = resp.Body.Close()
		<-done
		if err == nil {
			err = ctx.Err()
		}
	case <-done:
	}

	return resp, body, err
}

// apiClient wraps a regular client and processes successful API responses.
// Successful also includes responses that errored at the API level.
type apiClient struct {
	Client
}

type apiResponse struct {
	Status    string          `json:"status"`
	Data      json.RawMessage `json:"data"`
	ErrorType ErrorType       `json:"errorType"`
	Error     string          `json:"error"`
}

func (c apiClient) do(ctx context.Context, req *http.Request) (*http.Response, []byte, error) {
	resp, body, err := c.Client.do(ctx, req)
	if err != nil {
		return resp, body, err
	}

	code := resp.StatusCode

	if code/100 != 2 && code != statusAPIError {
		return resp, body, &Error{
			Type: ErrBadResponse,
			Msg:  fmt.Sprintf("bad response code %d", resp.StatusCode),
		}
	}

	var result apiResponse

	if err = json.Unmarshal(body, &result); err != nil {
		return resp, body, &Error{
			Type: ErrBadResponse,
			Msg:  err.Error(),
		}
	}

	if (code == statusAPIError) != (result.Status == "error") {
		err = &Error{
			Type: ErrBadResponse,
			Msg:  "inconsistent body for response code",
		}
	}

	if code == statusAPIError && result.Status == "error" {
		err = &Error{
			Type: result.ErrorType,
			Msg:  result.Error,
		}
	}

	return resp, []byte(result.Data), err
}

// Range represents a sliced time range.
type Range struct {
	// The boundaries of the time range.
	Start, End time.Time
	// The maximum time between two slices within the boundaries.
	Step time.Duration
}

// queryResult contains result data for a query.
type queryResult struct {
	Type   model.ValueType `json:"resultType"`
	Result interface{}     `json:"result"`

	// The decoded value.
	v model.Value
}

func (qr *queryResult) UnmarshalJSON(b []byte) error {
	v := struct {
		Type   model.ValueType `json:"resultType"`
		Result json.RawMessage `json:"result"`
	}{}

	err := json.Unmarshal(b, &v)
	if err != nil {
		return err
	}

	switch v.Type {
	case model.ValScalar:
		var sv model.Scalar
		err = json.Unmarshal(v.Result, &sv)
		qr.v = &sv

	case model.ValVector:
		var vv model.Vector
		err = json.Unmarshal(v.Result, &vv)
		qr.v = vv

	case model.ValMatrix:
		var mv model.Matrix
		err = json.Unmarshal(v.Result, &mv)
		qr.v = mv

	default:
		err = fmt.Errorf("unexpected value type %q", v.Type)
	}
	return err
}

// QueryAPI provides bindings the Prometheus's query API.
type QueryAPI interface {
	// Query performs a query for the given time.
	Query(ctx context.Context, query string, ts time.Time) (model.Value, error)
	// Query performs a query for the given range.
	QueryRange(ctx context.Context, query string, r Range) (model.Value, error)
}

// NewQueryAPI returns a new QueryAPI for the client.
func NewQueryAPI(c Client) QueryAPI {
	return &httpQueryAPI{client: apiClient{c}}
}

type httpQueryAPI struct {
	client Client
}

func (h *httpQueryAPI) Query(ctx context.Context, query string, ts time.Time) (model.Value, error) {
	u := h.client.url(epQuery, nil)
	q := u.Query()

	q.Set("query", query)
	q.Set("time", ts.Format(time.RFC3339Nano))

	u.RawQuery = q.Encode()

	req, _ := http.NewRequest("GET", u.String(), nil)

	_, body, err := h.client.do(ctx, req)
	if err != nil {
		return nil, err
	}

	var qres queryResult
	err = json.Unmarshal(body, &qres)

	return model.Value(qres.v), err
}

func (h *httpQueryAPI) QueryRange(ctx context.Context, query string, r Range) (model.Value, error) {
	u := h.client.url(epQueryRange, nil)
	q := u.Query()

	var (
		start = r.Start.Format(time.RFC3339Nano)
		end   = r.End.Format(time.RFC3339Nano)
		step  = strconv.FormatFloat(r.Step.Seconds(), 'f', 3, 64)
	)

	q.Set("query", query)
	q.Set("start", start)
	q.Set("end", end)
	q.Set("step", step)

	u.RawQuery = q.Encode()

	req, _ := http.NewRequest("GET", u.String(), nil)

	_, body, err := h.client.do(ctx, req)
	if err != nil {
		return nil, err
	}

	var qres queryResult
	err = json.Unmarshal(body, &qres)

	return model.Value(qres.v), err
}
