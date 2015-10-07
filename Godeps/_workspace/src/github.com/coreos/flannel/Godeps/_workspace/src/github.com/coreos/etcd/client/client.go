// Copyright 2015 CoreOS, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package client

import (
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"sync"
	"time"

	"github.com/coreos/flannel/Godeps/_workspace/src/golang.org/x/net/context"
)

var (
	ErrNoEndpoints           = errors.New("client: no endpoints available")
	ErrTooManyRedirects      = errors.New("client: too many redirects")
	ErrClusterUnavailable    = errors.New("client: etcd cluster is unavailable or misconfigured")
	errTooManyRedirectChecks = errors.New("client: too many redirect checks")
)

var DefaultRequestTimeout = 5 * time.Second

var DefaultTransport CancelableTransport = &http.Transport{
	Proxy: http.ProxyFromEnvironment,
	Dial: (&net.Dialer{
		Timeout:   30 * time.Second,
		KeepAlive: 30 * time.Second,
	}).Dial,
	TLSHandshakeTimeout: 10 * time.Second,
}

type Config struct {
	// Endpoints defines a set of URLs (schemes, hosts and ports only)
	// that can be used to communicate with a logical etcd cluster. For
	// example, a three-node cluster could be provided like so:
	//
	// 	Endpoints: []string{
	//		"http://node1.example.com:2379",
	//		"http://node2.example.com:2379",
	//		"http://node3.example.com:2379",
	//	}
	//
	// If multiple endpoints are provided, the Client will attempt to
	// use them all in the event that one or more of them are unusable.
	//
	// If Client.Sync is ever called, the Client may cache an alternate
	// set of endpoints to continue operation.
	Endpoints []string

	// Transport is used by the Client to drive HTTP requests. If not
	// provided, DefaultTransport will be used.
	Transport CancelableTransport

	// CheckRedirect specifies the policy for handling HTTP redirects.
	// If CheckRedirect is not nil, the Client calls it before
	// following an HTTP redirect. The sole argument is the number of
	// requests that have alrady been made. If CheckRedirect returns
	// an error, Client.Do will not make any further requests and return
	// the error back it to the caller.
	//
	// If CheckRedirect is nil, the Client uses its default policy,
	// which is to stop after 10 consecutive requests.
	CheckRedirect CheckRedirectFunc

	// Username specifies the user credential to add as an authorization header
	Username string

	// Password is the password for the specified user to add as an authorization header
	// to the request.
	Password string
}

func (cfg *Config) transport() CancelableTransport {
	if cfg.Transport == nil {
		return DefaultTransport
	}
	return cfg.Transport
}

func (cfg *Config) checkRedirect() CheckRedirectFunc {
	if cfg.CheckRedirect == nil {
		return DefaultCheckRedirect
	}
	return cfg.CheckRedirect
}

// CancelableTransport mimics net/http.Transport, but requires that
// the object also support request cancellation.
type CancelableTransport interface {
	http.RoundTripper
	CancelRequest(req *http.Request)
}

type CheckRedirectFunc func(via int) error

// DefaultCheckRedirect follows up to 10 redirects, but no more.
var DefaultCheckRedirect CheckRedirectFunc = func(via int) error {
	if via > 10 {
		return ErrTooManyRedirects
	}
	return nil
}

type Client interface {
	// Sync updates the internal cache of the etcd cluster's membership.
	Sync(context.Context) error

	// Endpoints returns a copy of the current set of API endpoints used
	// by Client to resolve HTTP requests. If Sync has ever been called,
	// this may differ from the initial Endpoints provided in the Config.
	Endpoints() []string

	httpClient
}

func New(cfg Config) (Client, error) {
	c := &httpClusterClient{
		clientFactory: newHTTPClientFactory(cfg.transport(), cfg.checkRedirect()),
		rand:          rand.New(rand.NewSource(int64(time.Now().Nanosecond()))),
	}
	if cfg.Username != "" {
		c.credentials = &credentials{
			username: cfg.Username,
			password: cfg.Password,
		}
	}
	if err := c.reset(cfg.Endpoints); err != nil {
		return nil, err
	}
	return c, nil
}

type httpClient interface {
	Do(context.Context, httpAction) (*http.Response, []byte, error)
}

func newHTTPClientFactory(tr CancelableTransport, cr CheckRedirectFunc) httpClientFactory {
	return func(ep url.URL) httpClient {
		return &redirectFollowingHTTPClient{
			checkRedirect: cr,
			client: &simpleHTTPClient{
				transport: tr,
				endpoint:  ep,
			},
		}
	}
}

type credentials struct {
	username string
	password string
}

type httpClientFactory func(url.URL) httpClient

type httpAction interface {
	HTTPRequest(url.URL) *http.Request
}

type httpClusterClient struct {
	clientFactory httpClientFactory
	endpoints     []url.URL
	pinned        int
	credentials   *credentials
	sync.RWMutex
	rand *rand.Rand
}

func (c *httpClusterClient) reset(eps []string) error {
	if len(eps) == 0 {
		return ErrNoEndpoints
	}

	neps := make([]url.URL, len(eps))
	for i, ep := range eps {
		u, err := url.Parse(ep)
		if err != nil {
			return err
		}
		neps[i] = *u
	}

	c.endpoints = shuffleEndpoints(c.rand, neps)
	// TODO: pin old endpoint if possible, and rebalance when new endpoint appears
	c.pinned = 0

	return nil
}

func (c *httpClusterClient) Do(ctx context.Context, act httpAction) (*http.Response, []byte, error) {
	action := act
	c.RLock()
	leps := len(c.endpoints)
	eps := make([]url.URL, leps)
	n := copy(eps, c.endpoints)
	pinned := c.pinned

	if c.credentials != nil {
		action = &authedAction{
			act:         act,
			credentials: *c.credentials,
		}
	}
	c.RUnlock()

	if leps == 0 {
		return nil, nil, ErrNoEndpoints
	}

	if leps != n {
		return nil, nil, errors.New("unable to pick endpoint: copy failed")
	}

	var resp *http.Response
	var body []byte
	var err error
	cerr := &ClusterError{}

	for i := pinned; i < leps+pinned; i++ {
		k := i % leps
		hc := c.clientFactory(eps[k])
		resp, body, err = hc.Do(ctx, action)
		if err != nil {
			cerr.Errors = append(cerr.Errors, err)
			if err == context.DeadlineExceeded || err == context.Canceled {
				return nil, nil, cerr
			}
			continue
		}
		if resp.StatusCode/100 == 5 {
			// TODO: make sure this is a no leader response
			cerr.Errors = append(cerr.Errors, fmt.Errorf("client: etcd member %s has no leader", eps[k].String()))
			continue
		}
		if k != pinned {
			c.Lock()
			c.pinned = k
			c.Unlock()
		}
		return resp, body, nil
	}

	return nil, nil, cerr
}

func (c *httpClusterClient) Endpoints() []string {
	c.RLock()
	defer c.RUnlock()

	eps := make([]string, len(c.endpoints))
	for i, ep := range c.endpoints {
		eps[i] = ep.String()
	}

	return eps
}

func (c *httpClusterClient) Sync(ctx context.Context) error {
	mAPI := NewMembersAPI(c)
	ms, err := mAPI.List(ctx)
	if err != nil {
		return err
	}

	c.Lock()
	defer c.Unlock()

	eps := make([]string, 0)
	for _, m := range ms {
		eps = append(eps, m.ClientURLs...)
	}

	return c.reset(eps)
}

type roundTripResponse struct {
	resp *http.Response
	err  error
}

type simpleHTTPClient struct {
	transport CancelableTransport
	endpoint  url.URL
}

func (c *simpleHTTPClient) Do(ctx context.Context, act httpAction) (*http.Response, []byte, error) {
	req := act.HTTPRequest(c.endpoint)

	if err := printcURL(req); err != nil {
		return nil, nil, err
	}

	rtchan := make(chan roundTripResponse, 1)
	go func() {
		resp, err := c.transport.RoundTrip(req)
		rtchan <- roundTripResponse{resp: resp, err: err}
		close(rtchan)
	}()

	var resp *http.Response
	var err error

	select {
	case rtresp := <-rtchan:
		resp, err = rtresp.resp, rtresp.err
	case <-ctx.Done():
		// cancel and wait for request to actually exit before continuing
		c.transport.CancelRequest(req)
		rtresp := <-rtchan
		resp = rtresp.resp
		err = ctx.Err()
	}

	// always check for resp nil-ness to deal with possible
	// race conditions between channels above
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
		done <- struct{}{}
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

type authedAction struct {
	act         httpAction
	credentials credentials
}

func (a *authedAction) HTTPRequest(url url.URL) *http.Request {
	r := a.act.HTTPRequest(url)
	r.SetBasicAuth(a.credentials.username, a.credentials.password)
	return r
}

type redirectFollowingHTTPClient struct {
	client        httpClient
	checkRedirect CheckRedirectFunc
}

func (r *redirectFollowingHTTPClient) Do(ctx context.Context, act httpAction) (*http.Response, []byte, error) {
	next := act
	for i := 0; i < 100; i++ {
		if i > 0 {
			if err := r.checkRedirect(i); err != nil {
				return nil, nil, err
			}
		}
		resp, body, err := r.client.Do(ctx, next)
		if err != nil {
			return nil, nil, err
		}
		if resp.StatusCode/100 == 3 {
			hdr := resp.Header.Get("Location")
			if hdr == "" {
				return nil, nil, fmt.Errorf("Location header not set")
			}
			loc, err := url.Parse(hdr)
			if err != nil {
				return nil, nil, fmt.Errorf("Location header not valid URL: %s", hdr)
			}
			next = &redirectedHTTPAction{
				action:   act,
				location: *loc,
			}
			continue
		}
		return resp, body, nil
	}

	return nil, nil, errTooManyRedirectChecks
}

type redirectedHTTPAction struct {
	action   httpAction
	location url.URL
}

func (r *redirectedHTTPAction) HTTPRequest(ep url.URL) *http.Request {
	orig := r.action.HTTPRequest(ep)
	orig.URL = &r.location
	return orig
}

func shuffleEndpoints(r *rand.Rand, eps []url.URL) []url.URL {
	p := r.Perm(len(eps))
	neps := make([]url.URL, len(eps))
	for i, k := range p {
		neps[i] = eps[k]
	}
	return neps
}
