/*
Copyright (c) 2014-2015 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package soap

import (
	"bytes"
	"crypto/tls"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"

	"github.com/vmware/govmomi/vim25/progress"
	"github.com/vmware/govmomi/vim25/types"
	"github.com/vmware/govmomi/vim25/xml"
	"golang.org/x/net/context"
)

type HasFault interface {
	Fault() *Fault
}

type RoundTripper interface {
	RoundTrip(ctx context.Context, req, res HasFault) error
}

var DefaultVimNamespace = "urn:vim25"
var DefaultVimVersion = "6.0"

type Client struct {
	http.Client

	u *url.URL
	k bool // Named after curl's -k flag
	d *debugContainer
	t *http.Transport
	p *url.URL

	Namespace string // Vim namespace
	Version   string // Vim version
}

var schemeMatch = regexp.MustCompile(`^\w+://`)

// ParseURL is wrapper around url.Parse, where Scheme defaults to "https" and Path defaults to "/sdk"
func ParseURL(s string) (*url.URL, error) {
	var err error
	var u *url.URL

	if s != "" {
		// Default the scheme to https
		if !schemeMatch.MatchString(s) {
			s = "https://" + s
		}

		u, err = url.Parse(s)
		if err != nil {
			return nil, err
		}

		// Default the path to /sdk
		if u.Path == "" {
			u.Path = "/sdk"
		}

		if u.User == nil {
			u.User = url.UserPassword("", "")
		}
	}

	return u, nil
}

func NewClient(u *url.URL, insecure bool) *Client {
	c := Client{
		u: u,
		k: insecure,
		d: newDebug(),
	}

	// Initialize http.RoundTripper on client, so we can customize it below
	c.t = &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		Dial: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).Dial,
	}

	if c.u.Scheme == "https" {
		c.t.TLSClientConfig = &tls.Config{InsecureSkipVerify: c.k}
		c.t.TLSHandshakeTimeout = 10 * time.Second
	}

	c.Client.Transport = c.t
	c.Client.Jar, _ = cookiejar.New(nil)

	// Remove user information from a copy of the URL
	c.u = c.URL()
	c.u.User = nil

	c.Namespace = DefaultVimNamespace
	c.Version = DefaultVimVersion

	return &c
}

// splitHostPort is similar to net.SplitHostPort,
// but rather than return error if there isn't a ':port',
// return an empty string for the port.
func splitHostPort(host string) (string, string) {
	ix := strings.LastIndex(host, ":")

	if ix <= strings.LastIndex(host, "]") {
		return host, ""
	}

	name := host[:ix]
	port := host[ix+1:]

	return name, port
}

const sdkTunnel = "sdkTunnel:8089"

func (c *Client) SetCertificate(cert tls.Certificate) {
	t := c.Client.Transport.(*http.Transport)

	// Extension certificate
	t.TLSClientConfig.Certificates = []tls.Certificate{cert}

	// Proxy to vCenter host on port 80
	host, _ := splitHostPort(c.u.Host)

	// Should be no reason to change the default port other than testing
	port := os.Getenv("GOVC_TUNNEL_PROXY_PORT")
	if port != "" {
		host += ":" + port
	}

	c.p = &url.URL{
		Scheme: "http",
		Host:   host,
	}
	t.Proxy = func(r *http.Request) (*url.URL, error) {
		// Only sdk requests should be proxied
		if r.URL.Path == "/sdk" {
			return c.p, nil
		}
		return http.ProxyFromEnvironment(r)
	}

	// Rewrite url Host to use the sdk tunnel, required for a certificate request.
	c.u.Host = sdkTunnel
}

func (c *Client) URL() *url.URL {
	urlCopy := *c.u
	return &urlCopy
}

type marshaledClient struct {
	Cookies  []*http.Cookie
	URL      *url.URL
	Insecure bool
}

func (c *Client) MarshalJSON() ([]byte, error) {
	m := marshaledClient{
		Cookies:  c.Jar.Cookies(c.u),
		URL:      c.u,
		Insecure: c.k,
	}

	return json.Marshal(m)
}

func (c *Client) UnmarshalJSON(b []byte) error {
	var m marshaledClient

	err := json.Unmarshal(b, &m)
	if err != nil {
		return err
	}

	*c = *NewClient(m.URL, m.Insecure)
	c.Jar.SetCookies(m.URL, m.Cookies)

	return nil
}

func (c *Client) do(ctx context.Context, req *http.Request) (*http.Response, error) {
	if nil == ctx || nil == ctx.Done() { // ctx.Done() is for context.TODO()
		return c.Client.Do(req)
	}

	var resc = make(chan *http.Response, 1)
	var errc = make(chan error, 1)

	// Perform request from separate routine.
	go func() {
		res, err := c.Client.Do(req)
		if err != nil {
			errc <- err
		} else {
			resc <- res
		}
	}()

	// Wait for request completion of context expiry.
	select {
	case <-ctx.Done():
		c.t.CancelRequest(req)
		return nil, ctx.Err()
	case err := <-errc:
		return nil, err
	case res := <-resc:
		return res, nil
	}
}

func (c *Client) RoundTrip(ctx context.Context, reqBody, resBody HasFault) error {
	var err error

	reqEnv := Envelope{Body: reqBody}
	resEnv := Envelope{Body: resBody}

	// Create debugging context for this round trip
	d := c.d.newRoundTrip()
	if d.enabled() {
		defer d.done()
	}

	b, err := xml.Marshal(reqEnv)
	if err != nil {
		panic(err)
	}

	rawReqBody := io.MultiReader(strings.NewReader(xml.Header), bytes.NewReader(b))
	req, err := http.NewRequest("POST", c.u.String(), rawReqBody)
	if err != nil {
		panic(err)
	}

	req.Header.Set(`Content-Type`, `text/xml; charset="utf-8"`)
	soapAction := fmt.Sprintf("%s/%s", c.Namespace, c.Version)
	req.Header.Set(`SOAPAction`, soapAction)

	if d.enabled() {
		d.debugRequest(req)
	}

	tstart := time.Now()
	res, err := c.do(ctx, req)
	tstop := time.Now()

	if d.enabled() {
		d.logf("%6dms (%T)", tstop.Sub(tstart)/time.Millisecond, resBody)
	}

	if err != nil {
		return err
	}

	if d.enabled() {
		d.debugResponse(res)
	}

	// Close response regardless of what happens next
	defer res.Body.Close()

	switch res.StatusCode {
	case http.StatusOK:
		// OK
	case http.StatusInternalServerError:
		// Error, but typically includes a body explaining the error
	default:
		return errors.New(res.Status)
	}

	dec := xml.NewDecoder(res.Body)
	dec.TypeFunc = types.TypeFunc()
	err = dec.Decode(&resEnv)
	if err != nil {
		return err
	}

	if f := resBody.Fault(); f != nil {
		return WrapSoapFault(f)
	}

	return err
}

func (c *Client) CloseIdleConnections() {
	c.t.CloseIdleConnections()
}

// ParseURL wraps url.Parse to rewrite the URL.Host field
// In the case of VM guest uploads or NFC lease URLs, a Host
// field with a value of "*" is rewritten to the Client's URL.Host.
func (c *Client) ParseURL(urlStr string) (*url.URL, error) {
	u, err := url.Parse(urlStr)
	if err != nil {
		return nil, err
	}

	host, _ := splitHostPort(u.Host)
	if host == "*" {
		// Also use Client's port, to support port forwarding
		u.Host = c.URL().Host
	}

	return u, nil
}

type Upload struct {
	Type          string
	Method        string
	ContentLength int64
	Headers       map[string]string
	Ticket        *http.Cookie
	Progress      progress.Sinker
}

var DefaultUpload = Upload{
	Type:   "application/octet-stream",
	Method: "PUT",
}

// Upload PUTs the local file to the given URL
func (c *Client) Upload(f io.Reader, u *url.URL, param *Upload) error {
	var err error

	if param.Progress != nil {
		pr := progress.NewReader(param.Progress, f, param.ContentLength)
		f = pr

		// Mark progress reader as done when returning from this function.
		defer func() {
			pr.Done(err)
		}()
	}

	req, err := http.NewRequest(param.Method, u.String(), f)
	if err != nil {
		return err
	}

	req.ContentLength = param.ContentLength
	req.Header.Set("Content-Type", param.Type)

	for k, v := range param.Headers {
		req.Header.Add(k, v)
	}

	if param.Ticket != nil {
		req.AddCookie(param.Ticket)
	}

	res, err := c.Client.Do(req)
	if err != nil {
		return err
	}

	switch res.StatusCode {
	case http.StatusOK:
	case http.StatusCreated:
	default:
		err = errors.New(res.Status)
	}

	return err
}

// UploadFile PUTs the local file to the given URL
func (c *Client) UploadFile(file string, u *url.URL, param *Upload) error {
	if param == nil {
		p := DefaultUpload // Copy since we set ContentLength
		param = &p
	}

	s, err := os.Stat(file)
	if err != nil {
		return err
	}

	f, err := os.Open(file)
	if err != nil {
		return err
	}
	defer f.Close()

	param.ContentLength = s.Size()

	return c.Upload(f, u, param)
}

type Download struct {
	Method   string
	Ticket   *http.Cookie
	Progress progress.Sinker
}

var DefaultDownload = Download{
	Method: "GET",
}

// Download GETs the remote file from the given URL
func (c *Client) Download(u *url.URL, param *Download) (io.ReadCloser, int64, error) {

	req, err := http.NewRequest(param.Method, u.String(), nil)
	if err != nil {
		return nil, 0, err
	}

	if param.Ticket != nil {
		req.AddCookie(param.Ticket)
	}

	res, err := c.Client.Do(req)
	if err != nil {
		return nil, 0, err
	}

	switch res.StatusCode {
	case http.StatusOK:
	default:
		err = errors.New(res.Status)
	}

	if err != nil {
		return nil, 0, err
	}

	var r io.ReadCloser = res.Body

	return r, res.ContentLength, nil
}

// DownloadFile GETs the given URL to a local file
func (c *Client) DownloadFile(file string, u *url.URL, param *Download) error {
	var err error
	if param == nil {
		param = &DefaultDownload
	}

	rc, contentLength, err := c.Download(u, param)
	if err != nil {
		return err
	}
	defer rc.Close()

	var r io.Reader = rc

	fh, err := os.Create(file)
	if err != nil {
		return err
	}
	defer fh.Close()

	if param.Progress != nil {
		pr := progress.NewReader(param.Progress, r, contentLength)
		r = pr

		// Mark progress reader as done when returning from this function.
		defer func() {
			pr.Done(err)
		}()
	}

	_, err = io.Copy(fh, r)
	if err != nil {
		return err
	}

	// Assign error before returning so that it gets picked up by the deferred
	// function marking the progress reader as done.
	err = fh.Close()
	if err != nil {
		return err
	}

	return nil
}
