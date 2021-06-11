/*
Copyright (c) 2018 VMware, Inc. All Rights Reserved.

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

package rest

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"
	"sync"
	"time"

	"github.com/vmware/govmomi/vapi/internal"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
)

// Client extends soap.Client to support JSON encoding, while inheriting security features, debug tracing and session persistence.
type Client struct {
	mu sync.Mutex

	*soap.Client
	sessionID string
}

// Session information
type Session struct {
	User         string    `json:"user"`
	Created      time.Time `json:"created_time"`
	LastAccessed time.Time `json:"last_accessed_time"`
}

// LocalizableMessage represents a localizable error
type LocalizableMessage struct {
	Args           []string `json:"args,omitempty"`
	DefaultMessage string   `json:"default_message,omitempty"`
	ID             string   `json:"id,omitempty"`
}

func (m *LocalizableMessage) Error() string {
	return m.DefaultMessage
}

// NewClient creates a new Client instance.
func NewClient(c *vim25.Client) *Client {
	sc := c.Client.NewServiceClient(Path, "")

	return &Client{Client: sc}
}

// SessionID is set by calling Login() or optionally with the given id param
func (c *Client) SessionID(id ...string) string {
	c.mu.Lock()
	defer c.mu.Unlock()
	if len(id) != 0 {
		c.sessionID = id[0]
	}
	return c.sessionID
}

type marshaledClient struct {
	SoapClient *soap.Client
	SessionID  string
}

func (c *Client) MarshalJSON() ([]byte, error) {
	m := marshaledClient{
		SoapClient: c.Client,
		SessionID:  c.sessionID,
	}

	return json.Marshal(m)
}

func (c *Client) UnmarshalJSON(b []byte) error {
	var m marshaledClient

	err := json.Unmarshal(b, &m)
	if err != nil {
		return err
	}

	*c = Client{
		Client:    m.SoapClient,
		sessionID: m.SessionID,
	}

	return nil
}

// isAPI returns true if path starts with "/api"
// This hack allows helpers to support both endpoints:
// "/rest" - value wrapped responses and structured error responses
// "/api" - raw responses and no structured error responses
func isAPI(path string) bool {
	return strings.HasPrefix(path, "/api")
}

// Resource helper for the given path.
func (c *Client) Resource(path string) *Resource {
	r := &Resource{u: c.URL()}
	if !isAPI(path) {
		path = Path + path
	}
	r.u.Path = path
	return r
}

type Signer interface {
	SignRequest(*http.Request) error
}

type signerContext struct{}

func (c *Client) WithSigner(ctx context.Context, s Signer) context.Context {
	return context.WithValue(ctx, signerContext{}, s)
}

type statusError struct {
	res *http.Response
}

func (e *statusError) Error() string {
	return fmt.Sprintf("%s %s: %s", e.res.Request.Method, e.res.Request.URL, e.res.Status)
}

// Do sends the http.Request, decoding resBody if provided.
func (c *Client) Do(ctx context.Context, req *http.Request, resBody interface{}) error {
	switch req.Method {
	case http.MethodPost, http.MethodPatch:
		req.Header.Set("Content-Type", "application/json")
	}

	req.Header.Set("Accept", "application/json")

	if id := c.SessionID(); id != "" {
		req.Header.Set(internal.SessionCookieName, id)
	}

	if s, ok := ctx.Value(signerContext{}).(Signer); ok {
		if err := s.SignRequest(req); err != nil {
			return err
		}
	}

	return c.Client.Do(ctx, req, func(res *http.Response) error {
		switch res.StatusCode {
		case http.StatusOK:
		case http.StatusCreated:
		case http.StatusNoContent:
		case http.StatusBadRequest:
			// TODO: structured error types
			detail, err := ioutil.ReadAll(res.Body)
			if err != nil {
				return err
			}
			return fmt.Errorf("%s: %s", res.Status, bytes.TrimSpace(detail))
		default:
			return &statusError{res}
		}

		if resBody == nil {
			return nil
		}

		switch b := resBody.(type) {
		case io.Writer:
			_, err := io.Copy(b, res.Body)
			return err
		default:
			d := json.NewDecoder(res.Body)
			if isAPI(req.URL.Path) {
				// Responses from the /api endpoint are not wrapped
				return d.Decode(resBody)
			}
			// Responses from the /rest endpoint are wrapped in this structure
			val := struct {
				Value interface{} `json:"value,omitempty"`
			}{
				resBody,
			}
			return d.Decode(&val)
		}
	})
}

// authHeaders ensures the given map contains a REST auth header
func (c *Client) authHeaders(h map[string]string) map[string]string {
	if _, exists := h[internal.SessionCookieName]; exists {
		return h
	}
	if h == nil {
		h = make(map[string]string)
	}

	h[internal.SessionCookieName] = c.SessionID()

	return h
}

// Download wraps soap.Client.Download, adding the REST authentication header
func (c *Client) Download(ctx context.Context, u *url.URL, param *soap.Download) (io.ReadCloser, int64, error) {
	p := *param
	p.Headers = c.authHeaders(p.Headers)
	return c.Client.Download(ctx, u, &p)
}

// DownloadFile wraps soap.Client.DownloadFile, adding the REST authentication header
func (c *Client) DownloadFile(ctx context.Context, file string, u *url.URL, param *soap.Download) error {
	p := *param
	p.Headers = c.authHeaders(p.Headers)
	return c.Client.DownloadFile(ctx, file, u, &p)
}

// Upload wraps soap.Client.Upload, adding the REST authentication header
func (c *Client) Upload(ctx context.Context, f io.Reader, u *url.URL, param *soap.Upload) error {
	p := *param
	p.Headers = c.authHeaders(p.Headers)
	return c.Client.Upload(ctx, f, u, &p)
}

// Login creates a new session via Basic Authentication with the given url.Userinfo.
func (c *Client) Login(ctx context.Context, user *url.Userinfo) error {
	req := c.Resource(internal.SessionPath).Request(http.MethodPost)

	req.Header.Set(internal.UseHeaderAuthn, "true")

	if user != nil {
		if password, ok := user.Password(); ok {
			req.SetBasicAuth(user.Username(), password)
		}
	}

	var id string
	err := c.Do(ctx, req, &id)
	if err != nil {
		return err
	}

	c.SessionID(id)

	return nil
}

func (c *Client) LoginByToken(ctx context.Context) error {
	return c.Login(ctx, nil)
}

// Session returns the user's current session.
// Nil is returned if the session is not authenticated.
func (c *Client) Session(ctx context.Context) (*Session, error) {
	var s Session
	req := c.Resource(internal.SessionPath).WithAction("get").Request(http.MethodPost)
	err := c.Do(ctx, req, &s)
	if err != nil {
		if e, ok := err.(*statusError); ok {
			if e.res.StatusCode == http.StatusUnauthorized {
				return nil, nil
			}
		}
		return nil, err
	}
	return &s, nil
}

// Logout deletes the current session.
func (c *Client) Logout(ctx context.Context) error {
	req := c.Resource(internal.SessionPath).Request(http.MethodDelete)
	return c.Do(ctx, req, nil)
}

// Valid returns whether or not the client is valid and ready for use.
// This should be called after unmarshalling the client.
func (c *Client) Valid() bool {
	if c == nil {
		return false
	}

	if c.Client == nil {
		return false
	}

	return true
}

// Path returns rest.Path (see cache.Client)
func (c *Client) Path() string {
	return Path
}
