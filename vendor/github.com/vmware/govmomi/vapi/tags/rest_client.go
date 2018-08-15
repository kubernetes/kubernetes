// Copyright 2017 VMware, Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tags

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

	"github.com/vmware/govmomi/vim25/soap"
)

const (
	RestPrefix          = "/rest"
	loginURL            = "/com/vmware/cis/session"
	sessionIDCookieName = "vmware-api-session-id"
)

type RestClient struct {
	mu       sync.Mutex
	host     string
	scheme   string
	endpoint *url.URL
	user     *url.Userinfo
	HTTP     *http.Client
	cookies  []*http.Cookie
}

func NewClient(u *url.URL, insecure bool, thumbprint string) *RestClient {
	endpoint := &url.URL{}
	*endpoint = *u
	endpoint.Path = RestPrefix
	// Ignore "#" anchor
	endpoint.Fragment = ""

	sc := soap.NewClient(endpoint, insecure)
	if thumbprint != "" {
		sc.SetThumbprint(endpoint.Host, thumbprint)
	}

	user := endpoint.User
	endpoint.User = nil

	return &RestClient{
		endpoint: endpoint,
		user:     user,
		host:     endpoint.Host,
		scheme:   endpoint.Scheme,
		HTTP:     &sc.Client,
	}
}

// NewClientWithSessionID creates a new REST client with a supplied session ID
// to re-connect to existing sessions.
//
// Note that the session is not checked for validity - to check for a valid
// session after creating the client, use the Valid method. If the session is
// no longer valid and the session needs to be re-saved, Login should be called
// again before calling SessionID to extract the new session ID. Clients
// created with this function function work in the exact same way as clients
// created with NewClient, including supporting re-login on invalid sessions on
// all SDK calls.
func NewClientWithSessionID(u *url.URL, insecure bool, thumbprint string, sessionID string) *RestClient {
	c := NewClient(u, insecure, thumbprint)
	c.SetSessionID(sessionID)

	return c
}

func (c *RestClient) encodeData(data interface{}) (*bytes.Buffer, error) {
	params := bytes.NewBuffer(nil)
	if data != nil {
		if err := json.NewEncoder(params).Encode(data); err != nil {
			return nil, err
		}
	}
	return params, nil
}

func (c *RestClient) call(ctx context.Context, method, path string, data interface{}, headers map[string][]string) (io.ReadCloser, http.Header, int, error) {
	//	Logger.Debugf("%s: %s, headers: %+v", method, path, headers)
	params, err := c.encodeData(data)
	if err != nil {
		return nil, nil, -1, err
	}

	if data != nil {
		if headers == nil {
			headers = make(map[string][]string)
		}
		headers["Content-Type"] = []string{"application/json"}
	}

	body, hdr, statusCode, err := c.clientRequest(ctx, method, path, params, headers)
	if statusCode == http.StatusUnauthorized && strings.Contains(err.Error(), "This method requires authentication") {
		c.Login(ctx)
		return c.clientRequest(ctx, method, path, params, headers)
	}

	return body, hdr, statusCode, err
}

func (c *RestClient) clientRequest(ctx context.Context, method, path string, in io.Reader, headers map[string][]string) (io.ReadCloser, http.Header, int, error) {
	expectedPayload := (method == http.MethodPost || method == http.MethodPut)
	if expectedPayload && in == nil {
		in = bytes.NewReader([]byte{})
	}

	req, err := c.newRequest(method, path, in)
	if err != nil {
		return nil, nil, -1, err
	}

	req = req.WithContext(ctx)
	c.mu.Lock()
	if c.cookies != nil {
		req.AddCookie(c.cookies[0])
	}
	c.mu.Unlock()

	if headers != nil {
		for k, v := range headers {
			req.Header[k] = v
		}
	}

	if expectedPayload && req.Header.Get("Content-Type") == "" {
		req.Header.Set("Content-Type", "application/json")
	}
	req.Header.Set("Accept", "application/json")

	resp, err := c.HTTP.Do(req)
	return c.handleResponse(resp, err)
}

func (c *RestClient) handleResponse(resp *http.Response, err error) (io.ReadCloser, http.Header, int, error) {
	statusCode := -1
	if resp != nil {
		statusCode = resp.StatusCode
	}
	if err != nil {
		if strings.Contains(err.Error(), "connection refused") {
			return nil, nil, statusCode, err
		}
		return nil, nil, statusCode, err
	}

	if statusCode < http.StatusOK || statusCode >= http.StatusBadRequest {
		body, err := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, nil, statusCode, err
		}
		if len(body) == 0 {
			return nil, nil, statusCode, err
		}
		return nil, nil, statusCode, fmt.Errorf("error response: %s", bytes.TrimSpace(body))

	}

	return resp.Body, resp.Header, statusCode, nil
}

func (c *RestClient) Login(ctx context.Context) error {

	request, err := c.newRequest(http.MethodPost, loginURL, nil)
	if err != nil {
		return err
	}
	if c.user != nil {
		password, _ := c.user.Password()
		request.SetBasicAuth(c.user.Username(), password)
	}
	resp, err := c.HTTP.Do(request)
	if err != nil {
		return err
	}
	if resp == nil {
		return err
	}
	if resp.StatusCode != http.StatusOK {
		resp.Body.Close()
		return err
	}

	c.cookies = resp.Cookies()

	return nil
}

func (c *RestClient) Logout(ctx context.Context) error {
	_, _, status, err := c.call(ctx, http.MethodDelete, loginURL, nil, nil)
	if status != http.StatusOK || err != nil {
		return err
	}
	c.SetSessionID("")
	return nil
}

func (c *RestClient) newRequest(method, urlStr string, body io.Reader) (*http.Request, error) {
	return http.NewRequest(method, c.endpoint.String()+urlStr, body)
}

// SessionID returns the current session ID of the REST client. An empty string
// means there was no session cookie currently loaded.
func (c *RestClient) SessionID() string {
	for _, cookie := range c.cookies {
		if cookie.Name == sessionIDCookieName {
			return cookie.Value
		}
	}
	return ""
}

// SetSessionID sets the session cookie with the supplied session ID.
//
// This does not necessarily mean the session is valid. The session should be
// checked with Valid before proceeding, and logged back in if it has expired.
//
// This function will overwrite any existing session.
func (c *RestClient) SetSessionID(sessionID string) {
	idx := -1
	for i, cookie := range c.cookies {
		if cookie.Name == sessionIDCookieName {
			idx = i
		}
	}
	sessionCookie := &http.Cookie{
		Name:  sessionIDCookieName,
		Value: sessionID,
		Path:  RestPrefix,
	}
	if idx > -1 {
		c.cookies[idx] = sessionCookie
	} else {
		c.cookies = append(c.cookies, sessionCookie)
	}
}

// Valid checks to see if the session cookies in a REST client are still valid.
// This should be used when restoring a session to determine if a new login is
// necessary.
func (c *RestClient) Valid(ctx context.Context) bool {
	_, _, statusCode, err := c.clientRequest(ctx, http.MethodPost, loginURL+"?~action=get", nil, nil)
	if err != nil {
		return false
	}

	if statusCode == http.StatusOK {
		return true
	}
	return false
}
