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

	"github.com/vmware/govmomi/vapi/internal"
	"github.com/vmware/govmomi/vim25"
	"github.com/vmware/govmomi/vim25/soap"
)

// Client extends soap.Client to support JSON encoding, while inheriting security features, debug tracing and session persistence.
type Client struct {
	*soap.Client
}

// NewClient creates a new Client instance.
func NewClient(c *vim25.Client) *Client {
	sc := c.Client.NewServiceClient(internal.Path, "")

	return &Client{sc}
}

// Do sends the http.Request, decoding resBody if provided.
func (c *Client) Do(ctx context.Context, req *http.Request, resBody interface{}) error {
	switch req.Method {
	case http.MethodPost, http.MethodPatch:
		req.Header.Set("Content-Type", "application/json")
	}

	req.Header.Set("Accept", "application/json")

	return c.Client.Do(ctx, req, func(res *http.Response) error {
		switch res.StatusCode {
		case http.StatusOK:
		case http.StatusBadRequest:
			// TODO: structured error types
			detail, err := ioutil.ReadAll(res.Body)
			if err != nil {
				return err
			}
			return fmt.Errorf("%s: %s", res.Status, bytes.TrimSpace(detail))
		default:
			return fmt.Errorf("%s %s: %s", req.Method, req.URL, res.Status)
		}

		if resBody == nil {
			return nil
		}

		switch b := resBody.(type) {
		case io.Writer:
			_, err := io.Copy(b, res.Body)
			return err
		default:
			val := struct {
				Value interface{} `json:"value,omitempty"`
			}{
				resBody,
			}
			return json.NewDecoder(res.Body).Decode(&val)
		}
	})
}

// Login creates a new session via Basic Authentication with the given url.Userinfo.
func (c *Client) Login(ctx context.Context, user *url.Userinfo) error {
	req := internal.URL(c, internal.SessionPath).Request(http.MethodPost)

	if user != nil {
		if password, ok := user.Password(); ok {
			req.SetBasicAuth(user.Username(), password)
		}
	}

	return c.Do(ctx, req, nil)
}

// Logout deletes the current session.
func (c *Client) Logout(ctx context.Context) error {
	req := internal.URL(c, internal.SessionPath).Request(http.MethodDelete)
	return c.Do(ctx, req, nil)
}
