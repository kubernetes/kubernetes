//
// Copyright (c) 2015 The heketi Authors
//
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
//

package client

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"time"

	jwt "github.com/dgrijalva/jwt-go"
	"github.com/heketi/heketi/pkg/utils"
)

const (
	MAX_CONCURRENT_REQUESTS = 32
)

// Client object
type Client struct {
	host     string
	key      string
	user     string
	throttle chan bool
}

// Creates a new client to access a Heketi server
func NewClient(host, user, key string) *Client {
	c := &Client{}

	c.key = key
	c.host = host
	c.user = user

	// Maximum concurrent requests
	c.throttle = make(chan bool, MAX_CONCURRENT_REQUESTS)

	return c
}

// Create a client to access a Heketi server without authentication enabled
func NewClientNoAuth(host string) *Client {
	return NewClient(host, "", "")
}

// Simple Hello test to check if the server is up
func (c *Client) Hello() error {
	// Create request
	req, err := http.NewRequest("GET", c.host+"/hello", nil)
	if err != nil {
		return err
	}

	// Set token
	err = c.setToken(req)
	if err != nil {
		return err
	}

	// Get info
	r, err := c.do(req)
	if err != nil {
		return err
	}
	if r.StatusCode != http.StatusOK {
		return utils.GetErrorFromResponse(r)
	}

	return nil
}

// Make sure we do not run out of fds by throttling the requests
func (c *Client) do(req *http.Request) (*http.Response, error) {
	c.throttle <- true
	defer func() {
		<-c.throttle
	}()

	httpClient := &http.Client{}
	httpClient.CheckRedirect = c.checkRedirect
	return httpClient.Do(req)
}

// This function is called by the http package if it detects that it needs to
// be redirected.  This happens when the server returns a 303 HTTP Status.
// Here we create a new token before it makes the next request.
func (c *Client) checkRedirect(req *http.Request, via []*http.Request) error {
	return c.setToken(req)
}

// Wait for the job to finish, waiting waitTime on every loop
func (c *Client) waitForResponseWithTimer(r *http.Response,
	waitTime time.Duration) (*http.Response, error) {

	// Get temp resource
	location, err := r.Location()
	if err != nil {
		return nil, err
	}

	for {
		// Create request
		req, err := http.NewRequest("GET", location.String(), nil)
		if err != nil {
			return nil, err
		}

		// Set token
		err = c.setToken(req)
		if err != nil {
			return nil, err
		}

		// Wait for response
		r, err = c.do(req)
		if err != nil {
			return nil, err
		}

		// Check if the request is pending
		if r.Header.Get("X-Pending") == "true" {
			if r.StatusCode != http.StatusOK {
				return nil, utils.GetErrorFromResponse(r)
			}
			time.Sleep(waitTime)
		} else {
			return r, nil
		}
	}

}

// Create JSON Web Token
func (c *Client) setToken(r *http.Request) error {

	// Create qsh hash
	qshstring := r.Method + "&" + r.URL.Path
	hash := sha256.New()
	hash.Write([]byte(qshstring))

	// Create Token
	token := jwt.NewWithClaims(jwt.SigningMethodHS256, jwt.MapClaims{
		// Set issuer
		"iss": c.user,

		// Set issued at time
		"iat": time.Now().Unix(),

		// Set expiration
		"exp": time.Now().Add(time.Minute * 5).Unix(),

		// Set qsh
		"qsh": hex.EncodeToString(hash.Sum(nil)),
	})

	// Sign the token
	signedtoken, err := token.SignedString([]byte(c.key))
	if err != nil {
		return err
	}

	// Save it in the header
	r.Header.Set("Authorization", "bearer "+signedtoken)

	return nil
}
