//
// Copyright (c) 2015 The heketi Authors
//
// This file is licensed to you under your choice of the GNU Lesser
// General Public License, version 3 or any later version (LGPLv3 or
// later), as published by the Free Software Foundation,
// or under the Apache License, Version 2.0 <LICENSE-APACHE2 or
// http://www.apache.org/licenses/LICENSE-2.0>.
//
// You may not use this file except in compliance with those terms.
//

package client

import (
	"crypto/sha256"
	"crypto/tls"
	"crypto/x509"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	jwt "github.com/dgrijalva/jwt-go"
	"github.com/heketi/heketi/pkg/utils"
)

const (
	MAX_CONCURRENT_REQUESTS = 32
)

type ClientTLSOptions struct {
	// directly borrow the field names from crypto/tls
	InsecureSkipVerify bool
	// one or more cert file paths (best for self-signed certs)
	VerifyCerts []string
}

// Client object
type Client struct {
	host     string
	key      string
	user     string
	throttle chan bool

	// configuration for TLS support
	tlsClientConfig *tls.Config
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

func NewClientTLS(host, user, key string, tlsOpts *ClientTLSOptions) (*Client, error) {
	c := NewClient(host, user, key)
	if err := c.SetTLSOptions(tlsOpts); err != nil {
		return nil, err
	}
	return c, nil
}

// Create a client to access a Heketi server without authentication enabled
func NewClientNoAuth(host string) *Client {
	return NewClient(host, "", "")
}

// SetTLSOptions configures an existing heketi client for
// TLS support based on the ClientTLSOptions.
func (c *Client) SetTLSOptions(o *ClientTLSOptions) error {
	if o == nil {
		c.tlsClientConfig = nil
		return nil
	}

	tlsConfig := &tls.Config{}
	tlsConfig.InsecureSkipVerify = o.InsecureSkipVerify
	if len(o.VerifyCerts) > 0 {
		tlsConfig.RootCAs = x509.NewCertPool()
		for _, path := range o.VerifyCerts {
			pem, err := ioutil.ReadFile(path)
			if err != nil {
				return fmt.Errorf("failed to read cert file %v: %v",
					path, err)
			}
			if ok := tlsConfig.RootCAs.AppendCertsFromPEM(pem); !ok {
				return fmt.Errorf("failed to load PEM encoded cert from %s",
					path)
			}
		}
	}
	c.tlsClientConfig = tlsConfig
	return nil
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
	defer r.Body.Close()
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
	if c.tlsClientConfig != nil {
		httpClient.Transport = &http.Transport{
			TLSClientConfig: c.tlsClientConfig,
		}
	}
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
