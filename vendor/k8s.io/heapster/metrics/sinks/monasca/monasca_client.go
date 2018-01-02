// Copyright 2015 Google Inc. All Rights Reserved.
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

package monasca

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"

	"github.com/golang/glog"
)

// Client specifies the methods of any client ot the Monasca API
type Client interface {
	SendRequest(method, path string, request interface{}) (int, string, error)
	GetURL() *url.URL
	CheckHealth() bool
}

// ClientImpl implements the monasca API client interface.
type ClientImpl struct {
	ksClient   KeystoneClient
	monascaURL *url.URL
}

// SendRequest to the Monasca API, authenticating on the fly.
// Returns 0, "", err if the request cannot be built.
// Returns statusCode, response, nil if communication with the server was OK.
func (monClient *ClientImpl) SendRequest(method, path string, request interface{}) (int, string, error) {
	req, err := monClient.prepareRequest(method, path, request)
	if err != nil {
		return 0, "", err
	}
	statusCode, response, err := monClient.receiveResponse(req)
	if err != nil {
		return 0, "", err
	}
	return statusCode, response, nil
}

func (monClient *ClientImpl) prepareRequest(method string, path string, request interface{}) (*http.Request, error) {
	// authenticate
	token, err := monClient.ksClient.GetToken()
	if err != nil {
		return nil, err
	}
	// marshal
	var rawRequest io.Reader
	if request != nil {
		jsonRequest, err := json.Marshal(request)
		if err != nil {
			return nil, err
		}
		rawRequest = bytes.NewReader(jsonRequest)
	}
	// build request
	req, err := http.NewRequest(method, monClient.GetURL().String()+path, rawRequest)
	if err != nil {
		return nil, err
	}
	req.Header.Add("Content-Type", "application/json")
	req.Header.Add("X-Auth-Token", token)
	return req, nil
}

func (monClient *ClientImpl) receiveResponse(req *http.Request) (int, string, error) {
	resp, err := (&http.Client{}).Do(req)
	if err != nil {
		return 0, "", err
	}
	defer resp.Body.Close()
	contents, err := ioutil.ReadAll(resp.Body)
	respString := ""
	if err != nil {
		glog.Warning("Cannot read monasca API's response.")
		respString = "Cannot read monasca API's response."
	} else {
		respString = string(contents)
	}
	return resp.StatusCode, fmt.Sprintf("%s", respString), nil
}

// GetURL of the Monasca API server.
func (monClient *ClientImpl) GetURL() *url.URL {
	return monClient.monascaURL
}

// CheckHealth of the monasca API server.
func (monClient *ClientImpl) CheckHealth() bool {
	code, _, _ := monClient.SendRequest("GET", "/", nil)
	return code == http.StatusOK
}

// NewMonascaClient creates a monasca client.
func NewMonascaClient(config Config) (Client, error) {
	// create keystone client
	ksClient, err := NewKeystoneClient(config)
	if err != nil {
		return nil, err
	}

	// detect monasca URL
	monascaURL := (*url.URL)(nil)
	if config.MonascaURL != "" {
		monascaURL, err = url.Parse(config.MonascaURL)
		if err != nil {
			return nil, fmt.Errorf("Malformed monasca-url sink parameter. %s", err)
		}
	} else {
		monascaURL, err = ksClient.MonascaURL()
		if err != nil {
			return nil, fmt.Errorf("Failed to automatically detect Monasca service: %s", err)
		}
	}

	// create monasca client
	client := &ClientImpl{ksClient: ksClient, monascaURL: monascaURL}
	healthy := client.CheckHealth()
	if !healthy {
		return nil, fmt.Errorf("Monasca is not running on the provided/discovered URL: %s", client.GetURL().String())
	}
	return client, nil
}
