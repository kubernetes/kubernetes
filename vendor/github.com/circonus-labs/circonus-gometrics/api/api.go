// Copyright 2016 Circonus, Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package api provides methods for interacting with the Circonus API
package api

import (
	"bytes"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	"github.com/hashicorp/go-retryablehttp"
)

const (
	// a few sensible defaults
	defaultAPIURL = "https://api.circonus.com/v2"
	defaultAPIApp = "circonus-gometrics"
	minRetryWait  = 10 * time.Millisecond
	maxRetryWait  = 50 * time.Millisecond
	maxRetries    = 3
)

// TokenKeyType - Circonus API Token key
type TokenKeyType string

// TokenAppType - Circonus API Token app name
type TokenAppType string

// IDType Circonus object id (numeric portion of cid)
type IDType int

// CIDType Circonus object cid
type CIDType string

// URLType submission url type
type URLType string

// SearchQueryType search query
type SearchQueryType string

// SearchTagType search/select tag type
type SearchTagType string

// Config options for Circonus API
type Config struct {
	URL      string
	TokenKey string
	TokenApp string
	Log      *log.Logger
	Debug    bool
}

// API Circonus API
type API struct {
	apiURL *url.URL
	key    TokenKeyType
	app    TokenAppType
	Debug  bool
	Log    *log.Logger
}

// NewAPI returns a new Circonus API
func NewAPI(ac *Config) (*API, error) {

	if ac == nil {
		return nil, errors.New("Invalid API configuration (nil)")
	}

	key := TokenKeyType(ac.TokenKey)
	if key == "" {
		return nil, errors.New("API Token is required")
	}

	app := TokenAppType(ac.TokenApp)
	if app == "" {
		app = defaultAPIApp
	}

	au := string(ac.URL)
	if au == "" {
		au = defaultAPIURL
	}
	if !strings.Contains(au, "/") {
		// if just a hostname is passed, ASSume "https" and a path prefix of "/v2"
		au = fmt.Sprintf("https://%s/v2", ac.URL)
	}
	if last := len(au) - 1; last >= 0 && au[last] == '/' {
		au = au[:last]
	}
	apiURL, err := url.Parse(au)
	if err != nil {
		return nil, err
	}

	a := &API{apiURL, key, app, ac.Debug, ac.Log}

	if a.Log == nil {
		if a.Debug {
			a.Log = log.New(os.Stderr, "", log.LstdFlags)
		} else {
			a.Log = log.New(ioutil.Discard, "", log.LstdFlags)
		}
	}

	return a, nil
}

// Get API request
func (a *API) Get(reqPath string) ([]byte, error) {
	return a.apiCall("GET", reqPath, nil)
}

// Delete API request
func (a *API) Delete(reqPath string) ([]byte, error) {
	return a.apiCall("DELETE", reqPath, nil)
}

// Post API request
func (a *API) Post(reqPath string, data []byte) ([]byte, error) {
	return a.apiCall("POST", reqPath, data)
}

// Put API request
func (a *API) Put(reqPath string, data []byte) ([]byte, error) {
	return a.apiCall("PUT", reqPath, data)
}

// apiCall call Circonus API
func (a *API) apiCall(reqMethod string, reqPath string, data []byte) ([]byte, error) {
	dataReader := bytes.NewReader(data)
	reqURL := a.apiURL.String()

	if reqPath[:1] != "/" {
		reqURL += "/"
	}
	if reqPath[:3] == "/v2" {
		reqURL += reqPath[3:len(reqPath)]
	} else {
		reqURL += reqPath
	}

	req, err := retryablehttp.NewRequest(reqMethod, reqURL, dataReader)
	if err != nil {
		return nil, fmt.Errorf("[ERROR] creating API request: %s %+v", reqURL, err)
	}
	req.Header.Add("Accept", "application/json")
	req.Header.Add("X-Circonus-Auth-Token", string(a.key))
	req.Header.Add("X-Circonus-App-Name", string(a.app))

	// keep last HTTP error in the event of retry failure
	var lastHTTPError error
	retryPolicy := func(resp *http.Response, err error) (bool, error) {
		if err != nil {
			lastHTTPError = err
			return true, err
		}
		// Check the response code. We retry on 500-range responses to allow
		// the server time to recover, as 500's are typically not permanent
		// errors and may relate to outages on the server side. This will catch
		// invalid response codes as well, like 0 and 999.
		if resp.StatusCode == 0 || resp.StatusCode >= 500 {
			defer resp.Body.Close()
			body, readErr := ioutil.ReadAll(resp.Body)
			if readErr != nil {
				lastHTTPError = fmt.Errorf("- last HTTP error: %d %+v", resp.StatusCode, readErr)
			} else {
				lastHTTPError = fmt.Errorf("- last HTTP error: %d %s", resp.StatusCode, string(body))
			}
			return true, nil
		}
		return false, nil
	}

	client := retryablehttp.NewClient()
	client.RetryWaitMin = minRetryWait
	client.RetryWaitMax = maxRetryWait
	client.RetryMax = maxRetries
	client.Logger = a.Log
	client.CheckRetry = retryPolicy

	resp, err := client.Do(req)
	if err != nil {
		if lastHTTPError != nil {
			return nil, fmt.Errorf("[ERROR] fetching: %+v %+v", err, lastHTTPError)
		}
		return nil, fmt.Errorf("[ERROR] fetching %s: %+v", reqURL, err)
	}

	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("[ERROR] reading body %+v", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		msg := fmt.Sprintf("API response code %d: %s", resp.StatusCode, string(body))
		if a.Debug {
			a.Log.Printf("[DEBUG] %s\n", msg)
		}

		return nil, fmt.Errorf("[ERROR] %s", msg)
	}

	return body, nil
}
