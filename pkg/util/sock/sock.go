/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package sock

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/docker/docker/pkg/tlsconfig"
	"github.com/golang/glog"
)

const (
	versionMimetype = "application/json"
	defaultTimeOut  = 30
)

// ConfigureTCPTransport configures the specified Transport according to the
// specified proto and addr.
// If the proto is unix (using a unix socket to communicate) the compression
// is disabled.
func ConfigureTCPTransport(tr *http.Transport, proto, addr string) {
	// Why 32? See https://github.com/docker/docker/pull/8035.
	timeout := 32 * time.Second
	if proto == "unix" {
		// No need for compression in local communications.
		tr.DisableCompression = true
		tr.Dial = func(_, _ string) (net.Conn, error) {
			return net.DialTimeout(proto, addr, timeout)
		}
	} else {
		tr.Proxy = http.ProxyFromEnvironment
		tr.Dial = (&net.Dialer{Timeout: timeout}).Dial
	}
}

// NewClient creates a new plugin client (http).
func NewClient(addr string, tlsConfig tlsconfig.Options) (*Client, error) {
	tr := &http.Transport{}

	c, err := tlsconfig.Client(tlsConfig)
	if err != nil {
		return nil, err
	}
	tr.TLSClientConfig = c

	protoAndAddr := strings.Split(addr, "://")
	ConfigureTCPTransport(tr, protoAndAddr[0], protoAndAddr[1])
	return &Client{&http.Client{Transport: tr}, protoAndAddr[1]}, nil
}

// Client represents a plugin client.
type Client struct {
	http *http.Client // http client to use
	addr string       // http address of the plugin
}

// Call calls the specified method with the specified arguments for the plugin.
// It will retry for 30 seconds if a failure occurs when calling.
func (c *Client) Call(serviceMethod string, args interface{}, ret interface{}) error {
	return c.callWithRetry(serviceMethod, args, ret, true)
}

func (c *Client) callWithRetry(serviceMethod string, args interface{}, ret interface{}, retry bool) error {
	var buf bytes.Buffer
	if err := json.NewEncoder(&buf).Encode(args); err != nil {
		return err
	}

	req, err := http.NewRequest("POST", "/"+serviceMethod, &buf)
	if err != nil {
		return err
	}
	req.Header.Add("Accept", versionMimetype)
	req.URL.Scheme = "http"
	req.URL.Host = c.addr

	var retries int
	start := time.Now()

	for {
		resp, err := c.http.Do(req)
		if err != nil {
			if !retry {
				return err
			}

			timeOff := backoff(retries)
			if abort(start, timeOff) {
				return err
			}
			retries++
			glog.Warningf("Unable to connect to plugin: %s, retrying in %v", c.addr, timeOff)
			time.Sleep(timeOff)
			continue
		}

		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			_, err := ioutil.ReadAll(resp.Body)
			if err != nil {
				return err
			}
		}

		return json.NewDecoder(resp.Body).Decode(&ret)
	}
}

func backoff(retries int) time.Duration {
	b, max := 1, defaultTimeOut
	for b < max && retries > 0 {
		b *= 2
		retries--
	}
	if b > max {
		b = max
	}
	return time.Duration(b) * time.Second
}

func abort(start time.Time, timeOff time.Duration) bool {
	return timeOff+time.Since(start) >= time.Duration(defaultTimeOut)*time.Second
}
