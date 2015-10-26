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

package unversioned

import (
	"crypto/tls"
	"crypto/x509"
	"fmt"
	"io/ioutil"
	"net/http"

	"k8s.io/kubernetes/pkg/util"
)

type userAgentRoundTripper struct {
	agent string
	rt    http.RoundTripper
}

func NewUserAgentRoundTripper(agent string, rt http.RoundTripper) http.RoundTripper {
	return &userAgentRoundTripper{agent, rt}
}

func (rt *userAgentRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("User-Agent")) != 0 {
		return rt.rt.RoundTrip(req)
	}
	req = cloneRequest(req)
	req.Header.Set("User-Agent", rt.agent)
	return rt.rt.RoundTrip(req)
}

var _ = util.RoundTripperWrapper(&userAgentRoundTripper{})

func (rt *userAgentRoundTripper) WrappedRoundTripper() http.RoundTripper {
	return rt.rt
}

type basicAuthRoundTripper struct {
	username string
	password string
	rt       http.RoundTripper
}

// NewBasicAuthRoundTripper will apply a BASIC auth authorization header to a request unless it has
// already been set.
func NewBasicAuthRoundTripper(username, password string, rt http.RoundTripper) http.RoundTripper {
	return &basicAuthRoundTripper{username, password, rt}
}

func (rt *basicAuthRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("Authorization")) != 0 {
		return rt.rt.RoundTrip(req)
	}
	req = cloneRequest(req)
	req.SetBasicAuth(rt.username, rt.password)
	return rt.rt.RoundTrip(req)
}

var _ = util.RoundTripperWrapper(&basicAuthRoundTripper{})

func (rt *basicAuthRoundTripper) WrappedRoundTripper() http.RoundTripper {
	return rt.rt
}

type bearerAuthRoundTripper struct {
	bearer string
	rt     http.RoundTripper
}

// NewBearerAuthRoundTripper adds the provided bearer token to a request unless the authorization
// header has already been set.
func NewBearerAuthRoundTripper(bearer string, rt http.RoundTripper) http.RoundTripper {
	return &bearerAuthRoundTripper{bearer, rt}
}

func (rt *bearerAuthRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	if len(req.Header.Get("Authorization")) != 0 {
		return rt.rt.RoundTrip(req)
	}

	req = cloneRequest(req)
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", rt.bearer))
	return rt.rt.RoundTrip(req)
}

var _ = util.RoundTripperWrapper(&bearerAuthRoundTripper{})

func (rt *bearerAuthRoundTripper) WrappedRoundTripper() http.RoundTripper {
	return rt.rt
}

// TLSConfigFor returns a tls.Config that will provide the transport level security defined
// by the provided Config. Will return nil if no transport level security is requested.
func TLSConfigFor(config *Config) (*tls.Config, error) {
	hasCA := len(config.CAFile) > 0 || len(config.CAData) > 0
	hasCert := len(config.CertFile) > 0 || len(config.CertData) > 0

	if !hasCA && !hasCert && !config.Insecure {
		return nil, nil
	}
	if hasCA && config.Insecure {
		return nil, fmt.Errorf("specifying a root certificates file with the insecure flag is not allowed")
	}
	if err := LoadTLSFiles(config); err != nil {
		return nil, err
	}

	tlsConfig := &tls.Config{
		// Change default from SSLv3 to TLSv1.0 (because of POODLE vulnerability)
		MinVersion:         tls.VersionTLS10,
		InsecureSkipVerify: config.Insecure,
	}

	if hasCA {
		tlsConfig.RootCAs = rootCertPool(config.CAData)
	}

	if hasCert {
		cert, err := tls.X509KeyPair(config.CertData, config.KeyData)
		if err != nil {
			return nil, err
		}
		tlsConfig.Certificates = []tls.Certificate{cert}
	}

	return tlsConfig, nil
}

// tlsConfigKey returns a unique key for tls.Config objects returned from TLSConfigFor
func tlsConfigKey(config *Config) (string, error) {
	// Make sure ca/key/cert content is loaded
	if err := LoadTLSFiles(config); err != nil {
		return "", err
	}
	// Only include the things that actually affect the tls.Config
	return fmt.Sprintf("%v/%x/%x/%x", config.Insecure, config.CAData, config.CertData, config.KeyData), nil
}

// LoadTLSFiles copies the data from the CertFile, KeyFile, and CAFile fields into the CertData,
// KeyData, and CAFile fields, or returns an error. If no error is returned, all three fields are
// either populated or were empty to start.
func LoadTLSFiles(config *Config) error {
	certData, err := dataFromSliceOrFile(config.CertData, config.CertFile)
	if err != nil {
		return err
	}
	config.CertData = certData
	keyData, err := dataFromSliceOrFile(config.KeyData, config.KeyFile)
	if err != nil {
		return err
	}
	config.KeyData = keyData
	caData, err := dataFromSliceOrFile(config.CAData, config.CAFile)
	if err != nil {
		return err
	}
	config.CAData = caData

	return nil
}

// dataFromSliceOrFile returns data from the slice (if non-empty), or from the file,
// or an error if an error occurred reading the file
func dataFromSliceOrFile(data []byte, file string) ([]byte, error) {
	if len(data) > 0 {
		return data, nil
	}
	if len(file) > 0 {
		fileData, err := ioutil.ReadFile(file)
		if err != nil {
			return []byte{}, err
		}
		return fileData, nil
	}
	return nil, nil
}

// rootCertPool returns nil if caData is empty.  When passed along, this will mean "use system CAs".
// When caData is not empty, it will be the ONLY information used in the CertPool.
func rootCertPool(caData []byte) *x509.CertPool {
	// What we really want is a copy of x509.systemRootsPool, but that isn't exposed.  It's difficult to build (see the go
	// code for a look at the platform specific insanity), so we'll use the fact that RootCAs == nil gives us the system values
	// It doesn't allow trusting either/or, but hopefully that won't be an issue
	if len(caData) == 0 {
		return nil
	}

	// if we have caData, use it
	certPool := x509.NewCertPool()
	certPool.AppendCertsFromPEM(caData)
	return certPool
}

// cloneRequest returns a clone of the provided *http.Request.
// The clone is a shallow copy of the struct and its Header map.
func cloneRequest(r *http.Request) *http.Request {
	// shallow copy of the struct
	r2 := new(http.Request)
	*r2 = *r
	// deep copy of the Header
	r2.Header = make(http.Header)
	for k, s := range r.Header {
		r2.Header[k] = s
	}
	return r2
}
