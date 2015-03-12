/*
Copyright 2014 Google Inc. All rights reserved.

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

package client

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"errors"
	"fmt"
	"github.com/nalind/gss/pkg/gss/proxy"
	"io/ioutil"
	"net"
	"net/http"
	"strings"
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

type basicAuthRoundTripper struct {
	username string
	password string
	rt       http.RoundTripper
}

func NewBasicAuthRoundTripper(username, password string, rt http.RoundTripper) http.RoundTripper {
	return &basicAuthRoundTripper{username, password, rt}
}

func (rt *basicAuthRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	req = cloneRequest(req)
	req.SetBasicAuth(rt.username, rt.password)
	return rt.rt.RoundTrip(req)
}

type bearerAuthRoundTripper struct {
	bearer string
	rt     http.RoundTripper
}

func NewBearerAuthRoundTripper(bearer string, rt http.RoundTripper) http.RoundTripper {
	return &bearerAuthRoundTripper{bearer, rt}
}

func (rt *bearerAuthRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	req = cloneRequest(req)
	req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", rt.bearer))
	return rt.rt.RoundTrip(req)
}

// TLSConfigFor returns a tls.Config that will provide the transport level security defined
// by the provided Config. Will return nil if no transport level security is requested.
func TLSConfigFor(config *Config) (*tls.Config, error) {
	hasCA := len(config.CAFile) > 0 || len(config.CAData) > 0
	hasCert := len(config.CertFile) > 0 || len(config.CertData) > 0

	if hasCA && config.Insecure {
		return nil, fmt.Errorf("specifying a root certificates file with the insecure flag is not allowed")
	}
	if err := LoadTLSFiles(config); err != nil {
		return nil, err
	}
	var tlsConfig *tls.Config
	switch {
	case hasCert:
		cfg, err := NewClientCertTLSConfig(config.CertData, config.KeyData, config.CAData)
		if err != nil {
			return nil, err
		}
		tlsConfig = cfg
	case hasCA:
		cfg, err := NewTLSConfig(config.CAData)
		if err != nil {
			return nil, err
		}
		tlsConfig = cfg
	case config.Insecure:
		tlsConfig = NewUnsafeTLSConfig()
	}

	return tlsConfig, nil
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

type negotiateAuthRoundTripper struct {
	proxyLocation string
	rt            http.RoundTripper
}

func NewNegotiateAuthRoundTripper(proxyLocation string, rt http.RoundTripper) (r http.RoundTripper) {
	r = &negotiateAuthRoundTripper{proxyLocation, rt}
	return
}

func (rt *negotiateAuthRoundTripper) RoundTrip(req *http.Request) (r *http.Response, err error) {
	var conn net.Conn
	var call proxy.CallCtx
	var ctx proxy.SecCtx
	var name proxy.Name
	var iscr proxy.InitSecContextResults
	var hostname, negotiate string
	var token []byte

	req = cloneRequest(req)

	/* Build the target service's name. */
	if len(req.Host) > 0 {
		hostname = req.Host
	} else {
		hostname = req.URL.Host
	}
	colon := strings.Index(hostname, ":")
	if colon > 0 {
		name.DisplayName = "HTTP@" + hostname[0:colon]
	} else {
		name.DisplayName = "HTTP@" + hostname
	}
	name.NameType = proxy.NT_HOSTBASED_SERVICE

	/* Connect to the proxy. */
	conn, err = net.Dial("unix", rt.proxyLocation)
	if err != nil {
		return
	}
	defer conn.Close()
	gccr, err := proxy.GetCallContext(&conn, &call, nil)
	if err != nil {
		return
	}
	if gccr.Status.MajorStatus != proxy.S_COMPLETE {
		err = errors.New("unable to get gss-proxy call context")
		return
	}

	/* Get the first token. */
	iscr, err = proxy.InitSecContext(&conn, &call, &ctx, nil, &name, proxy.MechSPNEGO, proxy.Flags{}, proxy.C_INDEFINITE, nil, nil, nil)
	if err != nil {
		return
	}
	if iscr.Status.MajorStatus != proxy.S_COMPLETE && iscr.Status.MajorStatus != proxy.S_CONTINUE_NEEDED {
		err = errors.New(iscr.Status.MajorStatusString + ", " + iscr.Status.MinorStatusString)
		return
	}
	if iscr.OutputToken == nil {
		err = errors.New("unable to generate authentication header")
		return
	}
	for iscr.OutputToken != nil {
		/* Send the token. */
		negotiate = base64.StdEncoding.EncodeToString(*iscr.OutputToken)
		req.Header.Set("Authorization", fmt.Sprintf("Negotiate %s", negotiate))
		r, err = rt.rt.RoundTrip(req)
		if iscr.Status.MajorStatus != proxy.S_CONTINUE_NEEDED {
			/* We're not expecting a reply token. */
			break
		}
		if r.StatusCode != 401 {
			/* If we got the content, then we have no use for response data. */
			break
		}
		/* Check if we got a reply token. */
		header := r.Header.Get("WWW-Authenticate")
		if len(header) == 0 {
			r = nil
			err = errors.New("no authorization response from server")
			return
		}
		parts := strings.SplitN(header, " ", 2)
		if len(parts) < 2 || strings.ToLower(parts[0]) != "negotiate" {
			r = nil
			err = errors.New(fmt.Sprintf("authorization response is not Negotiate ('%s')", header))
			return
		}
		b64data := strings.Replace(parts[1], " ", "", -1)
		token, err = base64.StdEncoding.DecodeString(b64data)
		if err != nil {
			r = nil
			return
		}
		/* Get the next token that we need to send. */
		iscr, err = proxy.InitSecContext(&conn, &call, &ctx, nil, &name, proxy.MechSPNEGO, proxy.Flags{}, proxy.C_INDEFINITE, nil, &token, nil)
		if err != nil {
			return
		}
		if iscr.Status.MajorStatus != proxy.S_COMPLETE && iscr.Status.MajorStatus != proxy.S_CONTINUE_NEEDED {
			break
		}
	}
	if iscr.Status.MajorStatus != proxy.S_COMPLETE && iscr.Status.MajorStatus != proxy.S_CONTINUE_NEEDED {
		r = nil
		err = errors.New(iscr.Status.MajorStatusString + ", " + iscr.Status.MinorStatusString)
		return
	}
	return
}

func NewClientCertTLSConfig(certData, keyData, caData []byte) (*tls.Config, error) {
	cert, err := tls.X509KeyPair(certData, keyData)
	if err != nil {
		return nil, err
	}
	certPool := x509.NewCertPool()
	certPool.AppendCertsFromPEM(caData)
	return &tls.Config{
		// Change default from SSLv3 to TLSv1.0 (because of POODLE vulnerability)
		MinVersion: tls.VersionTLS10,
		Certificates: []tls.Certificate{
			cert,
		},
		RootCAs:    certPool,
		ClientCAs:  certPool,
		ClientAuth: tls.RequireAndVerifyClientCert,
	}, nil
}

func NewTLSConfig(caData []byte) (*tls.Config, error) {
	certPool := x509.NewCertPool()
	certPool.AppendCertsFromPEM(caData)
	return &tls.Config{
		// Change default from SSLv3 to TLSv1.0 (because of POODLE vulnerability)
		MinVersion: tls.VersionTLS10,
		RootCAs:    certPool,
	}, nil
}

func NewUnsafeTLSConfig() *tls.Config {
	return &tls.Config{
		InsecureSkipVerify: true,
	}
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
