/*
Copyright 2015 The Kubernetes Authors.

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

package spdy

import (
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"strings"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/httpstream"
	utilnet "k8s.io/apimachinery/pkg/util/net"

	"golang.org/x/net/http/httpguts"
)

// SpdyRoundTripper knows how to upgrade an HTTP request to one that supports
// multiplexed streams. SpdyRoundTripper implements the UpgradeRoundTripper
// interface.
type SpdyRoundTripper struct {
	rt http.RoundTripper
}

var _ httpstream.UpgradeRoundTripper = &SpdyRoundTripper{}

// NewRoundTripper creates a new SpdyRoundTripper that will use the specified
// http.RoundTripper.
func NewRoundTripper(rt http.RoundTripper) *SpdyRoundTripper {
	if rt == nil {
		rt = http.DefaultTransport
	}
	return &SpdyRoundTripper{
		rt: rt,
	}
}

func (s *SpdyRoundTripper) WrappedRoundTripper() http.RoundTripper {
	return s.rt
}

// RoundTrip executes the Request and upgrades it. After a successful upgrade,
// clients may pass the response to SpdyRoundTripper.NewConnection() to
// complete the upgrade and retrieve the upgraded connection.
func (s *SpdyRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	req = utilnet.CloneRequest(req)
	req.Header.Add(httpstream.HeaderConnection, httpstream.HeaderUpgrade)
	req.Header.Add(httpstream.HeaderUpgrade, HeaderSpdy31)

	return s.rt.RoundTrip(req)
}

// NewConnection validates the upgrade response, creating and returning a new
// httpstream.Connection if there were no errors.
func (s *SpdyRoundTripper) NewConnection(resp *http.Response) (httpstream.Connection, error) {
	if (resp.StatusCode != http.StatusSwitchingProtocols) ||
		!httpguts.HeaderValuesContainsToken(resp.Header.Values(httpstream.HeaderConnection), httpstream.HeaderUpgrade) ||
		!httpguts.HeaderValuesContainsToken(resp.Header.Values(httpstream.HeaderUpgrade), HeaderSpdy31) {
		defer resp.Body.Close()
		responseError := ""
		responseErrorBytes, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			responseError = "unable to read error from server response"
		} else {
			// TODO: I don't belong here, I should be abstracted from this class
			if obj, _, err := statusCodecs.UniversalDecoder().Decode(responseErrorBytes, nil, &metav1.Status{}); err == nil {
				if status, ok := obj.(*metav1.Status); ok {
					return nil, &apierrors.StatusError{ErrStatus: *status}
				}
			}
			responseError = string(responseErrorBytes)
			responseError = strings.TrimSpace(responseError)
		}

		return nil, fmt.Errorf("unable to upgrade connection: %s", responseError)
	}

	rwc, ok := resp.Body.(io.ReadWriteCloser)
	if !ok {
		return nil, fmt.Errorf("internal error: 101 switching protocols response with non-writable body")
	}

	return NewClientConnectionWithPings(&conn{rwc: rwc}, 5*time.Second)
}

// statusScheme is private scheme for the decoding here until someone fixes the TODO in NewConnection
var statusScheme = runtime.NewScheme()

// ParameterCodec knows about query parameters used with the meta v1 API spec.
var statusCodecs = serializer.NewCodecFactory(statusScheme)

func init() {
	statusScheme.AddUnversionedTypes(metav1.SchemeGroupVersion,
		&metav1.Status{},
	)
}

type conn struct {
	net.Conn
	rwc io.ReadWriteCloser
}

func (c *conn) Read(b []byte) (n int, err error) {
	return c.rwc.Read(b)
}

func (c *conn) Write(b []byte) (n int, err error) {
	return c.rwc.Write(b)
}

func (c *conn) Close() error {
	return c.rwc.Close()
}
