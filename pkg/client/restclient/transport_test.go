/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package restclient

import (
	"fmt"
	"net/http"
	"testing"

	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
)

func TestTransportConfigAuthPlugins(t *testing.T) {
	if err := RegisterAuthProviderPlugin("pluginA", pluginAProvider); err != nil {
		t.Errorf("Unexpected error: failed to register pluginA: %v", err)
	}
	if err := RegisterAuthProviderPlugin("pluginB", pluginBProvider); err != nil {
		t.Errorf("Unexpected error: failed to register pluginB: %v", err)
	}
	if err := RegisterAuthProviderPlugin("pluginFail", pluginFailProvider); err != nil {
		t.Errorf("Unexpected error: failed to register pluginFail: %v", err)
	}
	testCases := []struct {
		useWrapTransport bool
		plugin           string
		expectErr        bool
		expectPluginA    bool
		expectPluginB    bool
	}{
		{false, "", false, false, false},
		{false, "pluginA", false, true, false},
		{false, "pluginB", false, false, true},
		{false, "pluginFail", true, false, false},
		{false, "pluginUnknown", true, false, false},
	}
	for i, tc := range testCases {
		c := Config{}
		if tc.useWrapTransport {
			// Specify an existing WrapTransport in the config to make sure that
			// plugins play nicely.
			c.WrapTransport = func(rt http.RoundTripper) http.RoundTripper {
				return &wrapTransport{rt}
			}
		}
		if len(tc.plugin) != 0 {
			c.AuthProvider = &clientcmdapi.AuthProviderConfig{Name: tc.plugin}
		}
		tConfig, err := c.transportConfig()
		if err != nil {
			// Unknown/bad plugins are expected to fail here.
			if !tc.expectErr {
				t.Errorf("%d. Did not expect errors loading Auth Plugin: %q. Got: %v", i, tc.plugin, err)
			}
			continue
		}
		var fullyWrappedTransport http.RoundTripper
		fullyWrappedTransport = &emptyTransport{}
		if tConfig.WrapTransport != nil {
			fullyWrappedTransport = tConfig.WrapTransport(&emptyTransport{})
		}
		res, err := fullyWrappedTransport.RoundTrip(&http.Request{})
		if err != nil {
			t.Errorf("%d. Unexpected error in RoundTrip: %v", i, err)
			continue
		}
		hasWrapTransport := res.Header.Get("wrapTransport") == "Y"
		hasPluginA := res.Header.Get("pluginA") == "Y"
		hasPluginB := res.Header.Get("pluginB") == "Y"
		if hasWrapTransport != tc.useWrapTransport {
			t.Errorf("%d. Expected Existing config.WrapTransport: %t; Got: %t", i, tc.useWrapTransport, hasWrapTransport)
		}
		if hasPluginA != tc.expectPluginA {
			t.Errorf("%d. Expected Plugin A: %t; Got: %t", i, tc.expectPluginA, hasPluginA)
		}
		if hasPluginB != tc.expectPluginB {
			t.Errorf("%d. Expected Plugin B: %t; Got: %t", i, tc.expectPluginB, hasPluginB)
		}
	}
}

// emptyTransport provides an empty http.Response with an initialized header
// to allow wrapping RoundTrippers to set header values.
type emptyTransport struct{}

func (*emptyTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	res := &http.Response{
		Header: make(map[string][]string),
	}
	return res, nil
}

// wrapTransport sets "wrapTransport" = "Y" on the response.
type wrapTransport struct {
	rt http.RoundTripper
}

func (w *wrapTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	res, err := w.rt.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	res.Header.Add("wrapTransport", "Y")
	return res, nil
}

// wrapTransportA sets "pluginA" = "Y" on the response.
type wrapTransportA struct {
	rt http.RoundTripper
}

func (w *wrapTransportA) RoundTrip(req *http.Request) (*http.Response, error) {
	res, err := w.rt.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	res.Header.Add("pluginA", "Y")
	return res, nil
}

type pluginA struct{}

func (*pluginA) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	return &wrapTransportA{rt}
}

func pluginAProvider() (AuthProvider, error) {
	return &pluginA{}, nil
}

// wrapTransportB sets "pluginB" = "Y" on the response.
type wrapTransportB struct {
	rt http.RoundTripper
}

func (w *wrapTransportB) RoundTrip(req *http.Request) (*http.Response, error) {
	res, err := w.rt.RoundTrip(req)
	if err != nil {
		return nil, err
	}
	res.Header.Add("pluginB", "Y")
	return res, nil
}

type pluginB struct{}

func (*pluginB) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	return &wrapTransportB{rt}
}

func pluginBProvider() (AuthProvider, error) {
	return &pluginB{}, nil
}

// pluginFailProvider simulates a registered AuthPlugin that fails to load.
func pluginFailProvider() (AuthProvider, error) {
	return nil, fmt.Errorf("Failed to load AuthProvider")
}
