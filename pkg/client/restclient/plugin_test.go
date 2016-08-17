/*
Copyright 2016 The Kubernetes Authors.

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
	"reflect"
	"strconv"
	"testing"

	clientcmdapi "k8s.io/kubernetes/pkg/client/unversioned/clientcmd/api"
)

func TestAuthPluginWrapTransport(t *testing.T) {
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

func TestAuthPluginPersist(t *testing.T) {
	// register pluginA by a different name so we don't collide across tests.
	if err := RegisterAuthProviderPlugin("pluginA2", pluginAProvider); err != nil {
		t.Errorf("Unexpected error: failed to register pluginA: %v", err)
	}
	if err := RegisterAuthProviderPlugin("pluginPersist", pluginPersistProvider); err != nil {
		t.Errorf("Unexpected error: failed to register pluginPersist: %v", err)
	}
	fooBarConfig := map[string]string{"foo": "bar"}
	testCases := []struct {
		plugin                       string
		startingConfig               map[string]string
		expectedConfigAfterLogin     map[string]string
		expectedConfigAfterRoundTrip map[string]string
	}{
		// non-persisting plugins should work fine without modifying config.
		{"pluginA2", map[string]string{}, map[string]string{}, map[string]string{}},
		{"pluginA2", fooBarConfig, fooBarConfig, fooBarConfig},
		// plugins that persist config should be able to persist when they want.
		{
			"pluginPersist",
			map[string]string{},
			map[string]string{
				"login": "Y",
			},
			map[string]string{
				"login":      "Y",
				"roundTrips": "1",
			},
		},
		{
			"pluginPersist",
			map[string]string{
				"login":      "Y",
				"roundTrips": "123",
			},
			map[string]string{
				"login":      "Y",
				"roundTrips": "123",
			},
			map[string]string{
				"login":      "Y",
				"roundTrips": "124",
			},
		},
	}
	for i, tc := range testCases {
		cfg := &clientcmdapi.AuthProviderConfig{
			Name:   tc.plugin,
			Config: tc.startingConfig,
		}
		persister := &inMemoryPersister{make(map[string]string)}
		persister.Persist(tc.startingConfig)
		plugin, err := GetAuthProvider("127.0.0.1", cfg, persister)
		if err != nil {
			t.Errorf("%d. Unexpected error: failed to get plugin %q: %v", i, tc.plugin, err)
		}
		if err := plugin.Login(); err != nil {
			t.Errorf("%d. Unexpected error calling Login() w/ plugin %q: %v", i, tc.plugin, err)
		}
		// Make sure the plugin persisted what we expect after Login().
		if !reflect.DeepEqual(persister.savedConfig, tc.expectedConfigAfterLogin) {
			t.Errorf("%d. Unexpected persisted config after calling %s.Login(): \nGot:\n%v\nExpected:\n%v",
				i, tc.plugin, persister.savedConfig, tc.expectedConfigAfterLogin)
		}
		if _, err := plugin.WrapTransport(&emptyTransport{}).RoundTrip(&http.Request{}); err != nil {
			t.Errorf("%d. Unexpected error round-tripping w/ plugin %q: %v", i, tc.plugin, err)
		}
		// Make sure the plugin persisted what we expect after RoundTrip().
		if !reflect.DeepEqual(persister.savedConfig, tc.expectedConfigAfterRoundTrip) {
			t.Errorf("%d. Unexpected persisted config after calling %s.WrapTransport.RoundTrip(): \nGot:\n%v\nExpected:\n%v",
				i, tc.plugin, persister.savedConfig, tc.expectedConfigAfterLogin)
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

func (*pluginA) Login() error { return nil }

func pluginAProvider(string, map[string]string, AuthProviderConfigPersister) (AuthProvider, error) {
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

func (*pluginB) Login() error { return nil }

func pluginBProvider(string, map[string]string, AuthProviderConfigPersister) (AuthProvider, error) {
	return &pluginB{}, nil
}

// pluginFailProvider simulates a registered AuthPlugin that fails to load.
func pluginFailProvider(string, map[string]string, AuthProviderConfigPersister) (AuthProvider, error) {
	return nil, fmt.Errorf("Failed to load AuthProvider")
}

type inMemoryPersister struct {
	savedConfig map[string]string
}

func (i *inMemoryPersister) Persist(config map[string]string) error {
	i.savedConfig = make(map[string]string)
	for k, v := range config {
		i.savedConfig[k] = v
	}
	return nil
}

// wrapTransportPersist increments the "roundTrips" entry from the config when
// roundTrip is called.
type wrapTransportPersist struct {
	rt        http.RoundTripper
	config    map[string]string
	persister AuthProviderConfigPersister
}

func (w *wrapTransportPersist) RoundTrip(req *http.Request) (*http.Response, error) {
	roundTrips := 0
	if rtVal, ok := w.config["roundTrips"]; ok {
		var err error
		roundTrips, err = strconv.Atoi(rtVal)
		if err != nil {
			return nil, err
		}
	}
	roundTrips++
	w.config["roundTrips"] = fmt.Sprintf("%d", roundTrips)
	if err := w.persister.Persist(w.config); err != nil {
		return nil, err
	}
	return w.rt.RoundTrip(req)
}

type pluginPersist struct {
	config    map[string]string
	persister AuthProviderConfigPersister
}

func (p *pluginPersist) WrapTransport(rt http.RoundTripper) http.RoundTripper {
	return &wrapTransportPersist{rt, p.config, p.persister}
}

// Login sets the config entry "login" to "Y".
func (p *pluginPersist) Login() error {
	p.config["login"] = "Y"
	p.persister.Persist(p.config)
	return nil
}

func pluginPersistProvider(_ string, config map[string]string, persister AuthProviderConfigPersister) (AuthProvider, error) {
	return &pluginPersist{config, persister}, nil
}
