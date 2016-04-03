// This file contains tests which depend on the race detector being enabled.
// +build race

package oidc

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
	"time"
)

type testProvider struct {
	baseURL *url.URL
}

func (p *testProvider) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != discoveryConfigPath {
		http.NotFound(w, r)
		return
	}

	cfg := ProviderConfig{
		Issuer:    p.baseURL,
		ExpiresAt: time.Now().Add(time.Second),
	}
	cfg = fillRequiredProviderFields(cfg)
	json.NewEncoder(w).Encode(&cfg)
}

// This test fails by triggering the race detector, not by calling t.Error or t.Fatal.
func TestProviderSyncRace(t *testing.T) {

	prov := &testProvider{}

	s := httptest.NewServer(prov)
	defer s.Close()
	u, err := url.Parse(s.URL)
	if err != nil {
		t.Fatal(err)
	}
	prov.baseURL = u

	prevValue := minimumProviderConfigSyncInterval
	defer func() { minimumProviderConfigSyncInterval = prevValue }()

	// Reduce the sync interval to increase the write frequencey.
	minimumProviderConfigSyncInterval = 5 * time.Millisecond

	cliCfg := ClientConfig{
		HTTPClient: http.DefaultClient,
	}
	cli, err := NewClient(cliCfg)
	if err != nil {
		t.Error(err)
		return
	}

	if !cli.providerConfig.Get().Empty() {
		t.Errorf("want c.ProviderConfig == nil, got c.ProviderConfig=%#v")
	}

	// SyncProviderConfig beings a goroutine which writes to the client's provider config.
	c := cli.SyncProviderConfig(s.URL)
	if cli.providerConfig.Get().Empty() {
		t.Errorf("want c.ProviderConfig != nil")
	}

	defer func() {
		// stop the background process
		c <- struct{}{}
	}()

	for i := 0; i < 10; i++ {
		time.Sleep(5 * time.Millisecond)
		// Creating an OAuth client reads from the provider config.
		cli.OAuthClient()
	}
}
