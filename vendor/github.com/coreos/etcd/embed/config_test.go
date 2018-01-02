// Copyright 2016 The etcd Authors
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

package embed

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/url"
	"os"
	"testing"

	"github.com/coreos/etcd/pkg/transport"

	"github.com/ghodss/yaml"
)

func TestConfigFileOtherFields(t *testing.T) {
	ctls := securityConfig{CAFile: "cca", CertFile: "ccert", KeyFile: "ckey"}
	ptls := securityConfig{CAFile: "pca", CertFile: "pcert", KeyFile: "pkey"}
	yc := struct {
		ClientSecurityCfgFile securityConfig `json:"client-transport-security"`
		PeerSecurityCfgFile   securityConfig `json:"peer-transport-security"`
		ForceNewCluster       bool           `json:"force-new-cluster"`
	}{
		ctls,
		ptls,
		true,
	}

	b, err := yaml.Marshal(&yc)
	if err != nil {
		t.Fatal(err)
	}

	tmpfile := mustCreateCfgFile(t, b)
	defer os.Remove(tmpfile.Name())

	cfg, err := ConfigFromFile(tmpfile.Name())
	if err != nil {
		t.Fatal(err)
	}

	if !cfg.ForceNewCluster {
		t.Errorf("ForceNewCluster = %v, want %v", cfg.ForceNewCluster, true)
	}

	if !ctls.equals(&cfg.ClientTLSInfo) {
		t.Errorf("ClientTLS = %v, want %v", cfg.ClientTLSInfo, ctls)
	}
	if !ptls.equals(&cfg.PeerTLSInfo) {
		t.Errorf("PeerTLS = %v, want %v", cfg.PeerTLSInfo, ptls)
	}
}

// TestUpdateDefaultClusterFromName ensures that etcd can start with 'etcd --name=abc'.
func TestUpdateDefaultClusterFromName(t *testing.T) {
	cfg := NewConfig()
	defaultInitialCluster := cfg.InitialCluster
	oldscheme := cfg.APUrls[0].Scheme
	origpeer := cfg.APUrls[0].String()
	origadvc := cfg.ACUrls[0].String()

	cfg.Name = "abc"
	_, lpport, _ := net.SplitHostPort(cfg.LPUrls[0].Host)

	// in case of 'etcd --name=abc'
	exp := fmt.Sprintf("%s=%s://localhost:%s", cfg.Name, oldscheme, lpport)
	cfg.UpdateDefaultClusterFromName(defaultInitialCluster)
	if exp != cfg.InitialCluster {
		t.Fatalf("initial-cluster expected %q, got %q", exp, cfg.InitialCluster)
	}
	// advertise peer URL should not be affected
	if origpeer != cfg.APUrls[0].String() {
		t.Fatalf("advertise peer url expected %q, got %q", origadvc, cfg.APUrls[0].String())
	}
	// advertise client URL should not be affected
	if origadvc != cfg.ACUrls[0].String() {
		t.Fatalf("advertise client url expected %q, got %q", origadvc, cfg.ACUrls[0].String())
	}
}

// TestUpdateDefaultClusterFromNameOverwrite ensures that machine's default host is only used
// if advertise URLs are default values(localhost:2379,2380) AND if listen URL is 0.0.0.0.
func TestUpdateDefaultClusterFromNameOverwrite(t *testing.T) {
	if defaultHostname == "" {
		t.Skip("machine's default host not found")
	}

	cfg := NewConfig()
	defaultInitialCluster := cfg.InitialCluster
	oldscheme := cfg.APUrls[0].Scheme
	origadvc := cfg.ACUrls[0].String()

	cfg.Name = "abc"
	_, lpport, _ := net.SplitHostPort(cfg.LPUrls[0].Host)
	cfg.LPUrls[0] = url.URL{Scheme: cfg.LPUrls[0].Scheme, Host: fmt.Sprintf("0.0.0.0:%s", lpport)}
	dhost, _ := cfg.UpdateDefaultClusterFromName(defaultInitialCluster)
	if dhost != defaultHostname {
		t.Fatalf("expected default host %q, got %q", defaultHostname, dhost)
	}
	aphost, apport, _ := net.SplitHostPort(cfg.APUrls[0].Host)
	if apport != lpport {
		t.Fatalf("advertise peer url got different port %s, expected %s", apport, lpport)
	}
	if aphost != defaultHostname {
		t.Fatalf("advertise peer url expected machine default host %q, got %q", defaultHostname, aphost)
	}
	expected := fmt.Sprintf("%s=%s://%s:%s", cfg.Name, oldscheme, defaultHostname, lpport)
	if expected != cfg.InitialCluster {
		t.Fatalf("initial-cluster expected %q, got %q", expected, cfg.InitialCluster)
	}

	// advertise client URL should not be affected
	if origadvc != cfg.ACUrls[0].String() {
		t.Fatalf("advertise-client-url expected %q, got %q", origadvc, cfg.ACUrls[0].String())
	}
}

func (s *securityConfig) equals(t *transport.TLSInfo) bool {
	return s.CAFile == t.CAFile &&
		s.CertFile == t.CertFile &&
		s.CertAuth == t.ClientCertAuth &&
		s.TrustedCAFile == t.TrustedCAFile
}

func mustCreateCfgFile(t *testing.T, b []byte) *os.File {
	tmpfile, err := ioutil.TempFile("", "servercfg")
	if err != nil {
		t.Fatal(err)
	}
	if _, err = tmpfile.Write(b); err != nil {
		t.Fatal(err)
	}
	if err = tmpfile.Close(); err != nil {
		t.Fatal(err)
	}
	return tmpfile
}
