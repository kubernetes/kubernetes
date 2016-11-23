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

package digitalocean

import (
	"os"
	"strings"
	"testing"
)

func TestReadConfig(t *testing.T) {
	_, err := readConfig(nil)
	if err == nil {
		t.Errorf("Should fail when no config is provided: %s", err)
	}

	cfg, err := readConfig(strings.NewReader(`
[Global]
Region = nyc1
ApiKey = 123456
`))
	if err != nil {
		t.Fatalf("Should succeed when a valid config is provided: %s", err)
	}
	if cfg.Global.Region != "nyc1" {
		t.Errorf("incorrect region: %s", cfg.Global.Region)
	}
}

func configFromEnv() (cfg Config, ok bool) {
	cfg.Global.ApiKey = os.Getenv("DO_API_KEY")
	cfg.Global.Region = os.Getenv("DO_REGION_NAME")

	ok = (cfg.Global.Region != "" &&
		cfg.Global.ApiKey != "")

	return
}

func TestNewDigitalOcean(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	_, err := newDigitalOcean(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate Digitalocean: %s", err)
	}
}
func TestInstances(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	do, err := newDigitalOcean(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate Digitalocean: %s", err)
	}

	i, ok := do.Instances()
	if !ok {
		t.Fatalf("Instances() returned false")
	}

	srvs, err := i.List(".")
	if err != nil {
		t.Fatalf("Instances.List() failed: %s", err)
	}
	if len(srvs) == 0 {
		t.Fatalf("Instances.List() returned zero servers")
	}
	t.Logf("Found servers (%d): %s\n", len(srvs), srvs)

	addrs, err := i.NodeAddresses(srvs[0])
	if err != nil {
		t.Fatalf("Instances.NodeAddresses(%s) failed: %s", srvs[0], err)
	}
	t.Logf("Found NodeAddresses(%s) = %s\n", srvs[0], addrs)
}
