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

package clc

import (
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/cloudprovider"
)

func checkInterface(provider cloudprovider.Interface) string {
	return provider.ProviderName()
}

func TestInterface(t *testing.T) {
	clc := &CLCCloud{}
	checkInterface(clc)
}

func checkLoadbalancer(provider cloudprovider.LoadBalancer) {
}

func TestLoadbalancer(t *testing.T) {
	clc := &CLCCloud{}
	checkLoadbalancer(clc)
}

func checkZones(provider cloudprovider.Zones) {
}

func TestZones(t *testing.T) {
	clc := &CLCCloud{}
	checkZones(clc)
}

func TestReadConfig(t *testing.T) {
	_, err := readConfig(nil)
	if err == nil {
		t.Errorf("fail on nil config: %s", err)
	}

	cfg, err := readConfig(strings.NewReader(`
[Global]
alias =
datacenter = VA1
username = admin
password-base64 = cGE1NXcwcmQ=
`))
	if err != nil {
		t.Fatalf("Error found with valid configuration: %s", err)
	}
	if cfg.Global.Username != "admin" {
		t.Errorf("incorrect username: %s", cfg.Global.Username)
	}
	if cfg.Global.Password != "pa55w0rd" {
		t.Errorf("incorrect base64-decoded password: %s", cfg.Global.Password)
	}
}
