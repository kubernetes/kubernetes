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

package vsphere

import (
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"testing"
)

func configFromEnv() (cfg VSphereConfig, ok bool) {
	fmt.Print("inside test")
	cfg.Global.VSphereServer = os.Getenv("VSPHERE_SERVER")
	cfg.Global.User = os.Getenv("VSPHERE_USER")
	cfg.Global.Password = os.Getenv("VSPHERE_PASSWORD")
	cfg.Global.Datacenter = os.Getenv("VSPHERE_DATACENTER")

	InsecureFlag, err := strconv.ParseBool(os.Getenv("VSPHERE_INSECURE"))
	if err != nil {
		log.Fatal(err)
	}

	cfg.Global.InsecureFlag = InsecureFlag

	ok = (cfg.Global.VSphereServer != "" &&
		cfg.Global.User != "")

	return
}

func TestReadConfig(t *testing.T) {
	_, err := readConfig(nil)
	if err == nil {
		t.Errorf("Should fail when no config is provided: %s", err)
	}

	cfg, err := readConfig(strings.NewReader(`
[Global]
server = 0.0.0.0
user = user
password = password
insecure-flag = true
datacenter = us-west
`))
	if err != nil {
		t.Fatalf("Should succeed when a valid config is provided: %s", err)
	}
	if cfg.Global.VSphereServer != "0.0.0.0" {
		t.Errorf("incorrect server: %s", cfg.Global.VSphereServer)
	}

	if cfg.Global.Datacenter != "us-west" {
		t.Errorf("incorrect datacenter: %s", cfg.Global.Datacenter)
	}
}

func TestNewVSphere(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	_, err := newVSphere(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}
}
