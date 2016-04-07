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

	"golang.org/x/net/context"
)

func configFromEnv() (cfg VSphereConfig, ok bool) {
	fmt.Print("inside test")
	cfg.Global.VCenterIp = os.Getenv("VSPHERE_VCENTER")
	cfg.Global.VCenterPort = os.Getenv("VSPHERE_VCENTER_PORT")
	cfg.Global.User = os.Getenv("VSPHERE_USER")
	cfg.Global.Password = os.Getenv("VSPHERE_PASSWORD")
	cfg.Global.Datacenter = os.Getenv("VSPHERE_DATACENTER")

	InsecureFlag, err := strconv.ParseBool(os.Getenv("VSPHERE_INSECURE"))
	if err != nil {
		log.Fatal(err)
	}

	cfg.Global.InsecureFlag = InsecureFlag



	ok = (cfg.Global.VCenterIp != "" &&
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
port = 80
user = user
password = password
insecure-flag = true
datacenter = us-west
`))
	if err != nil {
		t.Fatalf("Should succeed when a valid config is provided: %s", err)
	}

	if cfg.Global.VCenterIp != "0.0.0.0" {
		t.Errorf("incorrect vcenter ip: %s", cfg.Global.VCenterIp)
	}

	if cfg.Global.VCenterPort != "80" {
		t.Errorf("incorrect vcenter port: %s", cfg.Global.VCenterPort)
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

func TestVSphereLogin(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	// Create vSphere configuration object
	vs, err := newVSphere(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create vSphere client
	c, err := vsphereLogin(vs.cfg, ctx)
	if err != nil {
		t.Errorf("Failed to create vSpere client: %s", err)
	}
	defer c.Logout(ctx)
}

func TestZones(t *testing.T) {
	cfg := VSphereConfig{}
	cfg.Global.Datacenter = "myDatacenter"

	// Create vSphere configuration object
	vs, err := newVSphere(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}

	z, ok := vs.Zones()
	if !ok {
		t.Fatalf("Zones() returned false")
	}

	zone, err := z.GetZone()
	if err != nil {
		t.Fatalf("GetZone() returned error: %s", err)
	}

	if zone.Region != vs.cfg.Global.Datacenter {
		t.Fatalf("GetZone() returned wrong region (%s)", zone.Region)
	}
}
