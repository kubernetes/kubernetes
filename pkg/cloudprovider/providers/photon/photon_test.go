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

package photon

import (
	"context"
	"log"
	"os"
	"strconv"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	cloudprovider "k8s.io/cloud-provider"
)

func configFromEnv() (TestVM string, TestFlavor string, cfg PCConfig, ok bool) {
	var AuthEnabled bool
	var OverrideIP bool
	var err error
	cfg.Global.CloudTarget = os.Getenv("PHOTON_TARGET")
	cfg.Global.Project = os.Getenv("PHOTON_PROJECT")
	cfg.Global.VMID = os.Getenv("PHOTON_VMID")
	if os.Getenv("PHOTON_AUTH_ENABLED") != "" {
		AuthEnabled, err = strconv.ParseBool(os.Getenv("PHOTON_AUTH_ENABLED"))
	} else {
		AuthEnabled = false
	}
	if err != nil {
		log.Fatal(err)
	}
	cfg.Global.AuthEnabled = AuthEnabled
	if os.Getenv("PHOTON_OVERRIDE_IP") != "" {
		OverrideIP, err = strconv.ParseBool(os.Getenv("PHOTON_OVERRIDE_IP"))
	} else {
		OverrideIP = false
	}
	if err != nil {
		log.Fatal(err)
	}
	cfg.Global.OverrideIP = OverrideIP

	TestVM = os.Getenv("PHOTON_TEST_VM")
	if os.Getenv("PHOTON_TEST_FLAVOR") != "" {
		TestFlavor = os.Getenv("PHOTON_TEST_FLAVOR")
	} else {
		TestFlavor = ""
	}
	if err != nil {
		log.Fatal(err)
	}

	ok = (cfg.Global.CloudTarget != "" &&
		cfg.Global.Project != "" &&
		cfg.Global.VMID != "" &&
		TestVM != "")

	return
}

func TestReadConfig(t *testing.T) {
	_, err := readConfig(nil)
	if err == nil {
		t.Errorf("Should fail when no config is provided: %s", err)
	}

	cfg, err := readConfig(strings.NewReader(`
[Global]
target = 0.0.0.0
project = project
overrideIP = true
vmID = vmid
authentication = false
`))
	if err != nil {
		t.Fatalf("Should succeed when a valid config is provided: %s", err)
	}

	if cfg.Global.CloudTarget != "0.0.0.0" {
		t.Errorf("incorrect photon target ip: %s", cfg.Global.CloudTarget)
	}

	if cfg.Global.Project != "project" {
		t.Errorf("incorrect project: %s", cfg.Global.Project)
	}

	if cfg.Global.VMID != "vmid" {
		t.Errorf("incorrect vmid: %s", cfg.Global.VMID)
	}
}

func TestNewPCCloud(t *testing.T) {
	_, _, cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	_, err := newPCCloud(cfg)
	if err != nil {
		t.Fatalf("Failed to create new Photon client: %s", err)
	}
}

func TestInstances(t *testing.T) {
	testVM, _, cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}
	NodeName := types.NodeName(testVM)

	pc, err := newPCCloud(cfg)
	if err != nil {
		t.Fatalf("Failed to create new Photon client: %s", err)
	}

	i, ok := pc.Instances()
	if !ok {
		t.Fatalf("Instances() returned false")
	}

	nonExistingVM := types.NodeName(rand.String(15))
	instanceId, err := i.InstanceID(context.TODO(), NodeName)
	if err != nil {
		t.Fatalf("Instances.InstanceID(%s) failed: %s", testVM, err)
	}
	t.Logf("Found InstanceID(%s) = %s\n", testVM, instanceId)

	_, err = i.InstanceID(context.TODO(), nonExistingVM)
	if err == cloudprovider.InstanceNotFound {
		t.Logf("VM %s was not found as expected\n", nonExistingVM)
	} else if err == nil {
		t.Fatalf("Instances.InstanceID did not fail as expected, VM %s was found", nonExistingVM)
	} else {
		t.Fatalf("Instances.InstanceID did not fail as expected, err: %v", err)
	}

	addrs, err := i.NodeAddresses(context.TODO(), NodeName)
	if err != nil {
		t.Fatalf("Instances.NodeAddresses(%s) failed: %s", testVM, err)
	}
	t.Logf("Found NodeAddresses(%s) = %s\n", testVM, addrs)
}

func TestVolumes(t *testing.T) {
	testVM, testFlavor, cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	pc, err := newPCCloud(cfg)
	if err != nil {
		t.Fatalf("Failed to create new Photon client: %s", err)
	}

	NodeName := types.NodeName(testVM)

	volumeOptions := &VolumeOptions{
		CapacityGB: 2,
		Tags:       nil,
		Name:       "kubernetes-test-volume-" + rand.String(10),
		Flavor:     testFlavor}

	pdID, err := pc.CreateDisk(volumeOptions)
	if err != nil {
		t.Fatalf("Cannot create a Photon persistent disk: %v", err)
	}

	err = pc.AttachDisk(context.TODO(), pdID, NodeName)
	if err != nil {
		t.Fatalf("Cannot attach persistent disk(%s) to VM(%s): %v", pdID, testVM, err)
	}

	_, err = pc.DiskIsAttached(context.TODO(), pdID, NodeName)
	if err != nil {
		t.Fatalf("Cannot attach persistent disk(%s) to VM(%s): %v", pdID, testVM, err)
	}

	err = pc.DetachDisk(context.TODO(), pdID, NodeName)
	if err != nil {
		t.Fatalf("Cannot detach persisten disk(%s) from VM(%s): %v", pdID, testVM, err)
	}

	err = pc.DeleteDisk(pdID)
	if err != nil {
		t.Fatalf("Cannot delete persisten disk(%s): %v", pdID, err)
	}
}
