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
	"log"
	"os"
	"strconv"
	"strings"
	"testing"

	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/rand"
)

func configFromEnv() (TestVM string, TestFlavor string, cfg PCConfig, ok bool) {
	var IgnoreCertificate bool
	var OverrideIP bool
	var err error
	cfg.Global.CloudTarget = os.Getenv("PHOTON_TARGET")
	cfg.Global.Tenant = os.Getenv("PHOTON_TENANT")
	cfg.Global.Project = os.Getenv("PHOTON_PROJECT")
	if os.Getenv("PHOTON_IGNORE_CERTIFICATE") != "" {
		IgnoreCertificate, err = strconv.ParseBool(os.Getenv("PHOTON_IGNORE_CERTIFICATE"))
	} else {
		IgnoreCertificate = false
	}
	if err != nil {
		log.Fatal(err)
	}
	cfg.Global.IgnoreCertificate = IgnoreCertificate
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
		cfg.Global.Tenant != "" &&
		cfg.Global.Project != "" &&
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
ignoreCertificate = true
tenant = tenant
project = project
overrideIP = false
`))
	if err != nil {
		t.Fatalf("Should succeed when a valid config is provided: %s", err)
	}

	if cfg.Global.CloudTarget != "0.0.0.0" {
		t.Errorf("incorrect photon target ip: %s", cfg.Global.CloudTarget)
	}

	if cfg.Global.Tenant != "tenant" {
		t.Errorf("incorrect tenant: %s", cfg.Global.Tenant)
	}

	if cfg.Global.Project != "project" {
		t.Errorf("incorrect project: %s", cfg.Global.Project)
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

	externalId, err := i.ExternalID(NodeName)
	if err != nil {
		t.Fatalf("Instances.ExternalID(%s) failed: %s", testVM, err)
	}
	t.Logf("Found ExternalID(%s) = %s\n", testVM, externalId)

	nonExistingVM := types.NodeName(rand.String(15))
	externalId, err = i.ExternalID(nonExistingVM)
	if err == cloudprovider.InstanceNotFound {
		t.Logf("VM %s was not found as expected\n", nonExistingVM)
	} else if err == nil {
		t.Fatalf("Instances.ExternalID did not fail as expected, VM %s was found", nonExistingVM)
	} else {
		t.Fatalf("Instances.ExternalID did not fail as expected, err: %v", err)
	}

	instanceId, err := i.InstanceID(NodeName)
	if err != nil {
		t.Fatalf("Instances.InstanceID(%s) failed: %s", testVM, err)
	}
	t.Logf("Found InstanceID(%s) = %s\n", testVM, instanceId)

	instanceId, err = i.InstanceID(nonExistingVM)
	if err == cloudprovider.InstanceNotFound {
		t.Logf("VM %s was not found as expected\n", nonExistingVM)
	} else if err == nil {
		t.Fatalf("Instances.InstanceID did not fail as expected, VM %s was found", nonExistingVM)
	} else {
		t.Fatalf("Instances.InstanceID did not fail as expected, err: %v", err)
	}

	addrs, err := i.NodeAddresses(NodeName)
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

	err = pc.AttachDisk(pdID, NodeName)
	if err != nil {
		t.Fatalf("Cannot attach persistent disk(%s) to VM(%s): %v", pdID, testVM, err)
	}

	_, err = pc.DiskIsAttached(pdID, NodeName)
	if err != nil {
		t.Fatalf("Cannot attach persistent disk(%s) to VM(%s): %v", pdID, testVM, err)
	}

	err = pc.DetachDisk(pdID, NodeName)
	if err != nil {
		t.Fatalf("Cannot detach persisten disk(%s) from VM(%s): %v", pdID, testVM, err)
	}

	err = pc.DeleteDisk(pdID)
	if err != nil {
		t.Fatalf("Cannot delete persisten disk(%s): %v", pdID, err)
	}
}
