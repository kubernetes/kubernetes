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

package vsphere

import (
	"log"
	"os"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/rand"
)

func configFromEnv() (cfg VSphereConfig, ok bool) {
	var InsecureFlag bool
	var err error
	cfg.Global.VCenterIP = os.Getenv("VSPHERE_VCENTER")
	cfg.Global.VCenterPort = os.Getenv("VSPHERE_VCENTER_PORT")
	cfg.Global.User = os.Getenv("VSPHERE_USER")
	cfg.Global.Password = os.Getenv("VSPHERE_PASSWORD")
	cfg.Global.Datacenter = os.Getenv("VSPHERE_DATACENTER")
	cfg.Network.PublicNetwork = os.Getenv("VSPHERE_PUBLIC_NETWORK")
	cfg.Global.Datastore = os.Getenv("VSPHERE_DATASTORE")
	cfg.Disk.SCSIControllerType = os.Getenv("VSPHERE_SCSICONTROLLER_TYPE")
	cfg.Global.WorkingDir = os.Getenv("VSPHERE_WORKING_DIR")
	if os.Getenv("VSPHERE_INSECURE") != "" {
		InsecureFlag, err = strconv.ParseBool(os.Getenv("VSPHERE_INSECURE"))
	} else {
		InsecureFlag = false
	}
	if err != nil {
		log.Fatal(err)
	}
	cfg.Global.InsecureFlag = InsecureFlag

	ok = (cfg.Global.VCenterIP != "" &&
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
port = 443
user = user
password = password
insecure-flag = true
datacenter = us-west
`))
	if err != nil {
		t.Fatalf("Should succeed when a valid config is provided: %s", err)
	}

	if cfg.Global.VCenterIP != "0.0.0.0" {
		t.Errorf("incorrect vcenter ip: %s", cfg.Global.VCenterIP)
	}

	if cfg.Global.VCenterPort != "443" {
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
	err = vSphereLogin(vs, ctx)
	if err != nil {
		t.Errorf("Failed to create vSpere client: %s", err)
	}
	defer vs.client.Logout(ctx)
}

func TestZones(t *testing.T) {
	cfg := VSphereConfig{}
	cfg.Global.Datacenter = "myDatacenter"
	failureZone := "myCluster"

	// Create vSphere configuration object
	vs := VSphere{
		cfg:         &cfg,
		clusterName: failureZone,
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

	if zone.FailureDomain != failureZone {
		t.Fatalf("GetZone() returned wrong Failure Zone (%s)", zone.FailureDomain)
	}
}

func TestInstances(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	vs, err := newVSphere(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}

	i, ok := vs.Instances()
	if !ok {
		t.Fatalf("Instances() returned false")
	}

	srvs, err := i.List("*")
	if err != nil {
		t.Fatalf("Instances.List() failed: %s", err)
	}

	if len(srvs) == 0 {
		t.Fatalf("Instances.List() returned zero servers")
	}
	t.Logf("Found servers (%d): %s\n", len(srvs), srvs)

	externalId, err := i.ExternalID(srvs[0])
	if err != nil {
		t.Fatalf("Instances.ExternalID(%s) failed: %s", srvs[0], err)
	}
	t.Logf("Found ExternalID(%s) = %s\n", srvs[0], externalId)

	nonExistingVM := types.NodeName(rand.String(15))
	externalId, err = i.ExternalID(nonExistingVM)
	if err == cloudprovider.InstanceNotFound {
		t.Logf("VM %s was not found as expected\n", nonExistingVM)
	} else if err == nil {
		t.Fatalf("Instances.ExternalID did not fail as expected, VM %s was found", nonExistingVM)
	} else {
		t.Fatalf("Instances.ExternalID did not fail as expected, err: %v", err)
	}

	instanceId, err := i.InstanceID(srvs[0])
	if err != nil {
		t.Fatalf("Instances.InstanceID(%s) failed: %s", srvs[0], err)
	}
	t.Logf("Found InstanceID(%s) = %s\n", srvs[0], instanceId)

	instanceId, err = i.InstanceID(nonExistingVM)
	if err == cloudprovider.InstanceNotFound {
		t.Logf("VM %s was not found as expected\n", nonExistingVM)
	} else if err == nil {
		t.Fatalf("Instances.InstanceID did not fail as expected, VM %s was found", nonExistingVM)
	} else {
		t.Fatalf("Instances.InstanceID did not fail as expected, err: %v", err)
	}

	addrs, err := i.NodeAddresses(srvs[0])
	if err != nil {
		t.Fatalf("Instances.NodeAddresses(%s) failed: %s", srvs[0], err)
	}
	t.Logf("Found NodeAddresses(%s) = %s\n", srvs[0], addrs)
}

func TestVolumes(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	vs, err := newVSphere(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}

	i, ok := vs.Instances()
	if !ok {
		t.Fatalf("Instances() returned false")
	}

	srvs, err := i.List("*")
	if err != nil {
		t.Fatalf("Instances.List() failed: %s", err)
	}
	if len(srvs) == 0 {
		t.Fatalf("Instances.List() returned zero servers")
	}

	volumeOptions := &VolumeOptions{
		CapacityKB: 1 * 1024 * 1024,
		Tags:       nil,
		Name:       "kubernetes-test-volume-" + rand.String(10),
		DiskFormat: "thin"}

	volPath, err := vs.CreateVolume(volumeOptions)
	if err != nil {
		t.Fatalf("Cannot create a new VMDK volume: %v", err)
	}

	_, _, err = vs.AttachDisk(volPath, "")
	if err != nil {
		t.Fatalf("Cannot attach volume(%s) to VM(%s): %v", volPath, srvs[0], err)
	}

	err = vs.DetachDisk(volPath, "")
	if err != nil {
		t.Fatalf("Cannot detach disk(%s) from VM(%s): %v", volPath, srvs[0], err)
	}

	// todo: Deleting a volume after detach currently not working through API or UI (vSphere)
	// err = vs.DeleteVolume(volPath)
	// if err != nil {
	// 	t.Fatalf("Cannot delete VMDK volume %s: %v", volPath, err)
	// }
}
