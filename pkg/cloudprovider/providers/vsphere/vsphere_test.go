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
	"context"
	"crypto/tls"
	"crypto/x509"
	"io/ioutil"
	"log"
	"os"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"testing"

	"github.com/vmware/govmomi/find"
	lookup "github.com/vmware/govmomi/lookup/simulator"
	"github.com/vmware/govmomi/property"
	"github.com/vmware/govmomi/simulator"
	"github.com/vmware/govmomi/simulator/vpx"
	sts "github.com/vmware/govmomi/sts/simulator"
	"github.com/vmware/govmomi/vapi/rest"
	vapi "github.com/vmware/govmomi/vapi/simulator"
	"github.com/vmware/govmomi/vapi/tags"
	"github.com/vmware/govmomi/vim25/mo"
	vmwaretypes "github.com/vmware/govmomi/vim25/types"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	cloudprovider "k8s.io/cloud-provider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere/vclib"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere/vclib/fixtures"
)

// localhostCert was generated from crypto/tls/generate_cert.go with the following command:
//     go run generate_cert.go  --rsa-bits 512 --host 127.0.0.1,::1,example.com --ca --start-date "Jan 1 00:00:00 1970" --duration=1000000h
var localhostCert = `-----BEGIN CERTIFICATE-----
MIIBjzCCATmgAwIBAgIRAKpi2WmTcFrVjxrl5n5YDUEwDQYJKoZIhvcNAQELBQAw
EjEQMA4GA1UEChMHQWNtZSBDbzAgFw03MDAxMDEwMDAwMDBaGA8yMDg0MDEyOTE2
MDAwMFowEjEQMA4GA1UEChMHQWNtZSBDbzBcMA0GCSqGSIb3DQEBAQUAA0sAMEgC
QQC9fEbRszP3t14Gr4oahV7zFObBI4TfA5i7YnlMXeLinb7MnvT4bkfOJzE6zktn
59zP7UiHs3l4YOuqrjiwM413AgMBAAGjaDBmMA4GA1UdDwEB/wQEAwICpDATBgNV
HSUEDDAKBggrBgEFBQcDATAPBgNVHRMBAf8EBTADAQH/MC4GA1UdEQQnMCWCC2V4
YW1wbGUuY29thwR/AAABhxAAAAAAAAAAAAAAAAAAAAABMA0GCSqGSIb3DQEBCwUA
A0EAUsVE6KMnza/ZbodLlyeMzdo7EM/5nb5ywyOxgIOCf0OOLHsPS9ueGLQX9HEG
//yjTXuhNcUugExIjM/AIwAZPQ==
-----END CERTIFICATE-----`

// localhostKey is the private key for localhostCert.
var localhostKey = `-----BEGIN RSA PRIVATE KEY-----
MIIBOwIBAAJBAL18RtGzM/e3XgavihqFXvMU5sEjhN8DmLtieUxd4uKdvsye9Phu
R84nMTrOS2fn3M/tSIezeXhg66quOLAzjXcCAwEAAQJBAKcRxH9wuglYLBdI/0OT
BLzfWPZCEw1vZmMR2FF1Fm8nkNOVDPleeVGTWoOEcYYlQbpTmkGSxJ6ya+hqRi6x
goECIQDx3+X49fwpL6B5qpJIJMyZBSCuMhH4B7JevhGGFENi3wIhAMiNJN5Q3UkL
IuSvv03kaPR5XVQ99/UeEetUgGvBcABpAiBJSBzVITIVCGkGc7d+RCf49KTCIklv
bGWObufAR8Ni4QIgWpILjW8dkGg8GOUZ0zaNA6Nvt6TIv2UWGJ4v5PoV98kCIQDx
rIiZs5QbKdycsv9gQJzwQAogC8o04X3Zz3dsoX+h4A==
-----END RSA PRIVATE KEY-----`

func configFromEnv() (cfg VSphereConfig, ok bool) {
	var InsecureFlag bool
	var err error
	cfg.Global.VCenterIP = os.Getenv("VSPHERE_VCENTER")
	cfg.Global.VCenterPort = os.Getenv("VSPHERE_VCENTER_PORT")
	cfg.Global.User = os.Getenv("VSPHERE_USER")
	cfg.Global.Password = os.Getenv("VSPHERE_PASSWORD")
	cfg.Global.Datacenter = os.Getenv("VSPHERE_DATACENTER")
	cfg.Network.PublicNetwork = os.Getenv("VSPHERE_PUBLIC_NETWORK")
	cfg.Global.DefaultDatastore = os.Getenv("VSPHERE_DATASTORE")
	cfg.Disk.SCSIControllerType = os.Getenv("VSPHERE_SCSICONTROLLER_TYPE")
	cfg.Global.WorkingDir = os.Getenv("VSPHERE_WORKING_DIR")
	cfg.Global.VMName = os.Getenv("VSPHERE_VM_NAME")
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

// configFromSim starts a vcsim instance and returns config for use against the vcsim instance.
// The vcsim instance is configured with an empty tls.Config.
func configFromSim() (VSphereConfig, func()) {
	return configFromSimWithTLS(new(tls.Config), true)
}

// configFromSimWithTLS starts a vcsim instance and returns config for use against the vcsim instance.
// The vcsim instance is configured with a tls.Config. The returned client
// config can be configured to allow/decline insecure connections.
func configFromSimWithTLS(tlsConfig *tls.Config, insecureAllowed bool) (VSphereConfig, func()) {
	var cfg VSphereConfig
	model := simulator.VPX()

	err := model.Create()
	if err != nil {
		log.Fatal(err)
	}

	model.Service.TLS = tlsConfig
	s := model.Service.NewServer()

	// STS simulator
	path, handler := sts.New(s.URL, vpx.Setting)
	model.Service.ServeMux.Handle(path, handler)

	// vAPI simulator
	path, handler = vapi.New(s.URL, vpx.Setting)
	model.Service.ServeMux.Handle(path, handler)

	// Lookup Service simulator
	model.Service.RegisterSDK(lookup.New())

	cfg.Global.InsecureFlag = insecureAllowed

	cfg.Global.VCenterIP = s.URL.Hostname()
	cfg.Global.VCenterPort = s.URL.Port()
	cfg.Global.User = s.URL.User.Username()
	cfg.Global.Password, _ = s.URL.User.Password()
	cfg.Global.Datacenter = vclib.TestDefaultDatacenter
	cfg.Network.PublicNetwork = vclib.TestDefaultNetwork
	cfg.Global.DefaultDatastore = vclib.TestDefaultDatastore
	cfg.Disk.SCSIControllerType = os.Getenv("VSPHERE_SCSICONTROLLER_TYPE")
	cfg.Global.WorkingDir = os.Getenv("VSPHERE_WORKING_DIR")
	cfg.Global.VMName = os.Getenv("VSPHERE_VM_NAME")

	if cfg.Global.WorkingDir == "" {
		cfg.Global.WorkingDir = "vm" // top-level Datacenter.VmFolder
	}

	uuid := simulator.Map.Any("VirtualMachine").(*simulator.VirtualMachine).Config.Uuid
	getVMUUID = func() (string, error) { return uuid, nil }

	return cfg, func() {
		getVMUUID = GetVMUUID
		s.Close()
		model.Remove()
	}
}

// configFromEnvOrSim returns config from configFromEnv if set, otherwise returns configFromSim.
func configFromEnvOrSim() (VSphereConfig, func()) {
	cfg, ok := configFromEnv()
	if ok {
		return cfg, func() {}
	}
	return configFromSim()
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
vm-uuid = 1234
vm-name = vmname
ca-file = /some/path/to/a/ca.pem
`))
	if err != nil {
		t.Fatalf("Should succeed when a valid config is provided: %s", err)
	}

	if cfg.Global.VCenterIP != "0.0.0.0" {
		t.Errorf("incorrect vcenter ip: %s", cfg.Global.VCenterIP)
	}

	if cfg.Global.Datacenter != "us-west" {
		t.Errorf("incorrect datacenter: %s", cfg.Global.Datacenter)
	}

	if cfg.Global.VMUUID != "1234" {
		t.Errorf("incorrect vm-uuid: %s", cfg.Global.VMUUID)
	}

	if cfg.Global.VMName != "vmname" {
		t.Errorf("incorrect vm-name: %s", cfg.Global.VMName)
	}

	if cfg.Global.CAFile != "/some/path/to/a/ca.pem" {
		t.Errorf("incorrect ca-file: %s", cfg.Global.CAFile)
	}
}

func TestNewVSphere(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	_, err := newControllerNode(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}
}

func TestVSphereLogin(t *testing.T) {
	cfg, cleanup := configFromEnvOrSim()
	defer cleanup()

	// Create vSphere configuration object
	vs, err := newControllerNode(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}

	// Create context
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Create vSphere client
	vcInstance, ok := vs.vsphereInstanceMap[cfg.Global.VCenterIP]
	if !ok {
		t.Fatalf("Couldn't get vSphere instance: %s", cfg.Global.VCenterIP)
	}

	err = vcInstance.conn.Connect(ctx)
	if err != nil {
		t.Errorf("Failed to connect to vSphere: %s", err)
	}
	vcInstance.conn.Logout(ctx)
}

func TestVSphereLoginByToken(t *testing.T) {
	cfg, cleanup := configFromSim()
	defer cleanup()

	// Configure for SAML token auth
	cfg.Global.User = localhostCert
	cfg.Global.Password = localhostKey

	// Create vSphere configuration object
	vs, err := newControllerNode(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}

	ctx := context.Background()

	// Create vSphere client
	vcInstance, ok := vs.vsphereInstanceMap[cfg.Global.VCenterIP]
	if !ok {
		t.Fatalf("Couldn't get vSphere instance: %s", cfg.Global.VCenterIP)
	}

	err = vcInstance.conn.Connect(ctx)
	if err != nil {
		t.Errorf("Failed to connect to vSphere: %s", err)
	}
	vcInstance.conn.Logout(ctx)
}

func TestVSphereLoginWithCaCert(t *testing.T) {
	caCertPEM, err := ioutil.ReadFile(fixtures.CaCertPath)
	if err != nil {
		t.Fatalf("Could not read ca cert from file")
	}

	serverCert, err := tls.LoadX509KeyPair(fixtures.ServerCertPath, fixtures.ServerKeyPath)
	if err != nil {
		t.Fatalf("Could not load server cert and server key from files: %#v", err)
	}

	certPool := x509.NewCertPool()
	if ok := certPool.AppendCertsFromPEM(caCertPEM); !ok {
		t.Fatalf("Cannot add CA to CAPool")
	}

	tlsConfig := tls.Config{
		Certificates: []tls.Certificate{serverCert},
		RootCAs:      certPool,
	}

	cfg, cleanup := configFromSimWithTLS(&tlsConfig, false)
	defer cleanup()

	cfg.Global.CAFile = fixtures.CaCertPath

	// Create vSphere configuration object
	vs, err := newControllerNode(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}

	ctx := context.Background()

	// Create vSphere client
	vcInstance, ok := vs.vsphereInstanceMap[cfg.Global.VCenterIP]
	if !ok {
		t.Fatalf("Couldn't get vSphere instance: %s", cfg.Global.VCenterIP)
	}

	err = vcInstance.conn.Connect(ctx)
	if err != nil {
		t.Errorf("Failed to connect to vSphere: %s", err)
	}
	vcInstance.conn.Logout(ctx)
}

func TestZonesNoConfig(t *testing.T) {
	_, ok := new(VSphere).Zones()
	if ok {
		t.Fatalf("Zones() should return false without VCP configured")
	}
}

func TestZones(t *testing.T) {
	// Any context will do
	ctx := context.Background()

	// Create a vcsim instance
	cfg, cleanup := configFromSim()
	defer cleanup()

	// Create vSphere configuration object
	vs, err := newControllerNode(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}

	// Configure region and zone categories
	vs.cfg.Labels.Region = "k8s-region"
	vs.cfg.Labels.Zone = "k8s-zone"

	// Create vSphere client
	vsi, ok := vs.vsphereInstanceMap[cfg.Global.VCenterIP]
	if !ok {
		t.Fatalf("Couldn't get vSphere instance: %s", cfg.Global.VCenterIP)
	}

	err = vsi.conn.Connect(ctx)
	if err != nil {
		t.Errorf("Failed to connect to vSphere: %s", err)
	}

	// Lookup Datacenter for this test's Workspace
	dc, err := vclib.GetDatacenter(ctx, vsi.conn, vs.cfg.Workspace.Datacenter)
	if err != nil {
		t.Fatal(err)
	}

	// Lookup VM's host where we'll attach tags
	host, err := dc.GetHostByVMUUID(ctx, vs.vmUUID)
	if err != nil {
		t.Fatal(err)
	}

	// Property Collector instance
	pc := property.DefaultCollector(vsi.conn.Client)

	// Tag manager instance
	m := tags.NewManager(rest.NewClient(vsi.conn.Client))

	// Create a region category
	regionID, err := m.CreateCategory(ctx, &tags.Category{Name: vs.cfg.Labels.Region})
	if err != nil {
		t.Fatal(err)
	}

	// Create a region tag
	regionID, err = m.CreateTag(ctx, &tags.Tag{CategoryID: regionID, Name: "k8s-region-US"})
	if err != nil {
		t.Fatal(err)
	}

	// Create a zone category
	zoneID, err := m.CreateCategory(ctx, &tags.Category{Name: vs.cfg.Labels.Zone})
	if err != nil {
		t.Fatal(err)
	}

	// Create a zone tag
	zoneID, err = m.CreateTag(ctx, &tags.Tag{CategoryID: zoneID, Name: "k8s-zone-US-CA1"})
	if err != nil {
		t.Fatal(err)
	}

	// Create a random category
	randomID, err := m.CreateCategory(ctx, &tags.Category{Name: "random-cat"})
	if err != nil {
		t.Fatal(err)
	}

	// Create a random tag
	randomID, err = m.CreateTag(ctx, &tags.Tag{CategoryID: randomID, Name: "random-tag"})
	if err != nil {
		t.Fatal(err)
	}

	// Attach a random tag to VM's host
	if err = m.AttachTag(ctx, randomID, host); err != nil {
		t.Fatal(err)
	}

	// Expecting Zones() to return true, indicating VCP supports the Zones interface
	zones, ok := vs.Zones()
	if !ok {
		t.Fatalf("zones=%t", ok)
	}

	// GetZone() tests, covering error and success paths
	tests := []struct {
		name string // name of the test for logging
		fail bool   // expect GetZone() to return error if true
		prep func() // prepare vCenter state for the test
	}{
		{"no tags", true, func() {
			// no prep
		}},
		{"no zone tag", true, func() {
			if err = m.AttachTag(ctx, regionID, host); err != nil {
				t.Fatal(err)
			}
		}},
		{"host tags set", false, func() {
			if err = m.AttachTag(ctx, zoneID, host); err != nil {
				t.Fatal(err)
			}
		}},
		{"host tags removed", true, func() {
			if err = m.DetachTag(ctx, zoneID, host); err != nil {
				t.Fatal(err)
			}
			if err = m.DetachTag(ctx, regionID, host); err != nil {
				t.Fatal(err)
			}
		}},
		{"dc region, cluster zone", false, func() {
			var h mo.HostSystem
			if err = pc.RetrieveOne(ctx, host.Reference(), []string{"parent"}, &h); err != nil {
				t.Fatal(err)
			}
			// Attach region tag to Datacenter
			if err = m.AttachTag(ctx, regionID, dc); err != nil {
				t.Fatal(err)
			}
			// Attach zone tag to Cluster
			if err = m.AttachTag(ctx, zoneID, h.Parent); err != nil {
				t.Fatal(err)
			}
		}},
	}

	for _, test := range tests {
		test.prep()

		zone, err := zones.GetZone(ctx)
		if test.fail {
			if err == nil {
				t.Errorf("%s: expected error", test.name)
			} else {
				t.Logf("%s: expected error=%s", test.name, err)
			}
		} else {
			if err != nil {
				t.Errorf("%s: %s", test.name, err)
			}
			t.Logf("zone=%#v", zone)
		}
	}
}

func TestGetZoneToHosts(t *testing.T) {
	// Common setup for all testcases in this test
	ctx := context.TODO()

	// Create a vcsim instance
	cfg, cleanup := configFromSim()
	defer cleanup()

	// Create vSphere configuration object
	vs, err := newControllerNode(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}

	// Configure region and zone categories
	vs.cfg.Labels.Region = "k8s-region"
	vs.cfg.Labels.Zone = "k8s-zone"

	// Create vSphere client
	vsi, ok := vs.vsphereInstanceMap[cfg.Global.VCenterIP]
	if !ok {
		t.Fatalf("Couldn't get vSphere instance: %s", cfg.Global.VCenterIP)
	}

	err = vsi.conn.Connect(ctx)
	if err != nil {
		t.Errorf("Failed to connect to vSphere: %s", err)
	}

	// Lookup Datacenter for this test's Workspace
	dc, err := vclib.GetDatacenter(ctx, vsi.conn, vs.cfg.Workspace.Datacenter)
	if err != nil {
		t.Fatal(err)
	}

	// Property Collector instance
	pc := property.DefaultCollector(vsi.conn.Client)

	// find all hosts in VC
	finder := find.NewFinder(vsi.conn.Client, true)
	finder.SetDatacenter(dc.Datacenter)
	allVcHostsList, err := finder.HostSystemList(ctx, "*")
	if err != nil {
		t.Fatal(err)
	}
	var allVcHosts []vmwaretypes.ManagedObjectReference
	for _, h := range allVcHostsList {
		allVcHosts = append(allVcHosts, h.Reference())
	}

	// choose a cluster to apply zone/region tags
	cluster := simulator.Map.Any("ClusterComputeResource")
	var c mo.ClusterComputeResource
	if err := pc.RetrieveOne(ctx, cluster.Reference(), []string{"host"}, &c); err != nil {
		t.Fatal(err)
	}

	// choose one of the host inside this cluster to apply zone/region tags
	if c.Host == nil || len(c.Host) == 0 {
		t.Fatalf("This test needs a host inside a cluster.")
	}
	clusterHosts := c.Host
	sortHosts(clusterHosts)
	// pick the first host in the cluster to apply tags
	host := clusterHosts[0]
	remainingHostsInCluster := clusterHosts[1:]

	// Tag manager instance
	m := tags.NewManager(rest.NewClient(vsi.conn.Client))

	// Create a region category
	regionCat, err := m.CreateCategory(ctx, &tags.Category{Name: vs.cfg.Labels.Region})
	if err != nil {
		t.Fatal(err)
	}

	// Create a region tag
	regionName := "k8s-region-US"
	regionTag, err := m.CreateTag(ctx, &tags.Tag{CategoryID: regionCat, Name: regionName})
	if err != nil {
		t.Fatal(err)
	}

	// Create a zone category
	zoneCat, err := m.CreateCategory(ctx, &tags.Category{Name: vs.cfg.Labels.Zone})
	if err != nil {
		t.Fatal(err)
	}

	// Create a zone tag
	zone1Name := "k8s-zone-US-CA1"
	zone1Tag, err := m.CreateTag(ctx, &tags.Tag{CategoryID: zoneCat, Name: zone1Name})
	if err != nil {
		t.Fatal(err)
	}
	zone1 := cloudprovider.Zone{FailureDomain: zone1Name, Region: regionName}

	// Create a second zone tag
	zone2Name := "k8s-zone-US-CA2"
	zone2Tag, err := m.CreateTag(ctx, &tags.Tag{CategoryID: zoneCat, Name: zone2Name})
	if err != nil {
		t.Fatal(err)
	}
	zone2 := cloudprovider.Zone{FailureDomain: zone2Name, Region: regionName}

	testcases := []struct {
		name        string
		tags        map[string][]mo.Reference
		zoneToHosts map[cloudprovider.Zone][]vmwaretypes.ManagedObjectReference
	}{
		{
			name:        "Zone and Region tags on host",
			tags:        map[string][]mo.Reference{zone1Tag: {host}, regionTag: {host}},
			zoneToHosts: map[cloudprovider.Zone][]vmwaretypes.ManagedObjectReference{zone1: {host}},
		},
		{
			name:        "Zone on host Region on datacenter",
			tags:        map[string][]mo.Reference{zone1Tag: {host}, regionTag: {dc}},
			zoneToHosts: map[cloudprovider.Zone][]vmwaretypes.ManagedObjectReference{zone1: {host}},
		},
		{
			name:        "Zone on cluster Region on datacenter",
			tags:        map[string][]mo.Reference{zone1Tag: {cluster}, regionTag: {dc}},
			zoneToHosts: map[cloudprovider.Zone][]vmwaretypes.ManagedObjectReference{zone1: clusterHosts},
		},
		{
			name:        "Zone on cluster and override on host",
			tags:        map[string][]mo.Reference{zone2Tag: {cluster}, zone1Tag: {host}, regionTag: {dc}},
			zoneToHosts: map[cloudprovider.Zone][]vmwaretypes.ManagedObjectReference{zone1: {host}, zone2: remainingHostsInCluster},
		},
		{
			name:        "Zone and Region on datacenter",
			tags:        map[string][]mo.Reference{zone1Tag: {dc}, regionTag: {dc}},
			zoneToHosts: map[cloudprovider.Zone][]vmwaretypes.ManagedObjectReference{zone1: allVcHosts},
		},
	}

	for _, testcase := range testcases {
		t.Run(testcase.name, func(t *testing.T) {
			// apply tags to datacenter/cluster/host as per this testcase
			for tagId, objects := range testcase.tags {
				for _, object := range objects {
					if err := m.AttachTag(ctx, tagId, object); err != nil {
						t.Fatal(err)
					}
				}
			}

			// run the test
			zoneToHosts, err := vs.GetZoneToHosts(ctx, vsi)
			if err != nil {
				t.Errorf("unexpected error when calling GetZoneToHosts: %q", err)
			}

			// do not depend on the sort order of hosts in result
			sortHostsMap(zoneToHosts)
			if !reflect.DeepEqual(zoneToHosts, testcase.zoneToHosts) {
				t.Logf("expected result: %+v", testcase.zoneToHosts)
				t.Logf("actual result: %+v", zoneToHosts)
				t.Error("unexpected result from GetZoneToHosts")
			}

			// clean up tags applied on datacenter/cluster/host for this testcase
			for tagId, objects := range testcase.tags {
				for _, object := range objects {
					if err = m.DetachTag(ctx, tagId, object); err != nil {
						t.Fatal(err)
					}
				}
			}
		})
	}
}

func sortHostsMap(zoneToHosts map[cloudprovider.Zone][]vmwaretypes.ManagedObjectReference) {
	for _, hosts := range zoneToHosts {
		sortHosts(hosts)
	}
}

func sortHosts(hosts []vmwaretypes.ManagedObjectReference) {
	sort.Slice(hosts, func(i, j int) bool {
		return hosts[i].Value < hosts[j].Value
	})
}

func TestInstances(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	vs, err := newControllerNode(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}

	i, ok := vs.Instances()
	if !ok {
		t.Fatalf("Instances() returned false")
	}

	nodeName, err := vs.CurrentNodeName(context.TODO(), "")
	if err != nil {
		t.Fatalf("CurrentNodeName() failed: %s", err)
	}

	nonExistingVM := types.NodeName(rand.String(15))
	instanceID, err := i.InstanceID(context.TODO(), nodeName)
	if err != nil {
		t.Fatalf("Instances.InstanceID(%s) failed: %s", nodeName, err)
	}
	t.Logf("Found InstanceID(%s) = %s\n", nodeName, instanceID)

	_, err = i.InstanceID(context.TODO(), nonExistingVM)
	if err == cloudprovider.InstanceNotFound {
		t.Logf("VM %s was not found as expected\n", nonExistingVM)
	} else if err == nil {
		t.Fatalf("Instances.InstanceID did not fail as expected, VM %s was found", nonExistingVM)
	} else {
		t.Fatalf("Instances.InstanceID did not fail as expected, err: %v", err)
	}

	addrs, err := i.NodeAddresses(context.TODO(), nodeName)
	if err != nil {
		t.Fatalf("Instances.NodeAddresses(%s) failed: %s", nodeName, err)
	}
	found := false
	for _, addr := range addrs {
		if addr.Type == v1.NodeHostName {
			found = true
		}
	}
	if found == false {
		t.Fatalf("NodeAddresses does not report hostname, %s %s", nodeName, addrs)
	}
	t.Logf("Found NodeAddresses(%s) = %s\n", nodeName, addrs)
}

func TestVolumes(t *testing.T) {
	cfg, ok := configFromEnv()
	if !ok {
		t.Skipf("No config found in environment")
	}

	vs, err := newControllerNode(cfg)
	if err != nil {
		t.Fatalf("Failed to construct/authenticate vSphere: %s", err)
	}

	nodeName, err := vs.CurrentNodeName(context.TODO(), "")
	if err != nil {
		t.Fatalf("CurrentNodeName() failed: %s", err)
	}

	volumeOptions := &vclib.VolumeOptions{
		CapacityKB: 1 * 1024 * 1024,
		Tags:       nil,
		Name:       "kubernetes-test-volume-" + rand.String(10),
		DiskFormat: "thin"}

	volPath, err := vs.CreateVolume(volumeOptions)
	if err != nil {
		t.Fatalf("Cannot create a new VMDK volume: %v", err)
	}

	_, err = vs.AttachDisk(volPath, "", "")
	if err != nil {
		t.Fatalf("Cannot attach volume(%s) to VM(%s): %v", volPath, nodeName, err)
	}

	err = vs.DetachDisk(volPath, "")
	if err != nil {
		t.Fatalf("Cannot detach disk(%s) from VM(%s): %v", volPath, nodeName, err)
	}

	// todo: Deleting a volume after detach currently not working through API or UI (vSphere)
	// err = vs.DeleteVolume(volPath)
	// if err != nil {
	// 	t.Fatalf("Cannot delete VMDK volume %s: %v", volPath, err)
	// }
}

func TestSecretVSphereConfig(t *testing.T) {
	var vs *VSphere
	var (
		username = "user"
		password = "password"
	)
	var testcases = []struct {
		testName                 string
		conf                     string
		expectedIsSecretProvided bool
		expectedUsername         string
		expectedPassword         string
		expectedError            error
		expectedThumbprints      map[string]string
	}{
		{
			testName: "Username and password with old configuration",
			conf: `[Global]
			server = 0.0.0.0
			user = user
			password = password
			datacenter = us-west
			working-dir = kubernetes
			`,
			expectedUsername: username,
			expectedPassword: password,
			expectedError:    nil,
		},
		{
			testName: "SecretName and SecretNamespace in old configuration",
			conf: `[Global]
			server = 0.0.0.0
			datacenter = us-west
			secret-name = "vccreds"
			secret-namespace = "kube-system"
			working-dir = kubernetes
			`,
			expectedIsSecretProvided: true,
			expectedError:            nil,
		},
		{
			testName: "SecretName and SecretNamespace with Username and Password in old configuration",
			conf: `[Global]
			server = 0.0.0.0
			user = user
			password = password
			datacenter = us-west
			secret-name = "vccreds"
			secret-namespace = "kube-system"
			working-dir = kubernetes
			`,
			expectedIsSecretProvided: true,
			expectedError:            nil,
		},
		{
			testName: "SecretName and SecretNamespace with Username missing in old configuration",
			conf: `[Global]
			server = 0.0.0.0
			password = password
			datacenter = us-west
			secret-name = "vccreds"
			secret-namespace = "kube-system"
			working-dir = kubernetes
			`,
			expectedIsSecretProvided: true,
			expectedError:            nil,
		},
		{
			testName: "SecretNamespace missing with Username and Password in old configuration",
			conf: `[Global]
			server = 0.0.0.0
			user = user
			password = password
			datacenter = us-west
			secret-name = "vccreds"
			working-dir = kubernetes
			`,
			expectedUsername: username,
			expectedPassword: password,
			expectedError:    nil,
		},
		{
			testName: "SecretNamespace and Username missing in old configuration",
			conf: `[Global]
			server = 0.0.0.0
			password = password
			datacenter = us-west
			secret-name = "vccreds"
			working-dir = kubernetes
			`,
			expectedError: ErrUsernameMissing,
		},
		{
			testName: "SecretNamespace and Password missing in old configuration",
			conf: `[Global]
			server = 0.0.0.0
			user = user
			datacenter = us-west
			secret-name = "vccreds"
			working-dir = kubernetes
			`,
			expectedError: ErrPasswordMissing,
		},
		{
			testName: "SecretNamespace, Username and Password missing in old configuration",
			conf: `[Global]
			server = 0.0.0.0
			datacenter = us-west
			secret-name = "vccreds"
			working-dir = kubernetes
			`,
			expectedError: ErrUsernameMissing,
		},
		{
			testName: "Username and password with new configuration but username and password in global section",
			conf: `[Global]
			user = user
			password = password
			datacenter = us-west
			[VirtualCenter "0.0.0.0"]
			[Workspace]
			server = 0.0.0.0
			datacenter = us-west
			folder = kubernetes
			`,
			expectedUsername: username,
			expectedPassword: password,
			expectedError:    nil,
		},
		{
			testName: "Username and password with new configuration, username and password in virtualcenter section",
			conf: `[Global]
			server = 0.0.0.0
			port = 443
			insecure-flag = true
			datacenter = us-west
			[VirtualCenter "0.0.0.0"]
			user = user
			password = password
			[Workspace]
			server = 0.0.0.0
			datacenter = us-west
			folder = kubernetes
			`,
			expectedUsername: username,
			expectedPassword: password,
			expectedError:    nil,
		},
		{
			testName: "SecretName and SecretNamespace with new configuration",
			conf: `[Global]
			server = 0.0.0.0
			secret-name = "vccreds"
			secret-namespace = "kube-system"
			datacenter = us-west
			[VirtualCenter "0.0.0.0"]
			[Workspace]
			server = 0.0.0.0
			datacenter = us-west
			folder = kubernetes
			`,
			expectedIsSecretProvided: true,
			expectedError:            nil,
		},
		{
			testName: "SecretName and SecretNamespace with Username missing in new configuration",
			conf: `[Global]
			server = 0.0.0.0
			port = 443
			insecure-flag = true
			datacenter = us-west
			secret-name = "vccreds"
			secret-namespace = "kube-system"
			[VirtualCenter "0.0.0.0"]
			password = password
			[Workspace]
			server = 0.0.0.0
			datacenter = us-west
			folder = kubernetes
			`,
			expectedIsSecretProvided: true,
			expectedError:            nil,
		},
		{
			testName: "virtual centers with a thumbprint",
			conf: `[Global]
			server = global
			user = user
			password = password
			datacenter = us-west
			thumbprint = "thumbprint:global"
			working-dir = kubernetes
			`,
			expectedUsername: username,
			expectedPassword: password,
			expectedError:    nil,
			expectedThumbprints: map[string]string{
				"global": "thumbprint:global",
			},
		},
		{
			testName: "Multiple virtual centers with different thumbprints",
			conf: `[Global]
			user = user
			password = password
			datacenter = us-west
			[VirtualCenter "0.0.0.0"]
			thumbprint = thumbprint:0
			[VirtualCenter "no_thumbprint"]
			[VirtualCenter "1.1.1.1"]
			thumbprint = thumbprint:1
			[Workspace]
			server = 0.0.0.0
			datacenter = us-west
			folder = kubernetes
			`,
			expectedUsername: username,
			expectedPassword: password,
			expectedError:    nil,
			expectedThumbprints: map[string]string{
				"0.0.0.0": "thumbprint:0",
				"1.1.1.1": "thumbprint:1",
			},
		},
		{
			testName: "Multiple virtual centers use the global CA cert",
			conf: `[Global]
			user = user
			password = password
			datacenter = us-west
			ca-file = /some/path/to/my/trusted/ca.pem
			[VirtualCenter "0.0.0.0"]
			user = user
			password = password
			[VirtualCenter "1.1.1.1"]
			user = user
			password = password
			[Workspace]
			server = 0.0.0.0
			datacenter = us-west
			folder = kubernetes
			`,
			expectedUsername: username,
			expectedPassword: password,
			expectedError:    nil,
		},
	}

	for _, testcase := range testcases {
		t.Logf("Executing Testcase: %s", testcase.testName)
		cfg, err := readConfig(strings.NewReader(testcase.conf))
		if err != nil {
			t.Fatalf("Should succeed when a valid config is provided: %s", err)
		}
		vs, err = buildVSphereFromConfig(cfg)
		if err != testcase.expectedError {
			t.Fatalf("Should succeed when a valid config is provided: %s", err)
		}
		if err != nil {
			continue
		}
		if vs.isSecretInfoProvided != testcase.expectedIsSecretProvided {
			t.Fatalf("SecretName and SecretNamespace was expected in config %s. error: %s",
				testcase.conf, err)
		}
		if !testcase.expectedIsSecretProvided {
			for _, vsInstance := range vs.vsphereInstanceMap {
				if vsInstance.conn.Username != testcase.expectedUsername {
					t.Fatalf("Expected username %s doesn't match actual username %s in config %s. error: %s",
						testcase.expectedUsername, vsInstance.conn.Username, testcase.conf, err)
				}
				if vsInstance.conn.Password != testcase.expectedPassword {
					t.Fatalf("Expected password %s doesn't match actual password %s in config %s. error: %s",
						testcase.expectedPassword, vsInstance.conn.Password, testcase.conf, err)
				}
			}
		}
		// Check, if all the expected thumbprints are configured
		for instanceName, expectedThumbprint := range testcase.expectedThumbprints {
			instanceConfig, ok := vs.vsphereInstanceMap[instanceName]
			if !ok {
				t.Fatalf("Could not find configuration for instance %s", instanceName)
			}
			if actualThumbprint := instanceConfig.conn.Thumbprint; actualThumbprint != expectedThumbprint {
				t.Fatalf(
					"Expected thumbprint for instance '%s' to be '%s', got '%s'",
					instanceName, expectedThumbprint, actualThumbprint,
				)
			}
		}
		// Check, if all all connections are configured with the global CA certificate
		if expectedCaPath := cfg.Global.CAFile; expectedCaPath != "" {
			for name, instance := range vs.vsphereInstanceMap {
				if actualCaPath := instance.conn.CACert; actualCaPath != expectedCaPath {
					t.Fatalf(
						"Expected CA certificate path for instance '%s' to be the globally configured one ('%s'), got '%s'",
						name, expectedCaPath, actualCaPath,
					)
				}
			}
		}
	}
}
