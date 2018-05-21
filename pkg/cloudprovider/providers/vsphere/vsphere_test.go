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
	"log"
	"os"
	"strconv"
	"strings"
	"testing"

	lookup "github.com/vmware/govmomi/lookup/simulator"
	"github.com/vmware/govmomi/simulator"
	"github.com/vmware/govmomi/simulator/vpx"
	sts "github.com/vmware/govmomi/sts/simulator"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/cloudprovider/providers/vsphere/vclib"
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
func configFromSim() (VSphereConfig, func()) {
	var cfg VSphereConfig
	model := simulator.VPX()

	err := model.Create()
	if err != nil {
		log.Fatal(err)
	}

	model.Service.TLS = new(tls.Config)
	s := model.Service.NewServer()

	// STS simulator
	path, handler := sts.New(s.URL, vpx.Setting)
	model.Service.ServeMux.Handle(path, handler)

	// Lookup Service simulator
	model.Service.RegisterSDK(lookup.New())

	cfg.Global.InsecureFlag = true
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

func TestZones(t *testing.T) {
	cfg := VSphereConfig{}
	cfg.Global.Datacenter = "myDatacenter"

	// Create vSphere configuration object
	vs := VSphere{
		cfg: &cfg,
	}

	_, ok := vs.Zones()
	if ok {
		t.Fatalf("Zones() returned true")
	}
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

	instanceID, err = i.InstanceID(context.TODO(), nonExistingVM)
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

	}
}
