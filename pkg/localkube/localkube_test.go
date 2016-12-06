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

package localkube

import (
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/kubernetes/localkube/tests"
)

var testIPs = []net.IP{net.ParseIP("1.2.3.4")}

func TestGenerateCerts(t *testing.T) {
	tempDir := tests.MakeTempDir()
	defer os.RemoveAll(tempDir)
	os.Mkdir(filepath.Join(tempDir, "certs"), 0777)

	_, ipRange, _ := net.ParseCIDR("10.0.0.0/24")
	lk := LocalkubeServer{
		LocalkubeDirectory:    tempDir,
		ServiceClusterIPRange: *ipRange,
	}

	if err := lk.GenerateCerts(); err != nil {
		t.Fatalf("Unexpected error generating certs: %s", err)
	}

	for _, f := range []string{"apiserver.crt", "apiserver.key"} {
		p := filepath.Join(tempDir, "certs", f)
		_, err := os.Stat(p)
		if os.IsNotExist(err) {
			t.Fatalf("Certificate not created: %s", p)
		}
	}
	_, err := lk.loadCert(filepath.Join(tempDir, "certs", "apiserver.crt"))
	if err != nil {
		t.Fatalf("Error parsing cert: %s", err)
	}
}

func TestShouldGenerateCertsNoFiles(t *testing.T) {
	lk := LocalkubeServer{LocalkubeDirectory: "baddir"}
	if !lk.shouldGenerateCerts(testIPs) {
		t.Fatalf("No certs exist, we should generate.")
	}
}

func TestShouldGenerateCertsOneFile(t *testing.T) {
	tempDir := tests.MakeTempDir()
	defer os.RemoveAll(tempDir)
	os.Mkdir(filepath.Join(tempDir, "certs"), 0777)
	ioutil.WriteFile(filepath.Join(tempDir, "certs", "apiserver.crt"), []byte(""), 0644)
	lk := LocalkubeServer{LocalkubeDirectory: tempDir}
	if !lk.shouldGenerateCerts(testIPs) {
		t.Fatalf("Not all certs exist, we should generate.")
	}
}

func TestShouldGenerateCertsBadFiles(t *testing.T) {
	tempDir := tests.MakeTempDir()
	defer os.RemoveAll(tempDir)
	os.Mkdir(filepath.Join(tempDir, "certs"), 0777)
	for _, f := range []string{"apiserver.crt", "apiserver.key"} {
		ioutil.WriteFile(filepath.Join(tempDir, "certs", f), []byte(""), 0644)
	}
	lk := LocalkubeServer{LocalkubeDirectory: tempDir}
	if !lk.shouldGenerateCerts(testIPs) {
		t.Fatalf("Certs are badly formatted, we should generate.")
	}
}

func TestShouldGenerateCertsMismatchedIP(t *testing.T) {
	tempDir := tests.MakeTempDir()
	defer os.RemoveAll(tempDir)
	os.Mkdir(filepath.Join(tempDir, "certs"), 0777)

	_, ipRange, _ := net.ParseCIDR("10.0.0.0/24")
	lk := LocalkubeServer{
		LocalkubeDirectory:    tempDir,
		ServiceClusterIPRange: *ipRange,
	}

	lk.GenerateCerts()

	if !lk.shouldGenerateCerts([]net.IP{net.ParseIP("4.3.2.1")}) {
		t.Fatalf("IPs don't match, we should generate.")
	}
}

func TestShouldNotGenerateCerts(t *testing.T) {
	tempDir := tests.MakeTempDir()
	defer os.RemoveAll(tempDir)
	os.Mkdir(filepath.Join(tempDir, "certs"), 0777)

	_, ipRange, _ := net.ParseCIDR("10.0.0.0/24")
	lk := LocalkubeServer{
		LocalkubeDirectory:    tempDir,
		ServiceClusterIPRange: *ipRange,
	}
	lk.GenerateCerts()
	ips, _ := lk.getAllIPs()
	if lk.shouldGenerateCerts(ips) {
		t.Fatalf("IPs match, we should not generate.")
	}
}
