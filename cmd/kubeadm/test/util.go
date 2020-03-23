/*
Copyright 2017 The Kubernetes Authors.

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

package test

import (
	"fmt"
	"html/template"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/lithammer/dedent"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1beta2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta2"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certtestutil "k8s.io/kubernetes/cmd/kubeadm/app/util/certs"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
)

// SetupTempDir is a utility function for kubeadm testing, that creates a temporary directory
// NB. it is up to the caller to cleanup the folder at the end of the test with defer os.RemoveAll(tmpdir)
func SetupTempDir(t *testing.T) string {
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}

	return tmpdir
}

// SetupInitConfigurationFile is a utility function for kubeadm testing that writes a InitConfiguration file
// into /config subfolder of a given temporary directory.
// The function returns the path of the created InitConfiguration file.
func SetupInitConfigurationFile(t *testing.T, tmpdir string, cfg *kubeadmapi.InitConfiguration) string {

	cfgPath := filepath.Join(tmpdir, "config/masterconfig.yaml")
	if err := os.MkdirAll(filepath.Dir(cfgPath), os.FileMode(0755)); err != nil {
		t.Fatalf("Couldn't create cfgDir")
	}

	cfgTemplate := template.Must(template.New("init").Parse(dedent.Dedent(fmt.Sprintf(`
		apiVersion: kubeadm.k8s.io/v1beta2
		kind: InitConfiguration
		apiEndpoint:
		  advertiseAddress: {{.LocalAPIEndpoint.AdvertiseAddress}}
		  bindPort: {{.LocalAPIEndpoint.BindPort}}
		nodeRegistration:
		  name: {{.NodeRegistration.Name}}
		---
		apiVersion: kubeadm.k8s.io/v1beta2
		kind: ClusterConfiguration
		certificatesDir: {{.CertificatesDir}}
		kubernetesVersion: %s
		`, kubeadmconstants.MinimumControlPlaneVersion))))

	f, err := os.Create(cfgPath)
	if err != nil {
		t.Fatalf("error creating masterconfig file %s: %v", cfgPath, err)
	}

	err = cfgTemplate.Execute(f, cfg)
	if err != nil {
		t.Fatalf("error generating masterconfig file %s: %v", cfgPath, err)
	}
	f.Close()

	return cfgPath
}

// SetupEmptyFiles is a utility function for kubeadm testing that creates one or more empty files (touch)
func SetupEmptyFiles(t *testing.T, tmpdir string, fileNames ...string) {
	for _, fileName := range fileNames {
		newFile, err := os.Create(filepath.Join(tmpdir, fileName))
		if err != nil {
			t.Fatalf("Error creating file %s in %s: %v", fileName, tmpdir, err)
		}
		newFile.Close()
	}
}

// SetupPkiDirWithCertificateAuthority is a utility function for kubeadm testing that creates a
// CertificateAuthority cert/key pair into /pki subfolder of a given temporary directory.
// The function returns the path of the created pki.
func SetupPkiDirWithCertificateAuthority(t *testing.T, tmpdir string) string {
	caCert, caKey := certtestutil.SetupCertificateAuthority(t)

	certDir := filepath.Join(tmpdir, "pki")
	if err := pkiutil.WriteCertAndKey(certDir, kubeadmconstants.CACertAndKeyBaseName, caCert, caKey); err != nil {
		t.Fatalf("failure while saving CA certificate and key: %v", err)
	}

	return certDir
}

// AssertFilesCount is a utility function for kubeadm testing that asserts if the given folder contains
// count files.
func AssertFilesCount(t *testing.T, dirName string, count int) {
	files, err := ioutil.ReadDir(dirName)
	if err != nil {
		t.Fatalf("Couldn't read files from tmpdir: %s", err)
	}

	countFiles := 0
	for _, f := range files {
		if !f.IsDir() {
			countFiles++
		}
	}

	if countFiles != count {
		t.Errorf("dir does contains %d, %d expected", len(files), count)
		for _, f := range files {
			t.Error(f.Name())
		}
	}
}

// AssertFileExists is a utility function for kubeadm testing that asserts if the given folder contains
// the given files.
func AssertFileExists(t *testing.T, dirName string, fileNames ...string) {
	for _, fileName := range fileNames {
		path := filepath.Join(dirName, fileName)

		if _, err := os.Stat(path); os.IsNotExist(err) {
			t.Errorf("file %s does not exist", fileName)
		}
	}
}

// AssertError checks that the provided error matches the expected output
func AssertError(t *testing.T, err error, expected string) {
	if err == nil {
		t.Errorf("no error was found, but '%s' was expected", expected)
		return
	}
	if err.Error() != expected {
		t.Errorf("error '%s' does not match expected error: '%s'", err.Error(), expected)
	}
}

// GetDefaultInternalConfig returns a defaulted kubeadmapi.InitConfiguration
func GetDefaultInternalConfig(t *testing.T) *kubeadmapi.InitConfiguration {
	internalcfg, err := configutil.DefaultedInitConfiguration(&kubeadmapiv1beta2.InitConfiguration{}, &kubeadmapiv1beta2.ClusterConfiguration{})
	if err != nil {
		t.Fatalf("unexpected error getting default config: %v", err)
	}

	return internalcfg
}
