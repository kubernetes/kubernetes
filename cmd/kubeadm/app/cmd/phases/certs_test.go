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

package phases

import (
	"fmt"
	"html/template"
	"io/ioutil"
	"os"
	"path"
	"testing"

	"github.com/renstrom/dedent"
	"github.com/spf13/cobra"

	// required for triggering api machinery startup when running unit tests
	_ "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/install"

	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/pkiutil"
)

func TestSubCmdCertsCreateFiles(t *testing.T) {

	subCmds := newSubCmdCerts()

	var tests = []struct {
		subCmds       []string
		expectedFiles []string
	}{
		{
			subCmds: []string{"all"},
			expectedFiles: []string{
				kubeadmconstants.CACertName, kubeadmconstants.CAKeyName,
				kubeadmconstants.APIServerCertName, kubeadmconstants.APIServerKeyName,
				kubeadmconstants.APIServerKubeletClientCertName, kubeadmconstants.APIServerKubeletClientKeyName,
				kubeadmconstants.ServiceAccountPrivateKeyName, kubeadmconstants.ServiceAccountPublicKeyName,
				kubeadmconstants.FrontProxyCACertName, kubeadmconstants.FrontProxyCAKeyName,
				kubeadmconstants.FrontProxyClientCertName, kubeadmconstants.FrontProxyClientKeyName,
			},
		},
		{
			subCmds:       []string{"ca"},
			expectedFiles: []string{kubeadmconstants.CACertName, kubeadmconstants.CAKeyName},
		},
		{
			subCmds:       []string{"ca", "apiserver"},
			expectedFiles: []string{kubeadmconstants.CACertName, kubeadmconstants.CAKeyName, kubeadmconstants.APIServerCertName, kubeadmconstants.APIServerKeyName},
		},
		{
			subCmds:       []string{"ca", "apiserver-kubelet-client"},
			expectedFiles: []string{kubeadmconstants.CACertName, kubeadmconstants.CAKeyName, kubeadmconstants.APIServerKubeletClientCertName, kubeadmconstants.APIServerKubeletClientKeyName},
		},
		{
			subCmds:       []string{"sa"},
			expectedFiles: []string{kubeadmconstants.ServiceAccountPrivateKeyName, kubeadmconstants.ServiceAccountPublicKeyName},
		},
		{
			subCmds:       []string{"front-proxy-ca"},
			expectedFiles: []string{kubeadmconstants.FrontProxyCACertName, kubeadmconstants.FrontProxyCAKeyName},
		},
		{
			subCmds:       []string{"front-proxy-ca", "front-proxy-client"},
			expectedFiles: []string{kubeadmconstants.FrontProxyCACertName, kubeadmconstants.FrontProxyCAKeyName, kubeadmconstants.FrontProxyClientCertName, kubeadmconstants.FrontProxyClientKeyName},
		},
	}

	for _, test := range tests {
		// Temporary folder for the test case
		tmpdir, err := ioutil.TempDir("", "")
		if err != nil {
			t.Fatalf("Couldn't create tmpdir")
		}
		defer os.RemoveAll(tmpdir)

		// executes given sub commands
		for _, subCmdName := range test.subCmds {
			subCmd := getSubCmd(t, subCmdName, subCmds)
			subCmd.SetArgs([]string{fmt.Sprintf("--cert-dir=%s", tmpdir)})
			if err := subCmd.Execute(); err != nil {
				t.Fatalf("Could not execute subcommand: %s", subCmdName)
			}
		}

		// verify expected files are there
		assertFilesCount(t, tmpdir, len(test.expectedFiles))
		for _, file := range test.expectedFiles {
			assertFileExists(t, tmpdir, file)
		}
	}
}

func TestSubCmdApiServerFlags(t *testing.T) {

	subCmds := newSubCmdCerts()

	// Temporary folder for the test case
	tmpdir, err := ioutil.TempDir("", "")
	if err != nil {
		t.Fatalf("Couldn't create tmpdir")
	}
	defer os.RemoveAll(tmpdir)

	// creates ca cert
	subCmd := getSubCmd(t, "ca", subCmds)
	subCmd.SetArgs([]string{fmt.Sprintf("--cert-dir=%s", tmpdir)})
	if err := subCmd.Execute(); err != nil {
		t.Fatalf("Could not execute subcommand ca")
	}

	// creates apiserver cert
	subCmd = getSubCmd(t, "apiserver", subCmds)
	subCmd.SetArgs([]string{
		fmt.Sprintf("--cert-dir=%s", tmpdir),
		"--apiserver-cert-extra-sans=foo,boo",
		"--service-cidr=10.0.0.0/24",
		"--service-dns-domain=mycluster.local",
		"--apiserver-advertise-address=1.2.3.4",
	})
	if err := subCmd.Execute(); err != nil {
		t.Fatalf("Could not execute subcommand apiserver")
	}

	APIserverCert, err := pkiutil.TryLoadCertFromDisk(tmpdir, kubeadmconstants.APIServerCertAndKeyBaseName)
	if err != nil {
		t.Fatalf("Error loading API server certificate: %v", err)
	}

	hostname, err := os.Hostname()
	if err != nil {
		t.Errorf("couldn't get the hostname: %v", err)
	}
	for i, name := range []string{hostname, "kubernetes", "kubernetes.default", "kubernetes.default.svc", "kubernetes.default.svc.mycluster.local"} {
		if APIserverCert.DNSNames[i] != name {
			t.Errorf("APIserverCert.DNSNames[%d] is %s instead of %s", i, APIserverCert.DNSNames[i], name)
		}
	}
	for i, ip := range []string{"10.0.0.1", "1.2.3.4"} {
		if APIserverCert.IPAddresses[i].String() != ip {
			t.Errorf("APIserverCert.IPAddresses[%d] is %s instead of %s", i, APIserverCert.IPAddresses[i], ip)
		}
	}
}

func TestSubCmdReadsConfig(t *testing.T) {

	subCmds := newSubCmdCerts()

	var tests = []struct {
		subCmds           []string
		expectedFileCount int
	}{
		{
			subCmds:           []string{"sa"},
			expectedFileCount: 2,
		},
		{
			subCmds:           []string{"front-proxy-ca", "front-proxy-client"},
			expectedFileCount: 4,
		},
		{
			subCmds:           []string{"ca", "apiserver", "apiserver-kubelet-client"},
			expectedFileCount: 6,
		},
		{
			subCmds:           []string{"all"},
			expectedFileCount: 12,
		},
	}

	for _, test := range tests {
		// Temporary folder for the test case
		tmpdir, err := ioutil.TempDir("", "")
		if err != nil {
			t.Fatalf("Couldn't create tmpdir")
		}
		defer os.RemoveAll(tmpdir)

		configPath := saveDummyCfg(t, tmpdir)

		// executes given sub commands
		for _, subCmdName := range test.subCmds {
			subCmd := getSubCmd(t, subCmdName, subCmds)
			subCmd.SetArgs([]string{fmt.Sprintf("--config=%s", configPath)})
			if err := subCmd.Execute(); err != nil {
				t.Fatalf("Could not execute command: %s", subCmdName)
			}
		}

		// verify expected files are there
		// NB. test.expectedFileCount + 1 because in this test case the tempdir where key/certificates
		//     are saved contains also the dummy configuration file
		assertFilesCount(t, tmpdir, test.expectedFileCount+1)
	}
}

func getSubCmd(t *testing.T, name string, subCmds []*cobra.Command) *cobra.Command {
	for _, subCmd := range subCmds {
		if subCmd.Name() == name {
			return subCmd
		}
	}
	t.Fatalf("Unable to find sub command %s", name)

	return nil
}

func assertFilesCount(t *testing.T, dirName string, count int) {
	files, err := ioutil.ReadDir(dirName)
	if err != nil {
		t.Fatalf("Couldn't read files from tmpdir: %s", err)
	}

	if len(files) != count {
		t.Errorf("dir does contains %d, %d expected", len(files), count)
		for _, f := range files {
			t.Error(f.Name())
		}
	}
}

func assertFileExists(t *testing.T, dirName string, fileName string) {
	path := path.Join(dirName, fileName)
	if _, err := os.Stat(path); os.IsNotExist(err) {
		t.Errorf("file %s does not exist", fileName)
	}
}

func saveDummyCfg(t *testing.T, dirName string) string {

	path := path.Join(dirName, "dummyconfig.yaml")
	cfgTemplate := template.Must(template.New("init").Parse(dedent.Dedent(`
		apiVersion: kubeadm.k8s.io/v1alpha1
		kind: MasterConfiguration
		certificatesDir: {{.CertificatesDir}}
		`)))

	f, err := os.Create(path)
	if err != nil {
		t.Errorf("error creating dummyconfig file %s: %v", path, err)
	}

	templateData := struct {
		CertificatesDir string
	}{
		CertificatesDir: dirName,
	}

	err = cfgTemplate.Execute(f, templateData)
	if err != nil {
		t.Errorf("error generating dummyconfig file %s: %v", path, err)
	}
	f.Close()

	return path
}
