/*
Copyright 2019 The Kubernetes Authors.

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

package uploadcerts

import (
	"encoding/hex"
	"io/ioutil"
	"os"
	"path"
	"regexp"
	"testing"

	"github.com/lithammer/dedent"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	cryptoutil "k8s.io/kubernetes/cmd/kubeadm/app/util/crypto"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func TestUploadCerts(t *testing.T) {
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

}

//teste cert name, teste cert can be decrypted
func TestGetSecretData(t *testing.T) {
	certData := []byte("cert-data")
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)
	cfg := &kubeadmapi.InitConfiguration{}
	cfg.CertificatesDir = tmpdir

	key, err := CreateCertificateKey()
	if err != nil {
		t.Fatalf(dedent.Dedent("failed to create key.\nfatal error: %v"), err)
	}
	decodedKey, err := hex.DecodeString(key)
	if err != nil {
		t.Fatalf(dedent.Dedent("failed to decode key.\nfatal error: %v"), err)
	}

	if err := os.Mkdir(path.Join(tmpdir, "etcd"), 0755); err != nil {
		t.Fatalf(dedent.Dedent("failed to create etcd cert dir.\nfatal error: %v"), err)
	}

	certs := certsToUpload(cfg)
	for name, path := range certs {
		if err := ioutil.WriteFile(path, certData, 0644); err != nil {
			t.Fatalf(dedent.Dedent("failed to write cert: %s\nfatal error: %v"), name, err)
		}
	}

	secretData, err := getSecretData(cfg, decodedKey)
	if err != nil {
		t.Fatalf("failed to get secret data. fatal error: %v", err)
	}

	re := regexp.MustCompile(`[-._a-zA-Z0-9]+`)
	for name, data := range secretData {
		if !re.MatchString(name) {
			t.Fatalf(dedent.Dedent("failed to validate secretData\n %s isn't a valid secret key"), name)
		}

		decryptedData, err := cryptoutil.DecryptBytes(data, decodedKey)
		if string(certData) != string(decryptedData) {
			t.Fatalf(dedent.Dedent("can't decript cert: %s\nfatal error: %v"), name, err)
		}
	}
}

func TestCertsToUpload(t *testing.T) {
	localEtcdCfg := &kubeadmapi.InitConfiguration{}
	externalEtcdCfg := &kubeadmapi.InitConfiguration{}
	externalEtcdCfg.Etcd = kubeadmapi.Etcd{}
	externalEtcdCfg.Etcd.External = &kubeadmapi.ExternalEtcd{}

	tests := map[string]struct {
		config        *kubeadmapi.InitConfiguration
		expectedCerts []string
	}{
		"local etcd": {
			config:        localEtcdCfg,
			expectedCerts: []string{kubeadmconstants.EtcdCACertName, kubeadmconstants.EtcdCAKeyName},
		},
		"external etcd": {
			config:        externalEtcdCfg,
			expectedCerts: []string{externalEtcdCA, externalEtcdCert, externalEtcdKey},
		},
	}

	for name, test := range tests {
		t.Run(name, func(t2 *testing.T) {
			certList := certsToUpload(test.config)
			for _, cert := range test.expectedCerts {
				if _, found := certList[cert]; !found {
					t2.Fatalf(dedent.Dedent("failed to get list of certs to upload\ncert %s not found"), cert)
				}
			}
		})
	}
}
