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

package copycerts

import (
	"context"
	"encoding/hex"
	"io/ioutil"
	"os"
	"path"
	"regexp"
	"testing"

	"github.com/lithammer/dedent"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	fakeclient "k8s.io/client-go/kubernetes/fake"
	certutil "k8s.io/client-go/util/cert"
	keyutil "k8s.io/client-go/util/keyutil"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	cryptoutil "k8s.io/kubernetes/cmd/kubeadm/app/util/crypto"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func TestGetDataFromInitConfig(t *testing.T) {
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

	certs := certsToTransfer(cfg)
	for name, path := range certs {
		if err := ioutil.WriteFile(path, certData, 0644); err != nil {
			t.Fatalf(dedent.Dedent("failed to write cert: %s\nfatal error: %v"), name, err)
		}
	}

	secretData, err := getDataFromDisk(cfg, decodedKey)
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
			t.Fatalf(dedent.Dedent("can't decrypt cert: %s\nfatal error: %v"), name, err)
		}
	}
}

func TestCertsToTransfer(t *testing.T) {
	localEtcdCfg := &kubeadmapi.InitConfiguration{}
	externalEtcdCfg := &kubeadmapi.InitConfiguration{}
	externalEtcdCfg.Etcd = kubeadmapi.Etcd{}
	externalEtcdCfg.Etcd.External = &kubeadmapi.ExternalEtcd{}

	commonExpectedCerts := []string{
		kubeadmconstants.CACertName,
		kubeadmconstants.CAKeyName,
		kubeadmconstants.FrontProxyCACertName,
		kubeadmconstants.FrontProxyCAKeyName,
		kubeadmconstants.ServiceAccountPublicKeyName,
		kubeadmconstants.ServiceAccountPrivateKeyName,
	}

	tests := map[string]struct {
		config        *kubeadmapi.InitConfiguration
		expectedCerts []string
	}{
		"local etcd": {
			config: localEtcdCfg,
			expectedCerts: append(
				[]string{kubeadmconstants.EtcdCACertName, kubeadmconstants.EtcdCAKeyName},
				commonExpectedCerts...,
			),
		},
		"external etcd": {
			config: externalEtcdCfg,
			expectedCerts: append(
				[]string{externalEtcdCA, externalEtcdCert, externalEtcdKey},
				commonExpectedCerts...,
			),
		},
	}

	for name, test := range tests {
		t.Run(name, func(t2 *testing.T) {
			certList := certsToTransfer(test.config)
			for _, cert := range test.expectedCerts {
				if _, found := certList[cert]; !found {
					t2.Fatalf(dedent.Dedent("failed to get list of certs to upload\ncert %s not found"), cert)
				}
			}
		})
	}
}

func TestCertOrKeyNameToSecretName(t *testing.T) {
	tests := []struct {
		keyName            string
		expectedSecretName string
	}{
		{
			keyName:            "apiserver-kubelet-client.crt",
			expectedSecretName: "apiserver-kubelet-client.crt",
		},
		{
			keyName:            "etcd/ca.crt",
			expectedSecretName: "etcd-ca.crt",
		},
		{
			keyName:            "etcd/healthcheck-client.crt",
			expectedSecretName: "etcd-healthcheck-client.crt",
		},
	}

	for _, tc := range tests {
		secretName := certOrKeyNameToSecretName(tc.keyName)
		if secretName != tc.expectedSecretName {
			t.Fatalf("secret name %s didn't match expected name %s", secretName, tc.expectedSecretName)
		}
	}
}

func TestUploadCerts(t *testing.T) {
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)

	secretKey, err := CreateCertificateKey()
	if err != nil {
		t.Fatalf("could not create certificate key: %v", err)
	}

	initConfiguration := testutil.GetDefaultInternalConfig(t)
	initConfiguration.ClusterConfiguration.CertificatesDir = tmpdir

	if err := certs.CreatePKIAssets(initConfiguration); err != nil {
		t.Fatalf("error creating PKI assets: %v", err)
	}

	cs := fakeclient.NewSimpleClientset()
	if err := UploadCerts(cs, initConfiguration, secretKey); err != nil {
		t.Fatalf("error uploading certs: %v", err)
	}
	rawSecretKey, err := hex.DecodeString(secretKey)
	if err != nil {
		t.Fatalf("error decoding key: %v", err)
	}
	secretMap, err := cs.CoreV1().Secrets(metav1.NamespaceSystem).Get(context.TODO(), kubeadmconstants.KubeadmCertsSecret, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("could not fetch secret: %v", err)
	}
	for certName, certPath := range certsToTransfer(initConfiguration) {
		secretCertData, err := cryptoutil.DecryptBytes(secretMap.Data[certOrKeyNameToSecretName(certName)], rawSecretKey)
		if err != nil {
			t.Fatalf("error decrypting secret data: %v", err)
		}
		diskCertData, err := ioutil.ReadFile(certPath)
		if err != nil {
			t.Fatalf("error reading certificate from disk: %v", err)
		}
		// Check that the encrypted contents on the secret match the contents on disk, and that all
		// the expected certificates are in the secret
		if string(secretCertData) != string(diskCertData) {
			t.Fatalf("cert %s does not have the expected contents. contents: %q; expected contents: %q", certName, string(secretCertData), string(diskCertData))
		}
	}
}

func TestDownloadCerts(t *testing.T) {
	secretKey, err := CreateCertificateKey()
	if err != nil {
		t.Fatalf("could not create certificate key: %v", err)
	}

	// Temporary directory where certificates will be generated
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)
	initConfiguration := testutil.GetDefaultInternalConfig(t)
	initConfiguration.ClusterConfiguration.CertificatesDir = tmpdir

	// Temporary directory where certificates will be downloaded to
	targetTmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(targetTmpdir)
	initForDownloadConfiguration := testutil.GetDefaultInternalConfig(t)
	initForDownloadConfiguration.ClusterConfiguration.CertificatesDir = targetTmpdir

	if err := certs.CreatePKIAssets(initConfiguration); err != nil {
		t.Fatalf("error creating PKI assets: %v", err)
	}

	kubeadmCertsSecret := createKubeadmCertsSecret(t, initConfiguration, secretKey)
	cs := fakeclient.NewSimpleClientset(kubeadmCertsSecret)
	if err := DownloadCerts(cs, initForDownloadConfiguration, secretKey); err != nil {
		t.Fatalf("error downloading certs: %v", err)
	}

	const keyFileMode = 0600
	const certFileMode = 0644

	for certName, certPath := range certsToTransfer(initForDownloadConfiguration) {
		diskCertData, err := ioutil.ReadFile(certPath)
		if err != nil {
			t.Errorf("error reading certificate from disk: %v", err)
		}
		// Check that the written files are either certificates or keys, and that they have
		// the expected permissions
		if _, err := keyutil.ParsePublicKeysPEM(diskCertData); err == nil {
			if stat, err := os.Stat(certPath); err == nil {
				if stat.Mode() != keyFileMode {
					t.Errorf("key %q should have mode %#o, has %#o", certName, keyFileMode, stat.Mode())
				}
			} else {
				t.Errorf("could not stat key %q: %v", certName, err)
			}
		} else if _, err := certutil.ParseCertsPEM(diskCertData); err == nil {
			if stat, err := os.Stat(certPath); err == nil {
				if stat.Mode() != certFileMode {
					t.Errorf("cert %q should have mode %#o, has %#o", certName, certFileMode, stat.Mode())
				}
			} else {
				t.Errorf("could not stat cert %q: %v", certName, err)
			}
		} else {
			t.Errorf("secret %q was not identified as a cert or as a key", certName)
		}
	}
}

func createKubeadmCertsSecret(t *testing.T, cfg *kubeadmapi.InitConfiguration, secretKey string) *v1.Secret {
	decodedKey, err := hex.DecodeString(secretKey)
	if err != nil {
		t.Fatalf("error decoding key: %v", err)
	}
	secretData, err := getDataFromDisk(cfg, decodedKey)
	if err != nil {
		t.Fatalf("error creating secret data: %v", err)
	}
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.KubeadmCertsSecret,
			Namespace: metav1.NamespaceSystem,
		},
		Data: secretData,
	}
}
