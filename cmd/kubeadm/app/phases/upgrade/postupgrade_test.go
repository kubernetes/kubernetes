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

package upgrade

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func TestBackupAPIServerCertAndKey(t *testing.T) {
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)
	os.Chmod(tmpdir, 0766)

	certPath := filepath.Join(tmpdir, constants.APIServerCertName)
	certFile, err := os.OpenFile(certPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create cert file %s: %v", certPath, err)
	}
	defer certFile.Close()

	keyPath := filepath.Join(tmpdir, constants.APIServerKeyName)
	keyFile, err := os.OpenFile(keyPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create key file %s: %v", keyPath, err)
	}
	defer keyFile.Close()

	if err := backupAPIServerCertAndKey(tmpdir); err != nil {
		t.Fatalf("Failed to backup cert and key in dir %s: %v", tmpdir, err)
	}
}

func TestMoveFiles(t *testing.T) {
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)
	os.Chmod(tmpdir, 0766)

	certPath := filepath.Join(tmpdir, constants.APIServerCertName)
	certFile, err := os.OpenFile(certPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create cert file %s: %v", certPath, err)
	}
	defer certFile.Close()

	keyPath := filepath.Join(tmpdir, constants.APIServerKeyName)
	keyFile, err := os.OpenFile(keyPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create key file %s: %v", keyPath, err)
	}
	defer keyFile.Close()

	subDir := filepath.Join(tmpdir, "expired")
	if err := os.Mkdir(subDir, 0766); err != nil {
		t.Fatalf("Failed to create backup directory %s: %v", subDir, err)
	}

	filesToMove := map[string]string{
		filepath.Join(tmpdir, constants.APIServerCertName): filepath.Join(subDir, constants.APIServerCertName),
		filepath.Join(tmpdir, constants.APIServerKeyName):  filepath.Join(subDir, constants.APIServerKeyName),
	}

	if err := moveFiles(filesToMove); err != nil {
		t.Fatalf("Failed to move files %v: %v", filesToMove, err)
	}
}

func TestRollbackFiles(t *testing.T) {
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)
	os.Chmod(tmpdir, 0766)

	subDir := filepath.Join(tmpdir, "expired")
	if err := os.Mkdir(subDir, 0766); err != nil {
		t.Fatalf("Failed to create backup directory %s: %v", subDir, err)
	}

	certPath := filepath.Join(subDir, constants.APIServerCertName)
	certFile, err := os.OpenFile(certPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create cert file %s: %v", certPath, err)
	}
	defer certFile.Close()

	keyPath := filepath.Join(subDir, constants.APIServerKeyName)
	keyFile, err := os.OpenFile(keyPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create key file %s: %v", keyPath, err)
	}
	defer keyFile.Close()

	filesToRollBack := map[string]string{
		filepath.Join(subDir, constants.APIServerCertName): filepath.Join(tmpdir, constants.APIServerCertName),
		filepath.Join(subDir, constants.APIServerKeyName):  filepath.Join(tmpdir, constants.APIServerKeyName),
	}

	errString := "there are files need roll back"
	originalErr := errors.New(errString)
	err = rollbackFiles(filesToRollBack, originalErr)
	if err == nil {
		t.Fatalf("Expected error contains %q, got nil", errString)
	}
	if !strings.Contains(err.Error(), errString) {
		t.Fatalf("Expected error contains %q, got %v", errString, err)
	}
}

func TestShouldBackupAPIServerCertAndKey(t *testing.T) {
	cfg := &kubeadmapi.InitConfiguration{
		LocalAPIEndpoint: kubeadmapi.APIEndpoint{AdvertiseAddress: "1.2.3.4"},
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			Networking: kubeadmapi.Networking{ServiceSubnet: "10.96.0.0/12", DNSDomain: "cluster.local"},
		},
		NodeRegistration: kubeadmapi.NodeRegistrationOptions{Name: "test-node"},
	}

	for desc, test := range map[string]struct {
		adjustedExpiry time.Duration
		expected       bool
	}{
		"default: cert not older than 180 days doesn't needs to backup": {
			expected: false,
		},
		"cert older than 180 days need to backup": {
			adjustedExpiry: expiry + 100*time.Hour,
			expected:       true,
		},
	} {
		tmpdir := testutil.SetupTempDir(t)
		defer os.RemoveAll(tmpdir)
		cfg.CertificatesDir = tmpdir

		caCert, caKey, err := certsphase.KubeadmCertRootCA.CreateAsCA(cfg)
		if err != nil {
			t.Fatalf("failed creation of ca cert and key: %v", err)
		}
		caCert.NotBefore = caCert.NotBefore.Add(-test.adjustedExpiry).UTC()

		err = certsphase.KubeadmCertAPIServer.CreateFromCA(cfg, caCert, caKey)
		if err != nil {
			t.Fatalf("Test %s: failed creation of cert and key: %v", desc, err)
		}

		certAndKey := []string{filepath.Join(tmpdir, constants.APIServerCertName), filepath.Join(tmpdir, constants.APIServerKeyName)}
		for _, path := range certAndKey {
			if _, err := os.Stat(path); os.IsNotExist(err) {
				t.Fatalf("Test %s: %s not exist: %v", desc, path, err)
			}
		}

		shouldBackup, err := shouldBackupAPIServerCertAndKey(tmpdir)
		if err != nil {
			t.Fatalf("Test %s: failed to check shouldBackupAPIServerCertAndKey: %v", desc, err)
		}

		if shouldBackup != test.expected {
			t.Fatalf("Test %s: shouldBackupAPIServerCertAndKey expected %v, got %v", desc, test.expected, shouldBackup)
		}
	}
}
