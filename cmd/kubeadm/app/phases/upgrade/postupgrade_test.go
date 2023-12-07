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
	"context"
	"os"
	"path/filepath"
	"reflect"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/pkg/errors"

	errorsutil "k8s.io/apimachinery/pkg/util/errors"
	clientset "k8s.io/client-go/kubernetes"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/componentconfigs"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/preflight"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

func TestMoveFiles(t *testing.T) {
	tmpdir := testutil.SetupTempDir(t)
	defer os.RemoveAll(tmpdir)
	os.Chmod(tmpdir, 0766)

	certPath := filepath.Join(tmpdir, constants.APIServerCertName)
	certFile, err := os.OpenFile(certPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create cert file %s: %v", certPath, err)
	}
	certFile.Close()

	keyPath := filepath.Join(tmpdir, constants.APIServerKeyName)
	keyFile, err := os.OpenFile(keyPath, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0666)
	if err != nil {
		t.Fatalf("Failed to create key file %s: %v", keyPath, err)
	}
	keyFile.Close()

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

func TestWriteKubeletConfigFiles(t *testing.T) {
	// exit early if the user doesn't have root permission as the test needs to create /etc/kubernetes directory
	// while the permission should be granted to the user.
	isPrivileged := preflight.IsPrivilegedUserCheck{}
	if _, err := isPrivileged.Check(); err != nil {
		return
	}
	testCases := []struct {
		name       string
		dryrun     bool
		patchesDir string
		errPattern string
		cfg        *kubeadmapi.InitConfiguration
	}{
		// Be careful that if the dryrun is set to false and the test is run on a live cluster, the kubelet config file might be overwritten.
		// However, you should be able to find the original config file in /etc/kubernetes/tmp/kubeadm-kubelet-configxxx folder.
		// The test haven't clean up the temporary file created under /etc/kubernetes/tmp/ as that could be accidentally delete other files in
		// that folder as well which might be unexpected.
		{
			name:   "write kubelet config file successfully",
			dryrun: true,
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ComponentConfigs: kubeadmapi.ComponentConfigMap{
						componentconfigs.KubeletGroup: &componentConfig{},
					},
				},
			},
		},
		{
			name:       "aggregate errs: no kubelet config file and cannot read config file",
			dryrun:     true,
			errPattern: missingKubeletConfig,
			cfg:        &kubeadmapi.InitConfiguration{},
		},
		{
			name:       "only one err: patch dir does not exist",
			dryrun:     true,
			patchesDir: "Bogus",
			errPattern: "could not list patch files for path \"Bogus\"",
			cfg: &kubeadmapi.InitConfiguration{
				ClusterConfiguration: kubeadmapi.ClusterConfiguration{
					ComponentConfigs: kubeadmapi.ComponentConfigMap{
						componentconfigs.KubeletGroup: &componentConfig{},
					},
				},
			},
		},
	}
	for _, tc := range testCases {
		err := WriteKubeletConfigFiles(tc.cfg, tc.patchesDir, tc.dryrun, os.Stdout)
		if err != nil && tc.errPattern != "" {
			if match, _ := regexp.MatchString(tc.errPattern, err.Error()); !match {
				t.Fatalf("Expected error contains %q, got %v", tc.errPattern, err.Error())
			}
		}
		if err == nil && len(tc.errPattern) != 0 {
			t.Fatalf("WriteKubeletConfigFiles didn't return error expected %s", tc.errPattern)
		}
	}
}

// Just some stub code, the code could be enriched when necessary.
type componentConfig struct {
	userSupplied bool
}

func (cc *componentConfig) DeepCopy() kubeadmapi.ComponentConfig {
	result := &componentConfig{}
	return result
}

func (cc *componentConfig) Marshal() ([]byte, error) {
	return nil, nil
}

func (cc *componentConfig) Unmarshal(docmap kubeadmapi.DocumentMap) error {
	return nil
}

func (cc *componentConfig) Get() interface{} {
	return &cc
}

func (cc *componentConfig) Set(cfg interface{}) {
}

func (cc *componentConfig) Default(_ *kubeadmapi.ClusterConfiguration, _ *kubeadmapi.APIEndpoint, _ *kubeadmapi.NodeRegistrationOptions) {
}

func (cc *componentConfig) Mutate() error {
	return nil
}

func (cc *componentConfig) IsUserSupplied() bool {
	return false
}
func (cc *componentConfig) SetUserSupplied(userSupplied bool) {
	cc.userSupplied = userSupplied
}

// moveFiles moves files from one directory to another.
func moveFiles(files map[string]string) error {
	filesToRecover := make(map[string]string, len(files))
	for from, to := range files {
		if err := os.Rename(from, to); err != nil {
			return rollbackFiles(filesToRecover, err)
		}
		filesToRecover[to] = from
	}
	return nil
}

// rollbackFiles moves the files back to the original directory.
func rollbackFiles(files map[string]string, originalErr error) error {
	errs := []error{originalErr}
	for from, to := range files {
		if err := os.Rename(from, to); err != nil {
			errs = append(errs, err)
		}
	}
	return errors.Errorf("couldn't move these files: %v. Got errors: %v", files, errorsutil.NewAggregate(errs))
}

// TODO: Remove this unit test during the 1.30 release cycle:
// https://github.com/kubernetes/kubeadm/issues/2414
func TestCreateSuperAdminKubeConfig(t *testing.T) {
	dir := testutil.SetupTempDir(t)
	defer os.RemoveAll(dir)

	cfg := testutil.GetDefaultInternalConfig(t)
	cfg.CertificatesDir = dir

	ca := certsphase.KubeadmCertRootCA()
	_, _, err := ca.CreateAsCA(cfg)
	if err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name                  string
		kubeConfigExist       bool
		expectRBACError       bool
		expectedError         bool
		expectKubeConfigError bool
	}{
		{
			name: "no error",
		},
		{
			name:            "no error, kubeconfig files already exist",
			kubeConfigExist: true,
		},
		{
			name:            "return RBAC error",
			expectRBACError: true,
			expectedError:   true,
		},
		{
			name:                  "return kubeconfig error",
			expectKubeConfigError: true,
			expectedError:         true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {

			// Define a custom RBAC test function, so that there is no test coverage overlap.
			ensureRBACFunc := func(context.Context, clientset.Interface, clientset.Interface,
				time.Duration, time.Duration) (clientset.Interface, error) {

				if tc.expectRBACError {
					return nil, errors.New("ensureRBACFunc error")
				}
				return nil, nil
			}

			// Define a custom kubeconfig function so that we can fail at least one call.
			kubeConfigFunc := func(a string, b string, cfg *kubeadmapi.InitConfiguration) error {
				if tc.expectKubeConfigError {
					return errors.New("kubeConfigFunc error")
				}
				return kubeconfigphase.CreateKubeConfigFile(a, b, cfg)
			}

			// If kubeConfigExist is true, pre-create the admin.conf and super-admin.conf files.
			if tc.kubeConfigExist {
				b := []byte("foo")
				if err := os.WriteFile(filepath.Join(dir, constants.AdminKubeConfigFileName), b, 0644); err != nil {
					t.Fatal(err)
				}
				if err := os.WriteFile(filepath.Join(dir, constants.SuperAdminKubeConfigFileName), b, 0644); err != nil {
					t.Fatal(err)
				}
			}

			// Call createSuperAdminKubeConfig() with a custom ensureRBACFunc().
			err := createSuperAdminKubeConfig(cfg, dir, false, ensureRBACFunc, kubeConfigFunc)
			if (err != nil) != tc.expectedError {
				t.Fatalf("expected error: %v, got: %v, error: %v", err != nil, tc.expectedError, err)
			}

			// Obtain the list of files in the directory after createSuperAdminKubeConfig() is done.
			var files []string
			fileInfo, err := os.ReadDir(dir)
			if err != nil {
				t.Fatal(err)
			}
			for _, file := range fileInfo {
				files = append(files, file.Name())
			}

			// Verify the expected files.
			expectedFiles := []string{
				constants.AdminKubeConfigFileName,
				constants.CACertName,
				constants.CAKeyName,
				constants.SuperAdminKubeConfigFileName,
			}
			if !reflect.DeepEqual(expectedFiles, files) {
				t.Fatalf("expected files: %v, got: %v", expectedFiles, files)
			}
		})
	}
}
