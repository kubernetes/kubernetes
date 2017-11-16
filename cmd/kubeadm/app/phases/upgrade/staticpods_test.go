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
	"crypto/sha256"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/runtime"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiext "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha1"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
)

const (
	waitForHashes        = "wait-for-hashes"
	waitForHashChange    = "wait-for-hash-change"
	waitForPodsWithLabel = "wait-for-pods-with-label"

	testConfiguration = `
api:
  advertiseAddress: 1.2.3.4
  bindPort: 6443
apiServerCertSANs: null
apiServerExtraArgs: null
authorizationModes:
- Node
- RBAC
certificatesDir: /etc/kubernetes/pki
cloudProvider: ""
controllerManagerExtraArgs: null
etcd:
  caFile: ""
  certFile: ""
  dataDir: /var/lib/etcd
  endpoints: null
  extraArgs: null
  image: ""
  keyFile: ""
featureFlags: null
imageRepository: gcr.io/google_containers
kubernetesVersion: %s
networking:
  dnsDomain: cluster.local
  podSubnet: ""
  serviceSubnet: 10.96.0.0/12
nodeName: thegopher
schedulerExtraArgs: null
token: ce3aa5.5ec8455bb76b379f
tokenTTL: 24h
unifiedControlPlaneImage: ""
`
)

// fakeWaiter is a fake apiclient.Waiter that returns errors it was initialized with
type fakeWaiter struct {
	errsToReturn map[string]error
}

func NewFakeStaticPodWaiter(errsToReturn map[string]error) apiclient.Waiter {
	return &fakeWaiter{
		errsToReturn: errsToReturn,
	}
}

// WaitForAPI just returns a dummy nil, to indicate that the program should just proceed
func (w *fakeWaiter) WaitForAPI() error {
	return nil
}

// WaitForPodsWithLabel just returns an error if set from errsToReturn
func (w *fakeWaiter) WaitForPodsWithLabel(kvLabel string) error {
	return w.errsToReturn[waitForPodsWithLabel]
}

// WaitForPodToDisappear just returns a dummy nil, to indicate that the program should just proceed
func (w *fakeWaiter) WaitForPodToDisappear(podName string) error {
	return nil
}

// SetTimeout is a no-op; we don't use it in this implementation
func (w *fakeWaiter) SetTimeout(_ time.Duration) {}

// WaitForStaticPodControlPlaneHashes returns an error if set from errsToReturn
func (w *fakeWaiter) WaitForStaticPodControlPlaneHashes(_ string) (map[string]string, error) {
	return map[string]string{}, w.errsToReturn[waitForHashes]
}

// WaitForStaticPodControlPlaneHashChange returns an error if set from errsToReturn
func (w *fakeWaiter) WaitForStaticPodControlPlaneHashChange(_, _, _ string) error {
	return w.errsToReturn[waitForHashChange]
}

// WaitForHealthyKubelet returns a dummy nil just to implement the interface
func (w *fakeWaiter) WaitForHealthyKubelet(_ time.Duration, _ string) error {
	return nil
}

type fakeStaticPodPathManager struct {
	realManifestDir   string
	tempManifestDir   string
	backupManifestDir string
	MoveFileFunc      func(string, string) error
}

func NewFakeStaticPodPathManager(moveFileFunc func(string, string) error) (StaticPodPathManager, error) {
	realManifestsDir, err := ioutil.TempDir("", "kubeadm-upgraded-manifests")
	if err != nil {
		return nil, fmt.Errorf("couldn't create a temporary directory for the upgrade: %v", err)
	}

	upgradedManifestsDir, err := ioutil.TempDir("", "kubeadm-upgraded-manifests")
	if err != nil {
		return nil, fmt.Errorf("couldn't create a temporary directory for the upgrade: %v", err)
	}

	backupManifestsDir, err := ioutil.TempDir("", "kubeadm-backup-manifests")
	if err != nil {
		return nil, fmt.Errorf("couldn't create a temporary directory for the upgrade: %v", err)
	}

	return &fakeStaticPodPathManager{
		realManifestDir:   realManifestsDir,
		tempManifestDir:   upgradedManifestsDir,
		backupManifestDir: backupManifestsDir,
		MoveFileFunc:      moveFileFunc,
	}, nil
}

func (spm *fakeStaticPodPathManager) MoveFile(oldPath, newPath string) error {
	return spm.MoveFileFunc(oldPath, newPath)
}

func (spm *fakeStaticPodPathManager) RealManifestPath(component string) string {
	return constants.GetStaticPodFilepath(component, spm.realManifestDir)
}
func (spm *fakeStaticPodPathManager) RealManifestDir() string {
	return spm.realManifestDir
}

func (spm *fakeStaticPodPathManager) TempManifestPath(component string) string {
	return constants.GetStaticPodFilepath(component, spm.tempManifestDir)
}
func (spm *fakeStaticPodPathManager) TempManifestDir() string {
	return spm.tempManifestDir
}

func (spm *fakeStaticPodPathManager) BackupManifestPath(component string) string {
	return constants.GetStaticPodFilepath(component, spm.backupManifestDir)
}
func (spm *fakeStaticPodPathManager) BackupManifestDir() string {
	return spm.backupManifestDir
}

func TestStaticPodControlPlane(t *testing.T) {
	tests := []struct {
		waitErrsToReturn     map[string]error
		moveFileFunc         func(string, string) error
		expectedErr          bool
		manifestShouldChange bool
	}{
		{ // error-free case should succeed
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc: func(oldPath, newPath string) error {
				return os.Rename(oldPath, newPath)
			},
			expectedErr:          false,
			manifestShouldChange: true,
		},
		{ // any wait error should result in a rollback and an abort
			waitErrsToReturn: map[string]error{
				waitForHashes:        fmt.Errorf("boo! failed"),
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc: func(oldPath, newPath string) error {
				return os.Rename(oldPath, newPath)
			},
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{ // any wait error should result in a rollback and an abort
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    fmt.Errorf("boo! failed"),
				waitForPodsWithLabel: nil,
			},
			moveFileFunc: func(oldPath, newPath string) error {
				return os.Rename(oldPath, newPath)
			},
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{ // any wait error should result in a rollback and an abort
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: fmt.Errorf("boo! failed"),
			},
			moveFileFunc: func(oldPath, newPath string) error {
				return os.Rename(oldPath, newPath)
			},
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{ // any path-moving error should result in a rollback and an abort
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc: func(oldPath, newPath string) error {
				// fail for kube-apiserver move
				if strings.Contains(newPath, "kube-apiserver") {
					return fmt.Errorf("moving the kube-apiserver file failed")
				}
				return os.Rename(oldPath, newPath)
			},
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{ // any path-moving error should result in a rollback and an abort
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc: func(oldPath, newPath string) error {
				// fail for kube-controller-manager move
				if strings.Contains(newPath, "kube-controller-manager") {
					return fmt.Errorf("moving the kube-apiserver file failed")
				}
				return os.Rename(oldPath, newPath)
			},
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{ // any path-moving error should result in a rollback and an abort; even though this is the last component (kube-apiserver and kube-controller-manager healthy)
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc: func(oldPath, newPath string) error {
				// fail for kube-scheduler move
				if strings.Contains(newPath, "kube-scheduler") {
					return fmt.Errorf("moving the kube-apiserver file failed")
				}
				return os.Rename(oldPath, newPath)
			},
			expectedErr:          true,
			manifestShouldChange: false,
		},
	}

	for _, rt := range tests {

		waiter := NewFakeStaticPodWaiter(rt.waitErrsToReturn)
		pathMgr, err := NewFakeStaticPodPathManager(rt.moveFileFunc)
		if err != nil {
			t.Fatalf("couldn't run NewFakeStaticPodPathManager: %v", err)
		}
		defer os.RemoveAll(pathMgr.RealManifestDir())
		defer os.RemoveAll(pathMgr.TempManifestDir())
		defer os.RemoveAll(pathMgr.BackupManifestDir())

		oldcfg, err := getConfig("v1.7.0")
		if err != nil {
			t.Fatalf("couldn't create config: %v", err)
		}
		// Initialize the directory with v1.7 manifests; should then be upgraded to v1.8 using the method
		err = controlplane.CreateInitStaticPodManifestFiles(pathMgr.RealManifestDir(), oldcfg)
		if err != nil {
			t.Fatalf("couldn't run CreateInitStaticPodManifestFiles: %v", err)
		}
		// Get a hash of the v1.7 API server manifest to compare later (was the file re-written)
		oldHash, err := getAPIServerHash(pathMgr.RealManifestDir())
		if err != nil {
			t.Fatalf("couldn't read temp file: %v", err)
		}

		newcfg, err := getConfig("v1.8.0")
		if err != nil {
			t.Fatalf("couldn't create config: %v", err)
		}

		actualErr := StaticPodControlPlane(waiter, pathMgr, newcfg)
		if (actualErr != nil) != rt.expectedErr {
			t.Errorf(
				"failed UpgradeStaticPodControlPlane\n\texpected error: %t\n\tgot: %t",
				rt.expectedErr,
				(actualErr != nil),
			)
		}

		newHash, err := getAPIServerHash(pathMgr.RealManifestDir())
		if err != nil {
			t.Fatalf("couldn't read temp file: %v", err)
		}

		if (oldHash != newHash) != rt.manifestShouldChange {
			t.Errorf(
				"failed StaticPodControlPlane\n\texpected manifest change: %t\n\tgot: %t",
				rt.manifestShouldChange,
				(oldHash != newHash),
			)
		}

	}
}

func getAPIServerHash(dir string) (string, error) {
	manifestPath := constants.GetStaticPodFilepath(constants.KubeAPIServer, dir)

	fileBytes, err := ioutil.ReadFile(manifestPath)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%x", sha256.Sum256(fileBytes)), nil
}

func getConfig(version string) (*kubeadmapi.MasterConfiguration, error) {
	externalcfg := &kubeadmapiext.MasterConfiguration{}
	internalcfg := &kubeadmapi.MasterConfiguration{}
	if err := runtime.DecodeInto(legacyscheme.Codecs.UniversalDecoder(), []byte(fmt.Sprintf(testConfiguration, version)), externalcfg); err != nil {
		return nil, fmt.Errorf("unable to decode config: %v", err)
	}
	legacyscheme.Scheme.Convert(externalcfg, internalcfg, nil)
	return internalcfg, nil
}
