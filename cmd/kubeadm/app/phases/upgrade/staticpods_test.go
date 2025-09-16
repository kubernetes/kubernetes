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
	"crypto/x509"
	"fmt"
	"math/big"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"go.etcd.io/etcd/client/pkg/v3/transport"

	v1 "k8s.io/api/core/v1"
	"k8s.io/client-go/tools/clientcmd"
	certutil "k8s.io/client-go/util/cert"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
	certsphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/certs"
	"k8s.io/kubernetes/cmd/kubeadm/app/phases/certs/renewal"
	controlplanephase "k8s.io/kubernetes/cmd/kubeadm/app/phases/controlplane"
	etcdphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/etcd"
	kubeconfigphase "k8s.io/kubernetes/cmd/kubeadm/app/phases/kubeconfig"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/apiclient"
	certstestutil "k8s.io/kubernetes/cmd/kubeadm/app/util/certs"
	configutil "k8s.io/kubernetes/cmd/kubeadm/app/util/config"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/errors"
	etcdutil "k8s.io/kubernetes/cmd/kubeadm/app/util/etcd"
	"k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil"
	pkiutiltesting "k8s.io/kubernetes/cmd/kubeadm/app/util/pkiutil/testing"
	testutil "k8s.io/kubernetes/cmd/kubeadm/test"
)

const (
	waitForHashes        = "wait-for-hashes"
	waitForHashChange    = "wait-for-hash-change"
	waitForPodsWithLabel = "wait-for-pods-with-label"
)

var testConfiguration = fmt.Sprintf(`
apiVersion: %s
kind: InitConfiguration
nodeRegistration:
  name: foo
localAPIEndpoint:
  advertiseAddress: 192.168.2.2
  bindPort: 6443
bootstrapTokens:
- token: ce3aa5.5ec8455bb76b379f
  ttl: 24h
---
apiVersion: %[1]s
kind: ClusterConfiguration

apiServer:
  certSANs: null
  extraArgs: null
certificatesDir: %%s
etcd:
  local:
    dataDir: %%s
    image: ""
imageRepository: registry.k8s.io
kubernetesVersion: %%s
networking:
  dnsDomain: cluster.local
  podSubnet: ""
  serviceSubnet: 10.96.0.0/12
`, kubeadmapiv1.SchemeGroupVersion.String())

// fakeWaiter is a fake apiclient.Waiter that returns errors it was initialized with
type fakeWaiter struct {
	errsToReturn map[string]error
}

func NewFakeStaticPodWaiter(errsToReturn map[string]error) apiclient.Waiter {
	return &fakeWaiter{
		errsToReturn: errsToReturn,
	}
}

// WaitForControlPlaneComponents just returns a dummy nil, to indicate that the program should just proceed
func (w *fakeWaiter) WaitForControlPlaneComponents(podsMap map[string]*v1.Pod, apiServerAddress string) error {
	return nil
}

// WaitForPodsWithLabel just returns an error if set from errsToReturn
func (w *fakeWaiter) WaitForPodsWithLabel(kvLabel string) error {
	return w.errsToReturn[waitForPodsWithLabel]
}

// SetTimeout is a no-op; we don't use it in this implementation
func (w *fakeWaiter) SetTimeout(_ time.Duration) {}

// WaitForStaticPodControlPlaneHashes returns an error if set from errsToReturn
func (w *fakeWaiter) WaitForStaticPodControlPlaneHashes(_ string) (map[string]string, error) {
	return map[string]string{}, w.errsToReturn[waitForHashes]
}

// WaitForStaticPodSingleHash returns an error if set from errsToReturn
func (w *fakeWaiter) WaitForStaticPodSingleHash(_ string, _ string) (string, error) {
	return "", w.errsToReturn[waitForHashes]
}

// WaitForStaticPodHashChange returns an error if set from errsToReturn
func (w *fakeWaiter) WaitForStaticPodHashChange(_, _, _ string) error {
	return w.errsToReturn[waitForHashChange]
}

// WaitForHKubelet returns a dummy nil just to implement the interface
func (w *fakeWaiter) WaitForKubelet(_ string, _ int32) error {
	return nil
}

type fakeStaticPodPathManager struct {
	kubernetesDir     string
	patchesDir        string
	realManifestDir   string
	tempManifestDir   string
	backupManifestDir string
	backupEtcdDir     string
	MoveFileFunc      func(string, string) error
}

func NewFakeStaticPodPathManager(moveFileFunc func(string, string) error) (StaticPodPathManager, error) {
	kubernetesDir, err := os.MkdirTemp("", "kubeadm-pathmanager-")
	if err != nil {
		return nil, errors.Wrapf(err, "couldn't create a temporary directory for the upgrade")
	}

	realManifestDir := filepath.Join(kubernetesDir, constants.ManifestsSubDirName)
	if err := os.Mkdir(realManifestDir, 0700); err != nil {
		return nil, errors.Wrapf(err, "couldn't create a realManifestDir for the upgrade")
	}

	upgradedManifestDir := filepath.Join(kubernetesDir, "upgraded-manifests")
	if err := os.Mkdir(upgradedManifestDir, 0700); err != nil {
		return nil, errors.Wrapf(err, "couldn't create a upgradedManifestDir for the upgrade")
	}

	backupManifestDir := filepath.Join(kubernetesDir, "backup-manifests")
	if err := os.Mkdir(backupManifestDir, 0700); err != nil {
		return nil, errors.Wrap(err, "couldn't create a backupManifestDir for the upgrade")
	}

	backupEtcdDir := filepath.Join(kubernetesDir, "kubeadm-backup-etcd")
	if err := os.Mkdir(backupEtcdDir, 0700); err != nil {
		return nil, err
	}

	return &fakeStaticPodPathManager{
		kubernetesDir:     kubernetesDir,
		realManifestDir:   realManifestDir,
		tempManifestDir:   upgradedManifestDir,
		backupManifestDir: backupManifestDir,
		backupEtcdDir:     backupEtcdDir,
		MoveFileFunc:      moveFileFunc,
	}, nil
}

func (spm *fakeStaticPodPathManager) MoveFile(oldPath, newPath string) error {
	return spm.MoveFileFunc(oldPath, newPath)
}

func (spm *fakeStaticPodPathManager) KubernetesDir() string {
	return spm.kubernetesDir
}

func (spm *fakeStaticPodPathManager) PatchesDir() string {
	return spm.patchesDir
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

func (spm *fakeStaticPodPathManager) BackupEtcdDir() string {
	return spm.backupEtcdDir
}

func (spm *fakeStaticPodPathManager) CleanupDirs() error {
	if err := os.RemoveAll(spm.TempManifestDir()); err != nil {
		return err
	}
	if err := os.RemoveAll(spm.BackupManifestDir()); err != nil {
		return err
	}
	return os.RemoveAll(spm.BackupEtcdDir())
}

type fakeTLSEtcdClient struct{ TLS bool }

func (c fakeTLSEtcdClient) WaitForClusterAvailable(retries int, retryInterval time.Duration) (bool, error) {
	return true, nil
}

func (c fakeTLSEtcdClient) CheckClusterHealth() error {
	return nil
}

func (c fakeTLSEtcdClient) Sync() error { return nil }

func (c fakeTLSEtcdClient) ListMembers() ([]etcdutil.Member, error) {
	return []etcdutil.Member{}, nil
}

func (c fakeTLSEtcdClient) AddMemberAsLearner(name string, peerAddrs string) ([]etcdutil.Member, error) {
	return []etcdutil.Member{}, nil
}

func (c fakeTLSEtcdClient) AddMember(name string, peerAddrs string) ([]etcdutil.Member, error) {
	return []etcdutil.Member{}, nil
}

func (c fakeTLSEtcdClient) MemberPromote(learnerID uint64) error {
	return nil
}

func (c fakeTLSEtcdClient) GetMemberID(peerURL string) (uint64, error) {
	return 0, nil
}

func (c fakeTLSEtcdClient) RemoveMember(id uint64) ([]etcdutil.Member, error) {
	return []etcdutil.Member{}, nil
}

type fakePodManifestEtcdClient struct{ ManifestDir, CertificatesDir string }

func (c fakePodManifestEtcdClient) WaitForClusterAvailable(retries int, retryInterval time.Duration) (bool, error) {
	return true, nil
}

func (c fakePodManifestEtcdClient) CheckClusterHealth() error {
	// Make sure the certificates generated from the upgrade are readable from disk
	tlsInfo := transport.TLSInfo{
		CertFile:      filepath.Join(c.CertificatesDir, constants.EtcdCACertName),
		KeyFile:       filepath.Join(c.CertificatesDir, constants.EtcdHealthcheckClientCertName),
		TrustedCAFile: filepath.Join(c.CertificatesDir, constants.EtcdHealthcheckClientKeyName),
	}
	_, err := tlsInfo.ClientConfig()
	return err
}

func (c fakePodManifestEtcdClient) Sync() error { return nil }

func (c fakePodManifestEtcdClient) ListMembers() ([]etcdutil.Member, error) {
	return []etcdutil.Member{}, nil
}

func (c fakePodManifestEtcdClient) AddMemberAsLearner(name string, peerAddrs string) ([]etcdutil.Member, error) {
	return []etcdutil.Member{}, nil
}

func (c fakePodManifestEtcdClient) AddMember(name string, peerAddrs string) ([]etcdutil.Member, error) {
	return []etcdutil.Member{}, nil
}

func (c fakePodManifestEtcdClient) MemberPromote(learnerID uint64) error {
	return nil
}

func (c fakePodManifestEtcdClient) GetMemberID(peerURL string) (uint64, error) {
	return 0, nil
}

func (c fakePodManifestEtcdClient) RemoveMember(id uint64) ([]etcdutil.Member, error) {
	return []etcdutil.Member{}, nil
}

func TestStaticPodControlPlane(t *testing.T) {
	tests := []struct {
		description          string
		waitErrsToReturn     map[string]error
		moveFileFunc         func(string, string) error
		skipKubeConfig       string
		expectedErr          bool
		manifestShouldChange bool
	}{
		{
			description: "error-free case should succeed",
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc:         os.Rename,
			expectedErr:          false,
			manifestShouldChange: true,
		},
		{
			description: "any wait error should result in a rollback and an abort 1",
			waitErrsToReturn: map[string]error{
				waitForHashes:        errors.New("boo! failed"),
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc:         os.Rename,
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{
			description: "any wait error should result in a rollback and an abort 2",
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    errors.New("boo! failed"),
				waitForPodsWithLabel: nil,
			},
			moveFileFunc:         os.Rename,
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{
			description: "any wait error should result in a rollback and an abort 3",
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: errors.New("boo! failed"),
			},
			moveFileFunc:         os.Rename,
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{
			description: "any path-moving error should result in a rollback and an abort 1",
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc: func(oldPath, newPath string) error {
				// fail for kube-apiserver move
				if strings.Contains(newPath, "kube-apiserver") {
					return errors.New("moving the kube-apiserver file failed")
				}
				return os.Rename(oldPath, newPath)
			},
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{
			description: "any path-moving error should result in a rollback and an abort 2",
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc: func(oldPath, newPath string) error {
				// fail for kube-controller-manager move
				if strings.Contains(newPath, "kube-controller-manager") {
					return errors.New("moving the kube-apiserver file failed")
				}
				return os.Rename(oldPath, newPath)
			},
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{
			description: "any path-moving error should result in a rollback and an abort; even though this is the last component (kube-apiserver and kube-controller-manager healthy)",
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc: func(oldPath, newPath string) error {
				// fail for kube-scheduler move
				if strings.Contains(newPath, "kube-scheduler") {
					return errors.New("moving the kube-apiserver file failed")
				}
				return os.Rename(oldPath, newPath)
			},
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{
			description: "any cert renew error should result in a rollback and an abort; even though this is the last component (kube-apiserver and kube-controller-manager healthy)",
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc:         os.Rename,
			skipKubeConfig:       constants.SchedulerKubeConfigFileName,
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{
			description: "any cert renew error should result in a rollback and an abort; even though this is admin.conf (kube-apiserver and kube-controller-manager and kube-scheduler healthy)",
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc:         os.Rename,
			skipKubeConfig:       constants.AdminKubeConfigFileName,
			expectedErr:          true,
			manifestShouldChange: false,
		},
		{
			description: "super-admin.conf is renewed if it exists",
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc:         os.Rename,
			expectedErr:          false,
			manifestShouldChange: true,
		},
		{
			description: "no error is thrown if super-admin.conf does not exist",
			waitErrsToReturn: map[string]error{
				waitForHashes:        nil,
				waitForHashChange:    nil,
				waitForPodsWithLabel: nil,
			},
			moveFileFunc:         os.Rename,
			skipKubeConfig:       constants.SuperAdminKubeConfigFileName,
			expectedErr:          false,
			manifestShouldChange: true,
		},
	}

	for i := range tests {
		rt := tests[i]
		t.Run(rt.description, func(t *testing.T) {
			pkiutiltesting.Reset()
			waiter := NewFakeStaticPodWaiter(rt.waitErrsToReturn)
			pathMgr, err := NewFakeStaticPodPathManager(rt.moveFileFunc)
			if err != nil {
				t.Fatalf("couldn't run NewFakeStaticPodPathManager: %v", err)
			}
			defer os.RemoveAll(pathMgr.(*fakeStaticPodPathManager).KubernetesDir())
			tmpKubernetesDir := pathMgr.(*fakeStaticPodPathManager).KubernetesDir()

			tempCertsDir, err := os.MkdirTemp("", "kubeadm-certs")
			if err != nil {
				t.Fatalf("couldn't create temporary certificates directory: %v", err)
			}
			defer os.RemoveAll(tempCertsDir)
			tmpEtcdDataDir, err := os.MkdirTemp("", "kubeadm-etcd-data")
			if err != nil {
				t.Fatalf("couldn't create temporary etcd data directory: %v", err)
			}
			defer os.RemoveAll(tmpEtcdDataDir)

			oldcfg, err := getConfig("v1.3.0", tempCertsDir, tmpEtcdDataDir)
			if err != nil {
				t.Fatalf("couldn't create config: %v", err)
			}

			tree, err := certsphase.GetCertsWithoutEtcd().AsMap().CertTree()
			if err != nil {
				t.Fatalf("couldn't get cert tree: %v", err)
			}

			if err := tree.CreateTree(oldcfg); err != nil {
				t.Fatalf("couldn't get create cert tree: %v", err)
			}

			for _, kubeConfig := range []string{
				constants.AdminKubeConfigFileName,
				constants.SuperAdminKubeConfigFileName,
				constants.SchedulerKubeConfigFileName,
				constants.ControllerManagerKubeConfigFileName,
			} {
				if rt.skipKubeConfig == kubeConfig {
					continue
				}
				if err := kubeconfigphase.CreateKubeConfigFile(kubeConfig, tmpKubernetesDir, oldcfg); err != nil {
					t.Fatalf("couldn't create kubeconfig %q: %v", kubeConfig, err)
				}
			}

			// Initialize the directory with v1.7 manifests; should then be upgraded to v1.8 using the method
			err = controlplanephase.CreateInitStaticPodManifestFiles(pathMgr.RealManifestDir(), pathMgr.PatchesDir(), oldcfg, false /* isDryRun */)
			if err != nil {
				t.Fatalf("couldn't run CreateInitStaticPodManifestFiles: %v", err)
			}
			err = etcdphase.CreateLocalEtcdStaticPodManifestFile(pathMgr.RealManifestDir(), pathMgr.PatchesDir(), oldcfg.NodeRegistration.Name, &oldcfg.ClusterConfiguration, &oldcfg.LocalAPIEndpoint, false /* isDryRun */)
			if err != nil {
				t.Fatalf("couldn't run CreateLocalEtcdStaticPodManifestFile: %v", err)
			}
			// Get a hash of the v1.7 API server manifest to compare later (was the file re-written)
			oldHash, err := getAPIServerHash(pathMgr.RealManifestDir())
			if err != nil {
				t.Fatalf("couldn't read temp file: %v", err)
			}

			newcfg, err := getConfig(constants.CurrentKubernetesVersion.String(), tempCertsDir, tmpEtcdDataDir)
			if err != nil {
				t.Fatalf("couldn't create config: %v", err)
			}

			// create the kubeadm etcd certs
			caCert, caKey, err := certsphase.KubeadmCertEtcdCA().CreateAsCA(newcfg)
			if err != nil {
				t.Fatalf("couldn't create new CA certificate: %v", err)
			}
			for _, cert := range []*certsphase.KubeadmCert{
				certsphase.KubeadmCertEtcdServer(),
				certsphase.KubeadmCertEtcdPeer(),
				certsphase.KubeadmCertEtcdHealthcheck(),
				certsphase.KubeadmCertEtcdAPIClient(),
			} {
				if err := cert.CreateFromCA(newcfg, caCert, caKey); err != nil {
					t.Fatalf("couldn't create certificate %s: %v", cert.Name, err)
				}
			}

			actualErr := StaticPodControlPlane(
				nil,
				waiter,
				pathMgr,
				newcfg,
				true,
				true,
				fakeTLSEtcdClient{
					TLS: false,
				},
				fakePodManifestEtcdClient{
					ManifestDir:     pathMgr.RealManifestDir(),
					CertificatesDir: newcfg.CertificatesDir,
				},
			)
			if (actualErr != nil) != rt.expectedErr {
				t.Errorf(
					"failed UpgradeStaticPodControlPlane\n%s\n\texpected error: %t\n\tgot: %t\n\tactual error: %v",
					rt.description,
					rt.expectedErr,
					(actualErr != nil),
					actualErr,
				)
			}

			newHash, err := getAPIServerHash(pathMgr.RealManifestDir())
			if err != nil {
				t.Fatalf("couldn't read temp file: %v", err)
			}

			if (oldHash != newHash) != rt.manifestShouldChange {
				t.Errorf(
					"failed StaticPodControlPlane\n%s\n\texpected manifest change: %t\n\tgot: %t\n\tnewHash: %v",
					rt.description,
					rt.manifestShouldChange,
					(oldHash != newHash),
					newHash,
				)
			}
		})
	}
}

func getAPIServerHash(dir string) (string, error) {
	manifestPath := constants.GetStaticPodFilepath(constants.KubeAPIServer, dir)

	fileBytes, err := os.ReadFile(manifestPath)
	if err != nil {
		return "", err
	}

	return fmt.Sprintf("%x", sha256.Sum256(fileBytes)), nil
}

func getConfig(version, certsDir, etcdDataDir string) (*kubeadmapi.InitConfiguration, error) {
	configBytes := []byte(fmt.Sprintf(testConfiguration, certsDir, etcdDataDir, version))

	// Unmarshal the config
	return configutil.BytesToInitConfiguration(configBytes, true /* skipCRIDetect */)
}

func getTempDir(t *testing.T, name string) (string, func()) {
	dir, err := os.MkdirTemp(os.TempDir(), name)
	if err != nil {
		t.Fatalf("couldn't make temporary directory: %v", err)
	}

	return dir, func() {
		os.RemoveAll(dir)
	}
}

func TestCleanupDirs(t *testing.T) {
	tests := []struct {
		name                   string
		keepManifest, keepEtcd bool
	}{
		{
			name:         "save manifest backup",
			keepManifest: true,
		},
		{
			name:         "save both etcd and manifest",
			keepManifest: true,
			keepEtcd:     true,
		},
		{
			name: "save nothing",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			realKubernetesDir, cleanup := getTempDir(t, "realKubernetesDir")
			defer cleanup()

			tempManifestDir, cleanup := getTempDir(t, "tempManifestDir")
			defer cleanup()

			backupManifestDir, cleanup := getTempDir(t, "backupManifestDir")
			defer cleanup()

			backupEtcdDir, cleanup := getTempDir(t, "backupEtcdDir")
			defer cleanup()

			mgr := NewKubeStaticPodPathManager(realKubernetesDir, "", tempManifestDir, backupManifestDir, backupEtcdDir, test.keepManifest, test.keepEtcd)
			err := mgr.CleanupDirs()
			if err != nil {
				t.Errorf("unexpected error cleaning up: %v", err)
			}

			if _, err := os.Stat(tempManifestDir); !os.IsNotExist(err) {
				t.Errorf("%q should not have existed", tempManifestDir)
			}
			_, err = os.Stat(backupManifestDir)
			if test.keepManifest {
				if err != nil {
					t.Errorf("unexpected error getting backup manifest dir")
				}
			} else {
				if !os.IsNotExist(err) {
					t.Error("expected backup manifest to not exist")
				}
			}

			_, err = os.Stat(backupEtcdDir)
			if test.keepEtcd {
				if err != nil {
					t.Errorf("unexpected error getting backup etcd dir")
				}
			} else {
				if !os.IsNotExist(err) {
					t.Error("expected backup etcd dir to not exist")
				}
			}
		})
	}
}

func TestRenewCertsByComponent(t *testing.T) {
	caCert, caKey := certstestutil.SetupCertificateAuthority(t)

	tests := []struct {
		name                  string
		component             string
		externalCA            bool
		externalFrontProxyCA  bool
		skipCreateEtcdCA      bool
		shouldErrorOnRenew    bool
		certsShouldExist      []*certsphase.KubeadmCert
		certsShouldBeRenewed  []*certsphase.KubeadmCert // NB. If empty, it will assume certsShouldBeRenewed == certsShouldExist
		kubeConfigShouldExist []string
	}{
		{
			name:      "all CA exist, all certs should be rotated for etcd",
			component: constants.Etcd,
			certsShouldExist: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertEtcdServer(),
				certsphase.KubeadmCertEtcdPeer(),
				certsphase.KubeadmCertEtcdHealthcheck(),
			},
		},
		{
			name:      "all CA exist, all certs should be rotated for apiserver",
			component: constants.KubeAPIServer,
			certsShouldExist: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertEtcdAPIClient(),
				certsphase.KubeadmCertAPIServer(),
				certsphase.KubeadmCertKubeletClient(),
				certsphase.KubeadmCertFrontProxyClient(),
			},
		},
		{
			name:      "external CA, renew only certificates not signed by CA for apiserver",
			component: constants.KubeAPIServer,
			certsShouldExist: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertEtcdAPIClient(),
				certsphase.KubeadmCertFrontProxyClient(),
				certsphase.KubeadmCertAPIServer(),
				certsphase.KubeadmCertKubeletClient(),
			},
			certsShouldBeRenewed: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertEtcdAPIClient(),
				certsphase.KubeadmCertFrontProxyClient(),
			},
			externalCA: true,
		},
		{
			name:      "external front-proxy-CA, renew only certificates not signed by front-proxy-CA for apiserver",
			component: constants.KubeAPIServer,
			certsShouldExist: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertEtcdAPIClient(),
				certsphase.KubeadmCertFrontProxyClient(),
				certsphase.KubeadmCertAPIServer(),
				certsphase.KubeadmCertKubeletClient(),
			},
			certsShouldBeRenewed: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertEtcdAPIClient(),
				certsphase.KubeadmCertAPIServer(),
				certsphase.KubeadmCertKubeletClient(),
			},
			externalFrontProxyCA: true,
		},
		{
			name:      "all CA exist, should be rotated for scheduler",
			component: constants.KubeScheduler,
			kubeConfigShouldExist: []string{
				constants.SchedulerKubeConfigFileName,
			},
		},
		{
			name:      "all CA exist, should be rotated for controller manager",
			component: constants.KubeControllerManager,
			kubeConfigShouldExist: []string{
				constants.ControllerManagerKubeConfigFileName,
			},
		},
		{
			name:               "missing a cert to renew",
			component:          constants.Etcd,
			shouldErrorOnRenew: true,
			certsShouldExist: []*certsphase.KubeadmCert{
				certsphase.KubeadmCertEtcdServer(),
				certsphase.KubeadmCertEtcdPeer(),
			},
		},
		{
			name:               "no CA, cannot continue",
			component:          constants.Etcd,
			skipCreateEtcdCA:   true,
			shouldErrorOnRenew: true,
		},
	}

	for i := range tests {
		test := tests[i]
		t.Run(test.name, func(t *testing.T) {
			pkiutiltesting.Reset()

			// Setup up basic requisites
			tmpDir := t.TempDir()

			cfg := testutil.GetDefaultInternalConfig(t)
			cfg.CertificatesDir = tmpDir

			if err := pkiutil.WriteCertAndKey(tmpDir, constants.CACertAndKeyBaseName, caCert, caKey); err != nil {
				t.Fatalf("couldn't write out CA: %v", err)
			}
			if test.externalCA {
				os.Remove(filepath.Join(tmpDir, constants.CAKeyName))
			}
			if err := pkiutil.WriteCertAndKey(tmpDir, constants.FrontProxyCACertAndKeyBaseName, caCert, caKey); err != nil {
				t.Fatalf("couldn't write out front-proxy-CA: %v", err)
			}
			if test.externalFrontProxyCA {
				os.Remove(filepath.Join(tmpDir, constants.FrontProxyCAKeyName))
			}
			if !test.skipCreateEtcdCA {
				if err := pkiutil.WriteCertAndKey(tmpDir, constants.EtcdCACertAndKeyBaseName, caCert, caKey); err != nil {
					t.Fatalf("couldn't write out etcd-CA: %v", err)
				}
			}

			certMaps := make(map[string]big.Int)

			// Create expected certs and load to recorde the serial numbers
			for _, kubeCert := range test.certsShouldExist {
				if err := kubeCert.CreateFromCA(cfg, caCert, caKey); err != nil {
					t.Fatalf("couldn't create certificate %q: %v", kubeCert.Name, err)
				}

				cert, err := pkiutil.TryLoadCertFromDisk(tmpDir, kubeCert.BaseName)
				if err != nil {
					t.Fatalf("couldn't load certificate %q: %v", kubeCert.Name, err)
				}
				certMaps[kubeCert.Name] = *cert.SerialNumber
			}

			// Create expected kubeconfigs
			for _, kubeConfig := range test.kubeConfigShouldExist {
				if err := kubeconfigphase.CreateKubeConfigFile(kubeConfig, tmpDir, cfg); err != nil {
					t.Fatalf("couldn't create kubeconfig %q: %v", kubeConfig, err)
				}

				newCerts, err := getEmbeddedCerts(tmpDir, kubeConfig)
				if err != nil {
					t.Fatalf("error reading embedded certs from %s: %v", kubeConfig, err)
				}
				certMaps[kubeConfig] = *newCerts[0].SerialNumber
			}

			// Renew everything
			rm, err := renewal.NewManager(&cfg.ClusterConfiguration, tmpDir)
			if err != nil {
				t.Fatalf("Failed to create the certificate renewal manager: %v", err)
			}

			err = renewCertsByComponent(cfg, test.component, rm)
			if test.shouldErrorOnRenew {
				if err == nil {
					t.Fatal("expected renewal error, got nothing")
				}
				// expected error, got error
				return
			}
			if err != nil {
				t.Fatalf("couldn't renew certificates: %v", err)
			}

			// See if the certificate serial numbers change
			for _, kubeCert := range test.certsShouldExist {
				newCert, err := pkiutil.TryLoadCertFromDisk(tmpDir, kubeCert.BaseName)
				if err != nil {
					t.Errorf("couldn't load new certificate %q: %v", kubeCert.Name, err)
					continue
				}
				oldSerial := certMaps[kubeCert.Name]

				shouldBeRenewed := true
				if test.certsShouldBeRenewed != nil {
					shouldBeRenewed = false
					for _, x := range test.certsShouldBeRenewed {
						if x.Name == kubeCert.Name {
							shouldBeRenewed = true
						}
					}
				}

				if shouldBeRenewed && oldSerial.Cmp(newCert.SerialNumber) == 0 {
					t.Errorf("certifitate %v was not reissued when expected", kubeCert.Name)
				}
				if !shouldBeRenewed && oldSerial.Cmp(newCert.SerialNumber) != 0 {
					t.Errorf("certifitate %v was reissued when not expected", kubeCert.Name)
				}
			}

			// See if the embedded certificate serial numbers change
			for _, kubeConfig := range test.kubeConfigShouldExist {
				newCerts, err := getEmbeddedCerts(tmpDir, kubeConfig)
				if err != nil {
					t.Fatalf("error reading embedded certs from %s: %v", kubeConfig, err)
				}
				oldSerial := certMaps[kubeConfig]
				if oldSerial.Cmp(newCerts[0].SerialNumber) == 0 {
					t.Errorf("certifitate %v was not reissued", kubeConfig)
				}
			}
		})

	}
}

func getEmbeddedCerts(tmpDir, kubeConfig string) ([]*x509.Certificate, error) {
	kubeconfigPath := filepath.Join(tmpDir, kubeConfig)
	newConfig, err := clientcmd.LoadFromFile(kubeconfigPath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to load kubeconfig file %s", kubeconfigPath)
	}

	authInfoName := newConfig.Contexts[newConfig.CurrentContext].AuthInfo
	authInfo := newConfig.AuthInfos[authInfoName]

	return certutil.ParseCertsPEM(authInfo.ClientCertificateData)
}

func TestGetPathManagerForUpgrade(t *testing.T) {

	externalEtcd := &kubeadmapi.InitConfiguration{
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			Etcd: kubeadmapi.Etcd{
				External: &kubeadmapi.ExternalEtcd{
					Endpoints: []string{"10.100.0.1:2379", "10.100.0.2:2379", "10.100.0.3:2379"},
				},
			},
		},
	}

	stackedEtcd := &kubeadmapi.InitConfiguration{}

	tests := []struct {
		name             string
		cfg              *kubeadmapi.InitConfiguration
		etcdUpgrade      bool
		shouldDeleteEtcd bool
	}{
		{
			name:             "external etcd but no etcd upgrade",
			cfg:              externalEtcd,
			etcdUpgrade:      false,
			shouldDeleteEtcd: true,
		},
		{
			name:             "external etcd with etcd upgrade",
			cfg:              externalEtcd,
			etcdUpgrade:      true,
			shouldDeleteEtcd: true,
		},
		{
			name:             "stacked etcd but no etcd upgrade",
			cfg:              stackedEtcd,
			etcdUpgrade:      false,
			shouldDeleteEtcd: true,
		},
		{
			name:             "stacked etcd with etcd upgrade",
			cfg:              stackedEtcd,
			etcdUpgrade:      true,
			shouldDeleteEtcd: false,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			// Use a temporary directory
			tmpdir, err := os.MkdirTemp("", "TestGetPathManagerForUpgrade")
			if err != nil {
				t.Fatalf("unexpected error making temporary directory: %v", err)
			}
			defer func() {
				os.RemoveAll(tmpdir)
			}()

			pathmgr, err := GetPathManagerForUpgrade(tmpdir, "", test.cfg, test.etcdUpgrade)
			if err != nil {
				t.Fatalf("unexpected error creating path manager: %v", err)
			}

			if _, err := os.Stat(pathmgr.BackupManifestDir()); os.IsNotExist(err) {
				t.Errorf("expected manifest dir %s to exist, but it did not (%v)", pathmgr.BackupManifestDir(), err)
			}

			if _, err := os.Stat(pathmgr.BackupEtcdDir()); os.IsNotExist(err) {
				t.Errorf("expected etcd dir %s to exist, but it did not (%v)", pathmgr.BackupEtcdDir(), err)
			}

			if err := pathmgr.CleanupDirs(); err != nil {
				t.Fatalf("unexpected error cleaning up directories: %v", err)
			}

			if _, err := os.Stat(pathmgr.BackupManifestDir()); os.IsNotExist(err) {
				t.Errorf("expected manifest dir %s to exist, but it did not (%v)", pathmgr.BackupManifestDir(), err)
			}

			if test.shouldDeleteEtcd {
				if _, err := os.Stat(pathmgr.BackupEtcdDir()); !os.IsNotExist(err) {
					t.Errorf("expected etcd dir %s not to exist, but it did (%v)", pathmgr.BackupEtcdDir(), err)
				}
			} else {
				if _, err := os.Stat(pathmgr.BackupEtcdDir()); os.IsNotExist(err) {
					t.Errorf("expected etcd dir %s to exist, but it did not", pathmgr.BackupEtcdDir())
				}
			}
		})
	}

}

func TestGetEtcdImageTagFromStaticPod(t *testing.T) {
	const expectedEtcdVersion = "3.1.12"
	const etcdStaticPod = `apiVersion: v1
kind: Pod
metadata:
  labels:
    component: etcd
    tier: control-plane
  name: etcd
  namespace: kube-system
spec:
  containers:
  - name: etcd
    image: registry.k8s.io/etcd:` + expectedEtcdVersion

	manifestsDir, err := os.MkdirTemp("", "GetEtcdImageTagFromStaticPod-test-manifests")
	if err != nil {
		t.Fatalf("Unable to create temporary directory: %v", err)
	}
	defer os.RemoveAll(manifestsDir)

	if err = os.WriteFile(constants.GetStaticPodFilepath(constants.Etcd, manifestsDir), []byte(etcdStaticPod), 0644); err != nil {
		t.Fatalf("Unable to create test static pod manifest: %v", err)
	}

	got, err := GetEtcdImageTagFromStaticPod(manifestsDir)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if got != expectedEtcdVersion {
		t.Errorf("unexpected result:\n\tgot: %q\n\texpected: %q", got, expectedEtcdVersion)
	}
}
