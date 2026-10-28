/*
Copyright The Kubernetes Authors.

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
	"os"
	"path/filepath"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/util/sets"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func TestEtcdDataDirCheck(t *testing.T) {
	manifestDir := t.TempDir()
	manifestPath := filepath.Join(manifestDir, kubeadmconstants.Etcd+".yaml")
	writeEtcdManifest(t, manifestPath, "/var/lib/etcd-restore/")

	check := etcdDataDirCheck{manifestPath: manifestPath, configuredDir: "/var/lib/etcd-restore"}
	warnings, errs := check.Check()
	if len(warnings) != 0 || len(errs) != 0 {
		t.Fatalf("matching directories returned warnings %v errors %v", warnings, errs)
	}

	check.configuredDir = "/var/lib/etcd"
	_, errs = check.Check()
	if len(errs) != 1 {
		t.Fatalf("expected one mismatch error, got %v", errs)
	}
	for _, want := range []string{"/var/lib/etcd-restore/", "/var/lib/etcd", "etcd.local.dataDir", EtcdDataDirCheck} {
		if !strings.Contains(errs[0].Error(), want) {
			t.Errorf("mismatch error %q does not contain %q", errs[0].Error(), want)
		}
	}

	if err := os.Remove(manifestPath); err != nil {
		t.Fatal(err)
	}
	_, errs = check.Check()
	if len(errs) != 1 || !strings.Contains(errs[0].Error(), manifestPath) {
		t.Fatalf("missing manifest error = %v", errs)
	}

	writeEtcdManifest(t, manifestPath, "")
	_, errs = check.Check()
	if len(errs) != 1 || !strings.Contains(errs[0].Error(), etcdDataVolumeName) {
		t.Fatalf("missing volume error = %v", errs)
	}
}

func TestRunEtcdDataDirCheck(t *testing.T) {
	manifestDir := t.TempDir()
	writeEtcdManifest(t, filepath.Join(manifestDir, kubeadmconstants.Etcd+".yaml"), "/var/lib/etcd-restore")

	local := func(dataDir string) *kubeadmapi.ClusterConfiguration {
		return &kubeadmapi.ClusterConfiguration{
			Etcd: kubeadmapi.Etcd{Local: &kubeadmapi.LocalEtcd{DataDir: dataDir}},
		}
	}

	if err := RunEtcdDataDirCheck(manifestDir, &kubeadmapi.ClusterConfiguration{
		Etcd: kubeadmapi.Etcd{External: &kubeadmapi.ExternalEtcd{}},
	}, nil); err != nil {
		t.Fatalf("external etcd should skip the check: %v", err)
	}
	if err := RunEtcdDataDirCheck(manifestDir, &kubeadmapi.ClusterConfiguration{}, nil); err != nil {
		t.Fatalf("missing local etcd should skip the check: %v", err)
	}
	if err := RunEtcdDataDirCheck(manifestDir, local("/var/lib/etcd-restore"), nil); err != nil {
		t.Fatalf("matching data directory should pass: %v", err)
	}
	if err := RunEtcdDataDirCheck(manifestDir, local("/var/lib/etcd"), nil); err == nil {
		t.Fatal("mismatched data directory should fail")
	}
	if err := RunEtcdDataDirCheck(manifestDir, local("/var/lib/etcd"), sets.New(strings.ToLower(EtcdDataDirCheck))); err != nil {
		t.Fatalf("ignored check should pass: %v", err)
	}
	if err := RunEtcdDataDirCheck(manifestDir, local("/var/lib/etcd"), sets.New("all")); err != nil {
		t.Fatalf("ignoring all checks should pass: %v", err)
	}
	if err := RunEtcdDataDirCheck(manifestDir, nil, nil); err == nil {
		t.Fatal("nil configuration should fail")
	}
}

func TestPerformEtcdStaticPodUpgradeRejectsDataDirMismatch(t *testing.T) {
	pathMgr, err := NewFakeStaticPodPathManager(os.Rename)
	if err != nil {
		t.Fatal(err)
	}
	k8sDir := pathMgr.(*fakeStaticPodPathManager).KubernetesDir()
	t.Cleanup(func() {
		if err := os.RemoveAll(k8sDir); err != nil {
			t.Errorf("failed to remove %s: %v", k8sDir, err)
		}
	})

	writeEtcdManifest(t, pathMgr.RealManifestPath(kubeadmconstants.Etcd), "/var/lib/etcd-restore")
	cfg := &kubeadmapi.InitConfiguration{
		ClusterConfiguration: kubeadmapi.ClusterConfiguration{
			Etcd: kubeadmapi.Etcd{Local: &kubeadmapi.LocalEtcd{DataDir: "/var/lib/etcd"}},
		},
	}

	fatal, err := performEtcdStaticPodUpgrade(nil, nil, nil, pathMgr, cfg, nil, nil, nil)
	if !fatal || err == nil {
		t.Fatalf("fatal=%v err=%v, want a fatal preflight error before etcd is touched", fatal, err)
	}
}

func writeEtcdManifest(t *testing.T, manifestPath, dataDir string) {
	t.Helper()
	manifest := "apiVersion: v1\nkind: Pod\nmetadata:\n  name: etcd\n  namespace: kube-system\nspec:\n  containers:\n  - name: etcd\n    image: registry.k8s.io/etcd:3.5.0\n"
	if dataDir != "" {
		manifest += "  volumes:\n  - name: etcd-data\n    hostPath:\n      path: " + dataDir + "\n      type: DirectoryOrCreate\n"
	}
	if err := os.WriteFile(manifestPath, []byte(manifest), 0600); err != nil {
		t.Fatal(err)
	}
}
