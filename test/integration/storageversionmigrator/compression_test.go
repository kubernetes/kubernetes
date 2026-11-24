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

package storageversionmigrator

import (
	"context"
	"fmt"
	"maps"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	svmv1 "k8s.io/api/storagemigration/v1"
	metaconditions "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/ktesting"
	"k8s.io/kubernetes/test/utils"
)

const (
	// frameDiscriminator is byte 0 of every framed value, from format.go in
	// k8s.io/apiserver/pkg/storage/value/compression. Readers check only this
	// byte to tell a frame from a bare value.
	frameDiscriminator = 0x00

	// The test uses configmaps because svmSetup encrypts only secrets, so a
	// configmap's raw etcd value is the frame itself. Every configmap the test
	// creates is above the default 1Ki minSize.
	compressConfigMapsConfig = `
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: StorageCompressionConfiguration
compressionResources:
- resources: ["configmaps"]
  algorithm: Deflate
`
	// The first matching entry wins, so None keeps configmaps out of the
	// wildcard after it.
	excludeConfigMapsConfig = `
apiVersion: apiserver.config.k8s.io/v1alpha1
kind: StorageCompressionConfiguration
compressionResources:
- resources: ["configmaps"]
  algorithm: None
- resources: ["*."]
  algorithm: Deflate
`
	compressionTestObjects = 10
	writtenWithPolicy      = "written-with-policy"
	writtenWithoutPolicy   = "written-without-policy"
	svmTriggerConfigMap    = "trigger-for-svm"
)

// compressionFlags writes config to name in dir and returns the kube-apiserver
// flags that load it. Any policy that compresses something needs
// --min-compatibility-version 1.38 or the server won't start.
func compressionFlags(t *testing.T, dir, name, config string) []string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(config), 0o600); err != nil {
		t.Fatal(err)
	}
	return []string{
		"--storage-compression-config=" + path,
		"--min-compatibility-version=1.38",
	}
}

// compressibleData is well above the 1Ki minSize. The name keeps every value
// distinct, and requireReadBack uses it to check what came back.
func compressibleData(name string) string {
	return name + " " + strings.Repeat("compress me ", 200)
}

// createCompressibleConfigMaps creates compressionTestObjects configmaps in a new
// namespace, so nothing else stored can skew a count. It returns their resource
// versions by name.
func (svm *svmTest) createCompressibleConfigMaps(ctx context.Context, t *testing.T, namespace string) map[string]string {
	t.Helper()
	if _, err := svm.client.CoreV1().Namespaces().Create(ctx, &corev1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}, metav1.CreateOptions{}); err != nil {
		t.Fatalf("Failed to create namespace %s: %v", namespace, err)
	}
	rvs := make(map[string]string, compressionTestObjects)
	for i := range compressionTestObjects {
		name := "cm-" + strconv.Itoa(i)
		cm := &corev1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{Name: name},
			Data:       map[string]string{"data": compressibleData(name)},
		}
		created, err := svm.client.CoreV1().ConfigMaps(namespace).Create(ctx, cm, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Failed to create configmap %s/%s: %v", namespace, name, err)
		}
		rvs[name] = created.ResourceVersion
	}
	return rvs
}

// storedFraming reads configmap values straight from etcd, from one namespace or
// from all of them when namespace is empty. It counts every value it scans and
// the ones that lead with the frame discriminator.
func (svm *svmTest) storedFraming(t utils.TB, namespace string) (scanned, framed int) {
	t.Helper()
	prefix := getETCDPathForResource(t, svm.storageConfig.Prefix, "", "configmaps", namespace, "")
	resp, err := svm.readRawRecordFromETCD(t, prefix)
	if err != nil {
		t.Fatalf("Failed to read %s from etcd: %v", prefix, err)
	}
	for _, kv := range resp.Kvs {
		if len(kv.Value) > 0 && kv.Value[0] == frameDiscriminator {
			framed++
		}
	}
	return len(resp.Kvs), framed
}

// requireFraming checks that etcd holds exactly compressionTestObjects values in
// namespace and that wantFramed of them are framed. Checking the total too
// keeps an empty scan from passing.
func (svm *svmTest) requireFraming(t utils.TB, namespace string, wantFramed int) {
	t.Helper()
	scanned, framed := svm.storedFraming(t, namespace)
	if scanned != compressionTestObjects || framed != wantFramed {
		t.Fatalf("etcd holds %d configmaps in %s, %d of them framed; want %d, %d framed", scanned, namespace, framed, compressionTestObjects, wantFramed)
	}
}

// requireNoFramesStored fails the test if any configmap in etcd, in any
// namespace, leads with the frame discriminator, since an older apiserver fails
// on a frame anywhere under the resource. The scan has to see every configmap a
// LIST returns, so an empty scan can't pass.
func (svm *svmTest) requireNoFramesStored(ctx context.Context, t utils.TB) {
	t.Helper()
	list, err := svm.client.CoreV1().ConfigMaps(metav1.NamespaceAll).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list configmaps in all namespaces: %v", err)
	}
	if scanned, framed := svm.storedFraming(t, ""); scanned != len(list.Items) || framed != 0 {
		t.Errorf("etcd holds %d configmaps, %d of them framed; want %d, none framed", scanned, framed, len(list.Items))
	}
}

// requireReadBack fails the test unless namespace's configmaps read back with
// the data they were created with. A LIST without a resourceVersion is a
// consistent read, even when the watch cache serves it. It returns their
// resource versions by name.
func (svm *svmTest) requireReadBack(ctx context.Context, t utils.TB, namespace string) map[string]string {
	t.Helper()
	list, err := svm.client.CoreV1().ConfigMaps(namespace).List(ctx, metav1.ListOptions{})
	if err != nil {
		t.Fatalf("Failed to list configmaps in %s: %v", namespace, err)
	}
	if len(list.Items) != compressionTestObjects {
		t.Fatalf("Listed %d configmaps in %s, want %d", len(list.Items), namespace, compressionTestObjects)
	}
	rvs := make(map[string]string, len(list.Items))
	for _, cm := range list.Items {
		if got, want := cm.Data["data"], compressibleData(cm.Name); got != want {
			t.Errorf("Configmap %s/%s read back with %d bytes of data, want %d", namespace, cm.Name, len(got), len(want))
		}
		rvs[cm.Name] = cm.ResourceVersion
	}
	return rvs
}

// migrateConfigMaps runs a storage version migration for configmaps and waits
// for it to succeed.
func (svm *svmTest) migrateConfigMaps(ctx context.Context, t *testing.T, name string) {
	t.Helper()
	if _, err := svm.createSVMResource(ctx, t, name, metav1.GroupResource{Resource: "configmaps"}); err != nil {
		t.Fatalf("Failed to create SVM resource %s: %v", name, err)
	}

	triggered := false
	err := wait.PollUntilContextTimeout(ctx, 500*time.Millisecond, 5*time.Minute, true, func(ctx context.Context) (bool, error) {
		migration, err := svm.getSVM(ctx, t, name)
		if err != nil {
			return false, err
		}
		conditions := migration.Status.Conditions
		if metaconditions.IsStatusConditionTrue(conditions, string(svmv1.MigrationFailed)) {
			return false, fmt.Errorf("migration %s failed: %v", name, conditions)
		}
		if metaconditions.IsStatusConditionTrue(conditions, string(svmv1.MigrationSucceeded)) {
			return true, nil
		}
		// The controller migrates from the garbage collector's configmap cache, and
		// waits until that cache has seen the migration's resource version. Nothing
		// else writes configmaps here, so the cache would wait for a watch bookmark,
		// about a minute. Writing one configmap moves it past the checkpoint now.
		if migration.Status.ResourceVersion != "" && !triggered {
			trigger := &corev1.ConfigMap{ObjectMeta: metav1.ObjectMeta{Name: svmTriggerConfigMap}}
			if _, err := svm.client.CoreV1().ConfigMaps(defaultNamespace).Create(ctx, trigger, metav1.CreateOptions{}); err != nil {
				return false, err
			}
			if err := svm.client.CoreV1().ConfigMaps(defaultNamespace).Delete(ctx, svmTriggerConfigMap, metav1.DeleteOptions{}); err != nil {
				return false, err
			}
			triggered = true
		}
		return false, nil
	})
	if err != nil {
		t.Fatalf("Migration %s of configmaps didn't succeed: %v", name, err)
	}
}

// TestStorageVersionMigrationCompression runs the KEP-6264 pre-downgrade
// procedure and its mirror, since the staleness rule is symmetric. One etcd backs
// four kube-apiservers: configmaps compressed, excluded ahead of a wildcard,
// compressed again, then the flag dropped. The file is read only at startup, so
// each change needs a restart.
func TestStorageVersionMigrationCompression(t *testing.T) {
	ctx := ktesting.Init(t)

	configDir := t.TempDir()
	compressConfigMaps := compressionFlags(t, configDir, "compress-configmaps.yaml", compressConfigMapsConfig)
	excludeConfigMaps := compressionFlags(t, configDir, "exclude-configmaps.yaml", excludeConfigMapsConfig)

	svmTest := svmSetupWithFlags(ctx, t, compressConfigMaps)
	svmTest.createCompressibleConfigMaps(ctx, t, writtenWithPolicy)
	svmTest.requireFraming(t, writtenWithPolicy, compressionTestObjects)

	svmTest.restart(ctx, t, excludeConfigMaps)
	t.Run("resource excluded ahead of a wildcard migrates out of frames", func(t *testing.T) {
		// Frames survive a restart, so if they're gone below, the migration removed
		// them.
		svmTest.requireFraming(t, writtenWithPolicy, compressionTestObjects)
		plainRVs := svmTest.createCompressibleConfigMaps(ctx, t, writtenWithoutPolicy)
		svmTest.requireFraming(t, writtenWithoutPolicy, 0)

		svmTest.migrateConfigMaps(ctx, t, "compression-excluded")

		svmTest.requireFraming(t, writtenWithPolicy, 0)
		svmTest.requireNoFramesStored(ctx, t)
		svmTest.requireReadBack(ctx, t, writtenWithPolicy)
		// These were already stored the way the Rule writes them. If the migration
		// rewrote them, every later migration would too.
		if rvs := svmTest.requireReadBack(ctx, t, writtenWithoutPolicy); !maps.Equal(rvs, plainRVs) {
			t.Errorf("The migration rewrote configmaps that were already unframed: resource versions went from %v to %v", plainRVs, rvs)
		}
	})

	svmTest.restart(ctx, t, compressConfigMaps)
	t.Run("selected resource migrates into frames", func(t *testing.T) {
		svmTest.requireFraming(t, writtenWithoutPolicy, 0)

		svmTest.migrateConfigMaps(ctx, t, "compression-selected")

		// The first migration unframed writtenWithPolicy, so this one frames it again.
		for _, namespace := range []string{writtenWithoutPolicy, writtenWithPolicy} {
			svmTest.requireFraming(t, namespace, compressionTestObjects)
			svmTest.requireReadBack(ctx, t, namespace)
		}
	})

	// Without the flag no store gets a Rule from the file, so this deselects
	// configmaps through the zero Rule rather than a None entry.
	svmTest.restart(ctx, t, nil)
	t.Run("resource deselected by dropping the flag migrates out of frames", func(t *testing.T) {
		for _, namespace := range []string{writtenWithoutPolicy, writtenWithPolicy} {
			svmTest.requireFraming(t, namespace, compressionTestObjects)
		}

		svmTest.migrateConfigMaps(ctx, t, "compression-flag-dropped")

		for _, namespace := range []string{writtenWithoutPolicy, writtenWithPolicy} {
			svmTest.requireFraming(t, namespace, 0)
			svmTest.requireReadBack(ctx, t, namespace)
		}
		svmTest.requireNoFramesStored(ctx, t)
	})
}
