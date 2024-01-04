/*
Copyright 2024 The Kubernetes Authors.

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
	"bytes"
	"context"
	"testing"
	"time"

	"k8s.io/klog/v2/ktesting"
	"k8s.io/kubernetes/pkg/features"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	encryptionconfigcontroller "k8s.io/apiserver/pkg/server/options/encryptionconfig/controller"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
)

// TestStorageVersionMigration is an integration test that verifies storage version migration works.
// This test asserts following scenarios:
// 1. Start API server with encr at rest and hot reload of encryption config enabled
// 2. Create a secret
// 3. Update encryption config file to add a new key as write key
// 4. Perform Storage Version Migration for secrets
// 5. Verify that the secret is migrated to use the new key
// 6. Verify that the secret is updated with a new resource version
// 7. Perform another Storage Version Migration for secrets
// 8. Verify that the resource version of the secret is not updated. i.e. it was a no-op update
func TestStorageVersionMigration(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionMigrator, true)()
	// this makes the test super responsive. It's set to a default of 1 minute.
	encryptionconfigcontroller.EncryptionConfigFileChangePollDuration = time.Millisecond

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	svmTest := svmSetup(ctx, t)

	// ToDo: try to test with 1000 secrets
	secret, err := svmTest.createSecret(ctx, t, secretName, secretNamespace)
	if err != nil {
		t.Fatalf("Failed to create secret: %v", err)
	}

	metricBeforeUpdate := svmTest.getAutomaticReloadSuccessTotal(ctx, t)
	svmTest.updateFile(t, svmTest.filePathForEncryptionConfig, encryptionConfigFileName, []byte(resources["updatedEncryptionConfig"]))
	if !svmTest.isEncryptionConfigFileUpdated(ctx, t, metricBeforeUpdate) {
		t.Fatalf("Failed to update encryption config file")
	}

	svm, err := svmTest.createSVMResource(ctx, t, svmName)
	if err != nil {
		t.Fatalf("Failed to create SVM resource: %v", err)
	}
	if !svmTest.waitForResourceMigration(ctx, t, svm.Name, secret.Name, secret.Namespace, 1) {
		t.Fatalf("Failed to migrate resource %s/%s", secret.Namespace, secret.Name)
	}

	wantPrefix := "k8s:enc:aescbc:v1:key2"
	etcdSecret, err := svmTest.getRawSecretFromETCD(t, secret.Name, secret.Namespace)
	if err != nil {
		t.Fatalf("Failed to get secret from etcd: %v", err)
	}
	// assert that secret is prefixed with the new key
	if !bytes.HasPrefix(etcdSecret, []byte(wantPrefix)) {
		t.Fatalf("expected secret to be prefixed with %s, but got %s", wantPrefix, etcdSecret)
	}

	secretAfterMigration, err := svmTest.client.CoreV1().Secrets(secret.Namespace).Get(ctx, secret.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get secret: %v", err)
	}
	// assert that RV is different
	// rv is expected to be different as the secret was re-written to etcd with the new key
	if secret.ResourceVersion == secretAfterMigration.ResourceVersion {
		t.Fatalf("Expected resource version to be different, but got the same, rv before: %s, rv after: %s", secret.ResourceVersion, secretAfterMigration.ResourceVersion)
	}

	secondSVM, err := svmTest.createSVMResource(ctx, t, secondSVMName)
	if err != nil {
		t.Fatalf("Failed to create SVM resource: %v", err)
	}
	if !svmTest.waitForResourceMigration(ctx, t, secondSVM.Name, secretAfterMigration.Name, secretAfterMigration.Namespace, 2) {
		t.Fatalf("Failed to migrate resource %s/%s", secretAfterMigration.Namespace, secretAfterMigration.Name)
	}

	secretAfterSecondMigration, err := svmTest.client.CoreV1().Secrets(secretAfterMigration.Namespace).Get(ctx, secretAfterMigration.Name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Failed to get secret: %v", err)
	}
	// assert that RV is same
	if secretAfterMigration.ResourceVersion != secretAfterSecondMigration.ResourceVersion {
		t.Fatalf("Expected resource version to be same, but got different, rv before: %s, rv after: %s", secretAfterMigration.ResourceVersion, secretAfterSecondMigration.ResourceVersion)
	}
}
