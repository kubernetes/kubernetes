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
	"strconv"
	"sync"
	"testing"
	"time"

	"go.uber.org/goleak"

	svmv1alpha1 "k8s.io/api/storagemigration/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/util/wait"
	encryptionconfigcontroller "k8s.io/apiserver/pkg/server/options/encryptionconfig/controller"
	etcd3watcher "k8s.io/apiserver/pkg/storage/etcd3"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	clientgofeaturegate "k8s.io/client-go/features"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
)

// TestStorageVersionMigration is an integration test that verifies storage version migration works.
// This test asserts following scenarios:
// 1. Start API server with encryption at rest and hot reload of encryption config enabled
// 2. Create a secret
// 3. Update encryption config file to add a new key as write key
// 4. Perform Storage Version Migration for secrets
// 5. Verify that the secret is migrated to use the new key
// 6. Verify that the secret is updated with a new resource version
// 7. Perform another Storage Version Migration for secrets
// 8. Verify that the resource version of the secret is not updated. i.e. it was a no-op update
func TestStorageVersionMigration(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionMigrator, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, featuregate.Feature(clientgofeaturegate.InformerResourceVersion), true)

	// this makes the test super responsive. It's set to a default of 1 minute.
	encryptionconfigcontroller.EncryptionConfigFileChangePollDuration = time.Second

	ctx := ktesting.Init(t)

	svmTest := svmSetup(ctx, t)

	// ToDo: try to test with 1000 secrets
	secret, err := svmTest.createSecret(ctx, t, secretName, defaultNamespace)
	if err != nil {
		t.Fatalf("Failed to create secret: %v", err)
	}

	metricBeforeUpdate := svmTest.getAutomaticReloadSuccessTotal(ctx, t)
	svmTest.updateFile(t, svmTest.filePathForEncryptionConfig, encryptionConfigFileName, []byte(resources["updatedEncryptionConfig"]))
	if !svmTest.isEncryptionConfigFileUpdated(ctx, t, metricBeforeUpdate) {
		t.Fatalf("Failed to update encryption config file")
	}

	svm, err := svmTest.createSVMResource(
		ctx,
		t,
		svmName,
		svmv1alpha1.GroupVersionResource{
			Group:    "",
			Version:  "v1",
			Resource: "secrets",
		},
	)
	if err != nil {
		t.Fatalf("Failed to create SVM resource: %v", err)
	}
	if !svmTest.waitForResourceMigration(ctx, t, svm.Name, secret.Name, 1) {
		t.Fatalf("Failed to migrate resource %s/%s", secret.Namespace, secret.Name)
	}

	wantPrefix := "k8s:enc:aescbc:v1:key2"
	etcdSecret, err := svmTest.getRawSecretFromETCD(t, secret.Namespace, secret.Name)
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

	secondSVM, err := svmTest.createSVMResource(
		ctx,
		t,
		secondSVMName,
		svmv1alpha1.GroupVersionResource{
			Group:    "",
			Version:  "v1",
			Resource: "secrets",
		},
	)
	if err != nil {
		t.Fatalf("Failed to create SVM resource: %v", err)
	}
	if !svmTest.waitForResourceMigration(ctx, t, secondSVM.Name, secretAfterMigration.Name, 2) {
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

// TestStorageVersionMigrationWithCRD is an integration test that verifies storage version migration works with CRD.
// This test asserts following scenarios:
// 1. CRD is created with version v1 (serving and storage)
// 2. Verify that CRs are written and stored as v1
// 3. Update CRD to introduce v2 (for serving only), and a conversion webhook is added
// 4. Verify that CRs are written to v2 but are stored as v1
// 5. CRD storage version is changed from v1 to v2
// 6. Verify that CR written as either v1 or v2 version are stored as v2
// 7. Perform Storage Version Migration to migrate all v1 CRs to v2
// 8. CRD is updated to no longer serve v1
// 9. Shutdown conversion webhook
// 10. Verify RV and Generations of CRs
// 11. Verify the list of CRs at v2 works
func TestStorageVersionMigrationWithCRD(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionMigrator, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, featuregate.Feature(clientgofeaturegate.InformerResourceVersion), true)
	// decode errors are expected when using conversation webhooks
	etcd3watcher.TestOnlySetFatalOnDecodeError(false)
	t.Cleanup(func() { etcd3watcher.TestOnlySetFatalOnDecodeError(true) })
	framework.GoleakCheck(t, // block test clean up and let any lingering watches complete before making decode errors fatal again
		goleak.IgnoreTopFunction("k8s.io/kubernetes/vendor/gopkg.in/natefinch/lumberjack%2ev2.(*Logger).millRun"),
		goleak.IgnoreTopFunction("gopkg.in/natefinch/lumberjack%2ev2.(*Logger).millRun"),
		goleak.IgnoreTopFunction("github.com/moby/spdystream.(*Connection).shutdown"),
	)

	ctx := ktesting.Init(t)

	crVersions := make(map[string]versions)

	svmTest := svmSetup(ctx, t)
	certCtx := svmTest.setupServerCert(t)

	// simulate monkeys creating and deleting CRs
	svmTest.createChaos(ctx, t)

	// create CRD with v1 serving and storage
	crd := svmTest.createCRD(t, crdName, crdGroup, certCtx, v1CRDVersion)

	// create CR
	cr1 := svmTest.createCR(ctx, t, "cr1", "v1")
	if ok := svmTest.isCRStoredAtVersion(t, "v1", cr1.GetName()); !ok {
		t.Fatalf("CR not stored at version v1")
	}
	crVersions[cr1.GetName()] = versions{
		generation:  cr1.GetGeneration(),
		rv:          cr1.GetResourceVersion(),
		isRVUpdated: true,
	}

	// add conversion webhook
	shutdownServer := svmTest.createConversionWebhook(ctx, t, certCtx)

	// add v2 for serving only
	svmTest.updateCRD(ctx, t, crd.Name, v2CRDVersion, []string{"v1", "v2"}, "v1")

	// create another CR
	cr2 := svmTest.createCR(ctx, t, "cr2", "v2")
	if ok := svmTest.isCRStoredAtVersion(t, "v1", cr2.GetName()); !ok {
		t.Fatalf("CR not stored at version v1")
	}
	crVersions[cr2.GetName()] = versions{
		generation:  cr2.GetGeneration(),
		rv:          cr2.GetResourceVersion(),
		isRVUpdated: true,
	}

	// add v2 as storage version
	svmTest.updateCRD(ctx, t, crd.Name, v2StorageCRDVersion, []string{"v1", "v2"}, "v2")

	// create CR with v1
	var cr3 *unstructured.Unstructured
	// updateCRD checks discovery returns storageVersionHash matching storage version v2
	// to make sure the API server uses v2 but CRD controllers may race and the resource
	// might still get stored in v1.
	// Attempt to recreate the CR until it gets stored as v2.
	// https://github.com/kubernetes/kubernetes/issues/130235
	err := wait.PollUntilContextTimeout(ctx, 1*time.Second, 10*time.Second, true, func(waitCtx context.Context) (done bool, err error) {
		cr3 = svmTest.createCR(waitCtx, t, "cr3", "v1")
		if ok := svmTest.isCRStoredAtVersion(t, "v2", cr3.GetName()); !ok {
			svmTest.deleteCR(waitCtx, t, cr3.GetName(), "v1")
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Fatalf("timed out waiting for CR to be stored as v2: %v", err)
	}

	crVersions[cr3.GetName()] = versions{
		generation:  cr3.GetGeneration(),
		rv:          cr3.GetResourceVersion(),
		isRVUpdated: false,
	}

	// create CR with v2
	cr4 := svmTest.createCR(ctx, t, "cr4", "v2")
	if ok := svmTest.isCRStoredAtVersion(t, "v2", cr4.GetName()); !ok {
		t.Fatalf("CR not stored at version v2")
	}
	crVersions[cr4.GetName()] = versions{
		generation:  cr4.GetGeneration(),
		rv:          cr4.GetResourceVersion(),
		isRVUpdated: false,
	}

	// verify cr1 ans cr2 are still stored at v1
	if ok := svmTest.isCRStoredAtVersion(t, "v1", cr1.GetName()); !ok {
		t.Fatalf("CR not stored at version v1")
	}
	if ok := svmTest.isCRStoredAtVersion(t, "v1", cr2.GetName()); !ok {
		t.Fatalf("CR not stored at version v1")
	}

	// migrate CRs from v1 to v2
	svm, err := svmTest.createSVMResource(
		ctx, t, "crdsvm",
		svmv1alpha1.GroupVersionResource{
			Group:    crd.Spec.Group,
			Version:  "v1",
			Resource: crd.Spec.Names.Plural,
		})
	if err != nil {
		t.Fatalf("Failed to create SVM resource: %v", err)
	}
	if ok := svmTest.isCRDMigrated(ctx, t, svm.Name, "triggercr"); !ok {
		t.Fatalf("CRD not migrated")
	}

	// assert all the CRs are stored in the etcd at correct version
	if ok := svmTest.isCRStoredAtVersion(t, "v2", cr1.GetName()); !ok {
		t.Fatalf("CR not stored at version v2")
	}
	if ok := svmTest.isCRStoredAtVersion(t, "v2", cr2.GetName()); !ok {
		t.Fatalf("CR not stored at version v2")
	}
	if ok := svmTest.isCRStoredAtVersion(t, "v2", cr3.GetName()); !ok {
		t.Fatalf("CR not stored at version v2")
	}
	if ok := svmTest.isCRStoredAtVersion(t, "v2", cr4.GetName()); !ok {
		t.Fatalf("CR not stored at version v2")
	}

	// update CRD to v1 not serving and storage followed by webhook shutdown
	svmTest.updateCRD(ctx, t, crd.Name, v1NotServingCRDVersion, []string{"v2"}, "v2")
	shutdownServer()

	// assert RV and Generations of CRs
	svmTest.validateRVAndGeneration(ctx, t, crVersions, "v2")

	// assert v2 CRs can be listed
	if err := svmTest.listCR(ctx, t, "v2"); err != nil {
		t.Fatalf("Failed to list CRs at version v2: %v", err)
	}
}

// TestStorageVersionMigrationDuringChaos serves as a stress test for the SVM controller.
// It creates a CRD and a reasonable number of static instances for that resource.
// It also continuously creates and deletes instances of that resource.
// During all of this, it attempts to perform multiple parallel migrations of the resource.
// It asserts that all migrations are successful and that none of the static instances
// were changed after they were initially created (as the migrations must be a no-op).
func TestStorageVersionMigrationDuringChaos(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.StorageVersionMigrator, true)
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, featuregate.Feature(clientgofeaturegate.InformerResourceVersion), true)

	ctx := ktesting.Init(t)

	svmTest := svmSetup(ctx, t)

	svmTest.createChaos(ctx, t)

	crd := svmTest.createCRD(t, crdName, crdGroup, nil, v1CRDVersion)

	crVersions := make(map[string]versions)

	for i := range 50 { // a more realistic number of total resources
		cr := svmTest.createCR(ctx, t, "created-cr-"+strconv.Itoa(i), "v1")
		crVersions[cr.GetName()] = versions{
			generation:  cr.GetGeneration(),
			rv:          cr.GetResourceVersion(),
			isRVUpdated: false, // none of these CRs should change due to migrations
		}
	}

	var wg sync.WaitGroup
	const migrations = 10 // more than the total workers of SVM
	wg.Add(migrations)
	for i := range migrations {
		i := i
		go func() {
			defer wg.Done()

			svm, err := svmTest.createSVMResource(
				ctx, t, "chaos-svm-"+strconv.Itoa(i),
				svmv1alpha1.GroupVersionResource{
					Group:    crd.Spec.Group,
					Version:  "v1",
					Resource: crd.Spec.Names.Plural,
				},
			)
			if err != nil {
				t.Errorf("Failed to create SVM resource: %v", err)
				return
			}
			triggerCRName := "chaos-trigger-" + strconv.Itoa(i)
			if ok := svmTest.isCRDMigrated(ctx, t, svm.Name, triggerCRName); !ok {
				t.Errorf("CRD not migrated")
				return
			}
		}()
	}
	wg.Wait()

	svmTest.validateRVAndGeneration(ctx, t, crVersions, "v1")
}
