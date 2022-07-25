/*
Copyright 2018 The Kubernetes Authors.

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

package testsuites

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	e2evolume "k8s.io/kubernetes/test/e2e/framework/volume"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

// StorageClassTest represents parameters to be used by provisioning tests.
// Not all parameters are used by all tests.
type StorageClassTest struct {
	Client               clientset.Interface
	Timeouts             *framework.TimeoutContext
	Claim                *v1.PersistentVolumeClaim
	SourceClaim          *v1.PersistentVolumeClaim
	Class                *storagev1.StorageClass
	Name                 string
	CloudProviders       []string
	Provisioner          string
	Parameters           map[string]string
	DelayBinding         bool
	ClaimSize            string
	ExpectedSize         string
	PvCheck              func(claim *v1.PersistentVolumeClaim)
	VolumeMode           v1.PersistentVolumeMode
	AllowVolumeExpansion bool
	NodeSelection        e2epod.NodeSelection
}

type provisioningTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

// InitCustomProvisioningTestSuite returns provisioningTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomProvisioningTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &provisioningTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "provisioning",
			TestPatterns: patterns,
			SupportedSizeRange: e2evolume.SizeRange{
				Min: "1Mi",
			},
		},
	}
}

// InitProvisioningTestSuite returns provisioningTestSuite that implements TestSuite interface\
// using test suite default patterns
func InitProvisioningTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.DefaultFsDynamicPV,
		storageframework.BlockVolModeDynamicPV,
		storageframework.NtfsDynamicPV,
	}
	return InitCustomProvisioningTestSuite(patterns)
}

func (p *provisioningTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return p.tsInfo
}

func (p *provisioningTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	// Check preconditions.
	if pattern.VolType != storageframework.DynamicPV {
		e2eskipper.Skipf("Suite %q does not support %v", p.tsInfo.Name, pattern.VolType)
	}
	dInfo := driver.GetDriverInfo()
	if pattern.VolMode == v1.PersistentVolumeBlock && !dInfo.Capabilities[storageframework.CapBlock] {
		e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolMode)
	}
}

func (p *provisioningTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config        *storageframework.PerTestConfig
		driverCleanup func()

		testCase  *StorageClassTest
		cs        clientset.Interface
		pvc       *v1.PersistentVolumeClaim
		sourcePVC *v1.PersistentVolumeClaim
		sc        *storagev1.StorageClass

		migrationCheck *migrationOpCheck
	}
	var (
		dInfo   = driver.GetDriverInfo()
		dDriver storageframework.DynamicPVTestDriver
		l       local
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("provisioning", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	init := func() {
		l = local{}
		dDriver, _ = driver.(storageframework.DynamicPVTestDriver)
		// Now do the more expensive test initialization.
		l.config, l.driverCleanup = driver.PrepareTest(f)
		l.migrationCheck = newMigrationOpCheck(f.ClientSet, f.ClientConfig(), dInfo.InTreePluginName)
		l.cs = l.config.Framework.ClientSet
		testVolumeSizeRange := p.GetTestSuiteInfo().SupportedSizeRange
		driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
		claimSize, err := storageutils.GetSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
		framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, driverVolumeSizeRange)

		l.sc = dDriver.GetDynamicProvisionStorageClass(l.config, pattern.FsType)
		if l.sc == nil {
			e2eskipper.Skipf("Driver %q does not define Dynamic Provision StorageClass - skipping", dInfo.Name)
		}
		l.pvc = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(l.sc.Name),
			VolumeMode:       &pattern.VolMode,
		}, l.config.Framework.Namespace.Name)
		l.sourcePVC = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(l.sc.Name),
			VolumeMode:       &pattern.VolMode,
		}, l.config.Framework.Namespace.Name)
		framework.Logf("In creating storage class object and pvc objects for driver - sc: %v, pvc: %v, src-pvc: %v", l.sc, l.pvc, l.sourcePVC)
		l.testCase = &StorageClassTest{
			Client:       l.config.Framework.ClientSet,
			Timeouts:     f.Timeouts,
			Claim:        l.pvc,
			SourceClaim:  l.sourcePVC,
			Class:        l.sc,
			Provisioner:  l.sc.Provisioner,
			ClaimSize:    claimSize,
			ExpectedSize: claimSize,
			VolumeMode:   pattern.VolMode,
		}
	}

	cleanup := func() {
		err := storageutils.TryFunc(l.driverCleanup)
		l.driverCleanup = nil
		framework.ExpectNoError(err, "while cleaning up driver")

		l.migrationCheck.validateMigrationVolumeOpCounts()
	}

	ginkgo.It("should provision storage with mount options", func() {
		if dInfo.SupportedMountOption == nil {
			e2eskipper.Skipf("Driver %q does not define supported mount option - skipping", dInfo.Name)
		}
		if pattern.VolMode == v1.PersistentVolumeBlock {
			e2eskipper.Skipf("Block volumes do not support mount options - skipping")
		}

		init()
		defer cleanup()

		l.testCase.Class.MountOptions = dInfo.SupportedMountOption.Union(dInfo.RequiredMountOption).List()
		l.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim) {
			PVWriteReadSingleNodeCheck(l.cs, f.Timeouts, claim, l.config.ClientNodeSelection)
		}
		_, clearProvisionedStorageClass := SetupStorageClass(l.testCase.Client, l.testCase.Class)
		defer clearProvisionedStorageClass()

		l.testCase.TestDynamicProvisioning()
	})

	ginkgo.It("should provision storage with snapshot data source [Feature:VolumeSnapshotDataSource]", func() {
		if !dInfo.Capabilities[storageframework.CapSnapshotDataSource] {
			e2eskipper.Skipf("Driver %q does not support populate data from snapshot - skipping", dInfo.Name)
		}
		if !dInfo.SupportedFsType.Has(pattern.FsType) {
			e2eskipper.Skipf("Driver %q does not support %q fs type - skipping", dInfo.Name, pattern.FsType)
		}

		sDriver, ok := driver.(storageframework.SnapshottableTestDriver)
		if !ok {
			framework.Failf("Driver %q has CapSnapshotDataSource but does not implement SnapshottableTestDriver", dInfo.Name)
		}

		init()
		defer cleanup()

		dc := l.config.Framework.DynamicClient
		testConfig := storageframework.ConvertTestConfig(l.config)
		expectedContent := fmt.Sprintf("Hello from namespace %s", f.Namespace.Name)
		dataSource, cleanupFunc := prepareSnapshotDataSourceForProvisioning(f, testConfig, l.config, pattern, l.cs, dc, l.pvc, l.sc, sDriver, pattern.VolMode, expectedContent)
		defer cleanupFunc()

		l.pvc.Spec.DataSource = dataSource
		l.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim) {
			ginkgo.By("checking whether the created volume has the pre-populated data")
			tests := []e2evolume.Test{
				{
					Volume:          *storageutils.CreateVolumeSource(claim.Name, false /* readOnly */),
					Mode:            pattern.VolMode,
					File:            "index.html",
					ExpectedContent: expectedContent,
				},
			}
			e2evolume.TestVolumeClientSlow(f, testConfig, nil, "", tests)
		}
		l.testCase.TestDynamicProvisioning()
	})

	ginkgo.It("should provision storage with any volume data source [Serial]", func() {
		if len(dInfo.InTreePluginName) != 0 {
			e2eskipper.Skipf("AnyVolumeDataSource feature only works with CSI drivers - skipping")
		}
		if pattern.VolMode == v1.PersistentVolumeBlock {
			e2eskipper.Skipf("Test for Block volumes is not implemented - skipping")
		}

		init()
		defer cleanup()

		ginkgo.By("Creating validator namespace")
		valNamespace, err := f.CreateNamespace(fmt.Sprintf("%s-val", f.Namespace.Name), map[string]string{
			"e2e-framework":      f.BaseName,
			"e2e-test-namespace": f.Namespace.Name,
		})
		framework.ExpectNoError(err)

		defer func() {
			f.DeleteNamespace(valNamespace.Name)
		}()

		ginkgo.By("Deploying validator")
		valManifests := []string{
			"test/e2e/testing-manifests/storage-csi/any-volume-datasource/crd/populator.storage.k8s.io_volumepopulators.yaml",
			"test/e2e/testing-manifests/storage-csi/any-volume-datasource/volume-data-source-validator/rbac-data-source-validator.yaml",
			"test/e2e/testing-manifests/storage-csi/any-volume-datasource/volume-data-source-validator/setup-data-source-validator.yaml",
		}
		valCleanup, err := storageutils.CreateFromManifests(f, valNamespace,
			func(item interface{}) error { return nil },
			valManifests...)

		framework.ExpectNoError(err)
		defer valCleanup()

		ginkgo.By("Creating populator namespace")
		popNamespace, err := f.CreateNamespace(fmt.Sprintf("%s-pop", f.Namespace.Name), map[string]string{
			"e2e-framework":      f.BaseName,
			"e2e-test-namespace": f.Namespace.Name,
		})
		framework.ExpectNoError(err)

		defer func() {
			f.DeleteNamespace(popNamespace.Name)
		}()

		ginkgo.By("Deploying hello-populator")
		popManifests := []string{
			"test/e2e/testing-manifests/storage-csi/any-volume-datasource/crd/hello-populator-crd.yaml",
			"test/e2e/testing-manifests/storage-csi/any-volume-datasource/hello-populator-deploy.yaml",
		}
		popCleanup, err := storageutils.CreateFromManifests(f, popNamespace,
			func(item interface{}) error {
				switch item := item.(type) {
				case *appsv1.Deployment:
					for i, container := range item.Spec.Template.Spec.Containers {
						switch container.Name {
						case "hello":
							args := []string{}
							var foundNS, foundImage bool
							for _, arg := range container.Args {
								if strings.HasPrefix(arg, "--namespace=") {
									args = append(args, fmt.Sprintf("--namespace=%s", popNamespace.Name))
									foundNS = true
								} else if strings.HasPrefix(arg, "--image-name=") {
									args = append(args, fmt.Sprintf("--image-name=%s", container.Image))
									foundImage = true
								} else {
									args = append(args, arg)
								}
							}
							if !foundNS {
								args = append(args, fmt.Sprintf("--namespace=%s", popNamespace.Name))
								framework.Logf("container name: %s", container.Name)
							}
							if !foundImage {
								args = append(args, fmt.Sprintf("--image-name=%s", container.Image))
								framework.Logf("container image: %s", container.Image)
							}
							container.Args = args
							item.Spec.Template.Spec.Containers[i] = container
						default:
						}
					}
				}
				return nil
			},
			popManifests...)

		framework.ExpectNoError(err)
		defer popCleanup()

		dc := l.config.Framework.DynamicClient

		// Make hello-populator handle Hello resource in hello.example.com group
		ginkgo.By("Creating VolumePopulator CR datasource")
		volumePopulatorGVR := schema.GroupVersionResource{Group: "populator.storage.k8s.io", Version: "v1beta1", Resource: "volumepopulators"}
		helloPopulatorCR := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"kind":       "VolumePopulator",
				"apiVersion": "populator.storage.k8s.io/v1beta1",
				"metadata": map[string]interface{}{
					"name": fmt.Sprintf("%s-%s", "hello-populator", f.Namespace.Name),
				},
				"sourceKind": map[string]interface{}{
					"group": "hello.example.com",
					"kind":  "Hello",
				},
			},
		}

		_, err = dc.Resource(volumePopulatorGVR).Create(context.TODO(), helloPopulatorCR, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		defer func() {
			framework.Logf("deleting VolumePopulator CR datasource %q/%q", helloPopulatorCR.GetNamespace(), helloPopulatorCR.GetName())
			err = dc.Resource(volumePopulatorGVR).Delete(context.TODO(), helloPopulatorCR.GetName(), metav1.DeleteOptions{})
			if err != nil && !apierrors.IsNotFound(err) {
				framework.Failf("Error deleting VolumePopulator CR datasource %q. Error: %v", helloPopulatorCR.GetName(), err)
			}
		}()

		// Create Hello CR datasource
		ginkgo.By("Creating Hello CR datasource")
		helloCRName := "example-hello"
		fileName := fmt.Sprintf("example-%s.txt", f.Namespace.Name)
		expectedContent := fmt.Sprintf("Hello from namespace %s", f.Namespace.Name)
		helloGVR := schema.GroupVersionResource{Group: "hello.example.com", Version: "v1alpha1", Resource: "hellos"}
		helloCR := &unstructured.Unstructured{
			Object: map[string]interface{}{
				"kind":       "Hello",
				"apiVersion": "hello.example.com/v1alpha1",
				"metadata": map[string]interface{}{
					"name":      helloCRName,
					"namespace": f.Namespace.Name,
				},
				"spec": map[string]interface{}{
					"fileName":     fileName,
					"fileContents": expectedContent,
				},
			},
		}

		_, err = dc.Resource(helloGVR).Namespace(f.Namespace.Name).Create(context.TODO(), helloCR, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		defer func() {
			framework.Logf("deleting Hello CR datasource %q/%q", helloCR.GetNamespace(), helloCR.GetName())
			err = dc.Resource(helloGVR).Namespace(helloCR.GetNamespace()).Delete(context.TODO(), helloCR.GetName(), metav1.DeleteOptions{})
			if err != nil && !apierrors.IsNotFound(err) {
				framework.Failf("Error deleting Hello CR datasource %q. Error: %v", helloCR.GetName(), err)
			}
		}()

		apiGroup := "hello.example.com"
		l.pvc.Spec.DataSourceRef = &v1.TypedLocalObjectReference{
			APIGroup: &apiGroup,
			Kind:     "Hello",
			Name:     helloCRName,
		}

		testConfig := storageframework.ConvertTestConfig(l.config)
		l.testCase.NodeSelection = testConfig.ClientNodeSelection
		l.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim) {
			ginkgo.By("checking whether the created volume has the pre-populated data")
			tests := []e2evolume.Test{
				{
					Volume:          *storageutils.CreateVolumeSource(claim.Name, false /* readOnly */),
					Mode:            pattern.VolMode,
					File:            fileName,
					ExpectedContent: expectedContent,
				},
			}
			e2evolume.TestVolumeClientSlow(f, testConfig, nil, "", tests)
		}

		_, clearProvisionedStorageClass := SetupStorageClass(l.testCase.Client, l.testCase.Class)
		defer clearProvisionedStorageClass()

		l.testCase.TestDynamicProvisioning()
	})

	ginkgo.It("should provision storage with pvc data source", func() {
		if !dInfo.Capabilities[storageframework.CapPVCDataSource] {
			e2eskipper.Skipf("Driver %q does not support cloning - skipping", dInfo.Name)
		}
		init()
		defer cleanup()

		if l.config.ClientNodeSelection.Name == "" {
			// Schedule all pods to the same topology segment (e.g. a cloud availability zone), some
			// drivers don't support cloning across them.
			if err := ensureTopologyRequirements(&l.config.ClientNodeSelection, l.cs, dInfo, 1); err != nil {
				framework.Failf("Error setting topology requirements: %v", err)
			}
		}
		testConfig := storageframework.ConvertTestConfig(l.config)
		expectedContent := fmt.Sprintf("Hello from namespace %s", f.Namespace.Name)
		dataSource, dataSourceCleanup := preparePVCDataSourceForProvisioning(f, testConfig, l.cs, l.sourcePVC, l.sc, pattern.VolMode, expectedContent)
		defer dataSourceCleanup()
		l.pvc.Spec.DataSource = dataSource
		l.testCase.NodeSelection = testConfig.ClientNodeSelection
		l.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim) {
			ginkgo.By("checking whether the created volume has the pre-populated data")
			tests := []e2evolume.Test{
				{
					Volume:          *storageutils.CreateVolumeSource(claim.Name, false /* readOnly */),
					Mode:            pattern.VolMode,
					File:            "index.html",
					ExpectedContent: expectedContent,
				},
			}
			e2evolume.TestVolumeClientSlow(f, testConfig, nil, "", tests)
		}
		// Cloning fails if the source disk is still in the process of detaching, so we wait for the VolumeAttachment to be removed before cloning.
		volumeAttachment := e2evolume.GetVolumeAttachmentName(f.ClientSet, testConfig, l.testCase.Provisioner, dataSource.Name, l.sourcePVC.Namespace)
		e2evolume.WaitForVolumeAttachmentTerminated(volumeAttachment, f.ClientSet, f.Timeouts.DataSourceProvision)
		l.testCase.TestDynamicProvisioning()
	})

	ginkgo.It("should provision storage with pvc data source in parallel [Slow]", func() {
		// Test cloning a single volume multiple times.
		if !dInfo.Capabilities[storageframework.CapPVCDataSource] {
			e2eskipper.Skipf("Driver %q does not support cloning - skipping", dInfo.Name)
		}
		if pattern.VolMode == v1.PersistentVolumeBlock && !dInfo.Capabilities[storageframework.CapBlock] {
			e2eskipper.Skipf("Driver %q does not support block volumes - skipping", dInfo.Name)
		}

		init()
		defer cleanup()

		if l.config.ClientNodeSelection.Name == "" {
			// Schedule all pods to the same topology segment (e.g. a cloud availability zone), some
			// drivers don't support cloning across them.
			if err := ensureTopologyRequirements(&l.config.ClientNodeSelection, l.cs, dInfo, 1); err != nil {
				framework.Failf("Error setting topology requirements: %v", err)
			}
		}
		testConfig := storageframework.ConvertTestConfig(l.config)
		expectedContent := fmt.Sprintf("Hello from namespace %s", f.Namespace.Name)
		dataSource, dataSourceCleanup := preparePVCDataSourceForProvisioning(f, testConfig, l.cs, l.sourcePVC, l.sc, pattern.VolMode, expectedContent)
		defer dataSourceCleanup()
		l.pvc.Spec.DataSource = dataSource

		var wg sync.WaitGroup
		for i := 0; i < 5; i++ {
			wg.Add(1)
			go func(i int) {
				defer ginkgo.GinkgoRecover()
				defer wg.Done()
				ginkgo.By(fmt.Sprintf("Cloning volume nr. %d", i))
				// Each go routine must have its own pod prefix
				myTestConfig := testConfig
				myTestConfig.Prefix = fmt.Sprintf("%s-%d", myTestConfig.Prefix, i)

				t := *l.testCase
				t.NodeSelection = testConfig.ClientNodeSelection
				t.PvCheck = func(claim *v1.PersistentVolumeClaim) {
					ginkgo.By(fmt.Sprintf("checking whether the created volume %d has the pre-populated data", i))
					tests := []e2evolume.Test{
						{
							Volume:          *storageutils.CreateVolumeSource(claim.Name, false /* readOnly */),
							Mode:            pattern.VolMode,
							File:            "index.html",
							ExpectedContent: expectedContent,
						},
					}
					e2evolume.TestVolumeClientSlow(f, myTestConfig, nil, "", tests)
				}
				// Cloning fails if the source disk is still in the process of detaching, so we wait for the VolumeAttachment to be removed before cloning.
				volumeAttachment := e2evolume.GetVolumeAttachmentName(f.ClientSet, testConfig, l.testCase.Provisioner, dataSource.Name, l.sourcePVC.Namespace)
				e2evolume.WaitForVolumeAttachmentTerminated(volumeAttachment, f.ClientSet, f.Timeouts.DataSourceProvision)
				t.TestDynamicProvisioning()
			}(i)
		}
		wg.Wait()
	})

	ginkgo.It("should mount multiple PV pointing to the same storage on the same node", func() {
		// csi-hostpath driver does not support this test case. In this test case, we have 2 PV containing the same underlying storage.
		// during the NodeStage call for the second volume, csi-hostpath fails the call, because it thinks the volume is already staged at a different path.
		// Note: This is not an issue with driver like PD CSI where the NodeStage is a no-op for block mode.
		if pattern.VolMode == v1.PersistentVolumeBlock {
			e2eskipper.Skipf("skipping multiple PV mount test for block mode")
		}

		init()
		defer cleanup()

		l.testCase.PvCheck = func(claim *v1.PersistentVolumeClaim) {
			MultiplePVMountSingleNodeCheck(l.cs, f.Timeouts, claim, l.config.ClientNodeSelection)
		}
		_, clearProvisionedStorageClass := SetupStorageClass(l.testCase.Client, l.testCase.Class)
		defer clearProvisionedStorageClass()

		l.testCase.TestDynamicProvisioning()
	})
}

// SetupStorageClass ensures that a StorageClass from a spec exists, if the StorageClass already exists
// then it's returned as it is, if it doesn't exist then it's created first
// and then returned, if the spec is nil then we return the `default` StorageClass
func SetupStorageClass(
	client clientset.Interface,
	class *storagev1.StorageClass,
) (*storagev1.StorageClass, func()) {
	gomega.Expect(client).NotTo(gomega.BeNil(), "SetupStorageClass.client is required")

	var err error
	var computedStorageClass *storagev1.StorageClass
	var clearComputedStorageClass = func() {}
	if class != nil {
		computedStorageClass, err = client.StorageV1().StorageClasses().Get(context.TODO(), class.Name, metav1.GetOptions{})
		if err == nil {
			// skip storageclass creation if it already exists
			ginkgo.By("Storage class " + computedStorageClass.Name + " is already created, skipping creation.")
		} else {
			ginkgo.By("Creating a StorageClass")
			class, err = client.StorageV1().StorageClasses().Create(context.TODO(), class, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			computedStorageClass, err = client.StorageV1().StorageClasses().Get(context.TODO(), class.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)
			clearComputedStorageClass = func() {
				framework.Logf("deleting storage class %s", computedStorageClass.Name)
				err := client.StorageV1().StorageClasses().Delete(context.TODO(), computedStorageClass.Name, metav1.DeleteOptions{})
				if err != nil && !apierrors.IsNotFound(err) {
					framework.ExpectNoError(err, "delete storage class")
				}
			}
		}
	} else {
		// StorageClass is nil, so the default one will be used
		scName, err := e2epv.GetDefaultStorageClassName(client)
		framework.ExpectNoError(err)
		ginkgo.By("Wanted storage class is nil, fetching default StorageClass=" + scName)
		computedStorageClass, err = client.StorageV1().StorageClasses().Get(context.TODO(), scName, metav1.GetOptions{})
		framework.ExpectNoError(err)
	}

	return computedStorageClass, clearComputedStorageClass
}

// TestDynamicProvisioning tests dynamic provisioning with specified StorageClassTest
// it's assumed that the StorageClass `t.Class` is already provisioned,
// see #ProvisionStorageClass
func (t StorageClassTest) TestDynamicProvisioning() *v1.PersistentVolume {
	var err error
	client := t.Client
	gomega.Expect(client).NotTo(gomega.BeNil(), "StorageClassTest.Client is required")
	claim := t.Claim
	gomega.Expect(claim).NotTo(gomega.BeNil(), "StorageClassTest.Claim is required")
	gomega.Expect(claim.GenerateName).NotTo(gomega.BeEmpty(), "StorageClassTest.Claim.GenerateName must not be empty")
	class := t.Class
	gomega.Expect(class).NotTo(gomega.BeNil(), "StorageClassTest.Class is required")
	class, err = client.StorageV1().StorageClasses().Get(context.TODO(), class.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "StorageClass.Class "+class.Name+" couldn't be fetched from the cluster")

	ginkgo.By(fmt.Sprintf("creating claim=%+v", claim))
	claim, err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(context.TODO(), claim, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	defer func() {
		framework.Logf("deleting claim %q/%q", claim.Namespace, claim.Name)
		// typically this claim has already been deleted
		err = client.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(context.TODO(), claim.Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting claim %q. Error: %v", claim.Name, err)
		}
	}()

	// ensure that the claim refers to the provisioned StorageClass
	framework.ExpectEqual(*claim.Spec.StorageClassName, class.Name)

	// if late binding is configured, create and delete a pod to provision the volume
	if *class.VolumeBindingMode == storagev1.VolumeBindingWaitForFirstConsumer {
		ginkgo.By(fmt.Sprintf("creating a pod referring to the class=%+v claim=%+v", class, claim))
		var podConfig *e2epod.Config = &e2epod.Config{
			NS:            claim.Namespace,
			PVCs:          []*v1.PersistentVolumeClaim{claim},
			NodeSelection: t.NodeSelection,
		}

		var pod *v1.Pod
		pod, err := e2epod.CreateSecPod(client, podConfig, t.Timeouts.DataSourceProvision)
		// Delete pod now, otherwise PV can't be deleted below
		framework.ExpectNoError(err)
		e2epod.DeletePodOrFail(client, pod.Namespace, pod.Name)
	}

	// Run the checker
	if t.PvCheck != nil {
		t.PvCheck(claim)
	}

	pv := t.checkProvisioning(client, claim, class)

	ginkgo.By(fmt.Sprintf("deleting claim %q/%q", claim.Namespace, claim.Name))
	framework.ExpectNoError(client.CoreV1().PersistentVolumeClaims(claim.Namespace).Delete(context.TODO(), claim.Name, metav1.DeleteOptions{}))

	// Wait for the PV to get deleted if reclaim policy is Delete. (If it's
	// Retain, there's no use waiting because the PV won't be auto-deleted and
	// it's expected for the caller to do it.) Technically, the first few delete
	// attempts may fail, as the volume is still attached to a node because
	// kubelet is slowly cleaning up the previous pod, however it should succeed
	// in a couple of minutes. Wait 20 minutes (or whatever custom value is specified in
	// t.Timeouts.PVDeleteSlow) to recover from random cloud hiccups.
	if pv != nil && pv.Spec.PersistentVolumeReclaimPolicy == v1.PersistentVolumeReclaimDelete {
		ginkgo.By(fmt.Sprintf("deleting the claim's PV %q", pv.Name))
		framework.ExpectNoError(e2epv.WaitForPersistentVolumeDeleted(client, pv.Name, 5*time.Second, t.Timeouts.PVDeleteSlow))
	}

	return pv
}

// getBoundPV returns a PV details.
func getBoundPV(client clientset.Interface, pvc *v1.PersistentVolumeClaim) (*v1.PersistentVolume, error) {
	// Get new copy of the claim
	claim, err := client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(context.TODO(), pvc.Name, metav1.GetOptions{})
	if err != nil {
		return nil, err
	}

	// Get the bound PV
	pv, err := client.CoreV1().PersistentVolumes().Get(context.TODO(), claim.Spec.VolumeName, metav1.GetOptions{})
	return pv, err
}

// checkProvisioning verifies that the claim is bound and has the correct properities
func (t StorageClassTest) checkProvisioning(client clientset.Interface, claim *v1.PersistentVolumeClaim, class *storagev1.StorageClass) *v1.PersistentVolume {
	err := e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, claim.Namespace, claim.Name, framework.Poll, t.Timeouts.ClaimProvision)
	framework.ExpectNoError(err)

	ginkgo.By("checking the claim")
	pv, err := getBoundPV(client, claim)
	framework.ExpectNoError(err)

	// Check sizes
	expectedCapacity := resource.MustParse(t.ExpectedSize)
	pvCapacity := pv.Spec.Capacity[v1.ResourceName(v1.ResourceStorage)]
	gomega.Expect(pvCapacity.Value()).To(gomega.BeNumerically(">=", expectedCapacity.Value()), "pvCapacity is not greater or equal to expectedCapacity")

	requestedCapacity := resource.MustParse(t.ClaimSize)
	claimCapacity := claim.Spec.Resources.Requests[v1.ResourceName(v1.ResourceStorage)]
	gomega.Expect(claimCapacity.Value()).To(gomega.BeNumerically(">=", requestedCapacity.Value()), "claimCapacity is not greater or equal to requestedCapacity")

	// Check PV properties
	ginkgo.By("checking the PV")

	// Every access mode in PV should be in PVC
	gomega.Expect(pv.Spec.AccessModes).NotTo(gomega.BeZero())
	for _, pvMode := range pv.Spec.AccessModes {
		found := false
		for _, pvcMode := range claim.Spec.AccessModes {
			if pvMode == pvcMode {
				found = true
				break
			}
		}
		framework.ExpectEqual(found, true)
	}

	framework.ExpectEqual(pv.Spec.ClaimRef.Name, claim.ObjectMeta.Name)
	framework.ExpectEqual(pv.Spec.ClaimRef.Namespace, claim.ObjectMeta.Namespace)
	if class == nil {
		framework.ExpectEqual(pv.Spec.PersistentVolumeReclaimPolicy, v1.PersistentVolumeReclaimDelete)
	} else {
		framework.ExpectEqual(pv.Spec.PersistentVolumeReclaimPolicy, *class.ReclaimPolicy)
		framework.ExpectEqual(pv.Spec.MountOptions, class.MountOptions)
	}
	if claim.Spec.VolumeMode != nil {
		gomega.Expect(pv.Spec.VolumeMode).NotTo(gomega.BeNil())
		framework.ExpectEqual(*pv.Spec.VolumeMode, *claim.Spec.VolumeMode)
	}
	return pv
}

// PVWriteReadSingleNodeCheck checks that a PV retains data on a single node
// and returns the PV.
//
// It starts two pods:
// - The first pod writes 'hello word' to the /mnt/test (= the volume) on one node.
// - The second pod runs grep 'hello world' on /mnt/test on the same node.
//
// The node is selected by Kubernetes when scheduling the first
// pod. It's then selected via its name for the second pod.
//
// If both succeed, Kubernetes actually allocated something that is
// persistent across pods.
//
// This is a common test that can be called from a StorageClassTest.PvCheck.
func PVWriteReadSingleNodeCheck(client clientset.Interface, timeouts *framework.TimeoutContext, claim *v1.PersistentVolumeClaim, node e2epod.NodeSelection) *v1.PersistentVolume {
	ginkgo.By(fmt.Sprintf("checking the created volume is writable on node %+v", node))
	command := "echo 'hello world' > /mnt/test/data"
	pod := StartInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-volume-tester-writer", command, node)
	defer func() {
		// pod might be nil now.
		StopPod(client, pod)
	}()
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceTimeout(client, pod.Name, pod.Namespace, timeouts.PodStartSlow))
	runningPod, err := client.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	actualNodeName := runningPod.Spec.NodeName
	StopPod(client, pod)
	pod = nil // Don't stop twice.

	// Get a new copy of the PV
	e2evolume, err := getBoundPV(client, claim)
	framework.ExpectNoError(err)

	ginkgo.By(fmt.Sprintf("checking the created volume has the correct mount options, is readable and retains data on the same node %q", actualNodeName))
	command = "grep 'hello world' /mnt/test/data"

	// We give the second pod the additional responsibility of checking the volume has
	// been mounted with the PV's mount options, if the PV was provisioned with any
	for _, option := range e2evolume.Spec.MountOptions {
		// Get entry, get mount options at 6th word, replace brackets with commas
		command += fmt.Sprintf(" && ( mount | grep 'on /mnt/test' | awk '{print $6}' | sed 's/^(/,/; s/)$/,/' | grep -q ,%s, )", option)
	}
	command += " || (mount | grep 'on /mnt/test'; false)"

	if framework.NodeOSDistroIs("windows") {
		// agnhost doesn't support mount
		command = "grep 'hello world' /mnt/test/data"
	}
	RunInPodWithVolume(client, timeouts, claim.Namespace, claim.Name, "pvc-volume-tester-reader", command, e2epod.NodeSelection{Name: actualNodeName})

	return e2evolume
}

// PVMultiNodeCheck checks that a PV retains data when moved between nodes.
//
// It starts these pods:
// - The first pod writes 'hello word' to the /mnt/test (= the volume) on one node.
// - The second pod runs grep 'hello world' on /mnt/test on another node.
//
// The first node is selected by Kubernetes when scheduling the first pod. The second pod uses the same criteria, except that a special anti-affinity
// for the first node gets added. This test can only pass if the cluster has more than one
// suitable node. The caller has to ensure that.
//
// If all succeeds, Kubernetes actually allocated something that is
// persistent across pods and across nodes.
//
// This is a common test that can be called from a StorageClassTest.PvCheck.
func PVMultiNodeCheck(client clientset.Interface, timeouts *framework.TimeoutContext, claim *v1.PersistentVolumeClaim, node e2epod.NodeSelection) {
	framework.ExpectEqual(node.Name, "", "this test only works when not locked onto a single node")

	var pod *v1.Pod
	defer func() {
		// passing pod = nil is okay.
		StopPod(client, pod)
	}()

	ginkgo.By(fmt.Sprintf("checking the created volume is writable on node %+v", node))
	command := "echo 'hello world' > /mnt/test/data"
	pod = StartInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-writer-node1", command, node)
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceTimeout(client, pod.Name, pod.Namespace, timeouts.PodStartSlow))
	runningPod, err := client.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	actualNodeName := runningPod.Spec.NodeName
	StopPod(client, pod)
	pod = nil // Don't stop twice.

	// Add node-anti-affinity.
	secondNode := node
	e2epod.SetAntiAffinity(&secondNode, actualNodeName)
	ginkgo.By(fmt.Sprintf("checking the created volume is readable and retains data on another node %+v", secondNode))
	command = "grep 'hello world' /mnt/test/data"
	pod = StartInPodWithVolume(client, claim.Namespace, claim.Name, "pvc-reader-node2", command, secondNode)
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceTimeout(client, pod.Name, pod.Namespace, timeouts.PodStartSlow))
	runningPod, err = client.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err, "get pod")
	framework.ExpectNotEqual(runningPod.Spec.NodeName, actualNodeName, "second pod should have run on a different node")
	StopPod(client, pod)
	pod = nil
}

// TestBindingWaitForFirstConsumerMultiPVC tests the binding with WaitForFirstConsumer mode
func (t StorageClassTest) TestBindingWaitForFirstConsumerMultiPVC(claims []*v1.PersistentVolumeClaim, nodeSelector map[string]string, expectUnschedulable bool) ([]*v1.PersistentVolume, *v1.Node) {
	var err error
	framework.ExpectNotEqual(len(claims), 0)
	namespace := claims[0].Namespace

	ginkgo.By("creating claims")
	var claimNames []string
	var createdClaims []*v1.PersistentVolumeClaim
	for _, claim := range claims {
		c, err := t.Client.CoreV1().PersistentVolumeClaims(claim.Namespace).Create(context.TODO(), claim, metav1.CreateOptions{})
		claimNames = append(claimNames, c.Name)
		createdClaims = append(createdClaims, c)
		framework.ExpectNoError(err)
	}
	defer func() {
		errors := map[string]error{}
		for _, claim := range createdClaims {
			err := e2epv.DeletePersistentVolumeClaim(t.Client, claim.Name, claim.Namespace)
			if err != nil {
				errors[claim.Name] = err
			}
		}
		if len(errors) > 0 {
			for claimName, err := range errors {
				framework.Logf("Failed to delete PVC: %s due to error: %v", claimName, err)
			}
		}
	}()

	// Wait for ClaimProvisionTimeout (across all PVCs in parallel) and make sure the phase did not become Bound i.e. the Wait errors out
	ginkgo.By("checking the claims are in pending state")
	err = e2epv.WaitForPersistentVolumeClaimsPhase(v1.ClaimBound, t.Client, namespace, claimNames, 2*time.Second /* Poll */, t.Timeouts.ClaimProvisionShort, true)
	framework.ExpectError(err)
	verifyPVCsPending(t.Client, createdClaims)

	ginkgo.By("creating a pod referring to the claims")
	// Create a pod referring to the claim and wait for it to get to running
	var pod *v1.Pod
	if expectUnschedulable {
		pod, err = e2epod.CreateUnschedulablePod(t.Client, namespace, nodeSelector, createdClaims, true /* isPrivileged */, "" /* command */)
	} else {
		pod, err = e2epod.CreatePod(t.Client, namespace, nil /* nodeSelector */, createdClaims, true /* isPrivileged */, "" /* command */)
	}
	framework.ExpectNoError(err)
	defer func() {
		e2epod.DeletePodOrFail(t.Client, pod.Namespace, pod.Name)
		e2epod.WaitForPodToDisappear(t.Client, pod.Namespace, pod.Name, labels.Everything(), framework.Poll, t.Timeouts.PodDelete)
	}()
	if expectUnschedulable {
		// Verify that no claims are provisioned.
		verifyPVCsPending(t.Client, createdClaims)
		return nil, nil
	}

	// collect node details
	node, err := t.Client.CoreV1().Nodes().Get(context.TODO(), pod.Spec.NodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("re-checking the claims to see they bound")
	var pvs []*v1.PersistentVolume
	for _, claim := range createdClaims {
		// Get new copy of the claim
		claim, err = t.Client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(context.TODO(), claim.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		// make sure claim did bind
		err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, t.Client, claim.Namespace, claim.Name, framework.Poll, t.Timeouts.ClaimProvision)
		framework.ExpectNoError(err)

		pv, err := t.Client.CoreV1().PersistentVolumes().Get(context.TODO(), claim.Spec.VolumeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
		pvs = append(pvs, pv)
	}
	framework.ExpectEqual(len(pvs), len(createdClaims))
	return pvs, node
}

// RunInPodWithVolume runs a command in a pod with given claim mounted to /mnt directory.
// It starts, checks, collects output and stops it.
func RunInPodWithVolume(c clientset.Interface, t *framework.TimeoutContext, ns, claimName, podName, command string, node e2epod.NodeSelection) *v1.Pod {
	pod := StartInPodWithVolume(c, ns, claimName, podName, command, node)
	defer StopPod(c, pod)
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespaceTimeout(c, pod.Name, pod.Namespace, t.PodStartSlow))
	// get the latest status of the pod
	pod, err := c.CoreV1().Pods(pod.Namespace).Get(context.TODO(), pod.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)
	return pod
}

// StartInPodWithVolume starts a command in a pod with given claim mounted to /mnt directory
// The caller is responsible for checking the pod and deleting it.
func StartInPodWithVolume(c clientset.Interface, ns, claimName, podName, command string, node e2epod.NodeSelection) *v1.Pod {
	return StartInPodWithVolumeSource(c, v1.VolumeSource{
		PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
			ClaimName: claimName,
		},
	}, ns, podName, command, node)
}

// StartInPodWithVolumeSource starts a command in a pod with given volume mounted to /mnt directory
// The caller is responsible for checking the pod and deleting it.
func StartInPodWithVolumeSource(c clientset.Interface, volSrc v1.VolumeSource, ns, podName, command string, node e2epod.NodeSelection) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: podName + "-",
			Labels: map[string]string{
				"app": podName,
			},
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    "volume-tester",
					Image:   e2epod.GetDefaultTestImage(),
					Command: e2epod.GenerateScriptCmd(command),
					VolumeMounts: []v1.VolumeMount{
						{
							Name:      "my-volume",
							MountPath: "/mnt/test",
						},
					},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
			Volumes: []v1.Volume{
				{
					Name:         "my-volume",
					VolumeSource: volSrc,
				},
			},
		},
	}

	e2epod.SetNodeSelection(&pod.Spec, node)
	pod, err := c.CoreV1().Pods(ns).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	return pod
}

// StopPod first tries to log the output of the pod's container, then deletes the pod and
// waits for that to succeed.
func StopPod(c clientset.Interface, pod *v1.Pod) {
	if pod == nil {
		return
	}
	body, err := c.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &v1.PodLogOptions{}).Do(context.TODO()).Raw()
	if err != nil {
		framework.Logf("Error getting logs for pod %s: %v", pod.Name, err)
	} else {
		framework.Logf("Pod %s has the following logs: %s", pod.Name, body)
	}
	e2epod.DeletePodWithWait(c, pod)
}

// StopPodAndDependents first tries to log the output of the pod's container,
// then deletes the pod and waits for that to succeed. Also waits for all owned
// resources to be deleted.
func StopPodAndDependents(c clientset.Interface, timeouts *framework.TimeoutContext, pod *v1.Pod) {
	if pod == nil {
		return
	}
	body, err := c.CoreV1().Pods(pod.Namespace).GetLogs(pod.Name, &v1.PodLogOptions{}).Do(context.TODO()).Raw()
	if err != nil {
		framework.Logf("Error getting logs for pod %s: %v", pod.Name, err)
	} else {
		framework.Logf("Pod %s has the following logs: %s", pod.Name, body)
	}

	// We must wait explicitly for removal of the generic ephemeral volume PVs.
	// For that we must find them first...
	pvs, err := c.CoreV1().PersistentVolumes().List(context.TODO(), metav1.ListOptions{})
	framework.ExpectNoError(err, "list PVs")
	var podPVs []v1.PersistentVolume
	for _, pv := range pvs.Items {
		if pv.Spec.ClaimRef == nil ||
			pv.Spec.ClaimRef.Namespace != pod.Namespace {
			continue
		}
		pvc, err := c.CoreV1().PersistentVolumeClaims(pod.Namespace).Get(context.TODO(), pv.Spec.ClaimRef.Name, metav1.GetOptions{})
		if err != nil && apierrors.IsNotFound(err) {
			// Must have been some unrelated PV, otherwise the PVC should exist.
			continue
		}
		framework.ExpectNoError(err, "get PVC")
		if pv.Spec.ClaimRef.UID == pvc.UID && metav1.IsControlledBy(pvc, pod) {
			podPVs = append(podPVs, pv)
		}
	}

	framework.Logf("Deleting pod %q in namespace %q", pod.Name, pod.Namespace)
	deletionPolicy := metav1.DeletePropagationForeground
	err = c.CoreV1().Pods(pod.Namespace).Delete(context.TODO(), pod.Name,
		metav1.DeleteOptions{
			// If the pod is the owner of some resources (like ephemeral inline volumes),
			// then we want to be sure that those are also gone before we return.
			// Blocking pod deletion via metav1.DeletePropagationForeground achieves that.
			PropagationPolicy: &deletionPolicy,
		})
	if err != nil {
		if apierrors.IsNotFound(err) {
			return // assume pod was already deleted
		}
		framework.Logf("pod Delete API error: %v", err)
	}
	framework.Logf("Wait up to %v for pod %q to be fully deleted", timeouts.PodDelete, pod.Name)
	e2epod.WaitForPodNotFoundInNamespace(c, pod.Name, pod.Namespace, timeouts.PodDelete)
	if len(podPVs) > 0 {
		for _, pv := range podPVs {
			// As with CSI inline volumes, we use the pod delete timeout here because conceptually
			// the volume deletion needs to be that fast (whatever "that" is).
			framework.Logf("Wait up to %v for pod PV %s to be fully deleted", timeouts.PodDelete, pv.Name)
			e2epv.WaitForPersistentVolumeDeleted(c, pv.Name, 5*time.Second, timeouts.PodDelete)
		}
	}
}

func verifyPVCsPending(client clientset.Interface, pvcs []*v1.PersistentVolumeClaim) {
	for _, claim := range pvcs {
		// Get new copy of the claim
		claim, err := client.CoreV1().PersistentVolumeClaims(claim.Namespace).Get(context.TODO(), claim.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(claim.Status.Phase, v1.ClaimPending)
	}
}

func prepareSnapshotDataSourceForProvisioning(
	f *framework.Framework,
	config e2evolume.TestConfig,
	perTestConfig *storageframework.PerTestConfig,
	pattern storageframework.TestPattern,
	client clientset.Interface,
	dynamicClient dynamic.Interface,
	initClaim *v1.PersistentVolumeClaim,
	class *storagev1.StorageClass,
	sDriver storageframework.SnapshottableTestDriver,
	mode v1.PersistentVolumeMode,
	injectContent string,
) (*v1.TypedLocalObjectReference, func()) {
	_, clearComputedStorageClass := SetupStorageClass(client, class)

	if initClaim.ResourceVersion != "" {
		ginkgo.By("Skipping creation of PVC, it already exists")
	} else {
		ginkgo.By("[Initialize dataSource]creating a initClaim")
		updatedClaim, err := client.CoreV1().PersistentVolumeClaims(initClaim.Namespace).Create(context.TODO(), initClaim, metav1.CreateOptions{})
		if apierrors.IsAlreadyExists(err) {
			err = nil
		}
		framework.ExpectNoError(err)
		initClaim = updatedClaim
	}

	// write namespace to the /mnt/test (= the volume).
	tests := []e2evolume.Test{
		{
			Volume:          *storageutils.CreateVolumeSource(initClaim.Name, false /* readOnly */),
			Mode:            mode,
			File:            "index.html",
			ExpectedContent: injectContent,
		},
	}
	e2evolume.InjectContent(f, config, nil, "", tests)

	parameters := map[string]string{}
	snapshotResource := storageframework.CreateSnapshotResource(sDriver, perTestConfig, pattern, initClaim.GetName(), initClaim.GetNamespace(), f.Timeouts, parameters)
	group := "snapshot.storage.k8s.io"
	dataSourceRef := &v1.TypedLocalObjectReference{
		APIGroup: &group,
		Kind:     "VolumeSnapshot",
		Name:     snapshotResource.Vs.GetName(),
	}

	cleanupFunc := func() {
		framework.Logf("deleting initClaim %q/%q", initClaim.Namespace, initClaim.Name)
		err := client.CoreV1().PersistentVolumeClaims(initClaim.Namespace).Delete(context.TODO(), initClaim.Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting initClaim %q. Error: %v", initClaim.Name, err)
		}

		err = snapshotResource.CleanupResource(f.Timeouts)
		framework.ExpectNoError(err)

		clearComputedStorageClass()

	}

	return dataSourceRef, cleanupFunc
}

func preparePVCDataSourceForProvisioning(
	f *framework.Framework,
	config e2evolume.TestConfig,
	client clientset.Interface,
	source *v1.PersistentVolumeClaim,
	class *storagev1.StorageClass,
	mode v1.PersistentVolumeMode,
	injectContent string,
) (*v1.TypedLocalObjectReference, func()) {
	_, clearComputedStorageClass := SetupStorageClass(client, class)

	if source.ResourceVersion != "" {
		ginkgo.By("Skipping creation of PVC, it already exists")
	} else {
		ginkgo.By("[Initialize dataSource]creating a source PVC")
		var err error
		source, err = client.CoreV1().PersistentVolumeClaims(source.Namespace).Create(context.TODO(), source, metav1.CreateOptions{})
		framework.ExpectNoError(err)
	}

	tests := []e2evolume.Test{
		{
			Volume:          *storageutils.CreateVolumeSource(source.Name, false /* readOnly */),
			Mode:            mode,
			File:            "index.html",
			ExpectedContent: injectContent,
		},
	}
	e2evolume.InjectContent(f, config, nil, "", tests)

	dataSourceRef := &v1.TypedLocalObjectReference{
		Kind: "PersistentVolumeClaim",
		Name: source.GetName(),
	}

	cleanupFunc := func() {
		framework.Logf("deleting source PVC %q/%q", source.Namespace, source.Name)
		err := client.CoreV1().PersistentVolumeClaims(source.Namespace).Delete(context.TODO(), source.Name, metav1.DeleteOptions{})
		if err != nil && !apierrors.IsNotFound(err) {
			framework.Failf("Error deleting source PVC %q. Error: %v", source.Name, err)
		}

		clearComputedStorageClass()
	}

	return dataSourceRef, cleanupFunc
}

// MultiplePVMountSingleNodeCheck checks that multiple PV pointing to the same underlying storage can be mounted simultaneously on a single node.
//
// Steps:
// - Start Pod1 using PVC1, PV1 (which points to a underlying volume v) on node N1.
// - Create PVC2, PV2 and prebind them. PV2 points to the same underlying volume v.
// - Start Pod2 using PVC2, PV2 (which points to a underlying volume v) on node N1.
func MultiplePVMountSingleNodeCheck(client clientset.Interface, timeouts *framework.TimeoutContext, claim *v1.PersistentVolumeClaim, node e2epod.NodeSelection) {
	pod1Config := e2epod.Config{
		NS:            claim.Namespace,
		NodeSelection: node,
		PVCs:          []*v1.PersistentVolumeClaim{claim},
	}
	pod1, err := e2epod.CreateSecPodWithNodeSelection(client, &pod1Config, timeouts.PodStart)
	framework.ExpectNoError(err)
	defer func() {
		ginkgo.By(fmt.Sprintf("Deleting Pod %s/%s", pod1.Namespace, pod1.Name))
		framework.ExpectNoError(e2epod.DeletePodWithWait(client, pod1))
	}()
	ginkgo.By(fmt.Sprintf("Created Pod %s/%s on node %s", pod1.Namespace, pod1.Name, pod1.Spec.NodeName))

	// Create new PV which points to the same underlying storage. Retain policy is used so that deletion of second PVC does not trigger the deletion of its bound PV and underlying storage.
	e2evolume, err := getBoundPV(client, claim)
	framework.ExpectNoError(err)
	pv2Config := e2epv.PersistentVolumeConfig{
		NamePrefix:       fmt.Sprintf("%s-", "pv"),
		StorageClassName: *claim.Spec.StorageClassName,
		PVSource:         e2evolume.Spec.PersistentVolumeSource,
		AccessModes:      e2evolume.Spec.AccessModes,
		VolumeMode:       e2evolume.Spec.VolumeMode,
		ReclaimPolicy:    v1.PersistentVolumeReclaimRetain,
	}

	pvc2Config := e2epv.PersistentVolumeClaimConfig{
		NamePrefix:       fmt.Sprintf("%s-", "pvc"),
		StorageClassName: &claim.Namespace,
		AccessModes:      e2evolume.Spec.AccessModes,
		VolumeMode:       e2evolume.Spec.VolumeMode,
	}

	pv2, pvc2, err := e2epv.CreatePVCPV(client, timeouts, pv2Config, pvc2Config, claim.Namespace, true)
	framework.ExpectNoError(err, "PVC, PV creation failed")
	framework.Logf("Created PVC %s/%s and PV %s in namespace %s", pvc2.Namespace, pvc2.Name, pv2.Name)

	pod2Config := e2epod.Config{
		NS:            pvc2.Namespace,
		NodeSelection: e2epod.NodeSelection{Name: pod1.Spec.NodeName},
		PVCs:          []*v1.PersistentVolumeClaim{pvc2},
	}
	pod2, err := e2epod.CreateSecPodWithNodeSelection(client, &pod2Config, timeouts.PodStart)
	framework.ExpectNoError(err)
	ginkgo.By(fmt.Sprintf("Created Pod %s/%s on node %s", pod2.Namespace, pod2.Name, pod2.Spec.NodeName))

	ginkgo.By(fmt.Sprintf("Deleting Pod %s/%s", pod2.Namespace, pod2.Name))
	framework.ExpectNoError(e2epod.DeletePodWithWait(client, pod2))

	err = e2epv.DeletePersistentVolumeClaim(client, pvc2.Name, pvc2.Namespace)
	framework.ExpectNoError(err, "Failed to delete PVC: %s/%s", pvc2.Namespace, pvc2.Name)

	err = e2epv.DeletePersistentVolume(client, pv2.Name)
	framework.ExpectNoError(err, "Failed to delete PV: %s", pv2.Name)
}
