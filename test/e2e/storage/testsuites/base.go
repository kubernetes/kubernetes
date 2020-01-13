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
	"flag"
	"fmt"
	"math"
	"regexp"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/pkg/errors"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-base/metrics/testutil"
	csitrans "k8s.io/csi-translation-lib"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/framework/metrics"
	"k8s.io/kubernetes/test/e2e/framework/podlogs"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/framework/volume"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

var (
	migratedPlugins *string
	minValidSize    = "1Ki"
	maxValidSize    = "10Ei"
)

func init() {
	migratedPlugins = flag.String("storage.migratedPlugins", "", "comma separated list of in-tree plugin names of form 'kubernetes.io/{pluginName}' migrated to CSI")
}

type opCounts map[string]int64

// TestSuite represents an interface for a set of tests which works with TestDriver
type TestSuite interface {
	// GetTestSuiteInfo returns the TestSuiteInfo for this TestSuite
	GetTestSuiteInfo() TestSuiteInfo
	// DefineTests defines tests of the testpattern for the driver.
	// Called inside a Ginkgo context that reflects the current driver and test pattern,
	// so the test suite can define tests directly with ginkgo.It.
	DefineTests(TestDriver, testpatterns.TestPattern)
	// SkipRedundantSuite will skip the test suite based on the given TestPattern and TestDriver
	SkipRedundantSuite(TestDriver, testpatterns.TestPattern)
}

// TestSuiteInfo represents a set of parameters for TestSuite
type TestSuiteInfo struct {
	Name               string                     // name of the TestSuite
	FeatureTag         string                     // featureTag for the TestSuite
	TestPatterns       []testpatterns.TestPattern // Slice of TestPattern for the TestSuite
	SupportedSizeRange volume.SizeRange           // Size range supported by the test suite
}

func getTestNameStr(suite TestSuite, pattern testpatterns.TestPattern) string {
	tsInfo := suite.GetTestSuiteInfo()
	return fmt.Sprintf("[Testpattern: %s]%s %s%s", pattern.Name, pattern.FeatureTag, tsInfo.Name, tsInfo.FeatureTag)
}

// DefineTestSuite defines tests for all testpatterns and all testSuites for a driver
func DefineTestSuite(driver TestDriver, tsInits []func() TestSuite) {
	for _, testSuiteInit := range tsInits {
		suite := testSuiteInit()
		for _, pattern := range suite.GetTestSuiteInfo().TestPatterns {
			p := pattern
			ginkgo.Context(getTestNameStr(suite, p), func() {
				ginkgo.BeforeEach(func() {
					// Skip unsupported tests to avoid unnecessary resource initialization
					suite.SkipRedundantSuite(driver, p)
					skipUnsupportedTest(driver, p)
				})
				suite.DefineTests(driver, p)
			})
		}
	}
}

// skipUnsupportedTest will skip tests if the combination of driver,  and testpattern
// is not suitable to be tested.
// Whether it needs to be skipped is checked by following steps:
// 1. Check if Whether SnapshotType is supported by driver from its interface
// 2. Check if Whether volType is supported by driver from its interface
// 3. Check if fsType is supported
// 4. Check with driver specific logic
//
// Test suites can also skip tests inside their own DefineTests function or in
// individual tests.
func skipUnsupportedTest(driver TestDriver, pattern testpatterns.TestPattern) {
	dInfo := driver.GetDriverInfo()
	var isSupported bool

	// 1. Check if Whether SnapshotType is supported by driver from its interface
	// if isSupported, we still execute the driver and suite tests
	if len(pattern.SnapshotType) > 0 {
		switch pattern.SnapshotType {
		case testpatterns.DynamicCreatedSnapshot:
			_, isSupported = driver.(SnapshottableTestDriver)
		default:
			isSupported = false
		}
		if !isSupported {
			e2eskipper.Skipf("Driver %s doesn't support snapshot type %v -- skipping", dInfo.Name, pattern.SnapshotType)
		}
	} else {
		// 2. Check if Whether volType is supported by driver from its interface
		switch pattern.VolType {
		case testpatterns.InlineVolume:
			_, isSupported = driver.(InlineVolumeTestDriver)
		case testpatterns.PreprovisionedPV:
			_, isSupported = driver.(PreprovisionedPVTestDriver)
		case testpatterns.DynamicPV:
			_, isSupported = driver.(DynamicPVTestDriver)
		case testpatterns.CSIInlineVolume:
			_, isSupported = driver.(EphemeralTestDriver)
		default:
			isSupported = false
		}

		if !isSupported {
			e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
		}

		// 3. Check if fsType is supported
		if !dInfo.SupportedFsType.Has(pattern.FsType) {
			e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.FsType)
		}
		if pattern.FsType == "xfs" && framework.NodeOSDistroIs("gci", "cos", "windows") {
			e2eskipper.Skipf("Distro doesn't support xfs -- skipping")
		}
		if pattern.FsType == "ntfs" && !framework.NodeOSDistroIs("windows") {
			e2eskipper.Skipf("Distro %s doesn't support ntfs -- skipping", framework.TestContext.NodeOSDistro)
		}
	}

	// 4. Check with driver specific logic
	driver.SkipUnsupportedTest(pattern)
}

// VolumeResource is a generic implementation of TestResource that wil be able to
// be used in most of TestSuites.
// See volume_io.go or volumes.go in test/e2e/storage/testsuites/ for how to use this resource.
// Also, see subpath.go in the same directory for how to extend and use it.
type VolumeResource struct {
	Config    *PerTestConfig
	Pattern   testpatterns.TestPattern
	VolSource *v1.VolumeSource
	Pvc       *v1.PersistentVolumeClaim
	Pv        *v1.PersistentVolume
	Sc        *storagev1.StorageClass

	Volume TestVolume
}

// CreateVolumeResource constructs a VolumeResource for the current test. It knows how to deal with
// different test pattern volume types.
func CreateVolumeResource(driver TestDriver, config *PerTestConfig, pattern testpatterns.TestPattern, testVolumeSizeRange volume.SizeRange) *VolumeResource {
	r := VolumeResource{
		Config:  config,
		Pattern: pattern,
	}
	dInfo := driver.GetDriverInfo()
	f := config.Framework
	cs := f.ClientSet

	// Create volume for pre-provisioned volume tests
	r.Volume = CreateVolume(driver, config, pattern.VolType)

	switch pattern.VolType {
	case testpatterns.InlineVolume:
		framework.Logf("Creating resource for inline volume")
		if iDriver, ok := driver.(InlineVolumeTestDriver); ok {
			r.VolSource = iDriver.GetVolumeSource(false, pattern.FsType, r.Volume)
		}
	case testpatterns.PreprovisionedPV:
		framework.Logf("Creating resource for pre-provisioned PV")
		if pDriver, ok := driver.(PreprovisionedPVTestDriver); ok {
			pvSource, volumeNodeAffinity := pDriver.GetPersistentVolumeSource(false, pattern.FsType, r.Volume)
			if pvSource != nil {
				r.Pv, r.Pvc = createPVCPV(f, dInfo.Name, pvSource, volumeNodeAffinity, pattern.VolMode, dInfo.RequiredAccessModes)
				r.VolSource = createVolumeSource(r.Pvc.Name, false /* readOnly */)
			}
		}
	case testpatterns.DynamicPV:
		framework.Logf("Creating resource for dynamic PV")
		if dDriver, ok := driver.(DynamicPVTestDriver); ok {
			var err error
			driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
			claimSize, err := getSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
			framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, driverVolumeSizeRange)
			framework.Logf("Using claimSize:%s, test suite supported size:%v, driver(%s) supported size:%v ", claimSize, testVolumeSizeRange, dDriver.GetDriverInfo().Name, testVolumeSizeRange)
			r.Sc = dDriver.GetDynamicProvisionStorageClass(r.Config, pattern.FsType)

			if pattern.BindingMode != "" {
				r.Sc.VolumeBindingMode = &pattern.BindingMode
			}
			if pattern.AllowExpansion != false {
				r.Sc.AllowVolumeExpansion = &pattern.AllowExpansion
			}

			ginkgo.By("creating a StorageClass " + r.Sc.Name)

			r.Sc, err = cs.StorageV1().StorageClasses().Create(r.Sc)
			framework.ExpectNoError(err)

			if r.Sc != nil {
				r.Pv, r.Pvc = createPVCPVFromDynamicProvisionSC(
					f, dInfo.Name, claimSize, r.Sc, pattern.VolMode, dInfo.RequiredAccessModes)
				r.VolSource = createVolumeSource(r.Pvc.Name, false /* readOnly */)
			}
		}
	case testpatterns.CSIInlineVolume:
		framework.Logf("Creating resource for CSI ephemeral inline volume")
		if eDriver, ok := driver.(EphemeralTestDriver); ok {
			attributes, _, _ := eDriver.GetVolume(config, 0)
			r.VolSource = &v1.VolumeSource{
				CSI: &v1.CSIVolumeSource{
					Driver:           eDriver.GetCSIDriverName(config),
					VolumeAttributes: attributes,
				},
			}
		}
	default:
		framework.Failf("VolumeResource doesn't support: %s", pattern.VolType)
	}

	if r.VolSource == nil {
		e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
	}

	return &r
}

func createVolumeSource(pvcName string, readOnly bool) *v1.VolumeSource {
	return &v1.VolumeSource{
		PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
			ClaimName: pvcName,
			ReadOnly:  readOnly,
		},
	}

}

// CleanupResource cleans up VolumeResource
func (r *VolumeResource) CleanupResource() error {
	f := r.Config.Framework
	var cleanUpErrs []error
	if r.Pvc != nil || r.Pv != nil {
		switch r.Pattern.VolType {
		case testpatterns.PreprovisionedPV:
			ginkgo.By("Deleting pv and pvc")
			if errs := e2epv.PVPVCCleanup(f.ClientSet, f.Namespace.Name, r.Pv, r.Pvc); len(errs) != 0 {
				framework.Failf("Failed to delete PVC or PV: %v", utilerrors.NewAggregate(errs))
			}
		case testpatterns.DynamicPV:
			ginkgo.By("Deleting pvc")
			// We only delete the PVC so that PV (and disk) can be cleaned up by dynamic provisioner
			if r.Pv != nil && r.Pv.Spec.PersistentVolumeReclaimPolicy != v1.PersistentVolumeReclaimDelete {
				framework.Failf("Test framework does not currently support Dynamically Provisioned Persistent Volume %v specified with reclaim policy that isnt %v",
					r.Pv.Name, v1.PersistentVolumeReclaimDelete)
			}
			if r.Pvc != nil {
				err := e2epv.DeletePersistentVolumeClaim(f.ClientSet, r.Pvc.Name, f.Namespace.Name)
				if err != nil {
					cleanUpErrs = append(cleanUpErrs, errors.Wrapf(err, "Failed to delete PVC %v", r.Pvc.Name))
				}
				if r.Pv != nil {
					err = framework.WaitForPersistentVolumeDeleted(f.ClientSet, r.Pv.Name, 5*time.Second, 5*time.Minute)
					if err != nil {
						cleanUpErrs = append(cleanUpErrs, errors.Wrapf(err,
							"Persistent Volume %v not deleted by dynamic provisioner", r.Pv.Name))
					}
				}
			}
		default:
			framework.Failf("Found PVC (%v) or PV (%v) but not running Preprovisioned or Dynamic test pattern", r.Pvc, r.Pv)
		}
	}

	if r.Sc != nil {
		ginkgo.By("Deleting sc")
		if err := deleteStorageClass(f.ClientSet, r.Sc.Name); err != nil {
			cleanUpErrs = append(cleanUpErrs, errors.Wrapf(err, "Failed to delete StorageClass %v", r.Sc.Name))
		}
	}

	// Cleanup volume for pre-provisioned volume tests
	if r.Volume != nil {
		if err := tryFunc(r.Volume.DeleteVolume); err != nil {
			cleanUpErrs = append(cleanUpErrs, errors.Wrap(err, "Failed to delete Volume"))
		}
	}
	return utilerrors.NewAggregate(cleanUpErrs)
}

func createPVCPV(
	f *framework.Framework,
	name string,
	pvSource *v1.PersistentVolumeSource,
	volumeNodeAffinity *v1.VolumeNodeAffinity,
	volMode v1.PersistentVolumeMode,
	accessModes []v1.PersistentVolumeAccessMode,
) (*v1.PersistentVolume, *v1.PersistentVolumeClaim) {
	pvConfig := e2epv.PersistentVolumeConfig{
		NamePrefix:       fmt.Sprintf("%s-", name),
		StorageClassName: f.Namespace.Name,
		PVSource:         *pvSource,
		NodeAffinity:     volumeNodeAffinity,
		AccessModes:      accessModes,
	}

	pvcConfig := e2epv.PersistentVolumeClaimConfig{
		StorageClassName: &f.Namespace.Name,
		AccessModes:      accessModes,
	}

	if volMode != "" {
		pvConfig.VolumeMode = &volMode
		pvcConfig.VolumeMode = &volMode
	}

	framework.Logf("Creating PVC and PV")
	pv, pvc, err := e2epv.CreatePVCPV(f.ClientSet, pvConfig, pvcConfig, f.Namespace.Name, false)
	framework.ExpectNoError(err, "PVC, PV creation failed")

	err = e2epv.WaitOnPVandPVC(f.ClientSet, f.Namespace.Name, pv, pvc)
	framework.ExpectNoError(err, "PVC, PV failed to bind")

	return pv, pvc
}

func createPVCPVFromDynamicProvisionSC(
	f *framework.Framework,
	name string,
	claimSize string,
	sc *storagev1.StorageClass,
	volMode v1.PersistentVolumeMode,
	accessModes []v1.PersistentVolumeAccessMode,
) (*v1.PersistentVolume, *v1.PersistentVolumeClaim) {
	cs := f.ClientSet
	ns := f.Namespace.Name

	ginkgo.By("creating a claim")
	pvcCfg := e2epv.PersistentVolumeClaimConfig{
		NamePrefix:       name,
		ClaimSize:        claimSize,
		StorageClassName: &(sc.Name),
		AccessModes:      accessModes,
		VolumeMode:       &volMode,
	}

	pvc := e2epv.MakePersistentVolumeClaim(pvcCfg, ns)

	var err error
	pvc, err = e2epv.CreatePVC(cs, ns, pvc)
	framework.ExpectNoError(err)

	if !isDelayedBinding(sc) {
		err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, cs, pvc.Namespace, pvc.Name, framework.Poll, framework.ClaimProvisionTimeout)
		framework.ExpectNoError(err)
	}

	pvc, err = cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
	framework.ExpectNoError(err)

	var pv *v1.PersistentVolume
	if !isDelayedBinding(sc) {
		pv, err = cs.CoreV1().PersistentVolumes().Get(pvc.Spec.VolumeName, metav1.GetOptions{})
		framework.ExpectNoError(err)
	}

	return pv, pvc
}

func isDelayedBinding(sc *storagev1.StorageClass) bool {
	if sc.VolumeBindingMode != nil {
		return *sc.VolumeBindingMode == storagev1.VolumeBindingWaitForFirstConsumer
	}
	return false
}

// deleteStorageClass deletes the passed in StorageClass and catches errors other than "Not Found"
func deleteStorageClass(cs clientset.Interface, className string) error {
	err := cs.StorageV1().StorageClasses().Delete(className, nil)
	if err != nil && !apierrors.IsNotFound(err) {
		return err
	}
	return nil
}

// convertTestConfig returns a framework test config with the
// parameters specified for the testsuite or (if available) the
// dynamically created config for the volume server.
//
// This is done because TestConfig is the public API for
// the testsuites package whereas volume.TestConfig is merely
// an implementation detail. It contains fields that have no effect,
// which makes it unsuitable for use in the testsuits public API.
func convertTestConfig(in *PerTestConfig) volume.TestConfig {
	if in.ServerConfig != nil {
		return *in.ServerConfig
	}

	return volume.TestConfig{
		Namespace:      in.Framework.Namespace.Name,
		Prefix:         in.Prefix,
		ClientNodeName: in.ClientNodeName,
		NodeSelector:   in.ClientNodeSelector,
	}
}

// getSizeRangesIntersection takes two instances of storage size ranges and determines the
// intersection of the intervals (if it exists) and return the minimum of the intersection
// to be used as the claim size for the test.
// if value not set, that means there's no minimum or maximum size limitation and we set default size for it.
func getSizeRangesIntersection(first volume.SizeRange, second volume.SizeRange) (string, error) {
	var firstMin, firstMax, secondMin, secondMax resource.Quantity
	var err error

	//if SizeRange is not set, assign a minimum or maximum size
	if len(first.Min) == 0 {
		first.Min = minValidSize
	}
	if len(first.Max) == 0 {
		first.Max = maxValidSize
	}
	if len(second.Min) == 0 {
		second.Min = minValidSize
	}
	if len(second.Max) == 0 {
		second.Max = maxValidSize
	}

	if firstMin, err = resource.ParseQuantity(first.Min); err != nil {
		return "", err
	}
	if firstMax, err = resource.ParseQuantity(first.Max); err != nil {
		return "", err
	}
	if secondMin, err = resource.ParseQuantity(second.Min); err != nil {
		return "", err
	}
	if secondMax, err = resource.ParseQuantity(second.Max); err != nil {
		return "", err
	}

	interSectionStart := math.Max(float64(firstMin.Value()), float64(secondMin.Value()))
	intersectionEnd := math.Min(float64(firstMax.Value()), float64(secondMax.Value()))

	// the minimum of the intersection shall be returned as the claim size
	var intersectionMin resource.Quantity

	if intersectionEnd-interSectionStart >= 0 { //have intersection
		intersectionMin = *resource.NewQuantity(int64(interSectionStart), "BinarySI") //convert value to BinarySI format. E.g. 5Gi
		// return the minimum of the intersection as the claim size
		return intersectionMin.String(), nil
	}
	return "", fmt.Errorf("intersection of size ranges %+v, %+v is null", first, second)
}

func getSnapshot(claimName string, ns, snapshotClassName string) *unstructured.Unstructured {
	snapshot := &unstructured.Unstructured{
		Object: map[string]interface{}{
			"kind":       "VolumeSnapshot",
			"apiVersion": snapshotAPIVersion,
			"metadata": map[string]interface{}{
				"generateName": "snapshot-",
				"namespace":    ns,
			},
			"spec": map[string]interface{}{
				"volumeSnapshotClassName": snapshotClassName,
				"source": map[string]interface{}{
					"persistentVolumeClaimName": claimName,
				},
			},
		},
	}

	return snapshot
}

// StartPodLogs begins capturing log output and events from current
// and future pods running in the namespace of the framework. That
// ends when the returned cleanup function is called.
//
// The output goes to log files (when using --report-dir, as in the
// CI) or the output stream (otherwise).
func StartPodLogs(f *framework.Framework) func() {
	ctx, cancel := context.WithCancel(context.Background())
	cs := f.ClientSet
	ns := f.Namespace

	to := podlogs.LogOutput{
		StatusWriter: ginkgo.GinkgoWriter,
	}
	if framework.TestContext.ReportDir == "" {
		to.LogWriter = ginkgo.GinkgoWriter
	} else {
		test := ginkgo.CurrentGinkgoTestDescription()
		reg := regexp.MustCompile("[^a-zA-Z0-9_-]+")
		// We end the prefix with a slash to ensure that all logs
		// end up in a directory named after the current test.
		//
		// TODO: use a deeper directory hierarchy once gubernator
		// supports that (https://github.com/kubernetes/test-infra/issues/10289).
		to.LogPathPrefix = framework.TestContext.ReportDir + "/" +
			reg.ReplaceAllString(test.FullTestText, "_") + "/"
	}
	podlogs.CopyAllLogs(ctx, cs, ns.Name, to)

	// pod events are something that the framework already collects itself
	// after a failed test. Logging them live is only useful for interactive
	// debugging, not when we collect reports.
	if framework.TestContext.ReportDir == "" {
		podlogs.WatchPods(ctx, cs, ns.Name, ginkgo.GinkgoWriter)
	}

	return cancel
}

func getVolumeOpsFromMetricsForPlugin(ms testutil.Metrics, pluginName string) opCounts {
	totOps := opCounts{}

	for method, samples := range ms {
		switch method {
		case "storage_operation_status_count":
			for _, sample := range samples {
				plugin := string(sample.Metric["volume_plugin"])
				if pluginName != plugin {
					continue
				}
				opName := string(sample.Metric["operation_name"])
				if opName == "verify_controller_attached_volume" {
					// We ignore verify_controller_attached_volume because it does not call into
					// the plugin. It only watches Node API and updates Actual State of World cache
					continue
				}
				totOps[opName] = totOps[opName] + int64(sample.Value)
			}
		}
	}
	return totOps
}

func getVolumeOpCounts(c clientset.Interface, pluginName string) opCounts {
	if !framework.ProviderIs("gce", "gke", "aws") {
		return opCounts{}
	}

	nodeLimit := 25

	metricsGrabber, err := metrics.NewMetricsGrabber(c, nil, true, false, true, false, false)

	if err != nil {
		framework.ExpectNoError(err, "Error creating metrics grabber: %v", err)
	}

	if !metricsGrabber.HasRegisteredMaster() {
		framework.Logf("Warning: Environment does not support getting controller-manager metrics")
		return opCounts{}
	}

	controllerMetrics, err := metricsGrabber.GrabFromControllerManager()
	framework.ExpectNoError(err, "Error getting c-m metrics : %v", err)
	totOps := getVolumeOpsFromMetricsForPlugin(testutil.Metrics(controllerMetrics), pluginName)

	framework.Logf("Node name not specified for getVolumeOpCounts, falling back to listing nodes from API Server")
	nodes, err := c.CoreV1().Nodes().List(metav1.ListOptions{})
	framework.ExpectNoError(err, "Error listing nodes: %v", err)
	if len(nodes.Items) <= nodeLimit {
		// For large clusters with > nodeLimit nodes it is too time consuming to
		// gather metrics from all nodes. We just ignore the node metrics
		// for those clusters
		for _, node := range nodes.Items {
			nodeMetrics, err := metricsGrabber.GrabFromKubelet(node.GetName())
			framework.ExpectNoError(err, "Error getting Kubelet %v metrics: %v", node.GetName(), err)
			totOps = addOpCounts(totOps, getVolumeOpsFromMetricsForPlugin(testutil.Metrics(nodeMetrics), pluginName))
		}
	} else {
		framework.Logf("Skipping operation metrics gathering from nodes in getVolumeOpCounts, greater than %v nodes", nodeLimit)
	}

	return totOps
}

func addOpCounts(o1 opCounts, o2 opCounts) opCounts {
	totOps := opCounts{}
	seen := sets.NewString()
	for op, count := range o1 {
		seen.Insert(op)
		totOps[op] = totOps[op] + count + o2[op]
	}
	for op, count := range o2 {
		if !seen.Has(op) {
			totOps[op] = totOps[op] + count
		}
	}
	return totOps
}

func getMigrationVolumeOpCounts(cs clientset.Interface, pluginName string) (opCounts, opCounts) {
	if len(pluginName) > 0 {
		var migratedOps opCounts
		l := csitrans.New()
		csiName, err := l.GetCSINameFromInTreeName(pluginName)
		if err != nil {
			framework.Logf("Could not find CSI Name for in-tree plugin %v", pluginName)
			migratedOps = opCounts{}
		} else {
			csiName = "kubernetes.io/csi:" + csiName
			migratedOps = getVolumeOpCounts(cs, csiName)
		}
		return getVolumeOpCounts(cs, pluginName), migratedOps
	}
	// Not an in-tree driver
	framework.Logf("Test running for native CSI Driver, not checking metrics")
	return opCounts{}, opCounts{}
}

func validateMigrationVolumeOpCounts(cs clientset.Interface, pluginName string, oldInTreeOps, oldMigratedOps opCounts) {
	if len(pluginName) == 0 {
		// This is a native CSI Driver and we don't check ops
		return
	}

	if sets.NewString(strings.Split(*migratedPlugins, ",")...).Has(pluginName) {
		// If this plugin is migrated based on the test flag storage.migratedPlugins
		newInTreeOps, _ := getMigrationVolumeOpCounts(cs, pluginName)

		for op, count := range newInTreeOps {
			if count != oldInTreeOps[op] {
				framework.Failf("In-tree plugin %v migrated to CSI Driver, however found %v %v metrics for in-tree plugin", pluginName, count-oldInTreeOps[op], op)
			}
		}
		// We don't check for migrated metrics because some negative test cases
		// may not do any volume operations and therefore not emit any metrics
	} else {
		// In-tree plugin is not migrated
		framework.Logf("In-tree plugin %v is not migrated, not validating any metrics", pluginName)

		// We don't check in-tree plugin metrics because some negative test
		// cases may not do any volume operations and therefore not emit any
		// metrics

		// We don't check counts for the Migrated version of the driver because
		// if tests are running in parallel a test could be using the CSI Driver
		// natively and increase the metrics count

		// TODO(dyzz): Add a dimension to OperationGenerator metrics for
		// "migrated"->true/false so that we can disambiguate migrated metrics
		// and native CSI Driver metrics. This way we can check the counts for
		// migrated version of the driver for stronger negative test case
		// guarantees (as well as more informative metrics).
	}
}

// Skip skipVolTypes patterns if the driver supports dynamic provisioning
func skipVolTypePatterns(pattern testpatterns.TestPattern, driver TestDriver, skipVolTypes map[testpatterns.TestVolType]bool) {
	_, supportsProvisioning := driver.(DynamicPVTestDriver)
	if supportsProvisioning && skipVolTypes[pattern.VolType] {
		e2eskipper.Skipf("Driver supports dynamic provisioning, skipping %s pattern", pattern.VolType)
	}
}

func tryFunc(f func()) error {
	var err error
	if f == nil {
		return nil
	}
	defer func() {
		if recoverError := recover(); recoverError != nil {
			err = fmt.Errorf("%v", recoverError)
		}
	}()
	f()
	return err
}
