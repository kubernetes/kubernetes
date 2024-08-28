/*
Copyright 2019 The Kubernetes Authors.

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
	"regexp"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/component-helpers/storage/ephemeral"
	migrationplugins "k8s.io/csi-translation-lib/plugins" // volume plugin names are exported nicely there
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

type volumeLimitsTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

const (
	// The test uses generic pod startup / PV deletion timeouts. As it creates
	// much more volumes at once, these timeouts are multiplied by this number.
	// Using real nr. of volumes (e.g. 128 on GCE) would be really too much.
	testSlowMultiplier = 10

	// How long to wait until CSINode gets attach limit from installed CSI driver.
	csiNodeInfoTimeout = 2 * time.Minute
)

var _ storageframework.TestSuite = &volumeLimitsTestSuite{}

// InitCustomVolumeLimitsTestSuite returns volumeLimitsTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomVolumeLimitsTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &volumeLimitsTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "volumeLimits",
			TestPatterns: patterns,
		},
	}
}

// InitVolumeLimitsTestSuite returns volumeLimitsTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitVolumeLimitsTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.FsVolModeDynamicPV,
		storageframework.DefaultFsGenericEphemeralVolume,
	}
	return InitCustomVolumeLimitsTestSuite(patterns)
}

func (t *volumeLimitsTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return t.tsInfo
}

func (t *volumeLimitsTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	if pattern.VolType != storageframework.DynamicPV {
		e2eskipper.Skipf("Suite %q does not support %v", t.tsInfo.Name, pattern.VolType)
	}
	dInfo := driver.GetDriverInfo()
	if !dInfo.Capabilities[storageframework.CapVolumeLimits] {
		e2eskipper.Skipf("Driver %s does not support volume limits", dInfo.Name)
	}
}

func (t *volumeLimitsTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	type local struct {
		config *storageframework.PerTestConfig

		cs clientset.Interface
		ns *v1.Namespace
		// VolumeResource contains pv, pvc, sc, etc. of the first pod created
		resource *storageframework.VolumeResource

		// All created PVCs
		pvcNames []string

		// All created Pods
		podNames []string

		// All created PVs, incl. the one in resource
		pvNames sets.Set[string]
	}
	var (
		l local

		dDriver storageframework.DynamicPVTestDriver
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("volumelimits", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.BeforeEach(func() {
		dDriver = driver.(storageframework.DynamicPVTestDriver)
	})

	// This checks that CSIMaxVolumeLimitChecker works as expected.
	// A randomly chosen node should be able to handle as many CSI volumes as
	// it claims to handle in CSINode.Spec.Drivers[x].Allocatable.
	// The test uses one single pod with a lot of volumes to work around any
	// max pod limit on a node.
	// And one extra pod with a CSI volume should get Pending with a condition
	// that says it's unschedulable because of volume limit.
	// BEWARE: the test may create lot of volumes and it's really slow.
	f.It("should support volume limits", f.WithSerial(), func(ctx context.Context) {
		driverInfo := driver.GetDriverInfo()
		l.ns = f.Namespace
		l.cs = f.ClientSet

		l.config = driver.PrepareTest(ctx, f)

		ginkgo.By("Picking a node")
		// Some CSI drivers are deployed to a single node (e.g csi-hostpath),
		// so we use that node instead of picking a random one.
		nodeName := l.config.ClientNodeSelection.Name
		if nodeName == "" {
			node, err := e2enode.GetRandomReadySchedulableNode(ctx, f.ClientSet)
			framework.ExpectNoError(err)
			nodeName = node.Name
		}
		framework.Logf("Selected node %s", nodeName)

		ginkgo.By("Checking node limits")
		limit, err := getNodeLimits(ctx, l.cs, l.config, nodeName, dDriver)
		framework.ExpectNoError(err)

		framework.Logf("Node %s can handle %d volumes of driver %s", nodeName, limit, driverInfo.Name)
		// Create a storage class and generate a PVC. Do not instantiate the PVC yet, keep it for the last pod.
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
		claimSize, err := storageutils.GetSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
		framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, dDriver)

		l.resource = storageframework.CreateVolumeResource(ctx, driver, l.config, pattern, testVolumeSizeRange)
		ginkgo.DeferCleanup(l.resource.CleanupResource)
		ginkgo.DeferCleanup(cleanupTest, l.cs, l.ns.Name, l.podNames, l.pvcNames, l.pvNames, testSlowMultiplier*f.Timeouts.PVDelete)

		selection := e2epod.NodeSelection{Name: nodeName}

		if pattern.VolType == storageframework.GenericEphemeralVolume {
			// Create <limit> Pods.
			ginkgo.By(fmt.Sprintf("Creating %d Pod(s) with one volume each", limit))
			for i := 0; i < limit; i++ {
				pod := StartInPodWithVolumeSource(ctx, l.cs, *l.resource.VolSource, l.ns.Name, "volume-limits", e2epod.InfiniteSleepCommand, selection)
				l.podNames = append(l.podNames, pod.Name)
				l.pvcNames = append(l.pvcNames, ephemeral.VolumeClaimName(pod, &pod.Spec.Volumes[0]))
			}
		} else {
			// Create <limit> PVCs for one gigantic pod.
			var pvcs []*v1.PersistentVolumeClaim
			ginkgo.By(fmt.Sprintf("Creating %d PVC(s)", limit))
			for i := 0; i < limit; i++ {
				pvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
					ClaimSize:        claimSize,
					StorageClassName: &l.resource.Sc.Name,
				}, l.ns.Name)
				pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Create(ctx, pvc, metav1.CreateOptions{})
				framework.ExpectNoError(err)
				l.pvcNames = append(l.pvcNames, pvc.Name)
				pvcs = append(pvcs, pvc)
			}

			ginkgo.By("Creating pod to use all PVC(s)")
			podConfig := e2epod.Config{
				NS:            l.ns.Name,
				PVCs:          pvcs,
				SeLinuxLabel:  e2epv.SELinuxLabel,
				NodeSelection: selection,
			}
			pod, err := e2epod.MakeSecPod(&podConfig)
			framework.ExpectNoError(err)
			pod, err = l.cs.CoreV1().Pods(l.ns.Name).Create(ctx, pod, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			l.podNames = append(l.podNames, pod.Name)
		}

		ginkgo.By("Waiting for all PVCs to get Bound")
		l.pvNames = waitForAllPVCsBound(ctx, l.cs, testSlowMultiplier*f.Timeouts.PVBound, l.ns.Name, l.pvcNames)

		ginkgo.By("Waiting for the pod(s) running")
		for _, podName := range l.podNames {
			err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, l.cs, podName, l.ns.Name, testSlowMultiplier*f.Timeouts.PodStart)
			framework.ExpectNoError(err)
		}

		ginkgo.By("Creating an extra pod with one volume to exceed the limit")
		pod := StartInPodWithVolumeSource(ctx, l.cs, *l.resource.VolSource, l.ns.Name, "volume-limits-exceeded", e2epod.InfiniteSleepCommand, selection)
		l.podNames = append(l.podNames, pod.Name)

		ginkgo.By("Waiting for the pod to get unschedulable with the right message")
		err = e2epod.WaitForPodCondition(ctx, l.cs, l.ns.Name, pod.Name, "Unschedulable", f.Timeouts.PodStart, func(pod *v1.Pod) (bool, error) {
			if pod.Status.Phase == v1.PodPending {
				reg, err := regexp.Compile(`max.+volume.+count`)
				if err != nil {
					return false, err
				}
				for _, cond := range pod.Status.Conditions {
					matched := reg.MatchString(cond.Message)
					if cond.Type == v1.PodScheduled && cond.Status == v1.ConditionFalse && cond.Reason == "Unschedulable" && matched {
						return true, nil
					}
				}
			}
			if pod.Status.Phase != v1.PodPending {
				return true, fmt.Errorf("expected pod to be in phase Pending, but got phase: %v", pod.Status.Phase)
			}
			return false, nil
		})
		framework.ExpectNoError(err)
	})

	ginkgo.It("should verify that all csinodes have volume limits", func(ctx context.Context) {
		driverInfo := driver.GetDriverInfo()
		if !driverInfo.Capabilities[storageframework.CapVolumeLimits] {
			ginkgo.Skip(fmt.Sprintf("driver %s does not support volume limits", driverInfo.Name))
		}

		l.ns = f.Namespace
		l.cs = f.ClientSet

		l.config = driver.PrepareTest(ctx, f)

		nodeNames := []string{}
		if l.config.ClientNodeSelection.Name != "" {
			// Some CSI drivers are deployed to a single node (e.g csi-hostpath),
			// so we check that node instead of checking all of them
			nodeNames = append(nodeNames, l.config.ClientNodeSelection.Name)
		} else {
			nodeList, err := e2enode.GetReadySchedulableNodes(ctx, f.ClientSet)
			framework.ExpectNoError(err)
			for _, node := range nodeList.Items {
				nodeNames = append(nodeNames, node.Name)
			}
		}

		for _, nodeName := range nodeNames {
			ginkgo.By("Checking csinode limits")
			_, err := getNodeLimits(ctx, l.cs, l.config, nodeName, dDriver)
			if err != nil {
				framework.Failf("Expected volume limits to be set, error: %v", err)
			}
		}
	})
}

func cleanupTest(ctx context.Context, cs clientset.Interface, ns string, podNames, pvcNames []string, pvNames sets.Set[string], timeout time.Duration) error {
	var cleanupErrors []string
	for _, podName := range podNames {
		err := cs.CoreV1().Pods(ns).Delete(ctx, podName, metav1.DeleteOptions{})
		if err != nil {
			cleanupErrors = append(cleanupErrors, fmt.Sprintf("failed to delete pod %s: %s", podName, err))
		}
	}
	for _, pvcName := range pvcNames {
		err := cs.CoreV1().PersistentVolumeClaims(ns).Delete(ctx, pvcName, metav1.DeleteOptions{})
		if !apierrors.IsNotFound(err) {
			cleanupErrors = append(cleanupErrors, fmt.Sprintf("failed to delete PVC %s: %s", pvcName, err))
		}
	}
	// Wait for the PVs to be deleted. It includes also pod and PVC deletion because of PVC protection.
	// We use PVs to make sure that the test does not leave orphan PVs when a CSI driver is destroyed
	// just after the test ends.
	err := wait.PollUntilContextTimeout(ctx, 5*time.Second, timeout, false, func(ctx context.Context) (bool, error) {
		existing := 0
		for _, pvName := range pvNames.UnsortedList() {
			_, err := cs.CoreV1().PersistentVolumes().Get(ctx, pvName, metav1.GetOptions{})
			if err == nil {
				existing++
			} else {
				if apierrors.IsNotFound(err) {
					pvNames.Delete(pvName)
				} else {
					framework.Logf("Failed to get PV %s: %s", pvName, err)
				}
			}
		}
		if existing > 0 {
			framework.Logf("Waiting for %d PVs to be deleted", existing)
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		cleanupErrors = append(cleanupErrors, fmt.Sprintf("timed out waiting for PVs to be deleted: %s", err))
	}
	if len(cleanupErrors) != 0 {
		return fmt.Errorf("test cleanup failed: " + strings.Join(cleanupErrors, "; "))
	}
	return nil
}

// waitForAllPVCsBound waits until the given PVCs are all bound. It then returns the bound PVC names as a set.
func waitForAllPVCsBound(ctx context.Context, cs clientset.Interface, timeout time.Duration, ns string, pvcNames []string) sets.Set[string] {
	pvNames := sets.New[string]()
	gomega.Eventually(ctx, func() (int, error) {
		unbound := 0
		for _, pvcName := range pvcNames {
			pvc, err := cs.CoreV1().PersistentVolumeClaims(ns).Get(ctx, pvcName, metav1.GetOptions{})
			if err != nil {
				gomega.StopTrying("failed to fetch PVCs").Wrap(err).Now()
			}
			if pvc.Status.Phase != v1.ClaimBound {
				unbound++
			} else {
				pvNames.Insert(pvc.Spec.VolumeName)
			}
		}
		framework.Logf("%d/%d of PVCs are Bound", pvNames.Len(), len(pvcNames))
		return unbound, nil
	}).WithPolling(5*time.Second).WithTimeout(timeout).Should(gomega.BeZero(), "error waiting for all PVCs to be bound")
	return pvNames
}

func getNodeLimits(ctx context.Context, cs clientset.Interface, config *storageframework.PerTestConfig, nodeName string, driver storageframework.DynamicPVTestDriver) (int, error) {
	driverInfo := driver.GetDriverInfo()
	if len(driverInfo.InTreePluginName) > 0 {
		return getInTreeNodeLimits(ctx, cs, nodeName, driverInfo.InTreePluginName)
	}

	sc := driver.GetDynamicProvisionStorageClass(ctx, config, "")
	return getCSINodeLimits(ctx, cs, config, nodeName, sc.Provisioner)
}

func getInTreeNodeLimits(ctx context.Context, cs clientset.Interface, nodeName, driverName string) (int, error) {
	node, err := cs.CoreV1().Nodes().Get(ctx, nodeName, metav1.GetOptions{})
	if err != nil {
		return 0, err
	}

	var allocatableKey string
	switch driverName {
	case migrationplugins.AWSEBSInTreePluginName:
		allocatableKey = volumeutil.EBSVolumeLimitKey
	case migrationplugins.GCEPDInTreePluginName:
		allocatableKey = volumeutil.GCEVolumeLimitKey
	case migrationplugins.CinderInTreePluginName:
		allocatableKey = volumeutil.CinderVolumeLimitKey
	case migrationplugins.AzureDiskInTreePluginName:
		allocatableKey = volumeutil.AzureVolumeLimitKey
	default:
		return 0, fmt.Errorf("unknown in-tree volume plugin name: %s", driverName)
	}

	limit, ok := node.Status.Allocatable[v1.ResourceName(allocatableKey)]
	if !ok {
		return 0, fmt.Errorf("node %s does not contain status.allocatable[%s] for volume plugin %s", nodeName, allocatableKey, driverName)
	}
	return int(limit.Value()), nil
}

func getCSINodeLimits(ctx context.Context, cs clientset.Interface, config *storageframework.PerTestConfig, nodeName, driverName string) (int, error) {
	// Retry with a timeout, the driver might just have been installed and kubelet takes a while to publish everything.
	var limit int
	err := wait.PollUntilContextTimeout(ctx, 2*time.Second, csiNodeInfoTimeout, true, func(ctx context.Context) (bool, error) {
		csiNode, err := cs.StorageV1().CSINodes().Get(ctx, nodeName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("%s", err)
			return false, nil
		}
		var csiDriver *storagev1.CSINodeDriver
		for i, c := range csiNode.Spec.Drivers {
			if c.Name == driverName || c.Name == config.GetUniqueDriverName() {
				csiDriver = &csiNode.Spec.Drivers[i]
				break
			}
		}
		if csiDriver == nil {
			framework.Logf("CSINodeInfo does not have driver %s yet", driverName)
			return false, nil
		}
		if csiDriver.Allocatable == nil {
			return false, fmt.Errorf("CSINodeInfo does not have Allocatable for driver %s", driverName)
		}
		if csiDriver.Allocatable.Count == nil {
			return false, fmt.Errorf("CSINodeInfo does not have Allocatable.Count for driver %s", driverName)
		}
		limit = int(*csiDriver.Allocatable.Count)
		return true, nil
	})
	if err != nil {
		return 0, fmt.Errorf("could not get CSINode limit for driver %s: %w", driverName, err)
	}
	return limit, nil
}
