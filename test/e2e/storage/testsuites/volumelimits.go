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

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	migrationplugins "k8s.io/csi-translation-lib/plugins" // volume plugin names are exported nicely there
	volumeutil "k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

type volumeLimitsTestSuite struct {
	tsInfo TestSuiteInfo
}

const (
	// The test uses generic pod startup / PV deletion timeouts. As it creates
	// much more volumes at once, these timeouts are multiplied by this number.
	// Using real nr. of volumes (e.g. 128 on GCE) would be really too much.
	testSlowMultiplier = 10

	// How long to wait until CSINode gets attach limit from installed CSI driver.
	csiNodeInfoTimeout = 1 * time.Minute
)

var _ TestSuite = &volumeLimitsTestSuite{}

// InitVolumeLimitsTestSuite returns volumeLimitsTestSuite that implements TestSuite interface
func InitVolumeLimitsTestSuite() TestSuite {
	return &volumeLimitsTestSuite{
		tsInfo: TestSuiteInfo{
			Name: "volumeLimits",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.FsVolModeDynamicPV,
			},
		},
	}
}

func (t *volumeLimitsTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *volumeLimitsTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func (t *volumeLimitsTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	type local struct {
		config      *PerTestConfig
		testCleanup func()

		cs clientset.Interface
		ns *v1.Namespace
		// VolumeResource contains pv, pvc, sc, etc. of the first pod created
		resource *VolumeResource

		// All created PVCs, incl. the one in resource
		pvcs []*v1.PersistentVolumeClaim

		// All created PVs, incl. the one in resource
		pvNames sets.String

		runningPod       *v1.Pod
		unschedulablePod *v1.Pod
	}
	var (
		l local
	)

	// No preconditions to test. Normally they would be in a BeforeEach here.
	f := framework.NewDefaultFramework("volumelimits")

	// This checks that CSIMaxVolumeLimitChecker works as expected.
	// A randomly chosen node should be able to handle as many CSI volumes as
	// it claims to handle in CSINode.Spec.Drivers[x].Allocatable.
	// The test uses one single pod with a lot of volumes to work around any
	// max pod limit on a node.
	// And one extra pod with a CSI volume should get Pending with a condition
	// that says it's unschedulable because of volume limit.
	// BEWARE: the test may create lot of volumes and it's really slow.
	ginkgo.It("should support volume limits [Serial]", func() {
		driverInfo := driver.GetDriverInfo()
		if !driverInfo.Capabilities[CapVolumeLimits] {
			ginkgo.Skip(fmt.Sprintf("driver %s does not support volume limits", driverInfo.Name))
		}
		var dDriver DynamicPVTestDriver
		if dDriver = driver.(DynamicPVTestDriver); dDriver == nil {
			framework.Failf("Test driver does not provide dynamically created volumes")
		}

		l.ns = f.Namespace
		l.cs = f.ClientSet
		l.config, l.testCleanup = driver.PrepareTest(f)
		defer l.testCleanup()

		ginkgo.By("Picking a node")
		// Some CSI drivers are deployed to a single node (e.g csi-hostpath),
		// so we use that node instead of picking a random one.
		nodeName := l.config.ClientNodeSelection.Name
		if nodeName == "" {
			node, err := e2enode.GetRandomReadySchedulableNode(f.ClientSet)
			framework.ExpectNoError(err)
			nodeName = node.Name
		}
		framework.Logf("Selected node %s", nodeName)

		ginkgo.By("Checking node limits")
		limit, err := getNodeLimits(l.cs, l.config, nodeName, driverInfo)
		framework.ExpectNoError(err)

		framework.Logf("Node %s can handle %d volumes of driver %s", nodeName, limit, driverInfo.Name)
		// Create a storage class and generate a PVC. Do not instantiate the PVC yet, keep it for the last pod.
		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
		claimSize, err := getSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
		framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, dDriver)

		l.resource = CreateVolumeResource(driver, l.config, pattern, testVolumeSizeRange)
		defer func() {
			err := l.resource.CleanupResource()
			framework.ExpectNoError(err, "while cleaning up resource")
		}()
		defer func() {
			cleanupTest(l.cs, l.ns.Name, l.runningPod.Name, l.unschedulablePod.Name, l.pvcs, l.pvNames)
		}()

		// Create <limit> PVCs for one gigantic pod.
		ginkgo.By(fmt.Sprintf("Creating %d PVC(s)", limit))

		for i := 0; i < limit; i++ {
			pvc := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
				ClaimSize:        claimSize,
				StorageClassName: &l.resource.Sc.Name,
			}, l.ns.Name)
			pvc, err = l.cs.CoreV1().PersistentVolumeClaims(l.ns.Name).Create(context.TODO(), pvc, metav1.CreateOptions{})
			framework.ExpectNoError(err)
			l.pvcs = append(l.pvcs, pvc)
		}

		ginkgo.By("Creating pod to use all PVC(s)")
		selection := e2epod.NodeSelection{Name: nodeName}
		podConfig := e2epod.Config{
			NS:            l.ns.Name,
			PVCs:          l.pvcs,
			SeLinuxLabel:  e2epv.SELinuxLabel,
			NodeSelection: selection,
		}
		pod, err := e2epod.MakeSecPod(&podConfig)
		framework.ExpectNoError(err)
		l.runningPod, err = l.cs.CoreV1().Pods(l.ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for all PVCs to get Bound")
		l.pvNames, err = waitForAllPVCsBound(l.cs, testSlowMultiplier*e2epv.PVBindingTimeout, l.pvcs)
		framework.ExpectNoError(err)

		ginkgo.By("Waiting for the pod Running")
		err = e2epod.WaitTimeoutForPodRunningInNamespace(l.cs, l.runningPod.Name, l.ns.Name, testSlowMultiplier*framework.PodStartTimeout)
		framework.ExpectNoError(err)

		ginkgo.By("Creating an extra pod with one volume to exceed the limit")
		podConfig = e2epod.Config{
			NS:            l.ns.Name,
			PVCs:          []*v1.PersistentVolumeClaim{l.resource.Pvc},
			SeLinuxLabel:  e2epv.SELinuxLabel,
			NodeSelection: selection,
		}
		pod, err = e2epod.MakeSecPod(&podConfig)
		framework.ExpectNoError(err)
		l.unschedulablePod, err = l.cs.CoreV1().Pods(l.ns.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
		framework.ExpectNoError(err, "Failed to create an extra pod with one volume to exceed the limit")

		ginkgo.By("Waiting for the pod to get unschedulable with the right message")
		err = e2epod.WaitForPodCondition(l.cs, l.ns.Name, l.unschedulablePod.Name, "Unschedulable", framework.PodStartTimeout, func(pod *v1.Pod) (bool, error) {
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
				return true, fmt.Errorf("Expected pod to be in phase Pending, but got phase: %v", pod.Status.Phase)
			}
			return false, nil
		})
		framework.ExpectNoError(err)
	})
}

func cleanupTest(cs clientset.Interface, ns string, runningPodName, unschedulablePodName string, pvcs []*v1.PersistentVolumeClaim, pvNames sets.String) error {
	var cleanupErrors []string
	if runningPodName != "" {
		err := cs.CoreV1().Pods(ns).Delete(context.TODO(), runningPodName, metav1.DeleteOptions{})
		if err != nil {
			cleanupErrors = append(cleanupErrors, fmt.Sprintf("failed to delete pod %s: %s", runningPodName, err))
		}
	}
	if unschedulablePodName != "" {
		err := cs.CoreV1().Pods(ns).Delete(context.TODO(), unschedulablePodName, metav1.DeleteOptions{})
		if err != nil {
			cleanupErrors = append(cleanupErrors, fmt.Sprintf("failed to delete pod %s: %s", unschedulablePodName, err))
		}
	}
	for _, pvc := range pvcs {
		err := cs.CoreV1().PersistentVolumeClaims(ns).Delete(context.TODO(), pvc.Name, metav1.DeleteOptions{})
		if err != nil {
			cleanupErrors = append(cleanupErrors, fmt.Sprintf("failed to delete PVC %s: %s", pvc.Name, err))
		}
	}
	// Wait for the PVs to be deleted. It includes also pod and PVC deletion because of PVC protection.
	// We use PVs to make sure that the test does not leave orphan PVs when a CSI driver is destroyed
	// just after the test ends.
	err := wait.Poll(5*time.Second, testSlowMultiplier*e2epv.PVDeletingTimeout, func() (bool, error) {
		existing := 0
		for _, pvName := range pvNames.UnsortedList() {
			_, err := cs.CoreV1().PersistentVolumes().Get(context.TODO(), pvName, metav1.GetOptions{})
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
func waitForAllPVCsBound(cs clientset.Interface, timeout time.Duration, pvcs []*v1.PersistentVolumeClaim) (sets.String, error) {
	pvNames := sets.NewString()
	err := wait.Poll(5*time.Second, timeout, func() (bool, error) {
		unbound := 0
		for _, pvc := range pvcs {
			pvc, err := cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(context.TODO(), pvc.Name, metav1.GetOptions{})
			if err != nil {
				return false, err
			}
			if pvc.Status.Phase != v1.ClaimBound {
				unbound++
			} else {
				pvNames.Insert(pvc.Spec.VolumeName)
			}
		}
		if unbound > 0 {
			framework.Logf("%d/%d of PVCs are Bound", pvNames.Len(), len(pvcs))
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		return nil, fmt.Errorf("error waiting for all PVCs to be bound: %v", err)
	}
	return pvNames, nil
}

func getNodeLimits(cs clientset.Interface, config *PerTestConfig, nodeName string, driverInfo *DriverInfo) (int, error) {
	if len(driverInfo.InTreePluginName) == 0 {
		return getCSINodeLimits(cs, config, nodeName, driverInfo)
	}
	return getInTreeNodeLimits(cs, nodeName, driverInfo)
}

func getInTreeNodeLimits(cs clientset.Interface, nodeName string, driverInfo *DriverInfo) (int, error) {
	node, err := cs.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	if err != nil {
		return 0, err
	}

	var allocatableKey string
	switch driverInfo.InTreePluginName {
	case migrationplugins.AWSEBSInTreePluginName:
		allocatableKey = volumeutil.EBSVolumeLimitKey
	case migrationplugins.GCEPDInTreePluginName:
		allocatableKey = volumeutil.GCEVolumeLimitKey
	case migrationplugins.CinderInTreePluginName:
		allocatableKey = volumeutil.CinderVolumeLimitKey
	case migrationplugins.AzureDiskInTreePluginName:
		allocatableKey = volumeutil.AzureVolumeLimitKey
	default:
		return 0, fmt.Errorf("Unknown in-tree volume plugin name: %s", driverInfo.InTreePluginName)
	}

	limit, ok := node.Status.Allocatable[v1.ResourceName(allocatableKey)]
	if !ok {
		return 0, fmt.Errorf("Node %s does not contain status.allocatable[%s] for volume plugin %s", nodeName, allocatableKey, driverInfo.InTreePluginName)
	}
	return int(limit.Value()), nil
}

func getCSINodeLimits(cs clientset.Interface, config *PerTestConfig, nodeName string, driverInfo *DriverInfo) (int, error) {
	// Retry with a timeout, the driver might just have been installed and kubelet takes a while to publish everything.
	var limit int
	err := wait.PollImmediate(2*time.Second, csiNodeInfoTimeout, func() (bool, error) {
		csiNode, err := cs.StorageV1().CSINodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("%s", err)
			return false, nil
		}
		var csiDriver *storagev1.CSINodeDriver
		for _, c := range csiNode.Spec.Drivers {
			if c.Name == driverInfo.Name || c.Name == config.GetUniqueDriverName() {
				csiDriver = &c
				break
			}
		}
		if csiDriver == nil {
			framework.Logf("CSINodeInfo does not have driver %s yet", driverInfo.Name)
			return false, nil
		}
		if csiDriver.Allocatable == nil {
			return false, fmt.Errorf("CSINodeInfo does not have Allocatable for driver %s", driverInfo.Name)
		}
		if csiDriver.Allocatable.Count == nil {
			return false, fmt.Errorf("CSINodeInfo does not have Allocatable.Count for driver %s", driverInfo.Name)
		}
		limit = int(*csiDriver.Allocatable.Count)
		return true, nil
	})
	if err != nil {
		return 0, fmt.Errorf("could not get CSINode limit for driver %s: %v", driverInfo.Name, err)
	}
	return limit, nil
}
