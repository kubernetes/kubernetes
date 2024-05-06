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

// This suite tests volume topology

package testsuites

import (
	"context"
	"fmt"
	"math/rand"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	storageframework "k8s.io/kubernetes/test/e2e/storage/framework"
	storageutils "k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

type topologyTestSuite struct {
	tsInfo storageframework.TestSuiteInfo
}

type topologyTest struct {
	config *storageframework.PerTestConfig

	resource      storageframework.VolumeResource
	pod           *v1.Pod
	allTopologies []topology
}

type topology map[string]string

// InitCustomTopologyTestSuite returns topologyTestSuite that implements TestSuite interface
// using custom test patterns
func InitCustomTopologyTestSuite(patterns []storageframework.TestPattern) storageframework.TestSuite {
	return &topologyTestSuite{
		tsInfo: storageframework.TestSuiteInfo{
			Name:         "topology",
			TestPatterns: patterns,
		},
	}
}

// InitTopologyTestSuite returns topologyTestSuite that implements TestSuite interface
// using testsuite default patterns
func InitTopologyTestSuite() storageframework.TestSuite {
	patterns := []storageframework.TestPattern{
		storageframework.TopologyImmediate,
		storageframework.TopologyDelayed,
	}
	return InitCustomTopologyTestSuite(patterns)
}

func (t *topologyTestSuite) GetTestSuiteInfo() storageframework.TestSuiteInfo {
	return t.tsInfo
}

func (t *topologyTestSuite) SkipUnsupportedTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	dInfo := driver.GetDriverInfo()
	var ok bool
	_, ok = driver.(storageframework.DynamicPVTestDriver)
	if !ok {
		e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
	}

	if !dInfo.Capabilities[storageframework.CapTopology] {
		e2eskipper.Skipf("Driver %q does not support topology - skipping", dInfo.Name)
	}
}

func (t *topologyTestSuite) DefineTests(driver storageframework.TestDriver, pattern storageframework.TestPattern) {
	var (
		dInfo   = driver.GetDriverInfo()
		dDriver storageframework.DynamicPVTestDriver
		cs      clientset.Interface
		err     error
	)

	// Beware that it also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewFrameworkWithCustomTimeouts("topology", storageframework.GetDriverTimeouts(driver))
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	init := func(ctx context.Context) *topologyTest {
		dDriver, _ = driver.(storageframework.DynamicPVTestDriver)
		l := &topologyTest{}

		// Now do the more expensive test initialization.
		l.config = driver.PrepareTest(ctx, f)

		l.resource = storageframework.VolumeResource{
			Config:  l.config,
			Pattern: pattern,
		}

		// After driver is installed, check driver topologies on nodes
		cs = f.ClientSet
		keys := dInfo.TopologyKeys
		if len(keys) == 0 {
			e2eskipper.Skipf("Driver didn't provide topology keys -- skipping")
		}

		ginkgo.DeferCleanup(t.CleanupResources, cs, l)

		if dInfo.NumAllowedTopologies == 0 {
			// Any plugin that supports topology defaults to 1 topology
			dInfo.NumAllowedTopologies = 1
		}
		// We collect 1 additional topology, if possible, for the conflicting topology test
		// case, but it's not needed for the positive test
		l.allTopologies, err = t.getCurrentTopologies(ctx, cs, keys, dInfo.NumAllowedTopologies+1)
		framework.ExpectNoError(err, "failed to get current driver topologies")
		if len(l.allTopologies) < dInfo.NumAllowedTopologies {
			e2eskipper.Skipf("Not enough topologies in cluster -- skipping")
		}

		l.resource.Sc = dDriver.GetDynamicProvisionStorageClass(ctx, l.config, pattern.FsType)
		gomega.Expect(l.resource.Sc).ToNot(gomega.BeNil(), "driver failed to provide a StorageClass")
		l.resource.Sc.VolumeBindingMode = &pattern.BindingMode

		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
		claimSize, err := storageutils.GetSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
		framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, driverVolumeSizeRange)
		l.resource.Pvc = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(l.resource.Sc.Name),
		}, l.config.Framework.Namespace.Name)

		migrationCheck := newMigrationOpCheck(ctx, f.ClientSet, f.ClientConfig(), dInfo.InTreePluginName)
		ginkgo.DeferCleanup(migrationCheck.validateMigrationVolumeOpCounts)

		return l
	}

	ginkgo.It("should provision a volume and schedule a pod with AllowedTopologies", func(ctx context.Context) {
		l := init(ctx)

		// If possible, exclude one topology, otherwise allow them all
		excludedIndex := -1
		if len(l.allTopologies) > dInfo.NumAllowedTopologies {
			excludedIndex = rand.Intn(len(l.allTopologies))
		}
		allowedTopologies := t.setAllowedTopologies(l.resource.Sc, l.allTopologies, excludedIndex)

		t.createResources(ctx, cs, l, nil)

		err = e2epod.WaitTimeoutForPodRunningInNamespace(ctx, cs, l.pod.Name, l.pod.Namespace, f.Timeouts.PodStart)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying pod scheduled to correct node")
		pod, err := cs.CoreV1().Pods(l.pod.Namespace).Get(ctx, l.pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		node, err := cs.CoreV1().Nodes().Get(ctx, pod.Spec.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		t.verifyNodeTopology(node, allowedTopologies)
	})

	ginkgo.It("should fail to schedule a pod which has topologies that conflict with AllowedTopologies", func(ctx context.Context) {
		l := init(ctx)

		if len(l.allTopologies) < dInfo.NumAllowedTopologies+1 {
			e2eskipper.Skipf("Not enough topologies in cluster -- skipping")
		}

		// Exclude one topology
		excludedIndex := rand.Intn(len(l.allTopologies))
		t.setAllowedTopologies(l.resource.Sc, l.allTopologies, excludedIndex)

		// Set pod nodeSelector to the excluded topology
		exprs := []v1.NodeSelectorRequirement{}
		for k, v := range l.allTopologies[excludedIndex] {
			exprs = append(exprs, v1.NodeSelectorRequirement{
				Key:      k,
				Operator: v1.NodeSelectorOpIn,
				Values:   []string{v},
			})
		}

		affinity := &v1.Affinity{
			NodeAffinity: &v1.NodeAffinity{
				RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: exprs,
						},
					},
				},
			},
		}
		t.createResources(ctx, cs, l, affinity)

		// Wait for pod to fail scheduling
		// With delayed binding, the scheduler errors before provisioning
		// With immediate binding, the volume gets provisioned but cannot be scheduled
		err = e2epod.WaitForPodNameUnschedulableInNamespace(ctx, cs, l.pod.Name, l.pod.Namespace)
		framework.ExpectNoError(err)
	})
}

// getCurrentTopologies() goes through all Nodes and returns up to maxCount unique driver topologies
func (t *topologyTestSuite) getCurrentTopologies(ctx context.Context, cs clientset.Interface, keys []string, maxCount int) ([]topology, error) {
	nodes, err := e2enode.GetReadySchedulableNodes(ctx, cs)
	if err != nil {
		return nil, err
	}

	topos := []topology{}

	// TODO: scale?
	for _, n := range nodes.Items {
		topo := map[string]string{}
		for _, k := range keys {
			v, ok := n.Labels[k]
			if !ok {
				return nil, fmt.Errorf("node %v missing topology label %v", n.Name, k)
			}
			topo[k] = v
		}

		found := false
		for _, existingTopo := range topos {
			if topologyEqual(existingTopo, topo) {
				found = true
				break
			}
		}
		if !found {
			framework.Logf("found topology %v", topo)
			topos = append(topos, topo)
		}
		if len(topos) >= maxCount {
			break
		}
	}
	return topos, nil
}

// reflect.DeepEqual doesn't seem to work
func topologyEqual(t1, t2 topology) bool {
	if len(t1) != len(t2) {
		return false
	}
	for k1, v1 := range t1 {
		if v2, ok := t2[k1]; !ok || v1 != v2 {
			return false
		}
	}
	return true
}

// Set StorageClass.Allowed topologies from topos while excluding the topology at excludedIndex.
// excludedIndex can be -1 to specify nothing should be excluded.
// Return the list of allowed topologies generated.
func (t *topologyTestSuite) setAllowedTopologies(sc *storagev1.StorageClass, topos []topology, excludedIndex int) []topology {
	allowedTopologies := []topology{}
	sc.AllowedTopologies = []v1.TopologySelectorTerm{}

	for i := 0; i < len(topos); i++ {
		if i != excludedIndex {
			exprs := []v1.TopologySelectorLabelRequirement{}
			for k, v := range topos[i] {
				exprs = append(exprs, v1.TopologySelectorLabelRequirement{
					Key:    k,
					Values: []string{v},
				})
			}
			sc.AllowedTopologies = append(sc.AllowedTopologies, v1.TopologySelectorTerm{MatchLabelExpressions: exprs})
			allowedTopologies = append(allowedTopologies, topos[i])
		}
	}
	return allowedTopologies
}

func (t *topologyTestSuite) verifyNodeTopology(node *v1.Node, allowedTopos []topology) {
	for _, topo := range allowedTopos {
		for k, v := range topo {
			nodeV, _ := node.Labels[k]
			if nodeV == v {
				return
			}
		}
	}
	framework.Failf("node %v topology labels %+v doesn't match allowed topologies +%v", node.Name, node.Labels, allowedTopos)
}

func (t *topologyTestSuite) createResources(ctx context.Context, cs clientset.Interface, l *topologyTest, affinity *v1.Affinity) {
	var err error
	framework.Logf("Creating storage class object and pvc object for driver - sc: %v, pvc: %v", l.resource.Sc, l.resource.Pvc)

	ginkgo.By("Creating sc")
	l.resource.Sc, err = cs.StorageV1().StorageClasses().Create(ctx, l.resource.Sc, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("Creating pvc")
	l.resource.Pvc, err = cs.CoreV1().PersistentVolumeClaims(l.resource.Pvc.Namespace).Create(ctx, l.resource.Pvc, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	ginkgo.By("Creating pod")
	podConfig := e2epod.Config{
		NS:            l.config.Framework.Namespace.Name,
		PVCs:          []*v1.PersistentVolumeClaim{l.resource.Pvc},
		NodeSelection: e2epod.NodeSelection{Affinity: affinity, Selector: l.config.ClientNodeSelection.Selector},
		SeLinuxLabel:  e2epod.GetLinuxLabel(),
		ImageID:       e2epod.GetDefaultTestImageID(),
	}
	l.pod, err = e2epod.MakeSecPod(&podConfig)
	framework.ExpectNoError(err)
	l.pod, err = cs.CoreV1().Pods(l.pod.Namespace).Create(ctx, l.pod, metav1.CreateOptions{})
	framework.ExpectNoError(err)
}

func (t *topologyTestSuite) CleanupResources(ctx context.Context, cs clientset.Interface, l *topologyTest) {
	if l.pod != nil {
		ginkgo.By("Deleting pod")
		err := e2epod.DeletePodWithWait(ctx, cs, l.pod)
		framework.ExpectNoError(err, "while deleting pod")
	}

	err := l.resource.CleanupResource(ctx)
	framework.ExpectNoError(err, "while clean up resource")
}
