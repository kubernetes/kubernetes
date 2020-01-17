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
	"fmt"
	"math/rand"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/testpatterns"
)

type topologyTestSuite struct {
	tsInfo TestSuiteInfo
}

type topologyTest struct {
	config        *PerTestConfig
	driverCleanup func()

	intreeOps   opCounts
	migratedOps opCounts

	resource      VolumeResource
	pod           *v1.Pod
	allTopologies []topology
}

type topology map[string]string

var _ TestSuite = &topologyTestSuite{}

// InitTopologyTestSuite returns topologyTestSuite that implements TestSuite interface
func InitTopologyTestSuite() TestSuite {
	return &topologyTestSuite{
		tsInfo: TestSuiteInfo{
			Name: "topology",
			TestPatterns: []testpatterns.TestPattern{
				testpatterns.TopologyImmediate,
				testpatterns.TopologyDelayed,
			},
		},
	}
}

func (t *topologyTestSuite) GetTestSuiteInfo() TestSuiteInfo {
	return t.tsInfo
}

func (t *topologyTestSuite) SkipRedundantSuite(driver TestDriver, pattern testpatterns.TestPattern) {
}

func (t *topologyTestSuite) DefineTests(driver TestDriver, pattern testpatterns.TestPattern) {
	var (
		dInfo   = driver.GetDriverInfo()
		dDriver DynamicPVTestDriver
		cs      clientset.Interface
		err     error
	)

	ginkgo.BeforeEach(func() {
		// Check preconditions.
		ok := false
		dDriver, ok = driver.(DynamicPVTestDriver)
		if !ok {
			e2eskipper.Skipf("Driver %s doesn't support %v -- skipping", dInfo.Name, pattern.VolType)
		}

		if !dInfo.Capabilities[CapTopology] {
			e2eskipper.Skipf("Driver %q does not support topology - skipping", dInfo.Name)
		}

	})

	// This intentionally comes after checking the preconditions because it
	// registers its own BeforeEach which creates the namespace. Beware that it
	// also registers an AfterEach which renders f unusable. Any code using
	// f must run inside an It or Context callback.
	f := framework.NewDefaultFramework("topology")

	init := func() topologyTest {

		l := topologyTest{}

		// Now do the more expensive test initialization.
		l.config, l.driverCleanup = driver.PrepareTest(f)

		l.resource = VolumeResource{
			Config:  l.config,
			Pattern: pattern,
		}

		// After driver is installed, check driver topologies on nodes
		cs = f.ClientSet
		keys := dInfo.TopologyKeys
		if len(keys) == 0 {
			e2eskipper.Skipf("Driver didn't provide topology keys -- skipping")
		}
		if dInfo.NumAllowedTopologies == 0 {
			// Any plugin that supports topology defaults to 1 topology
			dInfo.NumAllowedTopologies = 1
		}
		// We collect 1 additional topology, if possible, for the conflicting topology test
		// case, but it's not needed for the positive test
		l.allTopologies, err = t.getCurrentTopologies(cs, keys, dInfo.NumAllowedTopologies+1)
		framework.ExpectNoError(err, "failed to get current driver topologies")
		if len(l.allTopologies) < dInfo.NumAllowedTopologies {
			e2eskipper.Skipf("Not enough topologies in cluster -- skipping")
		}

		l.resource.Sc = dDriver.GetDynamicProvisionStorageClass(l.config, pattern.FsType)
		framework.ExpectNotEqual(l.resource.Sc, nil, "driver failed to provide a StorageClass")
		l.resource.Sc.VolumeBindingMode = &pattern.BindingMode

		testVolumeSizeRange := t.GetTestSuiteInfo().SupportedSizeRange
		driverVolumeSizeRange := dDriver.GetDriverInfo().SupportedSizeRange
		claimSize, err := getSizeRangesIntersection(testVolumeSizeRange, driverVolumeSizeRange)
		framework.ExpectNoError(err, "determine intersection of test size range %+v and driver size range %+v", testVolumeSizeRange, driverVolumeSizeRange)
		l.resource.Pvc = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        claimSize,
			StorageClassName: &(l.resource.Sc.Name),
		}, l.config.Framework.Namespace.Name)

		l.intreeOps, l.migratedOps = getMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName)
		return l
	}

	cleanup := func(l topologyTest) {
		t.CleanupResources(cs, &l)
		err := tryFunc(l.driverCleanup)
		l.driverCleanup = nil
		framework.ExpectNoError(err, "while cleaning up driver")

		validateMigrationVolumeOpCounts(f.ClientSet, dInfo.InTreePluginName, l.intreeOps, l.migratedOps)
	}

	ginkgo.It("should provision a volume and schedule a pod with AllowedTopologies", func() {
		l := init()
		defer func() {
			cleanup(l)
		}()

		// If possible, exclude one topology, otherwise allow them all
		excludedIndex := -1
		if len(l.allTopologies) > dInfo.NumAllowedTopologies {
			excludedIndex = rand.Intn(len(l.allTopologies))
		}
		allowedTopologies := t.setAllowedTopologies(l.resource.Sc, l.allTopologies, excludedIndex)

		t.createResources(cs, &l, nil)

		err = e2epod.WaitForPodRunningInNamespace(cs, l.pod)
		framework.ExpectNoError(err)

		ginkgo.By("Verifying pod scheduled to correct node")
		pod, err := cs.CoreV1().Pods(l.pod.Namespace).Get(l.pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)

		node, err := cs.CoreV1().Nodes().Get(pod.Spec.NodeName, metav1.GetOptions{})
		framework.ExpectNoError(err)

		t.verifyNodeTopology(node, allowedTopologies)
	})

	ginkgo.It("should fail to schedule a pod which has topologies that conflict with AllowedTopologies", func() {
		l := init()
		defer func() {
			cleanup(l)
		}()

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
		t.createResources(cs, &l, affinity)

		// Wait for pod to fail scheduling
		// With delayed binding, the scheduler errors before provisioning
		// With immediate binding, the volume gets provisioned but cannot be scheduled
		err = e2epod.WaitForPodNameUnschedulableInNamespace(cs, l.pod.Name, l.pod.Namespace)
		framework.ExpectNoError(err)
	})
}

// getCurrentTopologies() goes through all Nodes and returns up to maxCount unique driver topologies
func (t *topologyTestSuite) getCurrentTopologies(cs clientset.Interface, keys []string, maxCount int) ([]topology, error) {
	nodes, err := e2enode.GetReadySchedulableNodes(cs)
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

func (t *topologyTestSuite) createResources(cs clientset.Interface, l *topologyTest, affinity *v1.Affinity) {
	var err error
	framework.Logf("Creating storage class object and pvc object for driver - sc: %v, pvc: %v", l.resource.Sc, l.resource.Pvc)

	ginkgo.By("Creating sc")
	l.resource.Sc, err = cs.StorageV1().StorageClasses().Create(l.resource.Sc)
	framework.ExpectNoError(err)

	ginkgo.By("Creating pvc")
	l.resource.Pvc, err = cs.CoreV1().PersistentVolumeClaims(l.resource.Pvc.Namespace).Create(l.resource.Pvc)
	framework.ExpectNoError(err)

	ginkgo.By("Creating pod")
	l.pod = e2epod.MakeSecPod(l.config.Framework.Namespace.Name,
		[]*v1.PersistentVolumeClaim{l.resource.Pvc},
		nil,
		false,
		"",
		false,
		false,
		e2epv.SELinuxLabel,
		nil)
	l.pod.Spec.Affinity = affinity
	l.pod, err = cs.CoreV1().Pods(l.pod.Namespace).Create(l.pod)
	framework.ExpectNoError(err)
}

func (t *topologyTestSuite) CleanupResources(cs clientset.Interface, l *topologyTest) {
	if l.pod != nil {
		ginkgo.By("Deleting pod")
		err := e2epod.DeletePodWithWait(cs, l.pod)
		framework.ExpectNoError(err, "while deleting pod")
	}

	err := l.resource.CleanupResource()
	framework.ExpectNoError(err, "while clean up resource")
}
