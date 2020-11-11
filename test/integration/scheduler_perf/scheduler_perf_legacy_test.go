/*
Copyright 2015 The Kubernetes Authors.

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

package benchmark

import (
	"fmt"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1beta1 "k8s.io/api/storage/v1beta1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/csi-translation-lib/plugins"
	csilibplugins "k8s.io/csi-translation-lib/plugins"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/volume/util"
	"k8s.io/kubernetes/test/integration/framework"
	testutils "k8s.io/kubernetes/test/utils"
)

var (
	defaultNodeStrategy = &testutils.TrivialNodePrepareStrategy{}

	testCSIDriver = plugins.AWSEBSDriverName
	// From PV controller
	annBindCompleted = "pv.kubernetes.io/bind-completed"

	defaultTests = []struct{ nodes, existingPods, minPods int }{
		{nodes: 500, existingPods: 500, minPods: 1000},
		{nodes: 600, existingPods: 10000, minPods: 1000},
		{nodes: 5000, existingPods: 5000, minPods: 1000},
	}
)

// BenchmarkScheduling benchmarks the scheduling rate when the cluster has
// various quantities of nodes and scheduled pods.
func BenchmarkScheduling(b *testing.B) {
	testStrategy := testutils.NewSimpleWithControllerCreatePodStrategy("rc1")
	for _, test := range defaultTests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			nodeStrategies := []testutils.CountToStrategy{{Count: test.nodes, Strategy: defaultNodeStrategy}}
			benchmarkScheduling(test.existingPods, test.minPods, nodeStrategies, testStrategy, b)
		})
	}
}

// BenchmarkSchedulingPodAntiAffinity benchmarks the scheduling rate of pods with
// PodAntiAffinity rules when the cluster has various quantities of nodes and
// scheduled pods.
func BenchmarkSchedulingPodAntiAffinity(b *testing.B) {
	// Since the pods has anti affinity to each other, the number of pods to schedule
	// can't exceed the number of nodes (the topology used in the test)
	tests := []struct{ nodes, existingPods, minPods int }{
		{nodes: 500, existingPods: 100, minPods: 400},
		{nodes: 5000, existingPods: 1000, minPods: 1000},
	}
	testBasePod := makeBasePodWithPodAntiAffinity(
		map[string]string{"name": "test", "color": "green"},
		map[string]string{"color": "green"})
	// The test strategy creates pods with anti-affinity to each other, each pod ending up in a separate node.
	testStrategy := testutils.NewCustomCreatePodStrategy(testBasePod)
	for _, test := range tests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			var nodeStrategies []testutils.CountToStrategy
			for i := 0; i < test.nodes; i++ {
				nodeStrategy := testutils.NewLabelNodePrepareStrategy(v1.LabelHostname, fmt.Sprintf("node-%d", i))
				nodeStrategies = append(nodeStrategies, testutils.CountToStrategy{Count: 1, Strategy: nodeStrategy})
			}
			benchmarkScheduling(test.existingPods, test.minPods, nodeStrategies, testStrategy, b)
		})
	}
}

// BenchmarkSchedulingSecrets benchmarks the scheduling rate of pods with
// volumes that don't require any special handling, such as Secrets.
// It can be used to compare scheduler efficiency with the other benchmarks
// that use volume scheduling predicates.
func BenchmarkSchedulingSecrets(b *testing.B) {
	// The test strategy creates pods with a secret.
	testBasePod := makeBasePodWithSecret()
	testStrategy := testutils.NewCustomCreatePodStrategy(testBasePod)
	for _, test := range defaultTests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			nodeStrategies := []testutils.CountToStrategy{{Count: test.nodes, Strategy: defaultNodeStrategy}}
			benchmarkScheduling(test.existingPods, test.minPods, nodeStrategies, testStrategy, b)
		})
	}
}

// BenchmarkSchedulingInTreePVs benchmarks the scheduling rate of pods with
// in-tree volumes (used via PV/PVC). Nodes have default hardcoded attach limits
// (39 for AWS EBS).
func BenchmarkSchedulingInTreePVs(b *testing.B) {
	// The test strategy creates pods with AWS EBS volume used via PV.
	baseClaim := makeBasePersistentVolumeClaim()
	basePod := makeBasePod()
	testStrategy := testutils.NewCreatePodWithPersistentVolumeStrategy(baseClaim, awsVolumeFactory, basePod)
	for _, test := range defaultTests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			nodeStrategies := []testutils.CountToStrategy{{Count: test.nodes, Strategy: defaultNodeStrategy}}
			benchmarkScheduling(test.existingPods, test.minPods, nodeStrategies, testStrategy, b)
		})
	}
}

// BenchmarkSchedulingWaitForFirstConsumerPVs benchmarks the scheduling rate
// of pods with volumes with VolumeBindingMode set to WaitForFirstConsumer.
func BenchmarkSchedulingWaitForFirstConsumerPVs(b *testing.B) {
	tests := []struct{ nodes, existingPods, minPods int }{
		{nodes: 500, existingPods: 500, minPods: 1000},
		// default 5000 existingPods is a way too much for now
	}
	basePod := makeBasePod()
	testStrategy := testutils.NewCreatePodWithPersistentVolumeWithFirstConsumerStrategy(gceVolumeFactory, basePod)
	nodeStrategy := testutils.NewLabelNodePrepareStrategy(v1.LabelFailureDomainBetaZone, "zone1")
	for _, test := range tests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			nodeStrategies := []testutils.CountToStrategy{{Count: test.nodes, Strategy: nodeStrategy}}
			benchmarkScheduling(test.existingPods, test.minPods, nodeStrategies, testStrategy, b)
		})
	}
}

// BenchmarkSchedulingMigratedInTreePVs benchmarks the scheduling rate of pods with
// in-tree volumes (used via PV/PVC) that are migrated to CSI. CSINode instances exist
// for all nodes and have proper annotation that AWS is migrated.
func BenchmarkSchedulingMigratedInTreePVs(b *testing.B) {
	// The test strategy creates pods with AWS EBS volume used via PV.
	baseClaim := makeBasePersistentVolumeClaim()
	basePod := makeBasePod()
	testStrategy := testutils.NewCreatePodWithPersistentVolumeStrategy(baseClaim, awsVolumeFactory, basePod)

	// Each node can use the same amount of CSI volumes as in-tree AWS volume
	// plugin, so the results should be comparable with BenchmarkSchedulingInTreePVs.
	driverKey := util.GetCSIAttachLimitKey(testCSIDriver)
	allocatable := map[v1.ResourceName]string{
		v1.ResourceName(driverKey): fmt.Sprintf("%d", util.DefaultMaxEBSVolumes),
	}
	var count int32 = util.DefaultMaxEBSVolumes
	csiAllocatable := map[string]*storagev1beta1.VolumeNodeResources{
		testCSIDriver: {
			Count: &count,
		},
	}
	nodeStrategy := testutils.NewNodeAllocatableStrategy(allocatable, csiAllocatable, []string{csilibplugins.AWSEBSInTreePluginName})
	for _, test := range defaultTests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.CSIMigration, true)()
			defer featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate, features.CSIMigrationAWS, true)()
			nodeStrategies := []testutils.CountToStrategy{{Count: test.nodes, Strategy: nodeStrategy}}
			benchmarkScheduling(test.existingPods, test.minPods, nodeStrategies, testStrategy, b)
		})
	}
}

// node.status.allocatable.
func BenchmarkSchedulingCSIPVs(b *testing.B) {
	// The test strategy creates pods with CSI volume via PV.
	baseClaim := makeBasePersistentVolumeClaim()
	basePod := makeBasePod()
	testStrategy := testutils.NewCreatePodWithPersistentVolumeStrategy(baseClaim, csiVolumeFactory, basePod)

	// Each node can use the same amount of CSI volumes as in-tree AWS volume
	// plugin, so the results should be comparable with BenchmarkSchedulingInTreePVs.
	driverKey := util.GetCSIAttachLimitKey(testCSIDriver)
	allocatable := map[v1.ResourceName]string{
		v1.ResourceName(driverKey): fmt.Sprintf("%d", util.DefaultMaxEBSVolumes),
	}
	var count int32 = util.DefaultMaxEBSVolumes
	csiAllocatable := map[string]*storagev1beta1.VolumeNodeResources{
		testCSIDriver: {
			Count: &count,
		},
	}
	nodeStrategy := testutils.NewNodeAllocatableStrategy(allocatable, csiAllocatable, []string{})
	for _, test := range defaultTests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			nodeStrategies := []testutils.CountToStrategy{{Count: test.nodes, Strategy: nodeStrategy}}
			benchmarkScheduling(test.existingPods, test.minPods, nodeStrategies, testStrategy, b)
		})
	}
}

// BenchmarkSchedulingPodAffinity benchmarks the scheduling rate of pods with
// PodAffinity rules when the cluster has various quantities of nodes and
// scheduled pods.
func BenchmarkSchedulingPodAffinity(b *testing.B) {
	testBasePod := makeBasePodWithPodAffinity(
		map[string]string{"foo": ""},
		map[string]string{"foo": ""},
	)
	// The test strategy creates pods with affinity for each other.
	testStrategy := testutils.NewCustomCreatePodStrategy(testBasePod)
	nodeStrategy := testutils.NewLabelNodePrepareStrategy(v1.LabelFailureDomainBetaZone, "zone1")
	for _, test := range defaultTests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			nodeStrategies := []testutils.CountToStrategy{{Count: test.nodes, Strategy: nodeStrategy}}
			benchmarkScheduling(test.existingPods, test.minPods, nodeStrategies, testStrategy, b)
		})
	}
}

// BenchmarkSchedulingPreferredPodAffinity benchmarks the scheduling rate of pods with
// preferred PodAffinity rules when the cluster has various quantities of nodes and
// scheduled pods.
func BenchmarkSchedulingPreferredPodAffinity(b *testing.B) {
	testBasePod := makeBasePodWithPreferredPodAffinity(
		map[string]string{"foo": ""},
		map[string]string{"foo": ""},
	)
	// The test strategy creates pods with affinity for each other.
	testStrategy := testutils.NewCustomCreatePodStrategy(testBasePod)
	for _, test := range defaultTests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			var nodeStrategies []testutils.CountToStrategy
			for i := 0; i < test.nodes; i++ {
				nodeStrategy := testutils.NewLabelNodePrepareStrategy(v1.LabelHostname, fmt.Sprintf("node-%d", i))
				nodeStrategies = append(nodeStrategies, testutils.CountToStrategy{Count: 1, Strategy: nodeStrategy})
			}
			benchmarkScheduling(test.existingPods, test.minPods, nodeStrategies, testStrategy, b)
		})
	}
}

// BenchmarkSchedulingPreferredPodAntiAffinity benchmarks the scheduling rate of pods with
// preferred PodAntiAffinity rules when the cluster has various quantities of nodes and
// scheduled pods.
func BenchmarkSchedulingPreferredPodAntiAffinity(b *testing.B) {
	testBasePod := makeBasePodWithPreferredPodAntiAffinity(
		map[string]string{"foo": ""},
		map[string]string{"foo": ""},
	)
	// The test strategy creates pods with anti affinity to each other.
	testStrategy := testutils.NewCustomCreatePodStrategy(testBasePod)
	for _, test := range defaultTests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			var nodeStrategies []testutils.CountToStrategy
			for i := 0; i < test.nodes; i++ {
				nodeStrategy := testutils.NewLabelNodePrepareStrategy(v1.LabelHostname, fmt.Sprintf("node-%d", i))
				nodeStrategies = append(nodeStrategies, testutils.CountToStrategy{Count: 1, Strategy: nodeStrategy})
			}
			benchmarkScheduling(test.existingPods, test.minPods, nodeStrategies, testStrategy, b)
		})
	}
}

// BenchmarkSchedulingNodeAffinity benchmarks the scheduling rate of pods with
// NodeAffinity rules when the cluster has various quantities of nodes and
// scheduled pods.
func BenchmarkSchedulingNodeAffinity(b *testing.B) {
	testBasePod := makeBasePodWithNodeAffinity(v1.LabelFailureDomainBetaZone, []string{"zone1", "zone2"})
	// The test strategy creates pods with node-affinity for each other.
	testStrategy := testutils.NewCustomCreatePodStrategy(testBasePod)
	nodeStrategy := testutils.NewLabelNodePrepareStrategy(v1.LabelFailureDomainBetaZone, "zone1")
	for _, test := range defaultTests {
		name := fmt.Sprintf("%vNodes/%vPods", test.nodes, test.existingPods)
		b.Run(name, func(b *testing.B) {
			nodeStrategies := []testutils.CountToStrategy{{Count: test.nodes, Strategy: nodeStrategy}}
			benchmarkScheduling(test.existingPods, test.minPods, nodeStrategies, testStrategy, b)
		})
	}
}

// makeBasePodWithPodAntiAffinity creates a Pod object to be used as a template.
// The Pod has a PodAntiAffinity requirement against pods with the given labels.
func makeBasePodWithPodAntiAffinity(podLabels, affinityLabels map[string]string) *v1.Pod {
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "anti-affinity-pod-",
			Labels:       podLabels,
		},
		Spec: testutils.MakePodSpec(),
	}
	basePod.Spec.Affinity = &v1.Affinity{
		PodAntiAffinity: &v1.PodAntiAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
				{
					LabelSelector: &metav1.LabelSelector{
						MatchLabels: affinityLabels,
					},
					TopologyKey: v1.LabelHostname,
					Namespaces:  []string{testNamespace, setupNamespace},
				},
			},
		},
	}
	return basePod
}

// makeBasePodWithPreferredPodAntiAffinity creates a Pod object to be used as a template.
// The Pod has a preferred PodAntiAffinity with pods with the given labels.
func makeBasePodWithPreferredPodAntiAffinity(podLabels, affinityLabels map[string]string) *v1.Pod {
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "preferred-affinity-pod-",
			Labels:       podLabels,
		},
		Spec: testutils.MakePodSpec(),
	}
	basePod.Spec.Affinity = &v1.Affinity{
		PodAntiAffinity: &v1.PodAntiAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
				{
					PodAffinityTerm: v1.PodAffinityTerm{
						LabelSelector: &metav1.LabelSelector{
							MatchLabels: affinityLabels,
						},
						TopologyKey: v1.LabelHostname,
						Namespaces:  []string{testNamespace, setupNamespace},
					},
					Weight: 1,
				},
			},
		},
	}
	return basePod
}

// makeBasePodWithPreferredPodAffinity creates a Pod object to be used as a template.
// The Pod has a preferred PodAffinity with pods with the given labels.
func makeBasePodWithPreferredPodAffinity(podLabels, affinityLabels map[string]string) *v1.Pod {
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "preferred-affinity-pod-",
			Labels:       podLabels,
		},
		Spec: testutils.MakePodSpec(),
	}
	basePod.Spec.Affinity = &v1.Affinity{
		PodAffinity: &v1.PodAffinity{
			PreferredDuringSchedulingIgnoredDuringExecution: []v1.WeightedPodAffinityTerm{
				{
					PodAffinityTerm: v1.PodAffinityTerm{
						LabelSelector: &metav1.LabelSelector{
							MatchLabels: affinityLabels,
						},
						TopologyKey: v1.LabelHostname,
						Namespaces:  []string{testNamespace, setupNamespace},
					},
					Weight: 1,
				},
			},
		},
	}
	return basePod
}

// makeBasePodWithPodAffinity creates a Pod object to be used as a template.
// The Pod has a PodAffinity requirement against pods with the given labels.
func makeBasePodWithPodAffinity(podLabels, affinityZoneLabels map[string]string) *v1.Pod {
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "affinity-pod-",
			Labels:       podLabels,
		},
		Spec: testutils.MakePodSpec(),
	}
	basePod.Spec.Affinity = &v1.Affinity{
		PodAffinity: &v1.PodAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: []v1.PodAffinityTerm{
				{
					LabelSelector: &metav1.LabelSelector{
						MatchLabels: affinityZoneLabels,
					},
					TopologyKey: v1.LabelFailureDomainBetaZone,
					Namespaces:  []string{testNamespace, setupNamespace},
				},
			},
		},
	}
	return basePod
}

// makeBasePodWithNodeAffinity creates a Pod object to be used as a template.
// The Pod has a NodeAffinity requirement against nodes with the given expressions.
func makeBasePodWithNodeAffinity(key string, vals []string) *v1.Pod {
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "node-affinity-",
		},
		Spec: testutils.MakePodSpec(),
	}
	basePod.Spec.Affinity = &v1.Affinity{
		NodeAffinity: &v1.NodeAffinity{
			RequiredDuringSchedulingIgnoredDuringExecution: &v1.NodeSelector{
				NodeSelectorTerms: []v1.NodeSelectorTerm{
					{
						MatchExpressions: []v1.NodeSelectorRequirement{
							{
								Key:      key,
								Operator: v1.NodeSelectorOpIn,
								Values:   vals,
							},
						},
					},
				},
			},
		},
	}
	return basePod
}

// benchmarkScheduling benchmarks scheduling rate with specific number of nodes
// and specific number of pods already scheduled.
// This will schedule numExistingPods pods before the benchmark starts, and at
// least minPods pods during the benchmark.
func benchmarkScheduling(numExistingPods, minPods int,
	nodeStrategies []testutils.CountToStrategy,
	testPodStrategy testutils.TestPodCreateStrategy,
	b *testing.B) {
	if b.N < minPods {
		b.N = minPods
	}
	finalFunc, podInformer, clientset := mustSetupScheduler()
	defer finalFunc()

	nodePreparer := framework.NewIntegrationTestNodePreparer(
		clientset,
		nodeStrategies,
		"scheduler-perf-")
	if err := nodePreparer.PrepareNodes(0); err != nil {
		klog.Fatalf("%v", err)
	}
	defer nodePreparer.CleanupNodes()

	config := testutils.NewTestPodCreatorConfig()
	config.AddStrategy(setupNamespace, numExistingPods, testPodStrategy)
	podCreator := testutils.NewTestPodCreator(clientset, config)
	podCreator.CreatePods()

	for {
		scheduled, err := getScheduledPods(podInformer)
		if err != nil {
			klog.Fatalf("%v", err)
		}
		if len(scheduled) >= numExistingPods {
			break
		}
		klog.Infof("got %d existing pods, required: %d", len(scheduled), numExistingPods)
		time.Sleep(1 * time.Second)
	}

	scheduled := int32(0)
	completedCh := make(chan struct{})
	podInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		UpdateFunc: func(old, cur interface{}) {
			curPod := cur.(*v1.Pod)
			oldPod := old.(*v1.Pod)

			if len(oldPod.Spec.NodeName) == 0 && len(curPod.Spec.NodeName) > 0 {
				if atomic.AddInt32(&scheduled, 1) >= int32(b.N) {
					completedCh <- struct{}{}
				}
			}
		},
	})

	// start benchmark
	b.ResetTimer()
	config = testutils.NewTestPodCreatorConfig()
	config.AddStrategy(testNamespace, b.N, testPodStrategy)
	podCreator = testutils.NewTestPodCreator(clientset, config)
	podCreator.CreatePods()

	<-completedCh

	// Note: without this line we're taking the overhead of defer() into account.
	b.StopTimer()
}

// makeBasePodWithSecrets creates a Pod object to be used as a template.
// The pod uses a single Secrets volume.
func makeBasePodWithSecret() *v1.Pod {
	basePod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "secret-volume-",
		},
		Spec: testutils.MakePodSpec(),
	}

	volumes := []v1.Volume{
		{
			Name: "secret",
			VolumeSource: v1.VolumeSource{
				Secret: &v1.SecretVolumeSource{
					SecretName: "secret",
				},
			},
		},
	}
	basePod.Spec.Volumes = volumes
	return basePod
}

func makeBasePersistentVolumeClaim() *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			// Name is filled in NewCreatePodWithPersistentVolumeStrategy
			Annotations: map[string]string{
				annBindCompleted: "true",
			},
		},
		Spec: v1.PersistentVolumeClaimSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany},
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("1Gi"),
				},
			},
		},
	}
}

func awsVolumeFactory(id int) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("vol-%d", id),
		},
		Spec: v1.PersistentVolumeSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany},
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("1Gi"),
			},
			PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimRetain,
			PersistentVolumeSource: v1.PersistentVolumeSource{
				AWSElasticBlockStore: &v1.AWSElasticBlockStoreVolumeSource{
					// VolumeID must be unique for each PV, so every PV is
					// counted as a separate volume in MaxPDVolumeCountChecker
					// predicate.
					VolumeID: fmt.Sprintf("vol-%d", id),
				},
			},
		},
	}
}

func gceVolumeFactory(id int) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("vol-%d", id),
		},
		Spec: v1.PersistentVolumeSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany},
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("1Gi"),
			},
			PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimRetain,
			PersistentVolumeSource: v1.PersistentVolumeSource{
				GCEPersistentDisk: &v1.GCEPersistentDiskVolumeSource{
					FSType: "ext4",
					PDName: fmt.Sprintf("vol-%d-pvc", id),
				},
			},
			NodeAffinity: &v1.VolumeNodeAffinity{
				Required: &v1.NodeSelector{
					NodeSelectorTerms: []v1.NodeSelectorTerm{
						{
							MatchExpressions: []v1.NodeSelectorRequirement{
								{
									Key:      v1.LabelFailureDomainBetaZone,
									Operator: v1.NodeSelectorOpIn,
									Values:   []string{"zone1"},
								},
							},
						},
					},
				},
			},
		},
	}
}

func csiVolumeFactory(id int) *v1.PersistentVolume {
	return &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name: fmt.Sprintf("vol-%d", id),
		},
		Spec: v1.PersistentVolumeSpec{
			AccessModes: []v1.PersistentVolumeAccessMode{v1.ReadOnlyMany},
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse("1Gi"),
			},
			PersistentVolumeReclaimPolicy: v1.PersistentVolumeReclaimRetain,
			PersistentVolumeSource: v1.PersistentVolumeSource{
				CSI: &v1.CSIPersistentVolumeSource{
					// Handle must be unique for each PV, so every PV is
					// counted as a separate volume in CSIMaxVolumeLimitChecker
					// predicate.
					VolumeHandle: fmt.Sprintf("vol-%d", id),
					Driver:       testCSIDriver,
				},
			},
		},
	}
}
