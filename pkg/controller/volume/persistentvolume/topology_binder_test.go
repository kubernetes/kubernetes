/*
Copyright 2017 The Kubernetes Authors.

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

package persistentvolume

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/controller"
)

var (
	unboundPVC          = makeTestPVC("unbound-pvc", "1G", pvcUnbound, "", &waitClass)
	unboundPVC2         = makeTestPVC("unbound-pvc2", "5G", pvcUnbound, "", &waitClass)
	preboundPVC         = makeTestPVC("prebound-pvc", "1G", pvcPrebound, "pv-node1a", &waitClass)
	boundPVC            = makeTestPVC("bound-pvc", "1G", pvcBound, "pv-bound", &waitClass)
	boundPVC2           = makeTestPVC("bound-pvc2", "1G", pvcBound, "pv-bound2", &waitClass)
	badPVC              = makeBadPVC()
	immediateUnboundPVC = makeTestPVC("immediate-unbound-pvc", "1G", pvcUnbound, "", &immediateClass)
	immediateBoundPVC   = makeTestPVC("immediate-bound-pvc", "1G", pvcBound, "pv-bound-immediate", &immediateClass)

	pvNoNode                   = makeTestPV("pv-no-node", "", "1G", "1", nil, waitClass)
	pvNode1a                   = makeTestPV("pv-node1a", "node1", "5G", "1", nil, waitClass)
	pvNode1b                   = makeTestPV("pv-node1b", "node1", "10G", "1", nil, waitClass)
	pvNode2                    = makeTestPV("pv-node2", "node2", "1G", "1", nil, waitClass)
	pvBound                    = makeTestPV("pv-bound", "node1", "1G", "1", boundPVC, waitClass)
	pvNode1aBound              = makeTestPV("pv-node1a", "node1", "1G", "1", unboundPVC, waitClass)
	pvNode1bBound              = makeTestPV("pv-node1b", "node1", "5G", "1", unboundPVC2, waitClass)
	pvNode1bBoundHigherVersion = makeTestPV("pv-node1b", "node1", "5G", "2", unboundPVC2, waitClass)
	pvBoundImmediate           = makeTestPV("pv-bound-immediate", "node1", "1G", "1", immediateBoundPVC, immediateClass)
	pvBoundImmediateNode2      = makeTestPV("pv-bound-immediate", "node2", "1G", "1", immediateBoundPVC, immediateClass)

	binding1a      = makeBinding(unboundPVC, pvNode1a)
	binding1b      = makeBinding(unboundPVC2, pvNode1b)
	bindingNoNode  = makeBinding(unboundPVC, pvNoNode)
	bindingBad     = makeBinding(badPVC, pvNode1b)
	binding1aBound = makeBinding(unboundPVC, pvNode1aBound)

	bindEmptyStr = ""
	bindOneStr   = "pv-node1a"
	bindTwoStr   = "pv-node1a,pv-node1b"

	waitClass      = "waitClass"
	immediateClass = "immediateClass"
)

type testEnv struct {
	client           clientset.Interface
	reactor          *volumeReactor
	binder           TopologyAwareVolumeBinder
	internalBinder   *topologyVolumeBinder
	internalPVCache  *pvTmpCache
	internalPVCCache cache.Indexer
}

func newTestBinder(t *testing.T) *testEnv {
	client := &fake.Clientset{}
	reactor := newVolumeReactor(client, nil, nil, nil, nil)
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())

	pvcInformer := informerFactory.Core().V1().PersistentVolumeClaims()
	nodeInformer := informerFactory.Core().V1().Nodes()
	classInformer := informerFactory.Storage().V1().StorageClasses()

	binder := NewTopologyAwareVolumeBinder(
		client,
		pvcInformer,
		informerFactory.Core().V1().PersistentVolumes(),
		nodeInformer,
		classInformer)

	// Add a node
	err := nodeInformer.Informer().GetIndexer().Add(&v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   "node1",
			Labels: map[string]string{"key1": "node1"},
		},
	})
	if err != nil {
		t.Fatalf("Failed to add node to internal cache: %v", err)
	}

	// Add storageclasses
	waitMode := storagev1.VolumeBindingWaitForFirstConsumer
	immediateMode := storagev1.VolumeBindingImmediate
	classes := []*storagev1.StorageClass{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: waitClass,
			},
			VolumeBindingMode: &waitMode,
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: immediateClass,
			},
			VolumeBindingMode: &immediateMode,
		},
	}
	for _, class := range classes {
		if err = classInformer.Informer().GetIndexer().Add(class); err != nil {
			t.Fatalf("Failed to add storage class to internal cache: %v", err)
		}
	}

	// Get internal types
	internalBinder, ok := binder.(*topologyVolumeBinder)
	if !ok {
		t.Fatalf("Failed to convert to internal binder")
	}

	pvCache := internalBinder.pvCache
	internalPVCache, ok := pvCache.(*pvTmpCache)
	if !ok {
		t.Fatalf("Failed to convert to internal PV cache")
	}

	return &testEnv{
		client:           client,
		reactor:          reactor,
		binder:           binder,
		internalBinder:   internalBinder,
		internalPVCache:  internalPVCache,
		internalPVCCache: pvcInformer.Informer().GetIndexer(),
	}
}

func (env *testEnv) initClaims(t *testing.T, pvcs []*v1.PersistentVolumeClaim) {
	for _, pvc := range pvcs {
		err := env.internalPVCCache.Add(pvc)
		if err != nil {
			t.Fatalf("Failed to add PVC %q to internal cache: %v", pvc.Name, err)
		}
		env.reactor.claims[pvc.Name] = pvc
	}
}

func (env *testEnv) initVolumes(cachedPVs []*v1.PersistentVolume, apiPVs []*v1.PersistentVolume) {
	internalPVCache := env.internalPVCache
	for _, pv := range cachedPVs {
		internalPVCache.add(pv)
		if apiPVs == nil {
			env.reactor.volumes[pv.Name] = pv
		}
	}
	for _, pv := range apiPVs {
		env.reactor.volumes[pv.Name] = pv
	}

}

func (env *testEnv) assumeVolumes(t *testing.T, name string, pvs []*v1.PersistentVolume) {
	pvCache := env.internalBinder.pvCache
	for _, pv := range pvs {
		if err := pvCache.TmpUpdate(pv); err != nil {
			t.Fatalf("Failed to setup test %q: error: %v", name, err)
		}
	}
}

func (env *testEnv) initTmpData(allBound bool, bindings []*bindingInfo) {
	tmpData := env.internalBinder.tmpData
	tmpData.allPVCsBound = allBound
	tmpData.nodeBindings["node1"] = bindings
}

func (env *testEnv) initPVsToBind(pod *v1.Pod, pvsStr *string) {
	if pvsStr != nil {
		metav1.SetMetaDataAnnotation(&pod.ObjectMeta, annPVsToBind, *pvsStr)
	}
}

func (env *testEnv) validateTmpData(t *testing.T, name, node string, expectedAllBound bool, expectedBindings []*bindingInfo) {
	tmpData := env.internalBinder.tmpData
	if expectedAllBound != tmpData.allPVCsBound {
		t.Errorf("Test %q failed: Expected allPVCsBound %v, got %v", name, expectedAllBound, tmpData.allPVCsBound)
	}

	if !reflect.DeepEqual(expectedBindings, tmpData.nodeBindings[node]) {
		t.Errorf("Test %q failed: Expected bindings %+v, got %+v", name, expectedBindings, tmpData.nodeBindings[node])
	}
}

func (env *testEnv) validateAssume(t *testing.T, name string, pod *v1.Pod, bindings []*bindingInfo, expectedPVsToBind *string) {
	// Check pvsToBind annotation in Pod
	result, ok := pod.Annotations[annPVsToBind]
	if expectedPVsToBind == nil && ok {
		t.Errorf("Test %q failed: Expected pvsToBind not set, got %q", name, result)
	} else if result != *expectedPVsToBind {
		t.Errorf("Test %q failed: OK %v, Expected pvsToBind %q, got %q", name, ok, *expectedPVsToBind, result)
	}

	// Check pv cache
	pvCache := env.internalBinder.pvCache
	for _, b := range bindings {
		pv := pvCache.GetPV(b.pv.Name)
		if pv == nil {
			t.Errorf("Test %q failed: PV %q not in cache", name, b.pv.Name)
			continue
		}
		if pv.Spec.ClaimRef == nil {
			t.Errorf("Test %q failed: PV %q ClaimRef is nil", name, b.pv.Name)
			continue
		}
		if pv.Spec.ClaimRef.Name != b.pvc.Name {
			t.Errorf("Test %q failed: expected PV.ClaimRef.Name %q, got %q", name, b.pvc.Name, pv.Spec.ClaimRef.Name)
		}
		if pv.Spec.ClaimRef.Namespace != b.pvc.Namespace {
			t.Errorf("Test %q failed: expected PV.ClaimRef.Namespace %q, got %q", name, b.pvc.Namespace, pv.Spec.ClaimRef.Namespace)
		}
	}
}

func (env *testEnv) validateFailedAssume(t *testing.T, name string, pod *v1.Pod, bindings []*bindingInfo) {
	// Check pvsToBind annotation not set in Pod
	if result, ok := pod.Annotations[annPVsToBind]; ok {
		t.Errorf("Test %q failed: expected pvsToBind not set, got %q", name, result)
	}

	// All PVs have been unmodified in cache
	pvCache := env.internalBinder.pvCache
	for _, b := range bindings {
		pv := pvCache.GetPV(b.pv.Name)
		// PV could be nil if it's missing from cache
		if pv != nil && pv != b.pv {
			t.Errorf("Test %q failed: PV %q was modified in cache", name, b.pv.Name)
		}
	}
}

func (env *testEnv) validateBind(
	t *testing.T,
	name string,
	pod *v1.Pod,
	expectedPVs []*v1.PersistentVolume,
	expectedAPIPVs []*v1.PersistentVolume) {

	// Check pvsToBind annotation is deleted
	result, ok := pod.Annotations[annPVsToBind]
	if ok {
		t.Errorf("Test %q failed: Expected pvsToBind not set, got %q", name, result)
	}

	// Check pv cache
	pvCache := env.internalBinder.pvCache
	for _, pv := range expectedPVs {
		cachedPV := pvCache.GetPV(pv.Name)
		if !reflect.DeepEqual(cachedPV, pv) {
			t.Errorf("Test %q failed: cached PV check failed [A-expected, B-got]:\n%s", name, diff.ObjectDiff(pv, cachedPV))
		}
	}

	// Check reactor for API updates
	if err := env.reactor.checkVolumes(expectedAPIPVs); err != nil {
		t.Errorf("Test %q failed: API reactor validation failed: %v", name, err)
	}
}

const (
	pvcUnbound = iota
	pvcPrebound
	pvcBound
)

func makeTestPVC(name, size string, pvcBoundState int, pvName string, className *string) *v1.PersistentVolumeClaim {
	pvc := &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       "testns",
			UID:             types.UID("pvc-uid"),
			ResourceVersion: "1",
			SelfLink:        testapi.Default.SelfLink("pvc", name),
		},
		Spec: v1.PersistentVolumeClaimSpec{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(size),
				},
			},
			StorageClassName: className,
		},
	}

	switch pvcBoundState {
	case pvcBound:
		metav1.SetMetaDataAnnotation(&pvc.ObjectMeta, annBindCompleted, "yes")
		fallthrough
	case pvcPrebound:
		pvc.Spec.VolumeName = pvName
	}
	return pvc
}

func makeBadPVC() *v1.PersistentVolumeClaim {
	return &v1.PersistentVolumeClaim{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "bad-pvc",
			Namespace:       "testns",
			UID:             types.UID("pvc-uid"),
			ResourceVersion: "1",
			// Don't include SefLink, so that GetReference will fail
		},
		Spec: v1.PersistentVolumeClaimSpec{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse("1G"),
				},
			},
			StorageClassName: &waitClass,
		},
	}
}

func makeTestPV(name, node, capacity, version string, boundToPVC *v1.PersistentVolumeClaim, className string) *v1.PersistentVolume {
	pv := &v1.PersistentVolume{
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			ResourceVersion: version,
		},
		Spec: v1.PersistentVolumeSpec{
			Capacity: v1.ResourceList{
				v1.ResourceName(v1.ResourceStorage): resource.MustParse(capacity),
			},
			StorageClassName: className,
		},
	}
	if node != "" {
		pv.Annotations = getAnnotationWithNodeAffinity("key1", node)
	}

	if boundToPVC != nil {
		pv.Spec.ClaimRef = &v1.ObjectReference{
			Name:      boundToPVC.Name,
			Namespace: boundToPVC.Namespace,
			UID:       boundToPVC.UID,
		}
		metav1.SetMetaDataAnnotation(&pv.ObjectMeta, annBoundByController, "yes")
	}

	return pv
}

func makePod(pvcs []*v1.PersistentVolumeClaim) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "testns",
		},
	}

	volumes := []v1.Volume{}
	for i, pvc := range pvcs {
		pvcVol := v1.Volume{
			Name: fmt.Sprintf("vol%v", i),
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: pvc.Name,
				},
			},
		}
		volumes = append(volumes, pvcVol)
	}
	pod.Spec.Volumes = volumes
	return pod
}

func makePodWithoutPVC() *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "test-pod",
			Namespace: "testns",
		},
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						EmptyDir: &v1.EmptyDirVolumeSource{},
					},
				},
			},
		},
	}
	return pod
}

func makeBinding(pvc *v1.PersistentVolumeClaim, pv *v1.PersistentVolume) *bindingInfo {
	return &bindingInfo{pvc: pvc, pv: pv}
}

func makeStringPtr(str string) *string {
	s := fmt.Sprintf("%v", str)
	return &s
}

func TestFindPodVolumes(t *testing.T) {
	scenarios := map[string]struct {
		// Inputs
		pvs     []*v1.PersistentVolume
		podPVCs []*v1.PersistentVolumeClaim
		// Defaults to node1
		node string
		// If nil, use pod PVCs
		cachePVCs []*v1.PersistentVolumeClaim
		// If nil, makePod with podPVCs
		pod *v1.Pod

		// Expected tmpData fields
		expectedBindings []*bindingInfo
		expectedAllBound bool

		// Expected return values
		expectedNeedsBinding bool
		expectedFoundPVs     bool
		shouldFail           bool
	}{
		"no-volumes": {
			pod:                  makePod(nil),
			expectedAllBound:     true,
			expectedFoundPVs:     true,
			expectedNeedsBinding: false,
		},
		"no-pvcs": {
			pod:                  makePodWithoutPVC(),
			expectedAllBound:     true,
			expectedFoundPVs:     true,
			expectedNeedsBinding: false,
		},
		"pvc-not-found": {
			cachePVCs:            []*v1.PersistentVolumeClaim{},
			podPVCs:              []*v1.PersistentVolumeClaim{boundPVC},
			expectedAllBound:     false,
			expectedFoundPVs:     false,
			expectedNeedsBinding: false,
			shouldFail:           true,
		},
		"bound-pvc": {
			podPVCs:              []*v1.PersistentVolumeClaim{boundPVC},
			pvs:                  []*v1.PersistentVolume{pvBound},
			expectedAllBound:     true,
			expectedFoundPVs:     true,
			expectedNeedsBinding: false,
		},
		"bound-pvc,pv-not-exists": {
			podPVCs:              []*v1.PersistentVolumeClaim{boundPVC},
			expectedAllBound:     true,
			expectedFoundPVs:     false,
			expectedNeedsBinding: false,
			shouldFail:           true,
		},
		"prebound-pvc": {
			podPVCs:              []*v1.PersistentVolumeClaim{preboundPVC},
			pvs:                  []*v1.PersistentVolume{pvNode1aBound},
			expectedAllBound:     false,
			expectedFoundPVs:     true,
			expectedNeedsBinding: false,
		},
		"unbound-pvc,node-not-exists": {
			podPVCs:              []*v1.PersistentVolumeClaim{unboundPVC},
			node:                 "node12",
			expectedAllBound:     false,
			expectedFoundPVs:     false,
			expectedNeedsBinding: false,
			shouldFail:           true,
		},
		"unbound-pvc,pv-same-node": {
			podPVCs:              []*v1.PersistentVolumeClaim{unboundPVC},
			pvs:                  []*v1.PersistentVolume{pvNode2, pvNode1a, pvNode1b},
			expectedBindings:     []*bindingInfo{binding1a},
			expectedAllBound:     false,
			expectedFoundPVs:     true,
			expectedNeedsBinding: true,
		},
		"unbound-pvc,pv-different-node": {
			podPVCs:              []*v1.PersistentVolumeClaim{unboundPVC},
			pvs:                  []*v1.PersistentVolume{pvNode2},
			expectedAllBound:     false,
			expectedFoundPVs:     false,
			expectedNeedsBinding: true,
		},
		"two-unbound-pvcs": {
			podPVCs:              []*v1.PersistentVolumeClaim{unboundPVC, unboundPVC2},
			pvs:                  []*v1.PersistentVolume{pvNode1a, pvNode1b},
			expectedBindings:     []*bindingInfo{binding1a, binding1b},
			expectedAllBound:     false,
			expectedFoundPVs:     true,
			expectedNeedsBinding: true,
		},
		"two-unbound-pvcs,order-by-size": {
			podPVCs:              []*v1.PersistentVolumeClaim{unboundPVC2, unboundPVC},
			pvs:                  []*v1.PersistentVolume{pvNode1a, pvNode1b},
			expectedBindings:     []*bindingInfo{binding1a, binding1b},
			expectedAllBound:     false,
			expectedFoundPVs:     true,
			expectedNeedsBinding: true,
		},
		"two-unbound-pvcs,partial-match": {
			podPVCs:              []*v1.PersistentVolumeClaim{unboundPVC, unboundPVC2},
			pvs:                  []*v1.PersistentVolume{pvNode1a},
			expectedAllBound:     false,
			expectedFoundPVs:     false,
			expectedNeedsBinding: true,
		},
		"one-bound,one-unbound": {
			podPVCs:              []*v1.PersistentVolumeClaim{unboundPVC, boundPVC},
			pvs:                  []*v1.PersistentVolume{pvBound, pvNode1a},
			expectedBindings:     []*bindingInfo{binding1a},
			expectedAllBound:     false,
			expectedFoundPVs:     true,
			expectedNeedsBinding: true,
		},
		"one-bound,one-unbound,no-match": {
			podPVCs:              []*v1.PersistentVolumeClaim{unboundPVC, boundPVC},
			pvs:                  []*v1.PersistentVolume{pvBound, pvNode2},
			expectedAllBound:     false,
			expectedFoundPVs:     false,
			expectedNeedsBinding: true,
		},
		"one-prebound,one-unbound": {
			podPVCs:              []*v1.PersistentVolumeClaim{unboundPVC, preboundPVC},
			pvs:                  []*v1.PersistentVolume{pvNode1a, pvNode1b},
			expectedBindings:     []*bindingInfo{binding1a},
			expectedAllBound:     false,
			expectedFoundPVs:     true,
			expectedNeedsBinding: true,
		},
		"immediate-bound-pvc": {
			podPVCs:              []*v1.PersistentVolumeClaim{immediateBoundPVC},
			pvs:                  []*v1.PersistentVolume{pvBoundImmediate},
			expectedAllBound:     true,
			expectedFoundPVs:     true,
			expectedNeedsBinding: false,
		},
		"immediate-bound-pvc-wrong-node": {
			podPVCs:              []*v1.PersistentVolumeClaim{immediateBoundPVC},
			pvs:                  []*v1.PersistentVolume{pvBoundImmediateNode2},
			expectedAllBound:     true,
			expectedFoundPVs:     false,
			expectedNeedsBinding: false,
		},
		"immediate-unbound-pvc": {
			podPVCs:              []*v1.PersistentVolumeClaim{immediateUnboundPVC},
			expectedAllBound:     false,
			expectedFoundPVs:     false,
			expectedNeedsBinding: true,
			shouldFail:           true,
		},
		"immediate-unbound-pvc,delayed-mode-bound": {
			podPVCs:              []*v1.PersistentVolumeClaim{immediateUnboundPVC, boundPVC},
			pvs:                  []*v1.PersistentVolume{pvBound},
			expectedAllBound:     false,
			expectedFoundPVs:     false,
			expectedNeedsBinding: true,
			shouldFail:           true,
		},
		"immediate-unbound-pvc,delayed-mode-unbound": {
			podPVCs:              []*v1.PersistentVolumeClaim{immediateUnboundPVC, unboundPVC},
			expectedAllBound:     false,
			expectedFoundPVs:     false,
			expectedNeedsBinding: true,
			shouldFail:           true,
		},
	}

	// Set feature gate
	err := utilfeature.DefaultFeatureGate.Set("VolumeTopologyBinding=true")
	if err != nil {
		t.Fatalf("Failed to enable feature gate for VolumeTopologyBinding: %v", err)
	}

	for name, scenario := range scenarios {
		glog.V(5).Infof("Running test case %q", name)

		// Setup
		testEnv := newTestBinder(t)
		testEnv.initVolumes(scenario.pvs, scenario.pvs)
		if scenario.node == "" {
			scenario.node = "node1"
		}

		// a. Init pvc cache
		if scenario.cachePVCs == nil {
			scenario.cachePVCs = scenario.podPVCs
		}
		testEnv.initClaims(t, scenario.cachePVCs)

		// b. Generate pod with given claims
		if scenario.pod == nil {
			scenario.pod = makePod(scenario.podPVCs)
		}
		testEnv.binder.InitTmpData(scenario.pod)

		// Execute
		needsBinding, foundPVs, err := testEnv.binder.FindPodVolumes(scenario.pod, scenario.node)

		// Validate
		if !scenario.shouldFail && err != nil {
			t.Errorf("Test %q failed: returned error: %v", name, err)
		}
		if scenario.shouldFail && err == nil {
			t.Errorf("Test %q failed: returned success but expected error", name)
		}
		if needsBinding != scenario.expectedNeedsBinding {
			t.Errorf("Test %q failed: expected needsBinding %v, got %v", name, scenario.expectedNeedsBinding, needsBinding)
		}
		if foundPVs != scenario.expectedFoundPVs {
			t.Errorf("Test %q failed: expected foundPVs %v, got %v", name, scenario.expectedFoundPVs, foundPVs)
		}
		testEnv.validateTmpData(t, name, scenario.node, scenario.expectedAllBound, scenario.expectedBindings)
	}

	err = utilfeature.DefaultFeatureGate.Set("VolumeTopologyBinding=false")
	if err != nil {
		t.Fatalf("Failed to disable feature gate for VolumeTopologyBinding: %v", err)
	}
}

func TestAssumePodVolumes(t *testing.T) {
	scenarios := map[string]struct {
		// Inputs
		allBound bool
		bindings []*bindingInfo
		pvs      []*v1.PersistentVolume

		// Expected return values
		shouldFail              bool
		expectedBindingRequired bool
		expectedPVsToBind       *string
		// if nil, use bindings
		expectedBindings []*bindingInfo
	}{
		"all-bound": {
			allBound: true,
		},
		"not-fully-bound": {
			expectedBindingRequired: true,
			expectedPVsToBind:       &bindEmptyStr,
		},
		"one-binding": {
			bindings: []*bindingInfo{binding1a},
			pvs:      []*v1.PersistentVolume{pvNode1a},
			expectedBindingRequired: true,
			expectedPVsToBind:       &bindOneStr,
		},
		"two-bindings": {
			bindings: []*bindingInfo{binding1a, binding1b},
			pvs:      []*v1.PersistentVolume{pvNode1a, pvNode1b},
			expectedBindingRequired: true,
			expectedPVsToBind:       &bindTwoStr,
		},
		"already-bound": {
			bindings: []*bindingInfo{binding1aBound},
			pvs:      []*v1.PersistentVolume{pvNode1aBound},
			expectedBindingRequired: true,
			expectedBindings:        []*bindingInfo{},
			expectedPVsToBind:       &bindEmptyStr,
		},
		"claimref-failed": {
			bindings:                []*bindingInfo{binding1a, bindingBad},
			pvs:                     []*v1.PersistentVolume{pvNode1a, pvNode1b},
			shouldFail:              true,
			expectedBindingRequired: true,
		},
		"tmpupdate-failed": {
			bindings:                []*bindingInfo{binding1a, binding1b},
			pvs:                     []*v1.PersistentVolume{pvNode1a},
			shouldFail:              true,
			expectedBindingRequired: true,
		},
	}

	for name, scenario := range scenarios {
		glog.V(5).Infof("Running test case %q", name)

		// Setup
		testEnv := newTestBinder(t)
		pod := makePod(nil)
		testEnv.binder.InitTmpData(pod)
		testEnv.initTmpData(scenario.allBound, scenario.bindings)
		testEnv.initVolumes(scenario.pvs, scenario.pvs)

		// Execute
		bindingRequired, err := testEnv.binder.AssumePodVolumes(pod, "node1")

		// Validate
		if !scenario.shouldFail && err != nil {
			t.Errorf("Test %q failed: returned error: %v", name, err)
		}
		if scenario.shouldFail && err == nil {
			t.Errorf("Test %q failed: returned success but expected error", name)
		}
		if scenario.expectedBindingRequired != bindingRequired {
			t.Errorf("Test %q failed: returned unexpected bindingRequired %v", name, scenario.expectedBindingRequired)
		}
		if scenario.expectedBindings == nil {
			scenario.expectedBindings = scenario.bindings
		}
		if scenario.shouldFail || scenario.allBound {
			testEnv.validateFailedAssume(t, name, pod, scenario.expectedBindings)
		} else {
			testEnv.validateAssume(t, name, pod, scenario.expectedBindings, scenario.expectedPVsToBind)
		}
	}
}

func TestBindPodVolumes(t *testing.T) {
	scenarios := map[string]struct {
		// Inputs
		pvsToBind  *string
		cachedPVCs []*v1.PersistentVolumeClaim
		cachedPVs  []*v1.PersistentVolume
		assumedPVs []*v1.PersistentVolume
		// if nil, use cachedPVs
		apiPVs []*v1.PersistentVolume

		// Expected return values
		shouldFail              bool
		expectedBindingRequired bool
		expectedPVs             []*v1.PersistentVolume
		// if nil, use expectedPVs
		expectedAPIPVs []*v1.PersistentVolume
	}{
		"all-bound": {},
		"not-fully-bound": {
			expectedBindingRequired: true,
			pvsToBind:               makeStringPtr(""),
		},
		"one-binding": {
			pvsToBind:               &bindOneStr,
			cachedPVCs:              []*v1.PersistentVolumeClaim{unboundPVC},
			cachedPVs:               []*v1.PersistentVolume{pvNode1a},
			assumedPVs:              []*v1.PersistentVolume{pvNode1aBound},
			expectedPVs:             []*v1.PersistentVolume{pvNode1aBound},
			expectedBindingRequired: true,
		},
		"two-bindings": {
			pvsToBind:               &bindTwoStr,
			cachedPVCs:              []*v1.PersistentVolumeClaim{unboundPVC, unboundPVC2},
			cachedPVs:               []*v1.PersistentVolume{pvNode1a, pvNode1b},
			assumedPVs:              []*v1.PersistentVolume{pvNode1aBound, pvNode1bBound},
			expectedPVs:             []*v1.PersistentVolume{pvNode1aBound, pvNode1bBound},
			expectedBindingRequired: true,
		},
		"pv-not-found": {
			pvsToBind:               &bindTwoStr,
			cachedPVCs:              []*v1.PersistentVolumeClaim{unboundPVC, unboundPVC2},
			cachedPVs:               []*v1.PersistentVolume{pvNode1a},
			assumedPVs:              []*v1.PersistentVolume{pvNode1aBound},
			expectedPVs:             []*v1.PersistentVolume{pvNode1a},
			expectedBindingRequired: true,
			shouldFail:              true,
		},
		"pv-claimref-not-set": {
			pvsToBind:               &bindTwoStr,
			cachedPVCs:              []*v1.PersistentVolumeClaim{unboundPVC, unboundPVC2},
			cachedPVs:               []*v1.PersistentVolume{pvNode1a, pvNode1b},
			assumedPVs:              []*v1.PersistentVolume{pvNode1aBound},
			expectedPVs:             []*v1.PersistentVolume{pvNode1a, pvNode1b},
			expectedBindingRequired: true,
			shouldFail:              true,
		},
		"pvc-not-found": {
			pvsToBind:               &bindTwoStr,
			cachedPVCs:              []*v1.PersistentVolumeClaim{unboundPVC},
			cachedPVs:               []*v1.PersistentVolume{pvNode1a, pvNode1b},
			assumedPVs:              []*v1.PersistentVolume{pvNode1aBound, pvNode1bBound},
			expectedPVs:             []*v1.PersistentVolume{pvNode1a, pvNode1b},
			expectedBindingRequired: true,
			shouldFail:              true,
		},
		"api-update-failed": {
			pvsToBind:               &bindTwoStr,
			cachedPVCs:              []*v1.PersistentVolumeClaim{unboundPVC, unboundPVC2},
			cachedPVs:               []*v1.PersistentVolume{pvNode1a, pvNode1b},
			apiPVs:                  []*v1.PersistentVolume{pvNode1a, pvNode1bBoundHigherVersion},
			assumedPVs:              []*v1.PersistentVolume{pvNode1aBound, pvNode1bBound},
			expectedPVs:             []*v1.PersistentVolume{pvNode1aBound, pvNode1b},
			expectedAPIPVs:          []*v1.PersistentVolume{pvNode1aBound, pvNode1bBoundHigherVersion},
			expectedBindingRequired: true,
			shouldFail:              true,
		},
	}
	for name, scenario := range scenarios {
		glog.V(5).Infof("Running test case %q", name)

		// Setup
		testEnv := newTestBinder(t)
		pod := makePod(nil)
		if scenario.apiPVs == nil {
			scenario.apiPVs = scenario.cachedPVs
		}
		testEnv.initVolumes(scenario.cachedPVs, scenario.apiPVs)
		testEnv.initClaims(t, scenario.cachedPVCs)
		testEnv.assumeVolumes(t, name, scenario.assumedPVs)
		testEnv.initPVsToBind(pod, scenario.pvsToBind)

		// Execute
		bindingRequired, err := testEnv.binder.BindPodVolumes(pod)

		// Validate
		if !scenario.shouldFail && err != nil {
			t.Errorf("Test %q failed: returned error: %v", name, err)
		}
		if scenario.shouldFail && err == nil {
			t.Errorf("Test %q failed: returned success but expected error", name)
		}
		if scenario.expectedBindingRequired != bindingRequired {
			t.Errorf("Test %q failed: returned unexpected bindingRequired %v", name, scenario.expectedBindingRequired)
		}
		if scenario.expectedAPIPVs == nil {
			scenario.expectedAPIPVs = scenario.expectedPVs
		}
		testEnv.validateBind(t, name, pod, scenario.expectedPVs, scenario.expectedAPIPVs)
	}
}
