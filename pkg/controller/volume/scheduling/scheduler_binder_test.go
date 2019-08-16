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

package scheduling

import (
	"context"
	"fmt"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/client-go/informers"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/controller"
	pvtesting "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/testing"
	pvutil "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
)

var (
	// PVCs for manual binding
	// TODO: clean up all of these
	unboundPVC          = makeTestPVC("unbound-pvc", "1G", "", pvcUnbound, "", "1", &waitClass)
	unboundPVC2         = makeTestPVC("unbound-pvc2", "5G", "", pvcUnbound, "", "1", &waitClass)
	preboundPVC         = makeTestPVC("prebound-pvc", "1G", "", pvcPrebound, "pv-node1a", "1", &waitClass)
	preboundPVCNode1a   = makeTestPVC("unbound-pvc", "1G", "", pvcPrebound, "pv-node1a", "1", &waitClass)
	boundPVC            = makeTestPVC("bound-pvc", "1G", "", pvcBound, "pv-bound", "1", &waitClass)
	boundPVC2           = makeTestPVC("bound-pvc2", "1G", "", pvcBound, "pv-bound2", "1", &waitClass)
	boundPVCNode1a      = makeTestPVC("unbound-pvc", "1G", "", pvcBound, "pv-node1a", "1", &waitClass)
	immediateUnboundPVC = makeTestPVC("immediate-unbound-pvc", "1G", "", pvcUnbound, "", "1", &immediateClass)
	immediateBoundPVC   = makeTestPVC("immediate-bound-pvc", "1G", "", pvcBound, "pv-bound-immediate", "1", &immediateClass)

	// PVCs for dynamic provisioning
	provisionedPVC              = makeTestPVC("provisioned-pvc", "1Gi", "", pvcUnbound, "", "1", &waitClassWithProvisioner)
	provisionedPVC2             = makeTestPVC("provisioned-pvc2", "1Gi", "", pvcUnbound, "", "1", &waitClassWithProvisioner)
	provisionedPVCHigherVersion = makeTestPVC("provisioned-pvc2", "1Gi", "", pvcUnbound, "", "2", &waitClassWithProvisioner)
	provisionedPVCBound         = makeTestPVC("provisioned-pvc", "1Gi", "", pvcBound, "pv-bound", "1", &waitClassWithProvisioner)
	noProvisionerPVC            = makeTestPVC("no-provisioner-pvc", "1Gi", "", pvcUnbound, "", "1", &waitClass)
	topoMismatchPVC             = makeTestPVC("topo-mismatch-pvc", "1Gi", "", pvcUnbound, "", "1", &topoMismatchClass)

	selectedNodePVC = makeTestPVC("provisioned-pvc", "1Gi", nodeLabelValue, pvcSelectedNode, "", "1", &waitClassWithProvisioner)

	// PVs for manual binding
	pvNoNode                   = makeTestPV("pv-no-node", "", "1G", "1", nil, waitClass)
	pvNode1a                   = makeTestPV("pv-node1a", "node1", "5G", "1", nil, waitClass)
	pvNode1b                   = makeTestPV("pv-node1b", "node1", "10G", "1", nil, waitClass)
	pvNode1c                   = makeTestPV("pv-node1b", "node1", "5G", "1", nil, waitClass)
	pvNode2                    = makeTestPV("pv-node2", "node2", "1G", "1", nil, waitClass)
	pvPrebound                 = makeTestPV("pv-prebound", "node1", "1G", "1", unboundPVC, waitClass)
	pvBound                    = makeTestPV("pv-bound", "node1", "1G", "1", boundPVC, waitClass)
	pvNode1aBound              = makeTestPV("pv-node1a", "node1", "5G", "1", unboundPVC, waitClass)
	pvNode1bBound              = makeTestPV("pv-node1b", "node1", "10G", "1", unboundPVC2, waitClass)
	pvNode1bBoundHigherVersion = makeTestPV("pv-node1b", "node1", "10G", "2", unboundPVC2, waitClass)
	pvBoundImmediate           = makeTestPV("pv-bound-immediate", "node1", "1G", "1", immediateBoundPVC, immediateClass)
	pvBoundImmediateNode2      = makeTestPV("pv-bound-immediate", "node2", "1G", "1", immediateBoundPVC, immediateClass)

	// storage class names
	waitClass                = "waitClass"
	immediateClass           = "immediateClass"
	waitClassWithProvisioner = "waitClassWithProvisioner"
	topoMismatchClass        = "topoMismatchClass"

	// nodes objects
	node1         = makeNode("node1", map[string]string{nodeLabelKey: "node1"})
	node2         = makeNode("node2", map[string]string{nodeLabelKey: "node2"})
	node1NoLabels = makeNode("node1", nil)

	// node topology
	nodeLabelKey   = "nodeKey"
	nodeLabelValue = "node1"
)

type testEnv struct {
	client               clientset.Interface
	reactor              *pvtesting.VolumeReactor
	binder               SchedulerVolumeBinder
	internalBinder       *volumeBinder
	internalNodeInformer coreinformers.NodeInformer
	internalPVCache      *assumeCache
	internalPVCCache     *assumeCache
}

func newTestBinder(t *testing.T, stopCh <-chan struct{}) *testEnv {
	client := &fake.Clientset{}
	reactor := pvtesting.NewVolumeReactor(client, nil, nil, nil)
	// TODO refactor all tests to use real watch mechanism, see #72327
	client.AddWatchReactor("*", func(action k8stesting.Action) (handled bool, ret watch.Interface, err error) {
		gvr := action.GetResource()
		ns := action.GetNamespace()
		watch, err := reactor.Watch(gvr, ns)
		if err != nil {
			return false, nil, err
		}
		return true, watch, nil
	})
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())

	nodeInformer := informerFactory.Core().V1().Nodes()
	pvcInformer := informerFactory.Core().V1().PersistentVolumeClaims()
	classInformer := informerFactory.Storage().V1().StorageClasses()
	binder := NewVolumeBinder(
		client,
		nodeInformer,
		pvcInformer,
		informerFactory.Core().V1().PersistentVolumes(),
		classInformer,
		10*time.Second)

	// Wait for informers cache sync
	informerFactory.Start(stopCh)
	for v, synced := range informerFactory.WaitForCacheSync(stopCh) {
		if !synced {
			klog.Fatalf("Error syncing informer for %v", v)
		}
	}

	// Add storageclasses
	waitMode := storagev1.VolumeBindingWaitForFirstConsumer
	immediateMode := storagev1.VolumeBindingImmediate
	classes := []*storagev1.StorageClass{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: waitClassWithProvisioner,
			},
			VolumeBindingMode: &waitMode,
			Provisioner:       "test-provisioner",
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    nodeLabelKey,
							Values: []string{nodeLabelValue, "reference-value"},
						},
					},
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: immediateClass,
			},
			VolumeBindingMode: &immediateMode,
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: waitClass,
			},
			VolumeBindingMode: &waitMode,
			Provisioner:       "kubernetes.io/no-provisioner",
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: topoMismatchClass,
			},
			VolumeBindingMode: &waitMode,
			Provisioner:       "test-provisioner",
			AllowedTopologies: []v1.TopologySelectorTerm{
				{
					MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
						{
							Key:    nodeLabelKey,
							Values: []string{"reference-value"},
						},
					},
				},
			},
		},
	}
	for _, class := range classes {
		if err := classInformer.Informer().GetIndexer().Add(class); err != nil {
			t.Fatalf("Failed to add storage class to internal cache: %v", err)
		}
	}

	// Get internal types
	internalBinder, ok := binder.(*volumeBinder)
	if !ok {
		t.Fatalf("Failed to convert to internal binder")
	}

	pvCache := internalBinder.pvCache
	internalPVCache, ok := pvCache.(*pvAssumeCache).AssumeCache.(*assumeCache)
	if !ok {
		t.Fatalf("Failed to convert to internal PV cache")
	}

	pvcCache := internalBinder.pvcCache
	internalPVCCache, ok := pvcCache.(*pvcAssumeCache).AssumeCache.(*assumeCache)
	if !ok {
		t.Fatalf("Failed to convert to internal PVC cache")
	}

	return &testEnv{
		client:               client,
		reactor:              reactor,
		binder:               binder,
		internalBinder:       internalBinder,
		internalNodeInformer: nodeInformer,
		internalPVCache:      internalPVCache,
		internalPVCCache:     internalPVCCache,
	}
}

func (env *testEnv) initNodes(cachedNodes []*v1.Node) {
	nodeInformer := env.internalNodeInformer.Informer()
	for _, node := range cachedNodes {
		nodeInformer.GetIndexer().Add(node)
	}
}

func (env *testEnv) initClaims(cachedPVCs []*v1.PersistentVolumeClaim, apiPVCs []*v1.PersistentVolumeClaim) {
	internalPVCCache := env.internalPVCCache
	for _, pvc := range cachedPVCs {
		internalPVCCache.add(pvc)
		if apiPVCs == nil {
			env.reactor.AddClaim(pvc)
		}
	}
	for _, pvc := range apiPVCs {
		env.reactor.AddClaim(pvc)
	}
}

func (env *testEnv) initVolumes(cachedPVs []*v1.PersistentVolume, apiPVs []*v1.PersistentVolume) {
	internalPVCache := env.internalPVCache
	for _, pv := range cachedPVs {
		internalPVCache.add(pv)
		if apiPVs == nil {
			env.reactor.AddVolume(pv)
		}
	}
	for _, pv := range apiPVs {
		env.reactor.AddVolume(pv)
	}

}

func (env *testEnv) updateVolumes(t *testing.T, pvs []*v1.PersistentVolume, waitCache bool) {
	for _, pv := range pvs {
		if _, err := env.client.CoreV1().PersistentVolumes().Update(pv); err != nil {
			t.Fatalf("failed to update PV %q", pv.Name)
		}
	}
	if waitCache {
		wait.Poll(100*time.Millisecond, 3*time.Second, func() (bool, error) {
			for _, pv := range pvs {
				obj, err := env.internalPVCache.GetAPIObj(pv.Name)
				if obj == nil || err != nil {
					return false, nil
				}
				pvInCache, ok := obj.(*v1.PersistentVolume)
				if !ok {
					return false, fmt.Errorf("PV %s invalid object", pvInCache.Name)
				}
				return versioner.CompareResourceVersion(pvInCache, pv) == 0, nil
			}
			return true, nil
		})
	}
}

func (env *testEnv) updateClaims(t *testing.T, pvcs []*v1.PersistentVolumeClaim, waitCache bool) {
	for _, pvc := range pvcs {
		if _, err := env.client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Update(pvc); err != nil {
			t.Fatalf("failed to update PVC %q", getPVCName(pvc))
		}
	}
	if waitCache {
		wait.Poll(100*time.Millisecond, 3*time.Second, func() (bool, error) {
			for _, pvc := range pvcs {
				obj, err := env.internalPVCCache.GetAPIObj(getPVCName(pvc))
				if obj == nil || err != nil {
					return false, nil
				}
				pvcInCache, ok := obj.(*v1.PersistentVolumeClaim)
				if !ok {
					return false, fmt.Errorf("PVC %s invalid object", pvcInCache.Name)
				}
				return versioner.CompareResourceVersion(pvcInCache, pvc) == 0, nil
			}
			return true, nil
		})
	}
}

func (env *testEnv) deleteVolumes(pvs []*v1.PersistentVolume) {
	for _, pv := range pvs {
		env.internalPVCache.delete(pv)
	}
}

func (env *testEnv) deleteClaims(pvcs []*v1.PersistentVolumeClaim) {
	for _, pvc := range pvcs {
		env.internalPVCCache.delete(pvc)
	}
}

func (env *testEnv) assumeVolumes(t *testing.T, name, node string, pod *v1.Pod, bindings []*bindingInfo, provisionings []*v1.PersistentVolumeClaim) {
	pvCache := env.internalBinder.pvCache
	for _, binding := range bindings {
		if err := pvCache.Assume(binding.pv); err != nil {
			t.Fatalf("Failed to setup test %q: error: %v", name, err)
		}
	}

	pvcCache := env.internalBinder.pvcCache
	for _, pvc := range provisionings {
		if err := pvcCache.Assume(pvc); err != nil {
			t.Fatalf("Failed to setup test %q: error: %v", name, err)
		}
	}

	env.internalBinder.podBindingCache.UpdateBindings(pod, node, bindings, provisionings)
}

func (env *testEnv) initPodCache(pod *v1.Pod, node string, bindings []*bindingInfo, provisionings []*v1.PersistentVolumeClaim) {
	cache := env.internalBinder.podBindingCache
	cache.UpdateBindings(pod, node, bindings, provisionings)
}

func (env *testEnv) validatePodCache(t *testing.T, name, node string, pod *v1.Pod, expectedBindings []*bindingInfo, expectedProvisionings []*v1.PersistentVolumeClaim) {
	cache := env.internalBinder.podBindingCache
	bindings := cache.GetBindings(pod, node)
	if aLen, eLen := len(bindings), len(expectedBindings); aLen != eLen {
		t.Errorf("Test %q failed. expected %v bindings, got %v", name, eLen, aLen)
	} else if expectedBindings == nil && bindings != nil {
		// nil and empty are different
		t.Errorf("Test %q failed. expected nil bindings, got empty", name)
	} else if expectedBindings != nil && bindings == nil {
		// nil and empty are different
		t.Errorf("Test %q failed. expected empty bindings, got nil", name)
	} else {
		for i := 0; i < aLen; i++ {
			// Validate PV
			if !reflect.DeepEqual(expectedBindings[i].pv, bindings[i].pv) {
				t.Errorf("Test %q failed. binding.pv doesn't match [A-expected, B-got]: %s", name, diff.ObjectDiff(expectedBindings[i].pv, bindings[i].pv))
			}

			// Validate PVC
			if !reflect.DeepEqual(expectedBindings[i].pvc, bindings[i].pvc) {
				t.Errorf("Test %q failed. binding.pvc doesn't match [A-expected, B-got]: %s", name, diff.ObjectDiff(expectedBindings[i].pvc, bindings[i].pvc))
			}
		}
	}

	provisionedClaims := cache.GetProvisionedPVCs(pod, node)
	if aLen, eLen := len(provisionedClaims), len(expectedProvisionings); aLen != eLen {
		t.Errorf("Test %q failed. expected %v provisioned claims, got %v", name, eLen, aLen)
	} else if expectedProvisionings == nil && provisionedClaims != nil {
		// nil and empty are different
		t.Errorf("Test %q failed. expected nil provisionings, got empty", name)
	} else if expectedProvisionings != nil && provisionedClaims == nil {
		// nil and empty are different
		t.Errorf("Test %q failed. expected empty provisionings, got nil", name)
	} else {
		for i := 0; i < aLen; i++ {
			if !reflect.DeepEqual(expectedProvisionings[i], provisionedClaims[i]) {
				t.Errorf("Test %q failed. provisioned claims doesn't match [A-expected, B-got]: %s", name, diff.ObjectDiff(expectedProvisionings[i], provisionedClaims[i]))
			}
		}
	}
}

func (env *testEnv) getPodBindings(t *testing.T, name, node string, pod *v1.Pod) []*bindingInfo {
	cache := env.internalBinder.podBindingCache
	return cache.GetBindings(pod, node)
}

func (env *testEnv) validateAssume(t *testing.T, name string, pod *v1.Pod, bindings []*bindingInfo, provisionings []*v1.PersistentVolumeClaim) {
	// Check pv cache
	pvCache := env.internalBinder.pvCache
	for _, b := range bindings {
		pv, err := pvCache.GetPV(b.pv.Name)
		if err != nil {
			t.Errorf("Test %q failed: GetPV %q returned error: %v", name, b.pv.Name, err)
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

	// Check pvc cache
	pvcCache := env.internalBinder.pvcCache
	for _, p := range provisionings {
		pvcKey := getPVCName(p)
		pvc, err := pvcCache.GetPVC(pvcKey)
		if err != nil {
			t.Errorf("Test %q failed: GetPVC %q returned error: %v", name, pvcKey, err)
			continue
		}
		if pvc.Annotations[pvutil.AnnSelectedNode] != nodeLabelValue {
			t.Errorf("Test %q failed: expected pvutil.AnnSelectedNode of pvc %q to be %q, but got %q", name, pvcKey, nodeLabelValue, pvc.Annotations[pvutil.AnnSelectedNode])
		}
	}
}

func (env *testEnv) validateFailedAssume(t *testing.T, name string, pod *v1.Pod, bindings []*bindingInfo, provisionings []*v1.PersistentVolumeClaim) {
	// All PVs have been unmodified in cache
	pvCache := env.internalBinder.pvCache
	for _, b := range bindings {
		pv, _ := pvCache.GetPV(b.pv.Name)
		// PV could be nil if it's missing from cache
		if pv != nil && pv != b.pv {
			t.Errorf("Test %q failed: PV %q was modified in cache", name, b.pv.Name)
		}
	}

	// Check pvc cache
	pvcCache := env.internalBinder.pvcCache
	for _, p := range provisionings {
		pvcKey := getPVCName(p)
		pvc, err := pvcCache.GetPVC(pvcKey)
		if err != nil {
			t.Errorf("Test %q failed: GetPVC %q returned error: %v", name, pvcKey, err)
			continue
		}
		if pvc.Annotations[pvutil.AnnSelectedNode] != "" {
			t.Errorf("Test %q failed: expected pvutil.AnnSelectedNode of pvc %q empty, but got %q", name, pvcKey, pvc.Annotations[pvutil.AnnSelectedNode])
		}
	}
}

func (env *testEnv) validateBind(
	t *testing.T,
	name string,
	pod *v1.Pod,
	expectedPVs []*v1.PersistentVolume,
	expectedAPIPVs []*v1.PersistentVolume) {

	// Check pv cache
	pvCache := env.internalBinder.pvCache
	for _, pv := range expectedPVs {
		cachedPV, err := pvCache.GetPV(pv.Name)
		if err != nil {
			t.Errorf("Test %q failed: GetPV %q returned error: %v", name, pv.Name, err)
		}
		// Cache may be overridden by API object with higher version, compare but ignore resource version.
		newCachedPV := cachedPV.DeepCopy()
		newCachedPV.ResourceVersion = pv.ResourceVersion
		if !reflect.DeepEqual(newCachedPV, pv) {
			t.Errorf("Test %q failed: cached PV check failed [A-expected, B-got]:\n%s", name, diff.ObjectDiff(pv, cachedPV))
		}
	}

	// Check reactor for API updates
	if err := env.reactor.CheckVolumes(expectedAPIPVs); err != nil {
		t.Errorf("Test %q failed: API reactor validation failed: %v", name, err)
	}
}

func (env *testEnv) validateProvision(
	t *testing.T,
	name string,
	pod *v1.Pod,
	expectedPVCs []*v1.PersistentVolumeClaim,
	expectedAPIPVCs []*v1.PersistentVolumeClaim) {

	// Check pvc cache
	pvcCache := env.internalBinder.pvcCache
	for _, pvc := range expectedPVCs {
		cachedPVC, err := pvcCache.GetPVC(getPVCName(pvc))
		if err != nil {
			t.Errorf("Test %q failed: GetPVC %q returned error: %v", name, getPVCName(pvc), err)
		}
		// Cache may be overridden by API object with higher version, compare but ignore resource version.
		newCachedPVC := cachedPVC.DeepCopy()
		newCachedPVC.ResourceVersion = pvc.ResourceVersion
		if !reflect.DeepEqual(newCachedPVC, pvc) {
			t.Errorf("Test %q failed: cached PVC check failed [A-expected, B-got]:\n%s", name, diff.ObjectDiff(pvc, cachedPVC))
		}
	}

	// Check reactor for API updates
	if err := env.reactor.CheckClaims(expectedAPIPVCs); err != nil {
		t.Errorf("Test %q failed: API reactor validation failed: %v", name, err)
	}
}

const (
	pvcUnbound = iota
	pvcPrebound
	pvcBound
	pvcSelectedNode
)

func makeTestPVC(name, size, node string, pvcBoundState int, pvName, resourceVersion string, className *string) *v1.PersistentVolumeClaim {
	fs := v1.PersistentVolumeFilesystem
	pvc := &v1.PersistentVolumeClaim{
		TypeMeta: metav1.TypeMeta{
			Kind:       "PersistentVolumeClaim",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            name,
			Namespace:       "testns",
			UID:             types.UID("pvc-uid"),
			ResourceVersion: resourceVersion,
			SelfLink:        testapi.Default.SelfLink("pvc", name),
		},
		Spec: v1.PersistentVolumeClaimSpec{
			Resources: v1.ResourceRequirements{
				Requests: v1.ResourceList{
					v1.ResourceName(v1.ResourceStorage): resource.MustParse(size),
				},
			},
			StorageClassName: className,
			VolumeMode:       &fs,
		},
	}

	switch pvcBoundState {
	case pvcSelectedNode:
		metav1.SetMetaDataAnnotation(&pvc.ObjectMeta, pvutil.AnnSelectedNode, node)
		// don't fallthrough
	case pvcBound:
		metav1.SetMetaDataAnnotation(&pvc.ObjectMeta, pvutil.AnnBindCompleted, "yes")
		fallthrough
	case pvcPrebound:
		pvc.Spec.VolumeName = pvName
	}
	return pvc
}

func makeTestPV(name, node, capacity, version string, boundToPVC *v1.PersistentVolumeClaim, className string) *v1.PersistentVolume {
	fs := v1.PersistentVolumeFilesystem
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
			VolumeMode:       &fs,
		},
		Status: v1.PersistentVolumeStatus{
			Phase: v1.VolumeAvailable,
		},
	}
	if node != "" {
		pv.Spec.NodeAffinity = pvutil.GetVolumeNodeAffinity(nodeLabelKey, node)
	}

	if boundToPVC != nil {
		pv.Spec.ClaimRef = &v1.ObjectReference{
			Kind:            boundToPVC.Kind,
			APIVersion:      boundToPVC.APIVersion,
			ResourceVersion: boundToPVC.ResourceVersion,
			Name:            boundToPVC.Name,
			Namespace:       boundToPVC.Namespace,
			UID:             boundToPVC.UID,
		}
		metav1.SetMetaDataAnnotation(&pv.ObjectMeta, pvutil.AnnBoundByController, "yes")
	}

	return pv
}

func pvcSetSelectedNode(pvc *v1.PersistentVolumeClaim, node string) *v1.PersistentVolumeClaim {
	newPVC := pvc.DeepCopy()
	metav1.SetMetaDataAnnotation(&newPVC.ObjectMeta, pvutil.AnnSelectedNode, node)
	return newPVC
}

func pvcSetEmptyAnnotations(pvc *v1.PersistentVolumeClaim) *v1.PersistentVolumeClaim {
	newPVC := pvc.DeepCopy()
	newPVC.Annotations = map[string]string{}
	return newPVC
}

func pvRemoveClaimUID(pv *v1.PersistentVolume) *v1.PersistentVolume {
	newPV := pv.DeepCopy()
	newPV.Spec.ClaimRef.UID = ""
	return newPV
}

func makeNode(name string, labels map[string]string) *v1.Node {
	return &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name:   name,
			Labels: labels,
		},
	}
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
	pod.Spec.NodeName = "node1"
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

func addProvisionAnn(pvc *v1.PersistentVolumeClaim) *v1.PersistentVolumeClaim {
	res := pvc.DeepCopy()
	// Add provision related annotations
	metav1.SetMetaDataAnnotation(&res.ObjectMeta, pvutil.AnnSelectedNode, nodeLabelValue)

	return res
}

func TestFindPodVolumesWithoutProvisioning(t *testing.T) {
	scenarios := map[string]struct {
		// Inputs
		pvs     []*v1.PersistentVolume
		podPVCs []*v1.PersistentVolumeClaim
		// If nil, use pod PVCs
		cachePVCs []*v1.PersistentVolumeClaim
		// If nil, makePod with podPVCs
		pod *v1.Pod

		// Expected podBindingCache fields
		expectedBindings []*bindingInfo

		// Expected return values
		expectedUnbound bool
		expectedBound   bool
		shouldFail      bool
	}{
		"no-volumes": {
			pod:             makePod(nil),
			expectedUnbound: true,
			expectedBound:   true,
		},
		"no-pvcs": {
			pod:             makePodWithoutPVC(),
			expectedUnbound: true,
			expectedBound:   true,
		},
		"pvc-not-found": {
			cachePVCs:       []*v1.PersistentVolumeClaim{},
			podPVCs:         []*v1.PersistentVolumeClaim{boundPVC},
			expectedUnbound: false,
			expectedBound:   false,
			shouldFail:      true,
		},
		"bound-pvc": {
			podPVCs:         []*v1.PersistentVolumeClaim{boundPVC},
			pvs:             []*v1.PersistentVolume{pvBound},
			expectedUnbound: true,
			expectedBound:   true,
		},
		"bound-pvc,pv-not-exists": {
			podPVCs:         []*v1.PersistentVolumeClaim{boundPVC},
			expectedUnbound: false,
			expectedBound:   false,
			shouldFail:      true,
		},
		"prebound-pvc": {
			podPVCs:    []*v1.PersistentVolumeClaim{preboundPVC},
			pvs:        []*v1.PersistentVolume{pvNode1aBound},
			shouldFail: true,
		},
		"unbound-pvc,pv-same-node": {
			podPVCs:          []*v1.PersistentVolumeClaim{unboundPVC},
			pvs:              []*v1.PersistentVolume{pvNode2, pvNode1a, pvNode1b},
			expectedBindings: []*bindingInfo{makeBinding(unboundPVC, pvNode1a)},
			expectedUnbound:  true,
			expectedBound:    true,
		},
		"unbound-pvc,pv-different-node": {
			podPVCs:         []*v1.PersistentVolumeClaim{unboundPVC},
			pvs:             []*v1.PersistentVolume{pvNode2},
			expectedUnbound: false,
			expectedBound:   true,
		},
		"two-unbound-pvcs": {
			podPVCs:          []*v1.PersistentVolumeClaim{unboundPVC, unboundPVC2},
			pvs:              []*v1.PersistentVolume{pvNode1a, pvNode1b},
			expectedBindings: []*bindingInfo{makeBinding(unboundPVC, pvNode1a), makeBinding(unboundPVC2, pvNode1b)},
			expectedUnbound:  true,
			expectedBound:    true,
		},
		"two-unbound-pvcs,order-by-size": {
			podPVCs:          []*v1.PersistentVolumeClaim{unboundPVC2, unboundPVC},
			pvs:              []*v1.PersistentVolume{pvNode1a, pvNode1b},
			expectedBindings: []*bindingInfo{makeBinding(unboundPVC, pvNode1a), makeBinding(unboundPVC2, pvNode1b)},
			expectedUnbound:  true,
			expectedBound:    true,
		},
		"two-unbound-pvcs,partial-match": {
			podPVCs:          []*v1.PersistentVolumeClaim{unboundPVC, unboundPVC2},
			pvs:              []*v1.PersistentVolume{pvNode1a},
			expectedBindings: []*bindingInfo{makeBinding(unboundPVC, pvNode1a)},
			expectedUnbound:  false,
			expectedBound:    true,
		},
		"one-bound,one-unbound": {
			podPVCs:          []*v1.PersistentVolumeClaim{unboundPVC, boundPVC},
			pvs:              []*v1.PersistentVolume{pvBound, pvNode1a},
			expectedBindings: []*bindingInfo{makeBinding(unboundPVC, pvNode1a)},
			expectedUnbound:  true,
			expectedBound:    true,
		},
		"one-bound,one-unbound,no-match": {
			podPVCs:         []*v1.PersistentVolumeClaim{unboundPVC, boundPVC},
			pvs:             []*v1.PersistentVolume{pvBound, pvNode2},
			expectedUnbound: false,
			expectedBound:   true,
		},
		"one-prebound,one-unbound": {
			podPVCs:    []*v1.PersistentVolumeClaim{unboundPVC, preboundPVC},
			pvs:        []*v1.PersistentVolume{pvNode1a, pvNode1b},
			shouldFail: true,
		},
		"immediate-bound-pvc": {
			podPVCs:         []*v1.PersistentVolumeClaim{immediateBoundPVC},
			pvs:             []*v1.PersistentVolume{pvBoundImmediate},
			expectedUnbound: true,
			expectedBound:   true,
		},
		"immediate-bound-pvc-wrong-node": {
			podPVCs:         []*v1.PersistentVolumeClaim{immediateBoundPVC},
			pvs:             []*v1.PersistentVolume{pvBoundImmediateNode2},
			expectedUnbound: true,
			expectedBound:   false,
		},
		"immediate-unbound-pvc": {
			podPVCs:         []*v1.PersistentVolumeClaim{immediateUnboundPVC},
			expectedUnbound: false,
			expectedBound:   false,
			shouldFail:      true,
		},
		"immediate-unbound-pvc,delayed-mode-bound": {
			podPVCs:         []*v1.PersistentVolumeClaim{immediateUnboundPVC, boundPVC},
			pvs:             []*v1.PersistentVolume{pvBound},
			expectedUnbound: false,
			expectedBound:   false,
			shouldFail:      true,
		},
		"immediate-unbound-pvc,delayed-mode-unbound": {
			podPVCs:         []*v1.PersistentVolumeClaim{immediateUnboundPVC, unboundPVC},
			expectedUnbound: false,
			expectedBound:   false,
			shouldFail:      true,
		},
	}

	testNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
			Labels: map[string]string{
				nodeLabelKey: "node1",
			},
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for name, scenario := range scenarios {
		klog.V(5).Infof("Running test case %q", name)

		// Setup
		testEnv := newTestBinder(t, ctx.Done())
		testEnv.initVolumes(scenario.pvs, scenario.pvs)

		// a. Init pvc cache
		if scenario.cachePVCs == nil {
			scenario.cachePVCs = scenario.podPVCs
		}
		testEnv.initClaims(scenario.cachePVCs, scenario.cachePVCs)

		// b. Generate pod with given claims
		if scenario.pod == nil {
			scenario.pod = makePod(scenario.podPVCs)
		}

		// Execute
		unboundSatisfied, boundSatisfied, err := testEnv.binder.FindPodVolumes(scenario.pod, testNode)

		// Validate
		if !scenario.shouldFail && err != nil {
			t.Errorf("Test %q failed: returned error: %v", name, err)
		}
		if scenario.shouldFail && err == nil {
			t.Errorf("Test %q failed: returned success but expected error", name)
		}
		if boundSatisfied != scenario.expectedBound {
			t.Errorf("Test %q failed: expected boundSatsified %v, got %v", name, scenario.expectedBound, boundSatisfied)
		}
		if unboundSatisfied != scenario.expectedUnbound {
			t.Errorf("Test %q failed: expected unboundSatsified %v, got %v", name, scenario.expectedUnbound, unboundSatisfied)
		}
		testEnv.validatePodCache(t, name, testNode.Name, scenario.pod, scenario.expectedBindings, nil)
	}
}

func TestFindPodVolumesWithProvisioning(t *testing.T) {
	scenarios := map[string]struct {
		// Inputs
		pvs     []*v1.PersistentVolume
		podPVCs []*v1.PersistentVolumeClaim
		// If nil, use pod PVCs
		cachePVCs []*v1.PersistentVolumeClaim
		// If nil, makePod with podPVCs
		pod *v1.Pod

		// Expected podBindingCache fields
		expectedBindings   []*bindingInfo
		expectedProvisions []*v1.PersistentVolumeClaim

		// Expected return values
		expectedUnbound bool
		expectedBound   bool
		shouldFail      bool
	}{
		"one-provisioned": {
			podPVCs:            []*v1.PersistentVolumeClaim{provisionedPVC},
			expectedProvisions: []*v1.PersistentVolumeClaim{provisionedPVC},
			expectedUnbound:    true,
			expectedBound:      true,
		},
		"two-unbound-pvcs,one-matched,one-provisioned": {
			podPVCs:            []*v1.PersistentVolumeClaim{unboundPVC, provisionedPVC},
			pvs:                []*v1.PersistentVolume{pvNode1a},
			expectedBindings:   []*bindingInfo{makeBinding(unboundPVC, pvNode1a)},
			expectedProvisions: []*v1.PersistentVolumeClaim{provisionedPVC},
			expectedUnbound:    true,
			expectedBound:      true,
		},
		"one-bound,one-provisioned": {
			podPVCs:            []*v1.PersistentVolumeClaim{boundPVC, provisionedPVC},
			pvs:                []*v1.PersistentVolume{pvBound},
			expectedProvisions: []*v1.PersistentVolumeClaim{provisionedPVC},
			expectedUnbound:    true,
			expectedBound:      true,
		},
		"one-binding,one-selected-node": {
			podPVCs:            []*v1.PersistentVolumeClaim{boundPVC, selectedNodePVC},
			pvs:                []*v1.PersistentVolume{pvBound},
			expectedProvisions: []*v1.PersistentVolumeClaim{selectedNodePVC},
			expectedUnbound:    true,
			expectedBound:      true,
		},
		"immediate-unbound-pvc": {
			podPVCs:         []*v1.PersistentVolumeClaim{immediateUnboundPVC},
			expectedUnbound: false,
			expectedBound:   false,
			shouldFail:      true,
		},
		"one-immediate-bound,one-provisioned": {
			podPVCs:            []*v1.PersistentVolumeClaim{immediateBoundPVC, provisionedPVC},
			pvs:                []*v1.PersistentVolume{pvBoundImmediate},
			expectedProvisions: []*v1.PersistentVolumeClaim{provisionedPVC},
			expectedUnbound:    true,
			expectedBound:      true,
		},
		"invalid-provisioner": {
			podPVCs:         []*v1.PersistentVolumeClaim{noProvisionerPVC},
			expectedUnbound: false,
			expectedBound:   true,
		},
		"volume-topology-unsatisfied": {
			podPVCs:         []*v1.PersistentVolumeClaim{topoMismatchPVC},
			expectedUnbound: false,
			expectedBound:   true,
		},
	}

	testNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
			Labels: map[string]string{
				nodeLabelKey: "node1",
			},
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for name, scenario := range scenarios {
		// Setup
		testEnv := newTestBinder(t, ctx.Done())
		testEnv.initVolumes(scenario.pvs, scenario.pvs)

		// a. Init pvc cache
		if scenario.cachePVCs == nil {
			scenario.cachePVCs = scenario.podPVCs
		}
		testEnv.initClaims(scenario.cachePVCs, scenario.cachePVCs)

		// b. Generate pod with given claims
		if scenario.pod == nil {
			scenario.pod = makePod(scenario.podPVCs)
		}

		// Execute
		unboundSatisfied, boundSatisfied, err := testEnv.binder.FindPodVolumes(scenario.pod, testNode)

		// Validate
		if !scenario.shouldFail && err != nil {
			t.Errorf("Test %q failed: returned error: %v", name, err)
		}
		if scenario.shouldFail && err == nil {
			t.Errorf("Test %q failed: returned success but expected error", name)
		}
		if boundSatisfied != scenario.expectedBound {
			t.Errorf("Test %q failed: expected boundSatsified %v, got %v", name, scenario.expectedBound, boundSatisfied)
		}
		if unboundSatisfied != scenario.expectedUnbound {
			t.Errorf("Test %q failed: expected unboundSatsified %v, got %v", name, scenario.expectedUnbound, unboundSatisfied)
		}
		testEnv.validatePodCache(t, name, testNode.Name, scenario.pod, scenario.expectedBindings, scenario.expectedProvisions)
	}
}

func TestAssumePodVolumes(t *testing.T) {
	scenarios := map[string]struct {
		// Inputs
		podPVCs         []*v1.PersistentVolumeClaim
		pvs             []*v1.PersistentVolume
		bindings        []*bindingInfo
		provisionedPVCs []*v1.PersistentVolumeClaim

		// Expected return values
		shouldFail       bool
		expectedAllBound bool

		expectedBindings      []*bindingInfo
		expectedProvisionings []*v1.PersistentVolumeClaim
	}{
		"all-bound": {
			podPVCs:          []*v1.PersistentVolumeClaim{boundPVC},
			pvs:              []*v1.PersistentVolume{pvBound},
			expectedAllBound: true,
		},
		"one-binding": {
			podPVCs:               []*v1.PersistentVolumeClaim{unboundPVC},
			bindings:              []*bindingInfo{makeBinding(unboundPVC, pvNode1a)},
			pvs:                   []*v1.PersistentVolume{pvNode1a},
			expectedBindings:      []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			expectedProvisionings: []*v1.PersistentVolumeClaim{},
		},
		"two-bindings": {
			podPVCs:               []*v1.PersistentVolumeClaim{unboundPVC, unboundPVC2},
			bindings:              []*bindingInfo{makeBinding(unboundPVC, pvNode1a), makeBinding(unboundPVC2, pvNode1b)},
			pvs:                   []*v1.PersistentVolume{pvNode1a, pvNode1b},
			expectedBindings:      []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound), makeBinding(unboundPVC2, pvNode1bBound)},
			expectedProvisionings: []*v1.PersistentVolumeClaim{},
		},
		"pv-already-bound": {
			podPVCs:               []*v1.PersistentVolumeClaim{unboundPVC},
			bindings:              []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			pvs:                   []*v1.PersistentVolume{pvNode1aBound},
			expectedBindings:      []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			expectedProvisionings: []*v1.PersistentVolumeClaim{},
		},
		"tmpupdate-failed": {
			podPVCs:    []*v1.PersistentVolumeClaim{unboundPVC},
			bindings:   []*bindingInfo{makeBinding(unboundPVC, pvNode1a), makeBinding(unboundPVC2, pvNode1b)},
			pvs:        []*v1.PersistentVolume{pvNode1a},
			shouldFail: true,
		},
		"one-binding, one-pvc-provisioned": {
			podPVCs:               []*v1.PersistentVolumeClaim{unboundPVC, provisionedPVC},
			bindings:              []*bindingInfo{makeBinding(unboundPVC, pvNode1a)},
			pvs:                   []*v1.PersistentVolume{pvNode1a},
			provisionedPVCs:       []*v1.PersistentVolumeClaim{provisionedPVC},
			expectedBindings:      []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			expectedProvisionings: []*v1.PersistentVolumeClaim{selectedNodePVC},
		},
		"one-binding, one-provision-tmpupdate-failed": {
			podPVCs:         []*v1.PersistentVolumeClaim{unboundPVC, provisionedPVCHigherVersion},
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1a)},
			pvs:             []*v1.PersistentVolume{pvNode1a},
			provisionedPVCs: []*v1.PersistentVolumeClaim{provisionedPVC2},
			shouldFail:      true,
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for name, scenario := range scenarios {
		klog.V(5).Infof("Running test case %q", name)

		// Setup
		testEnv := newTestBinder(t, ctx.Done())
		testEnv.initClaims(scenario.podPVCs, scenario.podPVCs)
		pod := makePod(scenario.podPVCs)
		testEnv.initPodCache(pod, "node1", scenario.bindings, scenario.provisionedPVCs)
		testEnv.initVolumes(scenario.pvs, scenario.pvs)

		// Execute
		allBound, err := testEnv.binder.AssumePodVolumes(pod, "node1")

		// Validate
		if !scenario.shouldFail && err != nil {
			t.Errorf("Test %q failed: returned error: %v", name, err)
		}
		if scenario.shouldFail && err == nil {
			t.Errorf("Test %q failed: returned success but expected error", name)
		}
		if scenario.expectedAllBound != allBound {
			t.Errorf("Test %q failed: returned unexpected allBound: %v", name, allBound)
		}
		if scenario.expectedBindings == nil {
			scenario.expectedBindings = scenario.bindings
		}
		if scenario.expectedProvisionings == nil {
			scenario.expectedProvisionings = scenario.provisionedPVCs
		}
		if scenario.shouldFail {
			testEnv.validateFailedAssume(t, name, pod, scenario.expectedBindings, scenario.expectedProvisionings)
		} else {
			testEnv.validateAssume(t, name, pod, scenario.expectedBindings, scenario.expectedProvisionings)
		}
		testEnv.validatePodCache(t, name, pod.Spec.NodeName, pod, scenario.expectedBindings, scenario.expectedProvisionings)
	}
}

func TestBindAPIUpdate(t *testing.T) {
	scenarios := map[string]struct {
		// Inputs
		bindings  []*bindingInfo
		cachedPVs []*v1.PersistentVolume
		// if nil, use cachedPVs
		apiPVs []*v1.PersistentVolume

		provisionedPVCs []*v1.PersistentVolumeClaim
		cachedPVCs      []*v1.PersistentVolumeClaim
		// if nil, use cachedPVCs
		apiPVCs []*v1.PersistentVolumeClaim

		// Expected return values
		shouldFail  bool
		expectedPVs []*v1.PersistentVolume
		// if nil, use expectedPVs
		expectedAPIPVs []*v1.PersistentVolume

		expectedPVCs []*v1.PersistentVolumeClaim
		// if nil, use expectedPVCs
		expectedAPIPVCs []*v1.PersistentVolumeClaim
	}{
		"nothing-to-bind-nil": {
			shouldFail: true,
		},
		"nothing-to-bind-bindings-nil": {
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
			shouldFail:      true,
		},
		"nothing-to-bind-provisionings-nil": {
			bindings:   []*bindingInfo{},
			shouldFail: true,
		},
		"nothing-to-bind-empty": {
			bindings:        []*bindingInfo{},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
		},
		"one-binding": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			cachedPVs:       []*v1.PersistentVolume{pvNode1a},
			expectedPVs:     []*v1.PersistentVolume{pvNode1aBound},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
		},
		"two-bindings": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound), makeBinding(unboundPVC2, pvNode1bBound)},
			cachedPVs:       []*v1.PersistentVolume{pvNode1a, pvNode1b},
			expectedPVs:     []*v1.PersistentVolume{pvNode1aBound, pvNode1bBound},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
		},
		"api-already-updated": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			cachedPVs:       []*v1.PersistentVolume{pvNode1aBound},
			expectedPVs:     []*v1.PersistentVolume{pvNode1aBound},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
		},
		"api-update-failed": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound), makeBinding(unboundPVC2, pvNode1bBound)},
			cachedPVs:       []*v1.PersistentVolume{pvNode1a, pvNode1b},
			apiPVs:          []*v1.PersistentVolume{pvNode1a, pvNode1bBoundHigherVersion},
			expectedPVs:     []*v1.PersistentVolume{pvNode1aBound, pvNode1b},
			expectedAPIPVs:  []*v1.PersistentVolume{pvNode1aBound, pvNode1bBoundHigherVersion},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
			shouldFail:      true,
		},
		"one-provisioned-pvc": {
			bindings:        []*bindingInfo{},
			provisionedPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC)},
			cachedPVCs:      []*v1.PersistentVolumeClaim{provisionedPVC},
			expectedPVCs:    []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC)},
		},
		"provision-api-update-failed": {
			bindings:        []*bindingInfo{},
			provisionedPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC), addProvisionAnn(provisionedPVC2)},
			cachedPVCs:      []*v1.PersistentVolumeClaim{provisionedPVC, provisionedPVC2},
			apiPVCs:         []*v1.PersistentVolumeClaim{provisionedPVC, provisionedPVCHigherVersion},
			expectedPVCs:    []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC), provisionedPVC2},
			expectedAPIPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC), provisionedPVCHigherVersion},
			shouldFail:      true,
		},
		"binding-succeed, provision-api-update-failed": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			cachedPVs:       []*v1.PersistentVolume{pvNode1a},
			expectedPVs:     []*v1.PersistentVolume{pvNode1aBound},
			provisionedPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC), addProvisionAnn(provisionedPVC2)},
			cachedPVCs:      []*v1.PersistentVolumeClaim{provisionedPVC, provisionedPVC2},
			apiPVCs:         []*v1.PersistentVolumeClaim{provisionedPVC, provisionedPVCHigherVersion},
			expectedPVCs:    []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC), provisionedPVC2},
			expectedAPIPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC), provisionedPVCHigherVersion},
			shouldFail:      true,
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for name, scenario := range scenarios {
		klog.V(4).Infof("Running test case %q", name)

		// Setup
		testEnv := newTestBinder(t, ctx.Done())
		pod := makePod(nil)
		if scenario.apiPVs == nil {
			scenario.apiPVs = scenario.cachedPVs
		}
		if scenario.apiPVCs == nil {
			scenario.apiPVCs = scenario.cachedPVCs
		}
		testEnv.initVolumes(scenario.cachedPVs, scenario.apiPVs)
		testEnv.initClaims(scenario.cachedPVCs, scenario.apiPVCs)
		testEnv.assumeVolumes(t, name, "node1", pod, scenario.bindings, scenario.provisionedPVCs)

		// Execute
		err := testEnv.internalBinder.bindAPIUpdate(pod.Name, scenario.bindings, scenario.provisionedPVCs)

		// Validate
		if !scenario.shouldFail && err != nil {
			t.Errorf("Test %q failed: returned error: %v", name, err)
		}
		if scenario.shouldFail && err == nil {
			t.Errorf("Test %q failed: returned success but expected error", name)
		}
		if scenario.expectedAPIPVs == nil {
			scenario.expectedAPIPVs = scenario.expectedPVs
		}
		if scenario.expectedAPIPVCs == nil {
			scenario.expectedAPIPVCs = scenario.expectedPVCs
		}
		testEnv.validateBind(t, name, pod, scenario.expectedPVs, scenario.expectedAPIPVs)
		testEnv.validateProvision(t, name, pod, scenario.expectedPVCs, scenario.expectedAPIPVCs)
	}
}

func TestCheckBindings(t *testing.T) {
	scenarios := map[string]struct {
		// Inputs
		initPVs  []*v1.PersistentVolume
		initPVCs []*v1.PersistentVolumeClaim

		bindings        []*bindingInfo
		provisionedPVCs []*v1.PersistentVolumeClaim

		// api updates before checking
		apiPVs  []*v1.PersistentVolume
		apiPVCs []*v1.PersistentVolumeClaim

		// delete objects before checking
		deletePVs  bool
		deletePVCs bool

		// Expected return values
		shouldFail    bool
		expectedBound bool
	}{
		"nothing-to-bind-nil": {
			shouldFail: true,
		},
		"nothing-to-bind-bindings-nil": {
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
			shouldFail:      true,
		},
		"nothing-to-bind-provisionings-nil": {
			bindings:   []*bindingInfo{},
			shouldFail: true,
		},
		"nothing-to-bind": {
			bindings:        []*bindingInfo{},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
			expectedBound:   true,
		},
		"binding-bound": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
			initPVs:         []*v1.PersistentVolume{pvNode1aBound},
			initPVCs:        []*v1.PersistentVolumeClaim{boundPVCNode1a},
			expectedBound:   true,
		},
		"binding-prebound": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
			initPVs:         []*v1.PersistentVolume{pvNode1aBound},
			initPVCs:        []*v1.PersistentVolumeClaim{preboundPVCNode1a},
		},
		"binding-unbound": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
			initPVs:         []*v1.PersistentVolume{pvNode1aBound},
			initPVCs:        []*v1.PersistentVolumeClaim{unboundPVC},
		},
		"binding-pvc-not-exists": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
			initPVs:         []*v1.PersistentVolume{pvNode1aBound},
			shouldFail:      true,
		},
		"binding-pv-not-exists": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
			initPVs:         []*v1.PersistentVolume{pvNode1aBound},
			initPVCs:        []*v1.PersistentVolumeClaim{boundPVCNode1a},
			deletePVs:       true,
			shouldFail:      true,
		},
		"binding-claimref-nil": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
			initPVs:         []*v1.PersistentVolume{pvNode1a},
			initPVCs:        []*v1.PersistentVolumeClaim{boundPVCNode1a},
			apiPVs:          []*v1.PersistentVolume{pvNode1a},
			apiPVCs:         []*v1.PersistentVolumeClaim{boundPVCNode1a},
			shouldFail:      true,
		},
		"binding-claimref-uid-empty": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
			initPVs:         []*v1.PersistentVolume{pvNode1aBound},
			initPVCs:        []*v1.PersistentVolumeClaim{boundPVCNode1a},
			apiPVs:          []*v1.PersistentVolume{pvRemoveClaimUID(pvNode1aBound)},
			apiPVCs:         []*v1.PersistentVolumeClaim{boundPVCNode1a},
			shouldFail:      true,
		},
		"binding-one-bound,one-unbound": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound), makeBinding(unboundPVC2, pvNode1bBound)},
			provisionedPVCs: []*v1.PersistentVolumeClaim{},
			initPVs:         []*v1.PersistentVolume{pvNode1aBound, pvNode1bBound},
			initPVCs:        []*v1.PersistentVolumeClaim{boundPVCNode1a, unboundPVC2},
		},
		"provisioning-pvc-bound": {
			bindings:        []*bindingInfo{},
			provisionedPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC)},
			initPVs:         []*v1.PersistentVolume{pvBound},
			initPVCs:        []*v1.PersistentVolumeClaim{provisionedPVCBound},
			apiPVCs:         []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVCBound)},
			expectedBound:   true,
		},
		"provisioning-pvc-unbound": {
			bindings:        []*bindingInfo{},
			provisionedPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC)},
			initPVCs:        []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC)},
		},
		"provisioning-pvc-not-exists": {
			bindings:        []*bindingInfo{},
			provisionedPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC)},
			initPVCs:        []*v1.PersistentVolumeClaim{provisionedPVC},
			deletePVCs:      true,
			shouldFail:      true,
		},
		"provisioning-pvc-annotations-nil": {
			bindings:        []*bindingInfo{},
			provisionedPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC)},
			initPVCs:        []*v1.PersistentVolumeClaim{provisionedPVC},
			apiPVCs:         []*v1.PersistentVolumeClaim{provisionedPVC},
			shouldFail:      true,
		},
		"provisioning-pvc-selected-node-dropped": {
			bindings:        []*bindingInfo{},
			provisionedPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC)},
			initPVCs:        []*v1.PersistentVolumeClaim{provisionedPVC},
			apiPVCs:         []*v1.PersistentVolumeClaim{pvcSetEmptyAnnotations(provisionedPVC)},
			shouldFail:      true,
		},
		"provisioning-pvc-selected-node-wrong-node": {
			initPVCs:        []*v1.PersistentVolumeClaim{provisionedPVC},
			bindings:        []*bindingInfo{},
			provisionedPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC)},
			apiPVCs:         []*v1.PersistentVolumeClaim{pvcSetSelectedNode(provisionedPVC, "wrong-node")},
			shouldFail:      true,
		},
		"binding-bound-provisioning-unbound": {
			bindings:        []*bindingInfo{makeBinding(unboundPVC, pvNode1aBound)},
			provisionedPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC)},
			initPVs:         []*v1.PersistentVolume{pvNode1aBound},
			initPVCs:        []*v1.PersistentVolumeClaim{boundPVCNode1a, addProvisionAnn(provisionedPVC)},
		},
		"tolerate-provisioning-pvc-bound-pv-not-found": {
			initPVs:         []*v1.PersistentVolume{pvNode1a},
			initPVCs:        []*v1.PersistentVolumeClaim{provisionedPVC},
			bindings:        []*bindingInfo{},
			provisionedPVCs: []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVC)},
			apiPVCs:         []*v1.PersistentVolumeClaim{addProvisionAnn(provisionedPVCBound)},
			deletePVs:       true,
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for name, scenario := range scenarios {
		klog.V(4).Infof("Running test case %q", name)

		// Setup
		pod := makePod(nil)
		testEnv := newTestBinder(t, ctx.Done())
		testEnv.initNodes([]*v1.Node{node1})
		testEnv.initVolumes(scenario.initPVs, nil)
		testEnv.initClaims(scenario.initPVCs, nil)
		testEnv.assumeVolumes(t, name, "node1", pod, scenario.bindings, scenario.provisionedPVCs)

		// Before execute
		if scenario.deletePVs {
			testEnv.deleteVolumes(scenario.initPVs)
		} else {
			testEnv.updateVolumes(t, scenario.apiPVs, true)
		}
		if scenario.deletePVCs {
			testEnv.deleteClaims(scenario.initPVCs)
		} else {
			testEnv.updateClaims(t, scenario.apiPVCs, true)
		}

		// Execute
		allBound, err := testEnv.internalBinder.checkBindings(pod, scenario.bindings, scenario.provisionedPVCs)

		// Validate
		if !scenario.shouldFail && err != nil {
			t.Errorf("Test %q failed: returned error: %v", name, err)
		}
		if scenario.shouldFail && err == nil {
			t.Errorf("Test %q failed: returned success but expected error", name)
		}
		if scenario.expectedBound != allBound {
			t.Errorf("Test %q failed: returned bound %v", name, allBound)
		}
	}
}

func TestBindPodVolumes(t *testing.T) {
	type scenarioType struct {
		// Inputs
		bindingsNil bool // Pass in nil bindings slice

		nodes []*v1.Node

		// before assume
		initPVs  []*v1.PersistentVolume
		initPVCs []*v1.PersistentVolumeClaim

		// assume PV & PVC with these binding results
		binding          *bindingInfo
		claimToProvision *v1.PersistentVolumeClaim

		// API updates after assume before bind
		apiPV  *v1.PersistentVolume
		apiPVC *v1.PersistentVolumeClaim

		// This function runs with a delay of 5 seconds
		delayFunc func(t *testing.T, testEnv *testEnv, pod *v1.Pod, pvs []*v1.PersistentVolume, pvcs []*v1.PersistentVolumeClaim)

		// Expected return values
		shouldFail bool
	}

	scenarios := map[string]scenarioType{
		"nothing-to-bind-nil": {
			bindingsNil: true,
			shouldFail:  true,
		},
		"nothing-to-bind-empty": {},
		"already-bound": {
			binding:  makeBinding(unboundPVC, pvNode1aBound),
			initPVs:  []*v1.PersistentVolume{pvNode1aBound},
			initPVCs: []*v1.PersistentVolumeClaim{boundPVCNode1a},
		},
		"binding-static-pv-succeeds-after-time": {
			initPVs:    []*v1.PersistentVolume{pvNode1a},
			initPVCs:   []*v1.PersistentVolumeClaim{unboundPVC},
			binding:    makeBinding(unboundPVC, pvNode1aBound),
			shouldFail: false, // Will succeed after PVC is fully bound to this PV by pv controller.
			delayFunc: func(t *testing.T, testEnv *testEnv, pod *v1.Pod, pvs []*v1.PersistentVolume, pvcs []*v1.PersistentVolumeClaim) {
				pvc := pvcs[0]
				pv := pvs[0]
				// Update PVC to be fully bound to PV
				newPVC := pvc.DeepCopy()
				newPVC.Spec.VolumeName = pv.Name
				metav1.SetMetaDataAnnotation(&newPVC.ObjectMeta, pvutil.AnnBindCompleted, "yes")
				if _, err := testEnv.client.CoreV1().PersistentVolumeClaims(newPVC.Namespace).Update(newPVC); err != nil {
					t.Errorf("failed to update PVC %q: %v", newPVC.Name, err)
				}
			},
		},
		"binding-dynamic-pv-succeeds-after-time": {
			claimToProvision: pvcSetSelectedNode(provisionedPVC, "node1"),
			initPVCs:         []*v1.PersistentVolumeClaim{provisionedPVC},
			delayFunc: func(t *testing.T, testEnv *testEnv, pod *v1.Pod, pvs []*v1.PersistentVolume, pvcs []*v1.PersistentVolumeClaim) {
				pvc := pvcs[0]
				// Update PVC to be fully bound to PV
				newPVC, err := testEnv.client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Get(pvc.Name, metav1.GetOptions{})
				if err != nil {
					t.Errorf("failed to get PVC %q: %v", pvc.Name, err)
					return
				}
				dynamicPV := makeTestPV("dynamic-pv", "node1", "1G", "1", newPVC, waitClass)
				dynamicPV, err = testEnv.client.CoreV1().PersistentVolumes().Create(dynamicPV)
				if err != nil {
					t.Errorf("failed to create PV %q: %v", dynamicPV.Name, err)
					return
				}
				newPVC.Spec.VolumeName = dynamicPV.Name
				metav1.SetMetaDataAnnotation(&newPVC.ObjectMeta, pvutil.AnnBindCompleted, "yes")
				if _, err := testEnv.client.CoreV1().PersistentVolumeClaims(newPVC.Namespace).Update(newPVC); err != nil {
					t.Errorf("failed to update PVC %q: %v", newPVC.Name, err)
				}
			},
		},
		"bound-by-pv-controller-before-bind": {
			initPVs:    []*v1.PersistentVolume{pvNode1a},
			initPVCs:   []*v1.PersistentVolumeClaim{unboundPVC},
			binding:    makeBinding(unboundPVC, pvNode1aBound),
			apiPV:      pvNode1aBound,
			apiPVC:     boundPVCNode1a,
			shouldFail: true, // bindAPIUpdate will fail because API conflict
		},
		"pod-deleted-after-time": {
			binding:  makeBinding(unboundPVC, pvNode1aBound),
			initPVs:  []*v1.PersistentVolume{pvNode1a},
			initPVCs: []*v1.PersistentVolumeClaim{unboundPVC},
			delayFunc: func(t *testing.T, testEnv *testEnv, pod *v1.Pod, pvs []*v1.PersistentVolume, pvcs []*v1.PersistentVolumeClaim) {
				bindingsCache := testEnv.binder.GetBindingsCache()
				if bindingsCache == nil {
					t.Fatalf("Failed to get bindings cache")
				}

				// Delete the pod from the cache
				bindingsCache.DeleteBindings(pod)

				// Check that it's deleted
				bindings := bindingsCache.GetBindings(pod, "node1")
				if bindings != nil {
					t.Fatalf("Failed to delete bindings")
				}
			},
			shouldFail: true,
		},
		"binding-times-out": {
			binding:    makeBinding(unboundPVC, pvNode1aBound),
			initPVs:    []*v1.PersistentVolume{pvNode1a},
			initPVCs:   []*v1.PersistentVolumeClaim{unboundPVC},
			shouldFail: true,
		},
		"binding-fails": {
			binding:    makeBinding(unboundPVC2, pvNode1bBound),
			initPVs:    []*v1.PersistentVolume{pvNode1b},
			initPVCs:   []*v1.PersistentVolumeClaim{unboundPVC2},
			shouldFail: true,
		},
		"check-fails": {
			binding:  makeBinding(unboundPVC, pvNode1aBound),
			initPVs:  []*v1.PersistentVolume{pvNode1a},
			initPVCs: []*v1.PersistentVolumeClaim{unboundPVC},
			delayFunc: func(t *testing.T, testEnv *testEnv, pod *v1.Pod, pvs []*v1.PersistentVolume, pvcs []*v1.PersistentVolumeClaim) {
				pvc := pvcs[0]
				// Delete PVC will fail check
				if err := testEnv.client.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(pvc.Name, &metav1.DeleteOptions{}); err != nil {
					t.Errorf("failed to delete PVC %q: %v", pvc.Name, err)
				}
			},
			shouldFail: true,
		},
		"node-affinity-fails": {
			binding:    makeBinding(unboundPVC, pvNode1aBound),
			initPVs:    []*v1.PersistentVolume{pvNode1aBound},
			initPVCs:   []*v1.PersistentVolumeClaim{boundPVCNode1a},
			nodes:      []*v1.Node{node1NoLabels},
			shouldFail: true,
		},
		"node-affinity-fails-dynamic-provisioning": {
			initPVs:          []*v1.PersistentVolume{pvNode1a, pvNode2},
			initPVCs:         []*v1.PersistentVolumeClaim{selectedNodePVC},
			claimToProvision: selectedNodePVC,
			nodes:            []*v1.Node{node1, node2},
			delayFunc: func(t *testing.T, testEnv *testEnv, pod *v1.Pod, pvs []*v1.PersistentVolume, pvcs []*v1.PersistentVolumeClaim) {
				// Update PVC to be fully bound to a PV with a different node
				newPVC := pvcs[0].DeepCopy()
				newPVC.Spec.VolumeName = pvNode2.Name
				metav1.SetMetaDataAnnotation(&newPVC.ObjectMeta, pvutil.AnnBindCompleted, "yes")
				if _, err := testEnv.client.CoreV1().PersistentVolumeClaims(newPVC.Namespace).Update(newPVC); err != nil {
					t.Errorf("failed to update PVC %q: %v", newPVC.Name, err)
				}
			},
			shouldFail: true,
		},
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	for name, scenario := range scenarios {
		klog.V(4).Infof("Running test case %q", name)

		// Setup
		pod := makePod(nil)
		testEnv := newTestBinder(t, ctx.Done())
		if scenario.nodes == nil {
			scenario.nodes = []*v1.Node{node1}
		}
		if !scenario.bindingsNil {
			bindings := []*bindingInfo{}
			if scenario.binding != nil {
				bindings = []*bindingInfo{scenario.binding}
			}
			claimsToProvision := []*v1.PersistentVolumeClaim{}
			if scenario.claimToProvision != nil {
				claimsToProvision = []*v1.PersistentVolumeClaim{scenario.claimToProvision}
			}
			testEnv.initNodes(scenario.nodes)
			testEnv.initVolumes(scenario.initPVs, scenario.initPVs)
			testEnv.initClaims(scenario.initPVCs, scenario.initPVCs)
			testEnv.assumeVolumes(t, name, "node1", pod, bindings, claimsToProvision)
		}

		// Before Execute
		if scenario.apiPV != nil {
			_, err := testEnv.client.CoreV1().PersistentVolumes().Update(scenario.apiPV)
			if err != nil {
				t.Fatalf("Test %q failed: failed to update PV %q", name, scenario.apiPV.Name)
			}
		}
		if scenario.apiPVC != nil {
			_, err := testEnv.client.CoreV1().PersistentVolumeClaims(scenario.apiPVC.Namespace).Update(scenario.apiPVC)
			if err != nil {
				t.Fatalf("Test %q failed: failed to update PVC %q", name, getPVCName(scenario.apiPVC))
			}
		}

		if scenario.delayFunc != nil {
			go func(scenario scenarioType) {
				time.Sleep(5 * time.Second)
				// Sleep a while to run after bindAPIUpdate in BindPodVolumes
				klog.V(5).Infof("Running delay function")
				scenario.delayFunc(t, testEnv, pod, scenario.initPVs, scenario.initPVCs)
			}(scenario)
		}

		// Execute
		err := testEnv.binder.BindPodVolumes(pod)

		// Validate
		if !scenario.shouldFail && err != nil {
			t.Errorf("Test %q failed: returned error: %v", name, err)
		}
		if scenario.shouldFail && err == nil {
			t.Errorf("Test %q failed: returned success but expected error", name)
		}
	}
}

func TestFindAssumeVolumes(t *testing.T) {
	// Test case
	podPVCs := []*v1.PersistentVolumeClaim{unboundPVC}
	pvs := []*v1.PersistentVolume{pvNode2, pvNode1a, pvNode1c}

	// Setup
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	testEnv := newTestBinder(t, ctx.Done())
	testEnv.initVolumes(pvs, pvs)
	testEnv.initClaims(podPVCs, podPVCs)
	pod := makePod(podPVCs)

	testNode := &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			Name: "node1",
			Labels: map[string]string{
				nodeLabelKey: "node1",
			},
		},
	}

	// Execute
	// 1. Find matching PVs
	unboundSatisfied, _, err := testEnv.binder.FindPodVolumes(pod, testNode)
	if err != nil {
		t.Errorf("Test failed: FindPodVolumes returned error: %v", err)
	}
	if !unboundSatisfied {
		t.Errorf("Test failed: couldn't find PVs for all PVCs")
	}
	expectedBindings := testEnv.getPodBindings(t, "before-assume", testNode.Name, pod)

	// 2. Assume matches
	allBound, err := testEnv.binder.AssumePodVolumes(pod, testNode.Name)
	if err != nil {
		t.Errorf("Test failed: AssumePodVolumes returned error: %v", err)
	}
	if allBound {
		t.Errorf("Test failed: detected unbound volumes as bound")
	}
	testEnv.validateAssume(t, "assume", pod, expectedBindings, nil)

	// After assume, claimref should be set on pv
	expectedBindings = testEnv.getPodBindings(t, "after-assume", testNode.Name, pod)

	// 3. Find matching PVs again
	// This should always return the original chosen pv
	// Run this many times in case sorting returns different orders for the two PVs.
	t.Logf("Testing FindPodVolumes after Assume")
	for i := 0; i < 50; i++ {
		unboundSatisfied, _, err := testEnv.binder.FindPodVolumes(pod, testNode)
		if err != nil {
			t.Errorf("Test failed: FindPodVolumes returned error: %v", err)
		}
		if !unboundSatisfied {
			t.Errorf("Test failed: couldn't find PVs for all PVCs")
		}
		testEnv.validatePodCache(t, "after-assume", testNode.Name, pod, expectedBindings, nil)
	}
}
