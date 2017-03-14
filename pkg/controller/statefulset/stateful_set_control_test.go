/*
Copyright 2016 The Kubernetes Authors.

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

package statefulset

import (
	"errors"
	"fmt"
	"sort"
	"strconv"
	"testing"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/v1"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset/fake"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions"
	appsinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/apps/v1beta1"
	coreinformers "k8s.io/kubernetes/pkg/client/informers/informers_generated/externalversions/core/v1"
	appslisters "k8s.io/kubernetes/pkg/client/listers/apps/v1beta1"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/v1"
	"k8s.io/kubernetes/pkg/controller"
)

func TestDefaultStatefulSetControlCreatesPods(t *testing.T) {
	set := newStatefulSet(3)
	client := fake.NewSimpleClientset(set)

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	spc := newFakeStatefulPodControl(informerFactory.Core().V1().Pods(), informerFactory.Apps().V1beta1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc)

	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	cache.WaitForCacheSync(
		stop,
		informerFactory.Apps().V1beta1().StatefulSets().Informer().HasSynced,
		informerFactory.Core().V1().Pods().Informer().HasSynced,
	)

	if err := scaleUpStatefulSetControl(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	var err error
	set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
	if err != nil {
		t.Fatalf("Error getting updated StatefulSet: %v", err)
	}
	if set.Status.Replicas != 3 {
		t.Error("Failed to scale statefulset to 3 replicas")
	}
}

func TestStatefulSetControlScaleUp(t *testing.T) {
	set := newStatefulSet(3)
	client := fake.NewSimpleClientset(set)

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	spc := newFakeStatefulPodControl(informerFactory.Core().V1().Pods(), informerFactory.Apps().V1beta1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc)

	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	cache.WaitForCacheSync(
		stop,
		informerFactory.Apps().V1beta1().StatefulSets().Informer().HasSynced,
		informerFactory.Core().V1().Pods().Informer().HasSynced,
	)

	if err := scaleUpStatefulSetControl(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	*set.Spec.Replicas = 4
	if err := scaleUpStatefulSetControl(set, ssc, spc); err != nil {
		t.Errorf("Failed to scale StatefulSet : %s", err)
	}
	var err error
	set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
	if err != nil {
		t.Fatalf("Error getting updated StatefulSet: %v", err)
	}
	if set.Status.Replicas != 4 {
		t.Error("Failed to scale statefulset to 4 replicas")
	}
}

func TestStatefulSetControlScaleDown(t *testing.T) {
	set := newStatefulSet(3)
	client := fake.NewSimpleClientset(set)

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	spc := newFakeStatefulPodControl(informerFactory.Core().V1().Pods(), informerFactory.Apps().V1beta1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc)

	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	cache.WaitForCacheSync(
		stop,
		informerFactory.Apps().V1beta1().StatefulSets().Informer().HasSynced,
		informerFactory.Core().V1().Pods().Informer().HasSynced,
	)

	if err := scaleUpStatefulSetControl(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	*set.Spec.Replicas = 0
	if err := scaleDownStatefulSetControl(set, ssc, spc); err != nil {
		t.Errorf("Failed to scale StatefulSet : %s", err)
	}
	if set.Status.Replicas != 0 {
		t.Error("Failed to scale statefulset to 0 replicas")
	}
}

func TestStatefulSetControlReplacesPods(t *testing.T) {
	set := newStatefulSet(5)
	client := fake.NewSimpleClientset(set)

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	spc := newFakeStatefulPodControl(informerFactory.Core().V1().Pods(), informerFactory.Apps().V1beta1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc)

	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	cache.WaitForCacheSync(
		stop,
		informerFactory.Apps().V1beta1().StatefulSets().Informer().HasSynced,
		informerFactory.Core().V1().Pods().Informer().HasSynced,
	)

	if err := scaleUpStatefulSetControl(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	var err error
	set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
	if err != nil {
		t.Fatalf("Error getting updated StatefulSet: %v", err)
	}
	if set.Status.Replicas != 5 {
		t.Error("Failed to scale statefulset to 5 replicas")
	}
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Error(err)
	}
	pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	sort.Sort(ascendingOrdinal(pods))
	spc.podsIndexer.Delete(pods[0])
	spc.podsIndexer.Delete(pods[2])
	spc.podsIndexer.Delete(pods[4])
	for i := 0; i < 5; i += 2 {
		pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
		if err != nil {
			t.Error(err)
		}
		if err = ssc.UpdateStatefulSet(set, pods); err != nil {
			t.Errorf("Failed to update StatefulSet : %s", err)
		}
		set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
		if err != nil {
			t.Fatalf("Error getting updated StatefulSet: %v", err)
		}
		if pods, err = spc.setPodRunning(set, i); err != nil {
			t.Error(err)
		}
		if err = ssc.UpdateStatefulSet(set, pods); err != nil {
			t.Errorf("Failed to update StatefulSet : %s", err)
		}
		set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
		if err != nil {
			t.Fatalf("Error getting updated StatefulSet: %v", err)
		}
		if pods, err = spc.setPodReady(set, i); err != nil {
			t.Error(err)
		}
	}
	pods, err = spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	if err := ssc.UpdateStatefulSet(set, pods); err != nil {
		t.Errorf("Failed to update StatefulSet : %s", err)
	}
	set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
	if err != nil {
		t.Fatalf("Error getting updated StatefulSet: %v", err)
	}
	if e, a := int32(5), set.Status.Replicas; e != a {
		t.Errorf("Expected to scale to %d, got %d", e, a)
	}
}

func TestStatefulSetDeletionTimestamp(t *testing.T) {
	set := newStatefulSet(5)
	client := fake.NewSimpleClientset(set)

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	spc := newFakeStatefulPodControl(informerFactory.Core().V1().Pods(), informerFactory.Apps().V1beta1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc)

	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	cache.WaitForCacheSync(
		stop,
		informerFactory.Apps().V1beta1().StatefulSets().Informer().HasSynced,
		informerFactory.Core().V1().Pods().Informer().HasSynced,
	)

	// Bring up a StatefulSet.
	if err := scaleUpStatefulSetControl(set, ssc, spc); err != nil {
		t.Errorf("failed to turn up StatefulSet : %s", err)
	}
	var err error
	set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
	if err != nil {
		t.Fatalf("error getting updated StatefulSet: %v", err)
	}
	if set.Status.Replicas != 5 {
		t.Error("failed to scale statefulset to 5 replicas")
	}
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Error(err)
	}
	pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	sort.Sort(ascendingOrdinal(pods))

	// Mark the StatefulSet as being deleted.
	set.DeletionTimestamp = new(metav1.Time)

	// Delete the first pod.
	spc.podsIndexer.Delete(pods[0])
	pods, err = spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}

	// The StatefulSet should update its replica count,
	// but not try to fix it.
	if err := ssc.UpdateStatefulSet(set, pods); err != nil {
		t.Errorf("failed to update StatefulSet : %s", err)
	}
	set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
	if err != nil {
		t.Fatalf("error getting updated StatefulSet: %v", err)
	}
	if e, a := int32(4), set.Status.Replicas; e != a {
		t.Errorf("expected to scale to %d, got %d", e, a)
	}
}

func TestDefaultStatefulSetControlRecreatesFailedPod(t *testing.T) {
	client := fake.NewSimpleClientset()
	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	spc := newFakeStatefulPodControl(informerFactory.Core().V1().Pods(), informerFactory.Apps().V1beta1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc)
	set := newStatefulSet(3)
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Error(err)
	}
	pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	if err := ssc.UpdateStatefulSet(set, pods); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	pods, err = spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	pods[0].Status.Phase = v1.PodFailed
	spc.podsIndexer.Update(pods[0])
	if err := ssc.UpdateStatefulSet(set, pods); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	pods, err = spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	if isCreated(pods[0]) {
		t.Error("StatefulSet did not recreate failed Pod")
	}
}

func TestDefaultStatefulSetControlInitAnnotation(t *testing.T) {
	set := newStatefulSet(3)
	client := fake.NewSimpleClientset(set)

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	spc := newFakeStatefulPodControl(informerFactory.Core().V1().Pods(), informerFactory.Apps().V1beta1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc)

	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	cache.WaitForCacheSync(
		stop,
		informerFactory.Apps().V1beta1().StatefulSets().Informer().HasSynced,
		informerFactory.Core().V1().Pods().Informer().HasSynced,
	)

	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Error(err)
	}
	pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	if err = ssc.UpdateStatefulSet(set, pods); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err = assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	if pods, err = spc.setPodRunning(set, 0); err != nil {
		t.Error(err)
	}
	if pods, err = spc.setPodReady(set, 0); err != nil {
		t.Error(err)
	}
	if pods, err = spc.setPodInitStatus(set, 0, false); err != nil {
		t.Error(err)
	}
	replicas := int(set.Status.Replicas)
	if err := ssc.UpdateStatefulSet(set, pods); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	if replicas != int(set.Status.Replicas) {
		t.Errorf("StatefulSetControl does not block on %s=false", apps.StatefulSetInitAnnotation)
	}
	if pods, err = spc.setPodInitStatus(set, 0, true); err != nil {
		t.Error(err)
	}
	if err := scaleUpStatefulSetControl(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
	if err != nil {
		t.Fatalf("Error getting updated StatefulSet: %v", err)
	}
	if int(set.Status.Replicas) != 3 {
		t.Errorf("StatefulSetControl does not unblock on %s=true", apps.StatefulSetInitAnnotation)
	}
}

func TestDefaultStatefulSetControlCreatePodFailure(t *testing.T) {
	set := newStatefulSet(3)
	client := fake.NewSimpleClientset(set)

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	spc := newFakeStatefulPodControl(informerFactory.Core().V1().Pods(), informerFactory.Apps().V1beta1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc)
	spc.SetCreateStatefulPodError(apierrors.NewInternalError(errors.New("API server failed")), 2)

	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	cache.WaitForCacheSync(
		stop,
		informerFactory.Apps().V1beta1().StatefulSets().Informer().HasSynced,
		informerFactory.Core().V1().Pods().Informer().HasSynced,
	)

	if err := scaleUpStatefulSetControl(set, ssc, spc); !apierrors.IsInternalError(err) {
		t.Errorf("StatefulSetControl did not return InternalError found %s", err)
	}
	if err := scaleUpStatefulSetControl(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	var err error
	set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
	if err != nil {
		t.Fatalf("Error getting updated StatefulSet: %v", err)
	}
	if set.Status.Replicas != 3 {
		t.Error("Failed to scale StatefulSet to 3 replicas")
	}
}

func TestDefaultStatefulSetControlUpdatePodFailure(t *testing.T) {
	set := newStatefulSet(3)
	client := fake.NewSimpleClientset(set)

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	spc := newFakeStatefulPodControl(informerFactory.Core().V1().Pods(), informerFactory.Apps().V1beta1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc)
	spc.SetUpdateStatefulPodError(apierrors.NewInternalError(errors.New("API server failed")), 0)

	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	cache.WaitForCacheSync(
		stop,
		informerFactory.Apps().V1beta1().StatefulSets().Informer().HasSynced,
		informerFactory.Core().V1().Pods().Informer().HasSynced,
	)

	// have to have 1 successful loop first
	if err := scaleUpStatefulSetControl(set, ssc, spc); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	var err error
	set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
	if err != nil {
		t.Fatalf("Error getting updated StatefulSet: %v", err)
	}
	if set.Status.Replicas != 3 {
		t.Error("Failed to scale StatefulSet to 3 replicas")
	}

	// now mutate a pod's identity
	pods, err := spc.podsLister.List(labels.Everything())
	if err != nil {
		t.Fatalf("Error listing pods: %v", err)
	}
	if len(pods) != 3 {
		t.Fatalf("Expected 3 pods, got %d", len(pods))
	}
	sort.Sort(ascendingOrdinal(pods))
	pods[0].Name = "goo-0"
	spc.podsIndexer.Update(pods[0])

	// now it should fail
	if err := ssc.UpdateStatefulSet(set, pods); !apierrors.IsInternalError(err) {
		t.Errorf("StatefulSetControl did not return InternalError found %s", err)
	}
}

func TestDefaultStatefulSetControlUpdateSetStatusFailure(t *testing.T) {
	set := newStatefulSet(3)
	client := fake.NewSimpleClientset(set)

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	spc := newFakeStatefulPodControl(informerFactory.Core().V1().Pods(), informerFactory.Apps().V1beta1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc)
	spc.SetUpdateStatefulSetStatusError(apierrors.NewInternalError(errors.New("API server failed")), 2)

	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	cache.WaitForCacheSync(
		stop,
		informerFactory.Apps().V1beta1().StatefulSets().Informer().HasSynced,
		informerFactory.Core().V1().Pods().Informer().HasSynced,
	)

	if err := scaleUpStatefulSetControl(set, ssc, spc); !apierrors.IsInternalError(err) {
		t.Errorf("StatefulSetControl did not return InternalError found %s", err)
	}
	if err := scaleUpStatefulSetControl(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	var err error
	set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
	if err != nil {
		t.Fatalf("Error getting updated StatefulSet: %v", err)
	}
	if set.Status.Replicas != 3 {
		t.Error("Failed to scale StatefulSet to 3 replicas")
	}
}

func TestDefaultStatefulSetControlPodRecreateDeleteError(t *testing.T) {
	set := newStatefulSet(3)
	client := fake.NewSimpleClientset(set)

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	spc := newFakeStatefulPodControl(informerFactory.Core().V1().Pods(), informerFactory.Apps().V1beta1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc)

	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	cache.WaitForCacheSync(
		stop,
		informerFactory.Apps().V1beta1().StatefulSets().Informer().HasSynced,
		informerFactory.Core().V1().Pods().Informer().HasSynced,
	)

	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		t.Error(err)
	}
	pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	if err := ssc.UpdateStatefulSet(set, pods); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	pods, err = spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	pods[0].Status.Phase = v1.PodFailed
	spc.podsIndexer.Update(pods[0])
	spc.SetDeleteStatefulPodError(apierrors.NewInternalError(errors.New("API server failed")), 0)
	if err := ssc.UpdateStatefulSet(set, pods); !apierrors.IsInternalError(err) {
		t.Errorf("StatefulSet failed to %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	if err := ssc.UpdateStatefulSet(set, pods); err != nil {
		t.Errorf("Error updating StatefulSet %s", err)
	}
	if err := assertInvariants(set, spc); err != nil {
		t.Error(err)
	}
	pods, err = spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		t.Error(err)
	}
	if isCreated(pods[0]) {
		t.Error("StatefulSet did not recreate failed Pod")
	}
}

func TestStatefulSetControlScaleDownDeleteError(t *testing.T) {
	set := newStatefulSet(3)
	client := fake.NewSimpleClientset(set)

	informerFactory := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
	spc := newFakeStatefulPodControl(informerFactory.Core().V1().Pods(), informerFactory.Apps().V1beta1().StatefulSets())
	ssc := NewDefaultStatefulSetControl(spc)

	stop := make(chan struct{})
	defer close(stop)
	informerFactory.Start(stop)
	cache.WaitForCacheSync(
		stop,
		informerFactory.Apps().V1beta1().StatefulSets().Informer().HasSynced,
		informerFactory.Core().V1().Pods().Informer().HasSynced,
	)

	if err := scaleUpStatefulSetControl(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn up StatefulSet : %s", err)
	}
	var err error
	set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
	if err != nil {
		t.Fatalf("Error getting updated StatefulSet: %v", err)
	}
	*set.Spec.Replicas = 0
	spc.SetDeleteStatefulPodError(apierrors.NewInternalError(errors.New("API server failed")), 2)
	if err := scaleDownStatefulSetControl(set, ssc, spc); !apierrors.IsInternalError(err) {
		t.Errorf("StatefulSetControl failed to throw error on delete %s", err)
	}
	if err := scaleDownStatefulSetControl(set, ssc, spc); err != nil {
		t.Errorf("Failed to turn down StatefulSet %s", err)
	}
	set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
	if err != nil {
		t.Fatalf("Error getting updated StatefulSet: %v", err)
	}
	if set.Status.Replicas != 0 {
		t.Error("Failed to scale statefulset to 0 replicas")
	}
}

type requestTracker struct {
	requests int
	err      error
	after    int
}

func (rt *requestTracker) errorReady() bool {
	return rt.err != nil && rt.requests >= rt.after
}

func (rt *requestTracker) inc() {
	rt.requests++
}

func (rt *requestTracker) reset() {
	rt.err = nil
	rt.after = 0
}

type fakeStatefulPodControl struct {
	podsLister          corelisters.PodLister
	claimsLister        corelisters.PersistentVolumeClaimLister
	setsLister          appslisters.StatefulSetLister
	podsIndexer         cache.Indexer
	claimsIndexer       cache.Indexer
	setsIndexer         cache.Indexer
	createPodTracker    requestTracker
	updatePodTracker    requestTracker
	deletePodTracker    requestTracker
	updateStatusTracker requestTracker
}

func newFakeStatefulPodControl(podInformer coreinformers.PodInformer, setInformer appsinformers.StatefulSetInformer) *fakeStatefulPodControl {
	claimsIndexer := cache.NewIndexer(controller.KeyFunc, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc})
	return &fakeStatefulPodControl{
		podInformer.Lister(),
		corelisters.NewPersistentVolumeClaimLister(claimsIndexer),
		setInformer.Lister(),
		podInformer.Informer().GetIndexer(),
		claimsIndexer,
		setInformer.Informer().GetIndexer(),
		requestTracker{0, nil, 0},
		requestTracker{0, nil, 0},
		requestTracker{0, nil, 0},
		requestTracker{0, nil, 0}}
}

func (spc *fakeStatefulPodControl) SetCreateStatefulPodError(err error, after int) {
	spc.createPodTracker.err = err
	spc.createPodTracker.after = after
}

func (spc *fakeStatefulPodControl) SetUpdateStatefulPodError(err error, after int) {
	spc.updatePodTracker.err = err
	spc.updatePodTracker.after = after
}

func (spc *fakeStatefulPodControl) SetDeleteStatefulPodError(err error, after int) {
	spc.deletePodTracker.err = err
	spc.deletePodTracker.after = after
}

func (spc *fakeStatefulPodControl) SetUpdateStatefulSetStatusError(err error, after int) {
	spc.updateStatusTracker.err = err
	spc.updateStatusTracker.after = after
}

func (spc *fakeStatefulPodControl) setPodPending(set *apps.StatefulSet, ordinal int) ([]*v1.Pod, error) {
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return nil, err
	}
	pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		return nil, err
	}
	if 0 > ordinal || ordinal >= len(pods) {
		return nil, fmt.Errorf("ordinal %d out of range [0,%d)", ordinal, len(pods))
	}
	sort.Sort(ascendingOrdinal(pods))
	pod := pods[ordinal]
	pod.Status.Phase = v1.PodPending
	fakeResourceVersion(pod)
	spc.podsIndexer.Update(pod)
	return spc.podsLister.Pods(set.Namespace).List(selector)
}

func (spc *fakeStatefulPodControl) setPodRunning(set *apps.StatefulSet, ordinal int) ([]*v1.Pod, error) {
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return nil, err
	}
	pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		return nil, err
	}
	if 0 > ordinal || ordinal >= len(pods) {
		return nil, fmt.Errorf("ordinal %d out of range [0,%d)", ordinal, len(pods))
	}
	sort.Sort(ascendingOrdinal(pods))
	pod := pods[ordinal]
	pod.Status.Phase = v1.PodRunning
	fakeResourceVersion(pod)
	spc.podsIndexer.Update(pod)
	return spc.podsLister.Pods(set.Namespace).List(selector)
}

func (spc *fakeStatefulPodControl) setPodReady(set *apps.StatefulSet, ordinal int) ([]*v1.Pod, error) {
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return nil, err
	}
	pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		return nil, err
	}
	if 0 > ordinal || ordinal >= len(pods) {
		return nil, fmt.Errorf("ordinal %d out of range [0,%d)", ordinal, len(pods))
	}
	sort.Sort(ascendingOrdinal(pods))
	pod := pods[ordinal]
	condition := v1.PodCondition{Type: v1.PodReady, Status: v1.ConditionTrue}
	v1.UpdatePodCondition(&pod.Status, &condition)
	fakeResourceVersion(pod)
	spc.podsIndexer.Update(pod)
	return spc.podsLister.Pods(set.Namespace).List(selector)
}

func (spc *fakeStatefulPodControl) setPodInitStatus(set *apps.StatefulSet, ordinal int, init bool) ([]*v1.Pod, error) {
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return nil, err
	}
	pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		return nil, err
	}
	if 0 > ordinal || ordinal >= len(pods) {
		return nil, fmt.Errorf("ordinal %d out of range [0,%d)", ordinal, len(pods))
	}
	sort.Sort(ascendingOrdinal(pods))
	pod := pods[ordinal]
	if init {
		pod.Annotations[apps.StatefulSetInitAnnotation] = "true"
	} else {
		pod.Annotations[apps.StatefulSetInitAnnotation] = "false"
	}
	fakeResourceVersion(pod)
	spc.podsIndexer.Update(pod)
	return spc.podsLister.Pods(set.Namespace).List(selector)
}

func (spc *fakeStatefulPodControl) addTerminatedPod(set *apps.StatefulSet, ordinal int) ([]*v1.Pod, error) {
	pod := newStatefulSetPod(set, ordinal)
	pod.Status.Phase = v1.PodRunning
	deleted := metav1.NewTime(time.Now())
	pod.DeletionTimestamp = &deleted
	condition := v1.PodCondition{Type: v1.PodReady, Status: v1.ConditionTrue}
	fakeResourceVersion(pod)
	v1.UpdatePodCondition(&pod.Status, &condition)
	spc.podsIndexer.Update(pod)
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return nil, err
	}
	return spc.podsLister.Pods(set.Namespace).List(selector)
}

func (spc *fakeStatefulPodControl) setPodTerminated(set *apps.StatefulSet, ordinal int) ([]*v1.Pod, error) {
	pod := newStatefulSetPod(set, ordinal)
	deleted := metav1.NewTime(time.Now())
	pod.DeletionTimestamp = &deleted
	fakeResourceVersion(pod)
	spc.podsIndexer.Update(pod)
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return nil, err
	}
	return spc.podsLister.Pods(set.Namespace).List(selector)
}

func (spc *fakeStatefulPodControl) CreateStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error {
	defer spc.createPodTracker.inc()
	if spc.createPodTracker.errorReady() {
		defer spc.createPodTracker.reset()
		return spc.createPodTracker.err
	}

	for _, claim := range getPersistentVolumeClaims(set, pod) {
		spc.claimsIndexer.Update(&claim)
	}
	spc.podsIndexer.Update(pod)
	return nil
}

func (spc *fakeStatefulPodControl) UpdateStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error {
	defer spc.updatePodTracker.inc()
	if spc.updatePodTracker.errorReady() {
		defer spc.updatePodTracker.reset()
		return spc.updatePodTracker.err
	}
	if !identityMatches(set, pod) {
		updateIdentity(set, pod)
	}
	if !storageMatches(set, pod) {
		updateStorage(set, pod)
		for _, claim := range getPersistentVolumeClaims(set, pod) {
			spc.claimsIndexer.Update(&claim)
		}
	}
	spc.podsIndexer.Update(pod)
	return nil
}

func (spc *fakeStatefulPodControl) DeleteStatefulPod(set *apps.StatefulSet, pod *v1.Pod) error {
	defer spc.deletePodTracker.inc()
	if spc.deletePodTracker.errorReady() {
		defer spc.deletePodTracker.reset()
		return spc.deletePodTracker.err
	}
	if key, err := controller.KeyFunc(pod); err != nil {
		return err
	} else if obj, found, err := spc.podsIndexer.GetByKey(key); err != nil {
		return err
	} else if found {
		spc.podsIndexer.Delete(obj)
	}

	return nil
}

func (spc *fakeStatefulPodControl) UpdateStatefulSetStatus(set *apps.StatefulSet, replicas int32, generation int64) error {
	defer spc.updateStatusTracker.inc()
	if spc.updateStatusTracker.errorReady() {
		defer spc.updateStatusTracker.reset()
		return spc.updateStatusTracker.err
	}
	set.Status.Replicas = replicas
	set.Status.ObservedGeneration = &generation
	spc.setsIndexer.Update(set)
	return nil
}

var _ StatefulPodControlInterface = &fakeStatefulPodControl{}

func assertInvariants(set *apps.StatefulSet, spc *fakeStatefulPodControl) error {
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return err
	}
	pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
	if err != nil {
		return err
	}
	sort.Sort(ascendingOrdinal(pods))
	for ord := 0; ord < len(pods); ord++ {
		if ord > 0 && isRunningAndReady(pods[ord]) && !isRunningAndReady(pods[ord-1]) {
			return fmt.Errorf("Predecessor %s is Running and Ready while %s is not",
				pods[ord-1].Name,
				pods[ord].Name)
		}
		if getOrdinal(pods[ord]) != ord {
			return fmt.Errorf("Pods %s deployed in the wrong order",
				pods[ord].Name)
		}
		if !storageMatches(set, pods[ord]) {
			return fmt.Errorf("Pods %s does not match the storage specification of StatefulSet %s ",
				pods[ord].
					Name, set.Name)
		} else {
			for _, claim := range getPersistentVolumeClaims(set, pods[ord]) {
				if claim, err := spc.claimsLister.PersistentVolumeClaims(set.Namespace).Get(claim.Name); err != nil {
					return err
				} else if claim == nil {
					return fmt.Errorf("claim %s for Pod %s was not created",
						claim.Name,
						pods[ord].Name)
				}
			}
		}
		if !identityMatches(set, pods[ord]) {
			return fmt.Errorf("Pods %s does not match the identity specification of StatefulSet %s ",
				pods[ord].Name,
				set.Name)
		}
	}
	return nil
}

func fakeResourceVersion(object interface{}) {
	obj, isObj := object.(metav1.Object)
	if !isObj {
		return
	} else if version := obj.GetResourceVersion(); version == "" {
		obj.SetResourceVersion("1")
	} else if intValue, err := strconv.ParseInt(version, 10, 32); err == nil {
		obj.SetResourceVersion(strconv.FormatInt(intValue+1, 10))
	}
}

func scaleUpStatefulSetControl(set *apps.StatefulSet, ssc StatefulSetControlInterface, spc *fakeStatefulPodControl) error {
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return err
	}
	for set.Status.Replicas < *set.Spec.Replicas {
		pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
		if err != nil {
			return err
		}
		if ord := len(pods) - 1; ord >= 0 {
			ord := len(pods) - 1
			if pods, err = spc.setPodPending(set, ord); err != nil {
				return err
			}
			if err = ssc.UpdateStatefulSet(set, pods); err != nil {
				return err
			}
			set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
			if err != nil {
				return err
			}
			if pods, err = spc.setPodRunning(set, ord); err != nil {
				return err
			}
			if err = ssc.UpdateStatefulSet(set, pods); err != nil {
				return err
			}
			set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
			if err != nil {
				return err
			}
			if pods, err = spc.setPodReady(set, ord); err != nil {
				return err
			}
		}
		if err := ssc.UpdateStatefulSet(set, pods); err != nil {
			return err
		}
		set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
		if err != nil {
			return err
		}
		if err := assertInvariants(set, spc); err != nil {
			return err
		}
	}
	return assertInvariants(set, spc)
}

func scaleDownStatefulSetControl(set *apps.StatefulSet, ssc StatefulSetControlInterface, spc *fakeStatefulPodControl) error {
	selector, err := metav1.LabelSelectorAsSelector(set.Spec.Selector)
	if err != nil {
		return err
	}
	for set.Status.Replicas > *set.Spec.Replicas {
		pods, err := spc.podsLister.Pods(set.Namespace).List(selector)
		if err != nil {
			return err
		}
		if ordinal := len(pods) - 1; ordinal >= 0 {
			if err := ssc.UpdateStatefulSet(set, pods); err != nil {
				return err
			}
			set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
			if err != nil {
				return err
			}
			if pods, err = spc.addTerminatedPod(set, ordinal); err != nil {
				return err
			}
			if err = ssc.UpdateStatefulSet(set, pods); err != nil {
				return err
			}
			set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
			if err != nil {
				return err
			}
			pods, err = spc.podsLister.Pods(set.Namespace).List(selector)
			if err != nil {
				return err
			}
			sort.Sort(ascendingOrdinal(pods))
			spc.podsIndexer.Delete(pods[ordinal])
		}
		if err := ssc.UpdateStatefulSet(set, pods); err != nil {
			return err
		}
		set, err = spc.setsLister.StatefulSets(set.Namespace).Get(set.Name)
		if err != nil {
			return err
		}
		if err := assertInvariants(set, spc); err != nil {
			return err
		}
	}
	return assertInvariants(set, spc)
}
