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

package clusterrolebinding

import (
	"encoding/json"
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/rbac/v1alpha1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	internalcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/informers"
	"k8s.io/kubernetes/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"

	"github.com/golang/glog"
)

type ClusterRoleBindingController struct {
	clusterRoleBindingControl controller.ClusterRoleBindingControlInterface

	kubeClient internalclientset.Interface

	syncHandler func(bindingKey string) error

	// bindingStoreSynced returns true if the clusterrolebinding store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	bindingStoreSynced cache.InformerSynced

	// A store of clusterrolebindings
	bindingLister cache.ClusterRoleBindingLister

	// clusterrolebinding that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder
}

func NewClusterRoleBindingController(bindingInformer informers.ClusterRoleBindingInformer, kubeClient internalclientset.Interface) *ClusterRoleBindingController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&internalcore.EventSinkImpl{Interface: kubeClient.Core().Events("")})

	if kubeClient != nil && kubeClient.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("clusterrolebinding_controller", kubeClient.Core().RESTClient().GetRateLimiter())
	}

	rb := &ClusterRoleBindingController{
		kubeClient: kubeClient,
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "clusterrolebinding"),
		recorder:   eventBroadcaster.NewRecorder(v1.EventSource{Component: "clusterrolebinding-controller"}),
	}
	rb.clusterRoleBindingControl = controller.RealClusterRoleBindingControl{
		KubeClient: kubeClient,
		Recorder:   rb.recorder,
	}

	bindingInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    rb.addClusterRoleBinding,
		UpdateFunc: rb.updateClusterRoleBinding,
		DeleteFunc: rb.deleteClusterRoleBinding,
	})
	rb.bindingLister = bindingInformer.Lister()
	rb.bindingStoreSynced = bindingInformer.Informer().HasSynced

	rb.syncHandler = rb.syncBinding
	return rb
}

// Run the main goroutine responsible for watching and syncing clusterrolebindings.
func (rb *ClusterRoleBindingController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer rb.queue.ShutDown()
	glog.Infof("Starting ClusterRoleBinding Manager")

	if !cache.WaitForCacheSync(stopCh, rb.bindingStoreSynced) {
		return
	}

	go wait.Until(rb.worker, time.Second, stopCh)
	<-stopCh
	glog.Infof("Shutting down ClusterRoleBinding Manager")
}

func (rb *ClusterRoleBindingController) addClusterRoleBinding(obj interface{}) {
	castObj := obj.(*rbac.ClusterRoleBinding)
	glog.V(4).Infof("Adding %s", castObj.Name)
	rb.enqueueController(castObj)
}

func (rb *ClusterRoleBindingController) updateClusterRoleBinding(oldObj, newObj interface{}) {
	castNewObj := newObj.(*rbac.ClusterRoleBinding)
	glog.V(4).Infof("Update %s", castNewObj.Name)
	rb.enqueueController(castNewObj)
}

func (rb *ClusterRoleBindingController) deleteClusterRoleBinding(obj interface{}) {
	castObj := obj.(*rbac.ClusterRoleBinding)
	glog.V(4).Infof("Deleting %s", castObj.Name)
	castObj.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	rb.enqueueController(castObj)
}

func (rb *ClusterRoleBindingController) enqueueController(obj interface{}) {
	key, err := clusterRoleBindingKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}
	glog.Infof("key %s", key)
	rb.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (rb *ClusterRoleBindingController) worker() {
	for rb.processNextWorkItem() {
	}
}

func (rb *ClusterRoleBindingController) processNextWorkItem() bool {
	key, quit := rb.queue.Get()
	if quit {
		return false
	}
	defer rb.queue.Done(key)

	err := rb.syncHandler(key.(string))
	if err == nil {
		rb.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("Error syncing clusterrolebinding: %v", err))
	rb.queue.AddRateLimited(key)

	return true
}

// syncBinding will sync the OwnerReferences with the given key.
func (rb *ClusterRoleBindingController) syncBinding(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing clusterrolebinding %q (%v)", key, time.Now().Sub(startTime))
	}()

	clusterRoleBinding, err := splitClusterRoleBindingKey(key)
	if err != nil {
		return err
	}

	if clusterRoleBinding.DeletionTimestamp != nil {
		return nil
	}

	switch clusterRoleBinding.RoleRef.Kind {
	case "ClusterRole":
		return rb.handleWithClusterRoles(clusterRoleBinding)
	}
	return nil
}

func (rb *ClusterRoleBindingController) handleWithClusterRoles(binding *rbac.ClusterRoleBinding) error {
	role, err := rb.kubeClient.Rbac().ClusterRoles().Get(binding.RoleRef.Name, metav1.GetOptions{})
	if err != nil {
		glog.V(4).Infof("Get ClusterRole failed: %v", err)
		return nil
	}

	controllerKind := getClusterRoleKind()

	owner := getControllerOf(binding.ObjectMeta)
	if owner != nil {
		controllerRef := &metav1.OwnerReference{
			APIVersion: controllerKind.GroupVersion().String(),
			Kind:       controllerKind.Kind,
			Name:       role.Name,
			UID:        role.UID,
		}
		// If the OwnerReference is latest, return nil
		if checkControllerOf(owner, controllerRef) {
			return nil
		}
	}

	addControllerPatch := fmt.Sprintf(
		`{"metadata":{"ownerReferences":[{"apiVersion":"%s","kind":"%s","name":"%s","uid":"%s","controller":true}],"uid":"%s"}}`,
		controllerKind.GroupVersion(), controllerKind.Kind, role.Name, role.UID, binding.UID)
	return rb.clusterRoleBindingControl.PatchClusterRoleBinding(binding.Name, []byte(addControllerPatch))
}

func getClusterRoleKind() schema.GroupVersionKind {
	return v1alpha1.SchemeGroupVersion.WithKind("ClusterRole")
}

// clusterRoleBindingKeyFunc encode obj into key
func clusterRoleBindingKeyFunc(obj interface{}) (string, error) {
	castObj := obj.(*rbac.ClusterRoleBinding)
	byteArray, err := json.Marshal(castObj)
	s := string(byteArray[:])
	return s, err
}

// splitClusterRoleBindingKey returns the obj that
// clusterRoleBindingKeyFunc encoded into key.
func splitClusterRoleBindingKey(key string) (*rbac.ClusterRoleBinding, error) {
	rolebinding := new(rbac.ClusterRoleBinding)
	err := json.Unmarshal([]byte(key), rolebinding)
	if err != nil {
		return nil, fmt.Errorf("unexpected key format: %q", key)
	}
	return rolebinding, nil
}

// getControllerOf returns the controllerRef if controllee has a controller,
// otherwise returns nil.
func getControllerOf(controllee api.ObjectMeta) *metav1.OwnerReference {
	for _, owner := range controllee.OwnerReferences {
		// controlled by other controller
		if owner.Controller != nil && *owner.Controller == true {
			return &owner
		}
	}
	return nil
}

// If the controllerRef is the latest, checkControllerOf returns true
// otherwise returns false.
func checkControllerOf(a *metav1.OwnerReference, b *metav1.OwnerReference) bool {
	if a.UID != b.UID {
		return false
	}
	if a.Name != b.Name {
		return false
	}
	if a.APIVersion != b.APIVersion {
		return false
	}
	if a.Kind != b.Kind {
		return false
	}
	return true
}
