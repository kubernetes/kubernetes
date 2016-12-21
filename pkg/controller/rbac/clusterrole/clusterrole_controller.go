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

package clusterrole

import (
	"encoding/json"
	"fmt"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
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
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"

	"github.com/golang/glog"
)

type ClusterRoleController struct {
	roleBindingControl controller.RoleBindingControlInterface

	clusterRoleBindingControl controller.ClusterRoleBindingControlInterface

	kubeClient internalclientset.Interface

	syncHandler func(roleKey string) error

	// roleStoreSynced returns true if the clusterrole store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	roleStoreSynced cache.InformerSynced

	// A store of clusterrole
	roleLister cache.ClusterRoleLister

	// role that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder
}

func NewClusterRoleController(roleInformer informers.ClusterRoleInformer, kubeClient internalclientset.Interface) *ClusterRoleController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&internalcore.EventSinkImpl{Interface: kubeClient.Core().Events("")})

	if kubeClient != nil && kubeClient.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("clusterrole_controller", kubeClient.Core().RESTClient().GetRateLimiter())
	}

	rc := &ClusterRoleController{
		kubeClient: kubeClient,
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "clusterrole"),
		recorder:   eventBroadcaster.NewRecorder(v1.EventSource{Component: "clusterrole-controller"}),
	}
	rc.roleBindingControl = controller.RealRoleBindingControl{
		KubeClient: kubeClient,
		Recorder:   rc.recorder,
	}
	rc.clusterRoleBindingControl = controller.RealClusterRoleBindingControl{
		KubeClient: kubeClient,
		Recorder:   rc.recorder,
	}

	roleInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    rc.addClusterRole,
		UpdateFunc: rc.updateClusterRole,
		DeleteFunc: rc.deleteClusterRole,
	})
	rc.roleLister = roleInformer.Lister()
	rc.roleStoreSynced = roleInformer.Informer().HasSynced

	rc.syncHandler = rc.syncRole
	return rc
}

// Run the main goroutine responsible for watching and syncing clusterroles.
func (rc *ClusterRoleController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer rc.queue.ShutDown()
	glog.Infof("Starting ClusterRole Manager")

	if !cache.WaitForCacheSync(stopCh, rc.roleStoreSynced) {
		return
	}

	go wait.Until(rc.worker, time.Second, stopCh)
	<-stopCh
	glog.Infof("Shutting down ClusterRole Manager")
}

func (rc *ClusterRoleController) addClusterRole(obj interface{}) {
	castObj := obj.(*rbac.ClusterRole)
	glog.V(4).Infof("Adding %s", castObj.Name)
	rc.enqueueController(castObj)
}

func (rc *ClusterRoleController) updateClusterRole(oldObj, newObj interface{}) {
	castNewObj := newObj.(*rbac.ClusterRole)
	glog.V(4).Infof("Update %s", castNewObj.Name)
	rc.enqueueController(castNewObj)
}

func (rc *ClusterRoleController) deleteClusterRole(obj interface{}) {
	castObj := obj.(*rbac.ClusterRole)
	glog.V(4).Infof("Deleting %s", castObj.Name)
	castObj.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	rc.enqueueController(castObj)
}

func (rc *ClusterRoleController) enqueueController(obj interface{}) {
	key, err := clusterRoleKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}
	glog.Infof("key %s", key)
	rc.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (rc *ClusterRoleController) worker() {
	for rc.processNextWorkItem() {
	}
}

func (rc *ClusterRoleController) processNextWorkItem() bool {
	key, quit := rc.queue.Get()
	if quit {
		return false
	}
	defer rc.queue.Done(key)

	err := rc.syncHandler(key.(string))
	if err == nil {
		rc.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("Error syncing clusterrole: %v", err))
	rc.queue.AddRateLimited(key)

	return true
}

// syncRole will sync the OwnerReferences with the given key.
func (rc *ClusterRoleController) syncRole(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing clusterrole %q (%v)", key, time.Now().Sub(startTime))
	}()

	role, err := splitClusterRoleKey(key)
	if err != nil {
		return err
	}

	var errs []error
	if err := rc.classifyRoleBindings(role); err != nil {
		errs = append(errs, err)
	}
	if err := rc.classifyClusterRoleBindings(role); err != nil {
		errs = append(errs, err)
	}
	return utilerrors.NewAggregate(errs)
}

// classifyRoleBindings uses NewRoleBindingControllerRefManager to classify RoleBindings
// and adopts them if their reference to the ClusterRoles but are missing the reference.
// It also removes the controllerRef for RoleBindings, whose no longer matches the ClusterRoles.
func (rc *ClusterRoleController) classifyRoleBindings(role *rbac.ClusterRole) error {
	bindings, err := rc.kubeClient.Rbac().RoleBindings(api.NamespaceAll).List(api.ListOptions{})
	if err != nil {
		return err
	}

	cm := controller.NewRoleBindingControllerRefManager(rc.roleBindingControl, role.ObjectMeta, getClusterRoleKind())
	matchesAndControlled, matchesNeedsController, controlledDoesNotMatch := cm.Classify(bindings.Items)
	// Adopt RoleBindings only if this role is not going to be deleted.
	if role.DeletionTimestamp == nil {
		for _, binding := range matchesNeedsController {
			err := cm.AdoptRoleBinding(binding)
			// continue to next RoleBinding if adoption fails.
			if err != nil {
				// If no longer exists, don't even log the error.
				if !errors.IsNotFound(err) {
					utilruntime.HandleError(err)
				}
			} else {
				matchesAndControlled = append(matchesAndControlled, binding)
			}
		}
	}
	// remove the controllerRef for the ClusterRole that no longer own the RoleBinding
	var errs []error
	for _, binding := range controlledDoesNotMatch {
		err := cm.ReleaseRoleBinding(binding)
		if err != nil {
			errs = append(errs, err)
		}
	}
	return utilerrors.NewAggregate(errs)
}

// classifyClusterRoleBindings uses NewClusterRoleBindingControllerRefManager to classify ClusterRoleBindings
// and adopts them if their reference to the ClusterRoles but are missing the reference.
// It also removes the controllerRef for ClusterRoleBindings, whose no longer matches the Roles.
func (rc *ClusterRoleController) classifyClusterRoleBindings(role *rbac.ClusterRole) error {
	bindings, err := rc.kubeClient.Rbac().ClusterRoleBindings().List(api.ListOptions{})
	if err != nil {
		return err
	}

	cm := controller.NewClusterRoleBindingControllerRefManager(rc.clusterRoleBindingControl, role.ObjectMeta, getClusterRoleKind())
	matchesAndControlled, matchesNeedsController, controlledDoesNotMatch := cm.Classify(bindings.Items)
	// Adopt ClusterRoleBindings only if this role is not going to be deleted.
	if role.DeletionTimestamp == nil {
		for _, binding := range matchesNeedsController {
			err := cm.AdoptClusterRoleBinding(binding)
			// continue to next ClusterRoleBinding if adoption fails.
			if err != nil {
				// If no longer exists, don't even log the error.
				if !errors.IsNotFound(err) {
					utilruntime.HandleError(err)
				}
			} else {
				matchesAndControlled = append(matchesAndControlled, binding)
			}
		}
	}
	// remove the controllerRef for the ClusterRole that no longer own the ClusterRoleBinding
	var errs []error
	for _, binding := range controlledDoesNotMatch {
		err := cm.ReleaseClusterRoleBinding(binding)
		if err != nil {
			errs = append(errs, err)
		}
	}
	return utilerrors.NewAggregate(errs)
}

func getClusterRoleKind() schema.GroupVersionKind {
	return v1alpha1.SchemeGroupVersion.WithKind("ClusterRole")
}

func (rc *ClusterRoleController) handleRoleBinding(uid string) error {
	bindings, err := rc.kubeClient.Rbac().RoleBindings(api.NamespaceAll).List(api.ListOptions{})
	if err != nil {
		return err
	}
	for _, binding := range bindings.Items {
		if string(binding.ObjectMeta.GetUID()) == uid {
			return rc.kubeClient.Rbac().RoleBindings(binding.Namespace).Delete(binding.Name, nil)
		}
	}
	return nil
}

func (rc *ClusterRoleController) handleClusterRoleBinding(name string) error {
	_, err := rc.kubeClient.Rbac().ClusterRoleBindings().Get(name, metav1.GetOptions{})
	if err != nil {
		return nil
	}
	return rc.kubeClient.Rbac().ClusterRoleBindings().Delete(name, nil)
}

// clusterRoleBindingKeyFunc encode obj into key
func clusterRoleKeyFunc(obj interface{}) (string, error) {
	castObj := obj.(*rbac.ClusterRole)
	byteArray, err := json.Marshal(castObj)
	s := string(byteArray[:])
	return s, err
}

// splitClusterRoleKey returns the obj that clusterRoleKeyFunc encoded into key.
func splitClusterRoleKey(key string) (*rbac.ClusterRole, error) {
	role := new(rbac.ClusterRole)
	err := json.Unmarshal([]byte(key), role)
	if err != nil {
		return nil, fmt.Errorf("unexpected key format: %q", key)
	}
	return role, nil
}
