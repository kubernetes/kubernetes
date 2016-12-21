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

package role

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

type RoleController struct {
	roleBindingControl controller.RoleBindingControlInterface

	kubeClient internalclientset.Interface

	syncHandler func(roleKey string) error

	// roleStoreSynced returns true if the clusterrole store has been synced at least once.
	// Added as a member to the struct to allow injection for testing.
	roleStoreSynced cache.InformerSynced

	// A store of clusterrole
	roleLister cache.RoleLister

	// role that need to be updated
	queue workqueue.RateLimitingInterface

	recorder record.EventRecorder
}

func NewRoleController(roleInformer informers.RoleInformer, kubeClient internalclientset.Interface) *RoleController {
	eventBroadcaster := record.NewBroadcaster()
	eventBroadcaster.StartLogging(glog.Infof)
	eventBroadcaster.StartRecordingToSink(&internalcore.EventSinkImpl{Interface: kubeClient.Core().Events("")})

	if kubeClient != nil && kubeClient.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("role_controller", kubeClient.Core().RESTClient().GetRateLimiter())
	}

	rc := &RoleController{
		kubeClient: kubeClient,
		queue:      workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "role"),
		recorder:   eventBroadcaster.NewRecorder(v1.EventSource{Component: "role-controller"}),
	}
	rc.roleBindingControl = controller.RealRoleBindingControl{
		KubeClient: kubeClient,
		Recorder:   rc.recorder,
	}

	roleInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    rc.addRole,
		UpdateFunc: rc.updateRole,
		DeleteFunc: rc.deleteRole,
	})
	rc.roleLister = roleInformer.Lister()
	rc.roleStoreSynced = roleInformer.Informer().HasSynced

	rc.syncHandler = rc.syncRole
	return rc
}

// Run the main goroutine responsible for watching and syncing clusterroles.
func (rc *RoleController) Run(stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer rc.queue.ShutDown()
	glog.Infof("Starting Role Manager")

	if !cache.WaitForCacheSync(stopCh, rc.roleStoreSynced) {
		return
	}

	go wait.Until(rc.worker, time.Second, stopCh)
	<-stopCh
	glog.Infof("Shutting down Role Manager")
}

func (rc *RoleController) addRole(obj interface{}) {
	castObj := obj.(*rbac.Role)
	glog.Infof("Adding %s", castObj.Name)
	rc.enqueueController(castObj)
}

func (rc *RoleController) updateRole(oldObj, newObj interface{}) {
	castNewObj := newObj.(*rbac.Role)
	glog.Infof("Update %s", castNewObj.Name)
	rc.enqueueController(castNewObj)
}

func (rc *RoleController) deleteRole(obj interface{}) {
	castObj := obj.(*rbac.Role)
	glog.Infof("Deleting %s", castObj.Name)
	castObj.DeletionTimestamp = &metav1.Time{Time: time.Now()}
	rc.enqueueController(castObj)
}

func (rc *RoleController) enqueueController(obj interface{}) {
	key, err := roleKeyFunc(obj)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("Couldn't get key for object %+v: %v", obj, err))
		return
	}
	glog.Infof("key %s", key)
	rc.queue.Add(key)
}

// worker runs a worker thread that just dequeues items, processes them, and marks them done.
// It enforces that the syncHandler is never invoked concurrently with the same key.
func (rc *RoleController) worker() {
	for rc.processNextWorkItem() {
	}
}

func (rc *RoleController) processNextWorkItem() bool {
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

	utilruntime.HandleError(fmt.Errorf("Error syncing role: %v", err))
	rc.queue.AddRateLimited(key)

	return true
}

// syncRole will sync the OwnerReferences with the given key.
func (rc *RoleController) syncRole(key string) error {
	startTime := time.Now()
	defer func() {
		glog.V(4).Infof("Finished syncing role %q (%v)", key, time.Now().Sub(startTime))
	}()

	role, err := splitRoleKey(key)
	if err != nil {
		return err
	}
	return rc.classifyRoleBindings(role)
}

// classifyRoleBindings uses NewRoleBindingControllerRefManager to classify RoleBindings
// and adopts them if their reference to the Roles but are missing the reference.
// It also removes the controllerRef for RoleBindings, whose no longer matches the Roles.
func (rc *RoleController) classifyRoleBindings(role *rbac.Role) error {
	bindings, err := rc.kubeClient.Rbac().RoleBindings(role.Namespace).List(api.ListOptions{})
	if err != nil {
		return err
	}

	cm := controller.NewRoleBindingControllerRefManager(rc.roleBindingControl, role.ObjectMeta, getRoleKind())
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
	// remove the controllerRef for the Role that no longer own the RoleBinding
	var errs []error
	for _, binding := range controlledDoesNotMatch {
		err := cm.ReleaseRoleBinding(binding)
		if err != nil {
			errs = append(errs, err)
		}
	}
	return utilerrors.NewAggregate(errs)
}

func getRoleKind() schema.GroupVersionKind {
	return v1alpha1.SchemeGroupVersion.WithKind("Role")
}

// roleBindingKeyFunc encode obj into key
func roleKeyFunc(obj interface{}) (string, error) {
	castObj := obj.(*rbac.Role)
	byteArray, err := json.Marshal(castObj)
	s := string(byteArray[:])
	return s, err
}

// splitRoleKey returns the obj that roleKeyFunc encoded into key.
func splitRoleKey(key string) (*rbac.Role, error) {
	role := new(rbac.Role)
	err := json.Unmarshal([]byte(key), role)
	if err != nil {
		return nil, fmt.Errorf("unexpected key format: %q", key)
	}
	return role, nil
}
