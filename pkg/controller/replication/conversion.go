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

// This file contains adapters that convert between RC and RS,
// as if ReplicationController were an older API version of ReplicaSet.
// It allows ReplicaSetController to directly replace the old ReplicationManager,
// which was previously a manually-maintained copy-paste of RSC.

package replication

import (
	"errors"
	"fmt"
	"time"

	"k8s.io/api/core/v1"
	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/watch"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	appsv1 "k8s.io/client-go/kubernetes/typed/apps/v1"
	appsv1beta2 "k8s.io/client-go/kubernetes/typed/apps/v1beta2"
	v1client "k8s.io/client-go/kubernetes/typed/core/v1"
	extensionsv1beta1client "k8s.io/client-go/kubernetes/typed/extensions/v1beta1"
	v1listers "k8s.io/client-go/listers/core/v1"
	extensionslisters "k8s.io/client-go/listers/extensions/v1beta1"
	"k8s.io/client-go/tools/cache"
	apiv1 "k8s.io/kubernetes/pkg/apis/core/v1"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsinternalv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/controller"
)

// informerAdapter implements ReplicaSetInformer by wrapping ReplicationControllerInformer
// and converting objects.
type informerAdapter struct {
	rcInformer coreinformers.ReplicationControllerInformer
}

func (i informerAdapter) Informer() cache.SharedIndexInformer {
	return conversionInformer{i.rcInformer.Informer()}
}

func (i informerAdapter) Lister() extensionslisters.ReplicaSetLister {
	return conversionLister{i.rcInformer.Lister()}
}

type conversionInformer struct {
	cache.SharedIndexInformer
}

func (i conversionInformer) AddEventHandler(handler cache.ResourceEventHandler) {
	i.SharedIndexInformer.AddEventHandler(conversionEventHandler{handler})
}

func (i conversionInformer) AddEventHandlerWithResyncPeriod(handler cache.ResourceEventHandler, resyncPeriod time.Duration) {
	i.SharedIndexInformer.AddEventHandlerWithResyncPeriod(conversionEventHandler{handler}, resyncPeriod)
}

type conversionLister struct {
	rcLister v1listers.ReplicationControllerLister
}

func (l conversionLister) List(selector labels.Selector) ([]*extensionsv1beta1.ReplicaSet, error) {
	rcList, err := l.rcLister.List(selector)
	if err != nil {
		return nil, err
	}
	return convertSlice(rcList)
}

func (l conversionLister) ReplicaSets(namespace string) extensionslisters.ReplicaSetNamespaceLister {
	return conversionNamespaceLister{l.rcLister.ReplicationControllers(namespace)}
}

func (l conversionLister) GetPodReplicaSets(pod *v1.Pod) ([]*extensionsv1beta1.ReplicaSet, error) {
	rcList, err := l.rcLister.GetPodControllers(pod)
	if err != nil {
		return nil, err
	}
	return convertSlice(rcList)
}

type conversionNamespaceLister struct {
	rcLister v1listers.ReplicationControllerNamespaceLister
}

func (l conversionNamespaceLister) List(selector labels.Selector) ([]*extensionsv1beta1.ReplicaSet, error) {
	rcList, err := l.rcLister.List(selector)
	if err != nil {
		return nil, err
	}
	return convertSlice(rcList)
}

func (l conversionNamespaceLister) Get(name string) (*extensionsv1beta1.ReplicaSet, error) {
	rc, err := l.rcLister.Get(name)
	if err != nil {
		return nil, err
	}
	return convertRCtoRS(rc, nil)
}

type conversionEventHandler struct {
	handler cache.ResourceEventHandler
}

func (h conversionEventHandler) OnAdd(obj interface{}) {
	rs, err := convertRCtoRS(obj.(*v1.ReplicationController), nil)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("dropping RC OnAdd event: can't convert object %#v to RS: %v", obj, err))
		return
	}
	h.handler.OnAdd(rs)
}

func (h conversionEventHandler) OnUpdate(oldObj, newObj interface{}) {
	oldRS, err := convertRCtoRS(oldObj.(*v1.ReplicationController), nil)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("dropping RC OnUpdate event: can't convert old object %#v to RS: %v", oldObj, err))
		return
	}
	newRS, err := convertRCtoRS(newObj.(*v1.ReplicationController), nil)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("dropping RC OnUpdate event: can't convert new object %#v to RS: %v", newObj, err))
		return
	}
	h.handler.OnUpdate(oldRS, newRS)
}

func (h conversionEventHandler) OnDelete(obj interface{}) {
	rc, ok := obj.(*v1.ReplicationController)
	if !ok {
		// Convert the Obj inside DeletedFinalStateUnknown.
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("dropping RC OnDelete event: couldn't get object from tombstone %+v", obj))
			return
		}
		rc, ok = tombstone.Obj.(*v1.ReplicationController)
		if !ok {
			utilruntime.HandleError(fmt.Errorf("dropping RC OnDelete event: tombstone contained object that is not a RC %#v", obj))
			return
		}
		rs, err := convertRCtoRS(rc, nil)
		if err != nil {
			utilruntime.HandleError(fmt.Errorf("dropping RC OnDelete event: can't convert object %#v to RS: %v", obj, err))
			return
		}
		h.handler.OnDelete(cache.DeletedFinalStateUnknown{Key: tombstone.Key, Obj: rs})
		return
	}

	// It's a regular RC object.
	rs, err := convertRCtoRS(rc, nil)
	if err != nil {
		utilruntime.HandleError(fmt.Errorf("dropping RC OnDelete event: can't convert object %#v to RS: %v", obj, err))
		return
	}
	h.handler.OnDelete(rs)
}

type clientsetAdapter struct {
	clientset.Interface
}

func (c clientsetAdapter) ExtensionsV1beta1() extensionsv1beta1client.ExtensionsV1beta1Interface {
	return conversionExtensionsClient{c.Interface, c.Interface.ExtensionsV1beta1()}
}

func (c clientsetAdapter) Extensions() extensionsv1beta1client.ExtensionsV1beta1Interface {
	return conversionExtensionsClient{c.Interface, c.Interface.ExtensionsV1beta1()}
}

func (c clientsetAdapter) AppsV1beta2() appsv1beta2.AppsV1beta2Interface {
	return conversionAppsV1beta2Client{c.Interface, c.Interface.AppsV1beta2()}
}

func (c clientsetAdapter) AppsV1() appsv1.AppsV1Interface {
	return conversionAppsV1Client{c.Interface, c.Interface.AppsV1()}
}

func (c clientsetAdapter) Apps() appsv1.AppsV1Interface {
	return conversionAppsV1Client{c.Interface, c.Interface.AppsV1()}
}

type conversionAppsV1beta2Client struct {
	clientset clientset.Interface
	appsv1beta2.AppsV1beta2Interface
}

func (c conversionAppsV1beta2Client) ReplicaSets(namespace string) appsv1beta2.ReplicaSetInterface {
	// TODO(enisoc): This will force RC integration tests to fail if anyone tries to update
	// ReplicaSetController to use apps/v1beta2 without updating this conversion adapter.
	// Please change conversionClient to use the new RS version instead of extensions/v1beta1,
	// and then return a conversionClient here.
	panic("need to update RC/RS conversionClient for apps/v1beta2")
}

type conversionAppsV1Client struct {
	clientset clientset.Interface
	appsv1.AppsV1Interface
}

func (c conversionAppsV1Client) ReplicaSets(namespace string) appsv1.ReplicaSetInterface {
	// TODO(enisoc): This will force RC integration tests to fail if anyone tries to update
	// ReplicaSetController to use apps/v1 without updating this conversion adapter.
	// Please change conversionClient to use the new RS version instead of extensions/v1beta1,
	// and then return a conversionClient here.
	panic("need to update RC/RS conversionClient for apps/v1")
}

type conversionExtensionsClient struct {
	clientset clientset.Interface
	extensionsv1beta1client.ExtensionsV1beta1Interface
}

func (c conversionExtensionsClient) ReplicaSets(namespace string) extensionsv1beta1client.ReplicaSetInterface {
	return conversionClient{c.clientset.CoreV1().ReplicationControllers(namespace)}
}

type conversionClient struct {
	v1client.ReplicationControllerInterface
}

func (c conversionClient) Create(rs *extensionsv1beta1.ReplicaSet) (*extensionsv1beta1.ReplicaSet, error) {
	return convertCall(c.ReplicationControllerInterface.Create, rs)
}

func (c conversionClient) Update(rs *extensionsv1beta1.ReplicaSet) (*extensionsv1beta1.ReplicaSet, error) {
	return convertCall(c.ReplicationControllerInterface.Update, rs)
}

func (c conversionClient) UpdateStatus(rs *extensionsv1beta1.ReplicaSet) (*extensionsv1beta1.ReplicaSet, error) {
	return convertCall(c.ReplicationControllerInterface.UpdateStatus, rs)
}

func (c conversionClient) Get(name string, options metav1.GetOptions) (*extensionsv1beta1.ReplicaSet, error) {
	rc, err := c.ReplicationControllerInterface.Get(name, options)
	if err != nil {
		return nil, err
	}
	return convertRCtoRS(rc, nil)
}

func (c conversionClient) List(opts metav1.ListOptions) (*extensionsv1beta1.ReplicaSetList, error) {
	rcList, err := c.ReplicationControllerInterface.List(opts)
	if err != nil {
		return nil, err
	}
	return convertList(rcList)
}

func (c conversionClient) Watch(opts metav1.ListOptions) (watch.Interface, error) {
	// This is not used by RSC because we wrap the shared informer instead.
	return nil, errors.New("Watch() is not implemented for conversionClient")
}

func (c conversionClient) Patch(name string, pt types.PatchType, data []byte, subresources ...string) (result *extensionsv1beta1.ReplicaSet, err error) {
	// This is not used by RSC.
	return nil, errors.New("Patch() is not implemented for conversionClient")
}

func convertSlice(rcList []*v1.ReplicationController) ([]*extensionsv1beta1.ReplicaSet, error) {
	rsList := make([]*extensionsv1beta1.ReplicaSet, 0, len(rcList))
	for _, rc := range rcList {
		rs, err := convertRCtoRS(rc, nil)
		if err != nil {
			return nil, err
		}
		rsList = append(rsList, rs)
	}
	return rsList, nil
}

func convertList(rcList *v1.ReplicationControllerList) (*extensionsv1beta1.ReplicaSetList, error) {
	rsList := &extensionsv1beta1.ReplicaSetList{Items: make([]extensionsv1beta1.ReplicaSet, len(rcList.Items))}
	for i := range rcList.Items {
		rc := &rcList.Items[i]
		_, err := convertRCtoRS(rc, &rsList.Items[i])
		if err != nil {
			return nil, err
		}
	}
	return rsList, nil
}

func convertCall(fn func(*v1.ReplicationController) (*v1.ReplicationController, error), rs *extensionsv1beta1.ReplicaSet) (*extensionsv1beta1.ReplicaSet, error) {
	rc, err := convertRStoRC(rs)
	if err != nil {
		return nil, err
	}
	result, err := fn(rc)
	if err != nil {
		return nil, err
	}
	return convertRCtoRS(result, nil)
}

func convertRCtoRS(rc *v1.ReplicationController, out *extensionsv1beta1.ReplicaSet) (*extensionsv1beta1.ReplicaSet, error) {
	var rsInternal extensions.ReplicaSet
	if err := apiv1.Convert_v1_ReplicationController_to_extensions_ReplicaSet(rc, &rsInternal, nil); err != nil {
		return nil, fmt.Errorf("can't convert ReplicationController %v/%v to ReplicaSet: %v", rc.Namespace, rc.Name, err)
	}
	if out == nil {
		out = new(extensionsv1beta1.ReplicaSet)
	}
	if err := extensionsinternalv1beta1.Convert_extensions_ReplicaSet_To_v1beta1_ReplicaSet(&rsInternal, out, nil); err != nil {
		return nil, fmt.Errorf("can't convert ReplicaSet (converted from ReplicationController %v/%v) from internal to extensions/v1beta1: %v", rc.Namespace, rc.Name, err)
	}
	return out, nil
}

func convertRStoRC(rs *extensionsv1beta1.ReplicaSet) (*v1.ReplicationController, error) {
	var rsInternal extensions.ReplicaSet
	if err := extensionsinternalv1beta1.Convert_v1beta1_ReplicaSet_To_extensions_ReplicaSet(rs, &rsInternal, nil); err != nil {
		return nil, fmt.Errorf("can't convert ReplicaSet (converting to ReplicationController %v/%v) from extensions/v1beta1 to internal: %v", rs.Namespace, rs.Name, err)
	}
	var rc v1.ReplicationController
	if err := apiv1.Convert_extensions_ReplicaSet_to_v1_ReplicationController(&rsInternal, &rc, nil); err != nil {
		return nil, fmt.Errorf("can't convert ReplicaSet to ReplicationController %v/%v: %v", rs.Namespace, rs.Name, err)
	}
	return &rc, nil
}

type podControlAdapter struct {
	controller.PodControlInterface
}

func (pc podControlAdapter) CreatePods(namespace string, template *v1.PodTemplateSpec, object runtime.Object) error {
	// This is not used by RSC.
	return errors.New("CreatePods() is not implemented for podControlAdapter")
}

func (pc podControlAdapter) CreatePodsOnNode(nodeName, namespace string, template *v1.PodTemplateSpec, object runtime.Object, controllerRef *metav1.OwnerReference) error {
	// This is not used by RSC.
	return errors.New("CreatePodsOnNode() is not implemented for podControlAdapter")
}

func (pc podControlAdapter) CreatePodsWithControllerRef(namespace string, template *v1.PodTemplateSpec, object runtime.Object, controllerRef *metav1.OwnerReference) error {
	rc, err := convertRStoRC(object.(*extensionsv1beta1.ReplicaSet))
	if err != nil {
		return err
	}
	return pc.PodControlInterface.CreatePodsWithControllerRef(namespace, template, rc, controllerRef)
}

func (pc podControlAdapter) DeletePod(namespace string, podID string, object runtime.Object) error {
	rc, err := convertRStoRC(object.(*extensionsv1beta1.ReplicaSet))
	if err != nil {
		return err
	}
	return pc.PodControlInterface.DeletePod(namespace, podID, rc)
}
