/*
Copyright 2018 The Kubernetes Authors.

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

package rootcacertpublisher

import (
	"fmt"
	"reflect"
	"time"

	"k8s.io/api/core/v1"
	apierrs "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/metrics"
)

// RootCACertCofigMapName is name of the configmap which stores certificates to access api-server
const RootCACertCofigMapName = "kube-root-ca.crt"

// NewPublisher construct a new controller which would manage the configmap which stores
// certificates in each namespace. It will make sure certificate configmap exists in each namespace.
func NewPublisher(cmInformer coreinformers.ConfigMapInformer, nsInformer coreinformers.NamespaceInformer, cl clientset.Interface, rootCA []byte) (*Publisher, error) {
	e := &Publisher{
		client: cl,
		configMap: v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name: RootCACertCofigMapName,
			},
			Data: map[string]string{
				"ca.crt": string(rootCA),
			},
		},
		queue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "root-ca-cert-publisher"),
	}
	if cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		if err := metrics.RegisterMetricAndTrackRateLimiterUsage("root_ca_cert_publisher", cl.CoreV1().RESTClient().GetRateLimiter()); err != nil {
			return nil, err
		}
	}

	cmInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		DeleteFunc: e.configMapDeleted,
		UpdateFunc: e.configMapUpdated,
	})
	e.cmLister = cmInformer.Lister()
	e.cmListerSynced = cmInformer.Informer().HasSynced

	nsInformer.Informer().AddEventHandler(cache.ResourceEventHandlerFuncs{
		AddFunc:    e.namespaceAdded,
		UpdateFunc: e.namespaceUpdated,
	})
	e.nsLister = nsInformer.Lister()
	e.nsListerSynced = nsInformer.Informer().HasSynced

	e.syncHandler = e.syncNamespace

	return e, nil

}

// Publisher manages certificate ConfigMap objects inside Namespaces
type Publisher struct {
	client    clientset.Interface
	configMap v1.ConfigMap

	// To allow injection for testing.
	syncHandler func(key string) error

	cmLister       corelisters.ConfigMapLister
	cmListerSynced cache.InformerSynced

	nsLister       corelisters.NamespaceLister
	nsListerSynced cache.InformerSynced

	queue workqueue.RateLimitingInterface
}

// Run starts process
func (c *Publisher) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting root CA certificate configmap publisher")
	defer klog.Infof("Shutting down root CA certificate configmap publisher")

	if !controller.WaitForCacheSync("crt configmap", stopCh, c.cmListerSynced, c.nsListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.Until(c.runWorker, time.Second, stopCh)
	}

	<-stopCh
}

func (c *Publisher) configMapDeleted(obj interface{}) {
	cm, err := convertToCM(obj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	if cm.Name != RootCACertCofigMapName {
		return
	}
	c.queue.Add(cm.Namespace)
}

func (c *Publisher) configMapUpdated(_, newObj interface{}) {
	newConfigMap, err := convertToCM(newObj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	if newConfigMap.Name != RootCACertCofigMapName {
		return
	}

	if reflect.DeepEqual(c.configMap.Data, newConfigMap.Data) {
		return
	}

	newConfigMap.Data = make(map[string]string)
	newConfigMap.Data["ca.crt"] = c.configMap.Data["ca.crt"]
	if _, err := c.client.CoreV1().ConfigMaps(newConfigMap.Namespace).Update(newConfigMap); err != nil && !apierrs.IsAlreadyExists(err) {
		utilruntime.HandleError(fmt.Errorf("configmap creation failure:%v", err))
	}
}

func (c *Publisher) namespaceAdded(obj interface{}) {
	namespace := obj.(*v1.Namespace)
	c.queue.Add(namespace.Name)
}

func (c *Publisher) namespaceUpdated(oldObj interface{}, newObj interface{}) {
	newNamespace := newObj.(*v1.Namespace)
	if newNamespace.Status.Phase != v1.NamespaceActive {
		return
	}
	c.queue.Add(newNamespace.Name)
}

func (c *Publisher) runWorker() {
	for c.processNextWorkItem() {
	}
}

// processNextWorkItem deals with one key off the queue.  It returns false when it's time to quit.
func (c *Publisher) processNextWorkItem() bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	err := c.syncHandler(key.(string))
	if err == nil {
		c.queue.Forget(key)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", key, err))
	c.queue.AddRateLimited(key)

	return true
}

func (c *Publisher) syncNamespace(key string) error {
	startTime := time.Now()
	defer func() {
		klog.V(4).Infof("Finished syncing namespace %q (%v)", key, time.Since(startTime))
	}()

	ns, err := c.nsLister.Get(key)
	if apierrs.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	switch _, err := c.cmLister.ConfigMaps(ns.Name).Get(c.configMap.Name); {
	case err == nil:
		return nil
	case apierrs.IsNotFound(err):
	case err != nil:
		return err
	}

	cm := c.configMap.DeepCopy()
	if _, err := c.client.CoreV1().ConfigMaps(ns.Name).Create(cm); err != nil && !apierrs.IsAlreadyExists(err) {
		return err
	}
	return nil
}

func convertToCM(obj interface{}) (*v1.ConfigMap, error) {
	cm, ok := obj.(*v1.ConfigMap)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			return nil, fmt.Errorf("Couldn't get object from tombstone %#v", obj)
		}
		cm, ok = tombstone.Obj.(*v1.ConfigMap)
		if !ok {
			return nil, fmt.Errorf("Tombstone contained object that is not a ConfigMap %#v", obj)
		}
	}
	return cm, nil
}
