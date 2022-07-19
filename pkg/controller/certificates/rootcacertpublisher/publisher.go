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
	"context"
	"fmt"
	"reflect"
	"time"

	"github.com/kcp-dev/logicalcluster/v2"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/clusters"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/component-base/metrics/prometheus/ratelimiter"
	"k8s.io/klog/v2"
)

// RootCACertConfigMapName is name of the configmap which stores certificates
// to access api-server
const (
	RootCACertConfigMapName = "kube-root-ca.crt"
	DescriptionAnnotation   = "kubernetes.io/description"
	Description             = "Contains a CA bundle that can be used to verify the kube-apiserver when using internal endpoints such as the internal service IP or kubernetes.default.svc. " +
		"No other usage is guaranteed across distributions of Kubernetes clusters."
)

func init() {
	registerMetrics()
}

// NewPublisher construct a new controller which would manage the configmap
// which stores certificates in each namespace. It will make sure certificate
// configmap exists in each namespace.
func NewPublisher(cmInformer coreinformers.ConfigMapInformer, nsInformer coreinformers.NamespaceInformer, cl clientset.Interface, rootCA []byte) (*Publisher, error) {
	e := &Publisher{
		client: cl,
		rootCA: rootCA,
		queue:  workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "root_ca_cert_publisher"),
	}
	if cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		if err := ratelimiter.RegisterMetricAndTrackRateLimiterUsage("root_ca_cert_publisher", cl.CoreV1().RESTClient().GetRateLimiter()); err != nil {
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
	e.nsListerSynced = nsInformer.Informer().HasSynced

	e.syncHandler = e.syncNamespace

	return e, nil

}

// Publisher manages certificate ConfigMap objects inside Namespaces
type Publisher struct {
	client clientset.Interface
	rootCA []byte

	// To allow injection for testing.
	syncHandler func(ctx context.Context, key string) error

	cmLister       corelisters.ConfigMapLister
	cmListerSynced cache.InformerSynced

	nsListerSynced cache.InformerSynced

	queue workqueue.RateLimitingInterface
}

// Run starts process
func (c *Publisher) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting root CA certificate configmap publisher")
	defer klog.Infof("Shutting down root CA certificate configmap publisher")

	if !cache.WaitForNamedCacheSync("crt configmap", ctx.Done(), c.cmListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.runWorker, time.Second)
	}

	<-ctx.Done()
}

func (c *Publisher) configMapDeleted(obj interface{}) {
	cm, err := convertToCM(obj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	if cm.Name != RootCACertConfigMapName {
		return
	}

	key := getNamespaceKey(cm)
	if key == "" {
		return
	}
	c.queue.Add(key)
}

func (c *Publisher) configMapUpdated(_, newObj interface{}) {
	cm, err := convertToCM(newObj)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}
	if cm.Name != RootCACertConfigMapName {
		return
	}

	key := getNamespaceKey(cm)
	if key == "" {
		return
	}
	c.queue.Add(key)
}

func (c *Publisher) namespaceAdded(obj interface{}) {
	namespace := obj.(*v1.Namespace)

	key, err := cache.MetaNamespaceKeyFunc(namespace)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}

	c.queue.Add(key)
}

func (c *Publisher) namespaceUpdated(oldObj interface{}, newObj interface{}) {
	newNamespace := newObj.(*v1.Namespace)
	if newNamespace.Status.Phase != v1.NamespaceActive {
		return
	}

	key, err := cache.MetaNamespaceKeyFunc(newNamespace)
	if err != nil {
		utilruntime.HandleError(err)
		return
	}

	c.queue.Add(key)
}

func (c *Publisher) runWorker(ctx context.Context) {
	for c.processNextWorkItem(ctx) {
	}
}

// processNextWorkItem deals with one key off the queue. It returns false when
// it's time to quit.
func (c *Publisher) processNextWorkItem(ctx context.Context) bool {
	key, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(key)

	if err := c.syncHandler(ctx, key.(string)); err != nil {
		utilruntime.HandleError(fmt.Errorf("syncing %q failed: %v", key, err))
		c.queue.AddRateLimited(key)
		return true
	}

	c.queue.Forget(key)
	return true
}

func (c *Publisher) syncNamespace(ctx context.Context, key string) (err error) {
	startTime := time.Now()
	defer func() {
		recordMetrics(startTime, err)
		klog.V(4).Infof("Finished syncing namespace %q (%v)", key, time.Since(startTime))
	}()

	// Get the clusterName and name from the key.
	_, clusterAwareName, err := cache.SplitMetaNamespaceKey(key)
	clusterName, name := clusters.SplitClusterAwareKey(clusterAwareName)

	clusterAwareConfigMapName := clusters.ToClusterAwareKey(clusterName, RootCACertConfigMapName)
	cm, err := c.cmLister.ConfigMaps(name).Get(clusterAwareConfigMapName)

	switch {
	case apierrors.IsNotFound(err):
		_, err = c.client.CoreV1().ConfigMaps(name).Create(genericapirequest.WithCluster(ctx, genericapirequest.Cluster{Name: clusterName}), &v1.ConfigMap{
			ObjectMeta: metav1.ObjectMeta{
				Name:        RootCACertConfigMapName,
				Annotations: map[string]string{DescriptionAnnotation: Description},
			},
			Data: map[string]string{
				"ca.crt": string(c.rootCA),
			},
		}, metav1.CreateOptions{})
		// don't retry a create if the namespace doesn't exist or is terminating
		if apierrors.IsNotFound(err) || apierrors.HasStatusCause(err, v1.NamespaceTerminatingCause) {
			return nil
		}
		return err
	case err != nil:
		return err
	}

	data := map[string]string{
		"ca.crt": string(c.rootCA),
	}

	// ensure the data and the one annotation describing usage of this configmap match.
	if reflect.DeepEqual(cm.Data, data) && len(cm.Annotations[DescriptionAnnotation]) > 0 {
		return nil
	}

	// copy so we don't modify the cache's instance of the configmap
	cm = cm.DeepCopy()
	cm.Data = data
	if cm.Annotations == nil {
		cm.Annotations = map[string]string{}
	}
	cm.Annotations[DescriptionAnnotation] = Description

	_, err = c.client.CoreV1().ConfigMaps(name).Update(genericapirequest.WithCluster(ctx, genericapirequest.Cluster{Name: clusterName}), cm, metav1.UpdateOptions{})
	return err
}

func convertToCM(obj interface{}) (*v1.ConfigMap, error) {
	cm, ok := obj.(*v1.ConfigMap)
	if !ok {
		tombstone, ok := obj.(cache.DeletedFinalStateUnknown)
		if !ok {
			return nil, fmt.Errorf("couldn't get object from tombstone %#v", obj)
		}
		cm, ok = tombstone.Obj.(*v1.ConfigMap)
		if !ok {
			return nil, fmt.Errorf("tombstone contained object that is not a ConfigMap %#v", obj)
		}
	}
	return cm, nil
}

func getNamespaceKey(configmap *v1.ConfigMap) string {
	clusterName := logicalcluster.From(configmap)
	if clusterName.Empty() {
		return ""
	}

	return clusters.ToClusterAwareKey(clusterName, configmap.GetNamespace())
}
