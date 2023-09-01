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

	kcpcache "github.com/kcp-dev/apimachinery/v2/pkg/cache"
	kcpcorev1informers "github.com/kcp-dev/client-go/informers/core/v1"
	kcpkubernetesclientset "github.com/kcp-dev/client-go/kubernetes"
	kcpcorev1listers "github.com/kcp-dev/client-go/listers/core/v1"
	"github.com/kcp-dev/logicalcluster/v3"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
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
func NewPublisher(cmInformer kcpcorev1informers.ConfigMapClusterInformer, nsInformer kcpcorev1informers.NamespaceClusterInformer, cl kcpkubernetesclientset.ClusterInterface, rootCA []byte) (*Publisher, error) {
	e := &Publisher{
		client: cl,
		rootCA: rootCA,
		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: "root_ca_cert_publisher",
			},
		),
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
	client kcpkubernetesclientset.ClusterInterface
	rootCA []byte

	// To allow injection for testing.
	syncHandler func(ctx context.Context, key string) error

	cmLister       kcpcorev1listers.ConfigMapClusterLister
	cmListerSynced cache.InformerSynced

	nsListerSynced cache.InformerSynced

	queue workqueue.TypedRateLimitingInterface[string]
}

// Run starts process
func (c *Publisher) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting root CA cert publisher controller")
	defer logger.Info("Shutting down root CA cert publisher controller")

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

	key, err := kcpcache.MetaClusterNamespaceKeyFunc(namespace)
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

	key, err := kcpcache.MetaClusterNamespaceKeyFunc(newNamespace)
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

	if err := c.syncHandler(ctx, key); err != nil {
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
		klog.FromContext(ctx).V(4).Info("Finished syncing namespace %q (%v)", key, time.Since(startTime))
	}()

	// Get the clusterName and name from the key.
	clusterName, _, name, err := kcpcache.SplitMetaClusterNamespaceKey(key)
	if err != nil {
		utilruntime.HandleError(err)
		return err
	}
	cm, err := c.cmLister.Cluster(clusterName).ConfigMaps(name).Get(RootCACertConfigMapName)

	switch {
	case apierrors.IsNotFound(err):
		_, err = c.client.Cluster(clusterName.Path()).CoreV1().ConfigMaps(name).Create(ctx, &v1.ConfigMap{
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

	_, err = c.client.Cluster(clusterName.Path()).CoreV1().ConfigMaps(name).Update(ctx, cm, metav1.UpdateOptions{})
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
	return kcpcache.ToClusterAwareKey(logicalcluster.From(configmap).String(), "", configmap.GetNamespace())
}
