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

	certificatesv1alpha1 "k8s.io/api/certificates/v1alpha1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	certinformers "k8s.io/client-go/informers/certificates/v1alpha1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	internalinterfaces "k8s.io/client-go/informers/internalinterfaces"
	clientset "k8s.io/client-go/kubernetes"
	certlisters "k8s.io/client-go/listers/certificates/v1alpha1"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
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
func NewPublisher(
	cmInformer coreinformers.ConfigMapInformer,
	nsInformer coreinformers.NamespaceInformer,
	cl clientset.Interface,
	rootCA []byte,
	signerName string,
) (*Publisher, error) {

	e := &Publisher{
		client:     cl,
		rootCA:     rootCA,
		signerName: signerName,
		configMapQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
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

	e.configMapsSyncer = e.syncNamespace

	if utilfeature.DefaultFeatureGate.Enabled(features.ClusterTrustBundle) && len(e.signerName) > 0 {
		e.trustBundleQueue = workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: "root_ca_cert_publisher_cluster_trust_bundles",
			},
		)

		e.ctbInformer = setupSignerNameFilteredCTBInformer(e.client, e.signerName)

		_, err := e.ctbInformer.AddEventHandler(cache.ResourceEventHandlerFuncs{
			AddFunc: func(obj interface{}) {
				e.trustBundleQueue.Add("")
			},
			UpdateFunc: func(_, _ interface{}) {
				e.trustBundleQueue.Add("")
			},
			DeleteFunc: func(_ interface{}) {
				e.trustBundleQueue.Add("")
			},
		})
		if err != nil {
			return nil, fmt.Errorf("failed to register ClusterTrustBundle event handler: %w", err)
		}

		e.ctbLister = certlisters.NewClusterTrustBundleLister(e.ctbInformer.GetIndexer())
		e.ctbListerSynced = e.ctbInformer.HasSynced

		e.clusterTrustBundleSyncer = e.syncClusterTrustBundle
	}

	return e, nil

}

// Publisher manages certificate ConfigMap objects inside Namespaces
type Publisher struct {
	client     clientset.Interface
	rootCA     []byte
	signerName string

	// To allow injection for testing.
	configMapsSyncer         func(ctx context.Context, key string) error
	clusterTrustBundleSyncer func(ctx context.Context, _ string) error

	cmLister       corelisters.ConfigMapLister
	cmListerSynced cache.InformerSynced

	ctbInformer     cache.SharedIndexInformer
	ctbLister       certlisters.ClusterTrustBundleLister
	ctbListerSynced cache.InformerSynced

	nsListerSynced cache.InformerSynced

	configMapQueue   workqueue.TypedRateLimitingInterface[string]
	trustBundleQueue workqueue.TypedRateLimitingInterface[string]
}

// Run starts process
func (c *Publisher) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrash()
	defer c.configMapQueue.ShutDown()

	logger := klog.FromContext(ctx)
	logger.Info("Starting root CA cert publisher controller")
	defer logger.Info("Shutting down root CA cert publisher controller")

	if !cache.WaitForNamedCacheSync("crt configmap", ctx.Done(), c.cmListerSynced) {
		return
	}

	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, c.runWorker(c.configMapQueue, c.configMapsSyncer), time.Second)
	}

	if c.clusterTrustBundleSyncer != nil {
		logger.Info("Starting root CA cert publisher controller - cluster trust bundle sync loop")
		go c.ctbInformer.Run(context.Background().Done())

		if !cache.WaitForNamedCacheSync("cluster trust bundle", ctx.Done(), c.ctbListerSynced) {
			return
		}

		for i := 0; i < workers; i++ {
			go wait.UntilWithContext(ctx, c.runWorker(c.trustBundleQueue, c.clusterTrustBundleSyncer), time.Second)
		}

		go wait.PollUntilContextCancel(ctx, 1*time.Minute, true, func(_ context.Context) (bool, error) {
			c.trustBundleQueue.Add("")
			return false, nil
		})

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
	c.configMapQueue.Add(cm.Namespace)
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
	c.configMapQueue.Add(cm.Namespace)
}

func (c *Publisher) namespaceAdded(obj interface{}) {
	namespace := obj.(*v1.Namespace)
	c.configMapQueue.Add(namespace.Name)
}

func (c *Publisher) namespaceUpdated(oldObj interface{}, newObj interface{}) {
	newNamespace := newObj.(*v1.Namespace)
	if newNamespace.Status.Phase != v1.NamespaceActive {
		return
	}
	c.configMapQueue.Add(newNamespace.Name)
}

func (c *Publisher) runWorker(queue workqueue.TypedRateLimitingInterface[string], syncer func(ctx context.Context, key string) error) func(context.Context) {
	return func(ctx context.Context) {
		for c.processNextWorkItem(ctx, queue, syncer) {
		}
	}
}

// processNextWorkItem deals with one key off the queue. It returns false when
// it's time to quit.
func (c *Publisher) processNextWorkItem(ctx context.Context, queue workqueue.TypedRateLimitingInterface[string], syncer func(ctx context.Context, key string) error) bool {
	key, quit := queue.Get()
	if quit {
		return false
	}
	defer queue.Done(key)

	if err := syncer(ctx, key); err != nil {
		utilruntime.HandleError(fmt.Errorf("syncing %q failed: %v", key, err))
		queue.AddRateLimited(key)
		return true
	}

	queue.Forget(key)
	return true
}

func (c *Publisher) syncNamespace(ctx context.Context, ns string) (err error) {
	startTime := time.Now()
	defer func() {
		recordMetrics(startTime, namespaceResource, err)
		klog.FromContext(ctx).V(4).Info("Finished syncing namespace", "namespace", ns, "elapsedTime", time.Since(startTime))
	}()

	cm, err := c.cmLister.ConfigMaps(ns).Get(RootCACertConfigMapName)
	switch {
	case apierrors.IsNotFound(err):
		_, err = c.client.CoreV1().ConfigMaps(ns).Create(ctx, &v1.ConfigMap{
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

	_, err = c.client.CoreV1().ConfigMaps(ns).Update(ctx, cm, metav1.UpdateOptions{})
	return err
}

func (c *Publisher) syncClusterTrustBundle(ctx context.Context, _ string) (err error) {
	startTime := time.Now()
	defer func() {
		recordMetrics(startTime, clusterTrustBundleResource, err)
		klog.FromContext(ctx).V(4).Info("Finished syncing ClusterTrustBundle", "signerName", c.signerName, "elapsedTime", time.Since(startTime))
	}()

	signerTrustBundles, err := c.ctbLister.List(labels.Everything())
	switch {
	case apierrors.IsNotFound(err) || (err == nil && len(signerTrustBundles) == 0):
		_, err := c.client.CertificatesV1alpha1().ClusterTrustBundles().Create(ctx, &certificatesv1alpha1.ClusterTrustBundle{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "kubernetes.io:kube-apiserver-serving:serving-trust",
			},
			Spec: certificatesv1alpha1.ClusterTrustBundleSpec{
				SignerName:  c.signerName,
				TrustBundle: string(c.rootCA),
			},
		}, metav1.CreateOptions{})

		return err

	case len(signerTrustBundles) > 1:
		err := c.client.CertificatesV1alpha1().ClusterTrustBundles().DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{
			FieldSelector: fields.OneTermEqualSelector("spec.signerName", c.signerName).String(),
		})

		if err != nil {
			return err
		}

		// requeue to recreate the bundle
		c.trustBundleQueue.Add("")
		return err
	case err != nil:
		return err
	}

	bundle := signerTrustBundles[0]
	if bundle.Spec.TrustBundle != string(c.rootCA) {
		bundle = bundle.DeepCopy()
		bundle.Spec.TrustBundle = string(c.rootCA)
		_, err = c.client.CertificatesV1alpha1().ClusterTrustBundles().Update(ctx, bundle, metav1.UpdateOptions{})
		return err
	}

	return nil
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

func setupSignerNameFilteredCTBInformer(client clientset.Interface, signerName string) cache.SharedIndexInformer {
	return certinformers.NewFilteredClusterTrustBundleInformer(client, 10*time.Minute, cache.Indexers{},
		internalinterfaces.TweakListOptionsFunc(func(options *metav1.ListOptions) {
			options.FieldSelector = fields.OneTermEqualSelector("spec.signerName", signerName).String()
		}))
}
