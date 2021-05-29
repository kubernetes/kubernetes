/*
Copyright 2020 The Kubernetes Authors.

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

package headerrequest

import (
	"context"
	"encoding/json"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	coreinformers "k8s.io/client-go/informers/core/v1"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"sync/atomic"
)

const (
	authenticationRoleName = "extension-apiserver-authentication-reader"
)

// RequestHeaderAuthRequestProvider a provider that knows how to dynamically fill parts of RequestHeaderConfig struct
type RequestHeaderAuthRequestProvider interface {
	UsernameHeaders() []string
	GroupHeaders() []string
	ExtraHeaderPrefixes() []string
	AllowedClientNames() []string
}

var _ RequestHeaderAuthRequestProvider = &RequestHeaderAuthRequestController{}

type requestHeaderBundle struct {
	UsernameHeaders     []string
	GroupHeaders        []string
	ExtraHeaderPrefixes []string
	AllowedClientNames  []string
}

// RequestHeaderAuthRequestController a controller that exposes a set of methods for dynamically filling parts of RequestHeaderConfig struct.
// The methods are sourced from the config map which is being monitored by this controller.
// The controller is primed from the server at the construction time for components that don't want to dynamically react to changes
// in the config map.
type RequestHeaderAuthRequestController struct {
	name string

	configmapName      string
	configmapNamespace string

	client                  kubernetes.Interface
	configmapLister         corev1listers.ConfigMapNamespaceLister
	configmapInformer       cache.SharedIndexInformer
	configmapInformerSynced cache.InformerSynced

	queue workqueue.RateLimitingInterface

	// exportedRequestHeaderBundle is a requestHeaderBundle that contains the last read, non-zero length content of the configmap
	exportedRequestHeaderBundle atomic.Value

	usernameHeadersKey     string
	groupHeadersKey        string
	extraHeaderPrefixesKey string
	allowedClientNamesKey  string
}

// NewRequestHeaderAuthRequestController creates a new controller that implements RequestHeaderAuthRequestController
func NewRequestHeaderAuthRequestController(
	cmName string,
	cmNamespace string,
	client kubernetes.Interface,
	usernameHeadersKey, groupHeadersKey, extraHeaderPrefixesKey, allowedClientNamesKey string) *RequestHeaderAuthRequestController {
	c := &RequestHeaderAuthRequestController{
		name: "RequestHeaderAuthRequestController",

		client: client,

		configmapName:      cmName,
		configmapNamespace: cmNamespace,

		usernameHeadersKey:     usernameHeadersKey,
		groupHeadersKey:        groupHeadersKey,
		extraHeaderPrefixesKey: extraHeaderPrefixesKey,
		allowedClientNamesKey:  allowedClientNamesKey,

		queue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "RequestHeaderAuthRequestController"),
	}

	// we construct our own informer because we need such a small subset of the information available.  Just one namespace.
	c.configmapInformer = coreinformers.NewFilteredConfigMapInformer(client, c.configmapNamespace, 12*time.Hour, cache.Indexers{cache.NamespaceIndex: cache.MetaNamespaceIndexFunc}, func(listOptions *metav1.ListOptions) {
		listOptions.FieldSelector = fields.OneTermEqualSelector("metadata.name", c.configmapName).String()
	})

	c.configmapInformer.AddEventHandler(cache.FilteringResourceEventHandler{
		FilterFunc: func(obj interface{}) bool {
			if cast, ok := obj.(*corev1.ConfigMap); ok {
				return cast.Name == c.configmapName && cast.Namespace == c.configmapNamespace
			}
			if tombstone, ok := obj.(cache.DeletedFinalStateUnknown); ok {
				if cast, ok := tombstone.Obj.(*corev1.ConfigMap); ok {
					return cast.Name == c.configmapName && cast.Namespace == c.configmapNamespace
				}
			}
			return true // always return true just in case.  The checks are fairly cheap
		},
		Handler: cache.ResourceEventHandlerFuncs{
			// we have a filter, so any time we're called, we may as well queue. We only ever check one configmap
			// so we don't have to be choosy about our key.
			AddFunc: func(obj interface{}) {
				c.queue.Add(c.keyFn())
			},
			UpdateFunc: func(oldObj, newObj interface{}) {
				c.queue.Add(c.keyFn())
			},
			DeleteFunc: func(obj interface{}) {
				c.queue.Add(c.keyFn())
			},
		},
	})

	c.configmapLister = corev1listers.NewConfigMapLister(c.configmapInformer.GetIndexer()).ConfigMaps(c.configmapNamespace)
	c.configmapInformerSynced = c.configmapInformer.HasSynced

	return c
}

func (c *RequestHeaderAuthRequestController) UsernameHeaders() []string {
	return c.loadRequestHeaderFor(c.usernameHeadersKey)
}

func (c *RequestHeaderAuthRequestController) GroupHeaders() []string {
	return c.loadRequestHeaderFor(c.groupHeadersKey)
}

func (c *RequestHeaderAuthRequestController) ExtraHeaderPrefixes() []string {
	return c.loadRequestHeaderFor(c.extraHeaderPrefixesKey)
}

func (c *RequestHeaderAuthRequestController) AllowedClientNames() []string {
	return c.loadRequestHeaderFor(c.allowedClientNamesKey)
}

// Run starts RequestHeaderAuthRequestController controller and blocks until stopCh is closed.
func (c *RequestHeaderAuthRequestController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()
	defer c.queue.ShutDown()

	klog.Infof("Starting %s", c.name)
	defer klog.Infof("Shutting down %s", c.name)

	go c.configmapInformer.Run(stopCh)

	// wait for caches to fill before starting your work
	if !cache.WaitForNamedCacheSync(c.name, stopCh, c.configmapInformerSynced) {
		return
	}

	// doesn't matter what workers say, only start one.
	go wait.Until(c.runWorker, time.Second, stopCh)

	<-stopCh
}

// // RunOnce runs a single sync loop
func (c *RequestHeaderAuthRequestController) RunOnce() error {
	configMap, err := c.client.CoreV1().ConfigMaps(c.configmapNamespace).Get(context.TODO(), c.configmapName, metav1.GetOptions{})
	switch {
	case errors.IsNotFound(err):
		// ignore, authConfigMap is nil now
		return nil
	case errors.IsForbidden(err):
		klog.Warningf("Unable to get configmap/%s in %s.  Usually fixed by "+
			"'kubectl create rolebinding -n %s ROLEBINDING_NAME --role=%s --serviceaccount=YOUR_NS:YOUR_SA'",
			c.configmapName, c.configmapNamespace, c.configmapNamespace, authenticationRoleName)
		return err
	case err != nil:
		return err
	}
	return c.syncConfigMap(configMap)
}

func (c *RequestHeaderAuthRequestController) runWorker() {
	for c.processNextWorkItem() {
	}
}

func (c *RequestHeaderAuthRequestController) processNextWorkItem() bool {
	dsKey, quit := c.queue.Get()
	if quit {
		return false
	}
	defer c.queue.Done(dsKey)

	err := c.sync()
	if err == nil {
		c.queue.Forget(dsKey)
		return true
	}

	utilruntime.HandleError(fmt.Errorf("%v failed with : %v", dsKey, err))
	c.queue.AddRateLimited(dsKey)

	return true
}

// sync reads the config and propagates the changes to exportedRequestHeaderBundle
// which is exposed by the set of methods that are used to fill RequestHeaderConfig struct
func (c *RequestHeaderAuthRequestController) sync() error {
	configMap, err := c.configmapLister.Get(c.configmapName)
	if err != nil {
		return err
	}
	return c.syncConfigMap(configMap)
}

func (c *RequestHeaderAuthRequestController) syncConfigMap(configMap *corev1.ConfigMap) error {
	hasChanged, newRequestHeaderBundle, err := c.hasRequestHeaderBundleChanged(configMap)
	if err != nil {
		return err
	}
	if hasChanged {
		c.exportedRequestHeaderBundle.Store(newRequestHeaderBundle)
		klog.V(2).Infof("Loaded a new request header values for %v", c.name)
	}
	return nil
}

func (c *RequestHeaderAuthRequestController) hasRequestHeaderBundleChanged(cm *corev1.ConfigMap) (bool, *requestHeaderBundle, error) {
	currentHeadersBundle, err := c.getRequestHeaderBundleFromConfigMap(cm)
	if err != nil {
		return false, nil, err
	}

	rawHeaderBundle := c.exportedRequestHeaderBundle.Load()
	if rawHeaderBundle == nil {
		return true, currentHeadersBundle, nil
	}

	// check to see if we have a change. If the values are the same, do nothing.
	loadedHeadersBundle, ok := rawHeaderBundle.(*requestHeaderBundle)
	if !ok {
		return true, currentHeadersBundle, nil
	}

	if !equality.Semantic.DeepEqual(loadedHeadersBundle, currentHeadersBundle) {
		return true, currentHeadersBundle, nil
	}
	return false, nil, nil
}

func (c *RequestHeaderAuthRequestController) getRequestHeaderBundleFromConfigMap(cm *corev1.ConfigMap) (*requestHeaderBundle, error) {
	usernameHeaderCurrentValue, err := deserializeStrings(cm.Data[c.usernameHeadersKey])
	if err != nil {
		return nil, err
	}

	groupHeadersCurrentValue, err := deserializeStrings(cm.Data[c.groupHeadersKey])
	if err != nil {
		return nil, err
	}

	extraHeaderPrefixesCurrentValue, err := deserializeStrings(cm.Data[c.extraHeaderPrefixesKey])
	if err != nil {
		return nil, err

	}

	allowedClientNamesCurrentValue, err := deserializeStrings(cm.Data[c.allowedClientNamesKey])
	if err != nil {
		return nil, err
	}

	return &requestHeaderBundle{
		UsernameHeaders:     usernameHeaderCurrentValue,
		GroupHeaders:        groupHeadersCurrentValue,
		ExtraHeaderPrefixes: extraHeaderPrefixesCurrentValue,
		AllowedClientNames:  allowedClientNamesCurrentValue,
	}, nil
}

func (c *RequestHeaderAuthRequestController) loadRequestHeaderFor(key string) []string {
	rawHeaderBundle := c.exportedRequestHeaderBundle.Load()
	if rawHeaderBundle == nil {
		return nil // this can happen if we've been unable load data from the apiserver for some reason
	}
	headerBundle := rawHeaderBundle.(*requestHeaderBundle)

	switch key {
	case c.usernameHeadersKey:
		return headerBundle.UsernameHeaders
	case c.groupHeadersKey:
		return headerBundle.GroupHeaders
	case c.extraHeaderPrefixesKey:
		return headerBundle.ExtraHeaderPrefixes
	case c.allowedClientNamesKey:
		return headerBundle.AllowedClientNames
	default:
		return nil
	}
}

func (c *RequestHeaderAuthRequestController) keyFn() string {
	// this format matches DeletionHandlingMetaNamespaceKeyFunc for our single key
	return c.configmapNamespace + "/" + c.configmapName
}

func deserializeStrings(in string) ([]string, error) {
	if len(in) == 0 {
		return nil, nil
	}
	var ret []string
	if err := json.Unmarshal([]byte(in), &ret); err != nil {
		return nil, err
	}
	return ret, nil
}
