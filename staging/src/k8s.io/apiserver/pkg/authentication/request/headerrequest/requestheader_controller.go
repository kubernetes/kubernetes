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
	"sync/atomic"
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
)

const (
	authenticationRoleName = "extension-apiserver-authentication-reader"
)

// RequestHeaderConfig is an immutable point-in-time snapshot of the front-proxy request
// header authentication configuration read from the extension-apiserver-authentication
// configmap. All fields of a snapshot returned by a RequestHeaderConfigProvider are
// mutually consistent, i.e. they originate from a single observation of the configmap.
type RequestHeaderConfig struct {
	// UsernameHeaders are the headers to check (in order, case-insensitively) for an identity. The first header with a value wins.
	UsernameHeaders []string
	// UIDHeaders are the headers to check (in order, case-insensitively) for an identity UID. The first header with a value wins.
	UIDHeaders []string
	// GroupHeaders are the headers to check (case-insensitively) for group membership. All values of all headers will be added.
	GroupHeaders []string
	// ExtraHeaderPrefixes are the header prefixes to check (case-insensitively) for filling in the user.Info.Extra. All values of all matching headers will be added.
	ExtraHeaderPrefixes []string
	// AllowedClientNames are the common names the front proxy client certificate may have. Empty means: accept any verified client certificate.
	AllowedClientNames []string
}

// Snapshot returns the receiver, so a static RequestHeaderConfig doubles as a
// RequestHeaderConfigProvider serving an immutable snapshot.
func (c *RequestHeaderConfig) Snapshot() *RequestHeaderConfig {
	return c
}

var _ RequestHeaderConfigProvider = &RequestHeaderConfig{}

// RequestHeaderConfigProvider provides atomic point-in-time snapshots of the request header
// authentication configuration.
type RequestHeaderConfigProvider interface {
	// Snapshot returns the current request header configuration, or nil if no configuration
	// is available. Nil is returned when the source configmap has never been read or has been
	// deleted, so those two states are indistinguishable. Callers must fail closed, i.e.
	// reject every request, when Snapshot returns nil.
	Snapshot() *RequestHeaderConfig
}

// RequestHeaderConfigProviderFunc is a function that matches the RequestHeaderConfigProvider interface
type RequestHeaderConfigProviderFunc func() *RequestHeaderConfig

// Snapshot returns the current request header configuration.
func (f RequestHeaderConfigProviderFunc) Snapshot() *RequestHeaderConfig {
	return f()
}

// NewStaticRequestHeaderConfig returns a RequestHeaderConfigProvider serving an immutable
// snapshot of the given configuration.
func NewStaticRequestHeaderConfig(usernameHeaders, uidHeaders, groupHeaders, extraHeaderPrefixes, allowedClientNames []string) *RequestHeaderConfig {
	return &RequestHeaderConfig{
		UsernameHeaders:     usernameHeaders,
		UIDHeaders:          uidHeaders,
		GroupHeaders:        groupHeaders,
		ExtraHeaderPrefixes: extraHeaderPrefixes,
		AllowedClientNames:  allowedClientNames,
	}
}

// RequestHeaderAuthRequestController a controller that exposes a snapshot of the request header
// configuration sourced from the config map which is being monitored by this controller.
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

	queue workqueue.TypedRateLimitingInterface[string]

	// exportedRequestHeaderConfig contains the last read content of the configmap.
	// It is nil when the configmap has never been read and nil again once the configmap
	// is deleted, see clearRequestHeaderConfig, so those two states are indistinguishable.
	exportedRequestHeaderConfig atomic.Pointer[RequestHeaderConfig]

	usernameHeadersKey     string
	uidHeadersKey          string
	groupHeadersKey        string
	extraHeaderPrefixesKey string
	allowedClientNamesKey  string
}

// NewRequestHeaderAuthRequestController creates a new controller that implements RequestHeaderAuthRequestController
func NewRequestHeaderAuthRequestController(
	cmName string,
	cmNamespace string,
	client kubernetes.Interface,
	usernameHeadersKey, uidHeadersKey, groupHeadersKey, extraHeaderPrefixesKey, allowedClientNamesKey string) *RequestHeaderAuthRequestController {
	c := &RequestHeaderAuthRequestController{
		name: "RequestHeaderAuthRequestController",

		client: client,

		configmapName:      cmName,
		configmapNamespace: cmNamespace,

		usernameHeadersKey:     usernameHeadersKey,
		uidHeadersKey:          uidHeadersKey,
		groupHeadersKey:        groupHeadersKey,
		extraHeaderPrefixesKey: extraHeaderPrefixesKey,
		allowedClientNamesKey:  allowedClientNamesKey,

		queue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{Name: "RequestHeaderAuthRequestController"},
		),
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

// Snapshot returns a point-in-time snapshot of the request header configuration, or nil
// when the configmap has never been read or has been deleted. All fields of the returned
// snapshot are mutually consistent.
func (c *RequestHeaderAuthRequestController) Snapshot() *RequestHeaderConfig {
	return c.exportedRequestHeaderConfig.Load()
}

// Run starts RequestHeaderAuthRequestController controller and blocks until stopCh is closed.
func (c *RequestHeaderAuthRequestController) Run(ctx context.Context, workers int) {
	defer utilruntime.HandleCrashWithContext(ctx)
	defer c.queue.ShutDown()

	klog.Infof("Starting %s", c.name)
	defer klog.Infof("Shutting down %s", c.name)

	go c.configmapInformer.Run(ctx.Done())

	// wait for caches to fill before starting your work
	if !cache.WaitForNamedCacheSyncWithContext(ctx, c.configmapInformerSynced) {
		return
	}

	// doesn't matter what workers say, only start one.
	go wait.Until(c.runWorker, time.Second, ctx.Done())

	<-ctx.Done()
}

// RunOnce runs a single sync loop
func (c *RequestHeaderAuthRequestController) RunOnce(ctx context.Context) error {
	configMap, err := c.client.CoreV1().ConfigMaps(c.configmapNamespace).Get(ctx, c.configmapName, metav1.GetOptions{})
	switch {
	case errors.IsNotFound(err):
		c.clearRequestHeaderConfig()
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

// sync reads the config and propagates the changes to exportedRequestHeaderConfig
// which is exposed via Snapshot
func (c *RequestHeaderAuthRequestController) sync() error {
	configMap, err := c.configmapLister.Get(c.configmapName)
	if err != nil {
		if errors.IsNotFound(err) {
			// returning nil keeps the workqueue from retrying a deletion forever
			c.clearRequestHeaderConfig()
			return nil
		}
		return err
	}
	return c.syncConfigMap(configMap)
}

// clearRequestHeaderConfig resets the configuration to the same unavailable state as when the
// controller has never read the configmap, so a deleted configmap is indistinguishable from one
// that never existed.
func (c *RequestHeaderAuthRequestController) clearRequestHeaderConfig() {
	c.exportedRequestHeaderConfig.Store(nil)
	klog.InfoS("Cleared request header values, configmap was deleted", "name", c.name)
}

func (c *RequestHeaderAuthRequestController) syncConfigMap(configMap *corev1.ConfigMap) error {
	hasChanged, newRequestHeaderConfig, err := c.hasRequestHeaderBundleChanged(configMap)
	if err != nil {
		return err
	}
	if hasChanged {
		c.exportedRequestHeaderConfig.Store(newRequestHeaderConfig)
		klog.V(2).Infof("Loaded a new request header values for %v", c.name)
	}
	return nil
}

func (c *RequestHeaderAuthRequestController) hasRequestHeaderBundleChanged(cm *corev1.ConfigMap) (bool, *RequestHeaderConfig, error) {
	currentHeadersBundle, err := c.getRequestHeaderBundleFromConfigMap(cm)
	if err != nil {
		return false, nil, err
	}

	rawHeaderBundle := c.exportedRequestHeaderConfig.Load()
	if rawHeaderBundle == nil {
		return true, currentHeadersBundle, nil
	}

	// check to see if we have a change. If the values are the same, do nothing.
	if !equality.Semantic.DeepEqual(*rawHeaderBundle, *currentHeadersBundle) {
		return true, currentHeadersBundle, nil
	}
	return false, nil, nil
}

func (c *RequestHeaderAuthRequestController) getRequestHeaderBundleFromConfigMap(cm *corev1.ConfigMap) (*RequestHeaderConfig, error) {
	usernameHeaderCurrentValue, err := deserializeStrings(cm.Data[c.usernameHeadersKey])
	if err != nil {
		return nil, err
	}

	uidHeaderCurrentValue, err := deserializeStrings(cm.Data[c.uidHeadersKey])
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

	return &RequestHeaderConfig{
		UsernameHeaders:     usernameHeaderCurrentValue,
		UIDHeaders:          uidHeaderCurrentValue,
		GroupHeaders:        groupHeadersCurrentValue,
		ExtraHeaderPrefixes: extraHeaderPrefixesCurrentValue,
		AllowedClientNames:  allowedClientNamesCurrentValue,
	}, nil
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
