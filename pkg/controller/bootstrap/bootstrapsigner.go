/*
Copyright 2016 The Kubernetes Authors.

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

package bootstrap

import (
	"strings"
	"time"

	"k8s.io/klog"

	"fmt"
	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/util/metrics"
)

// BootstrapSignerOptions contains options for the BootstrapSigner
type BootstrapSignerOptions struct {
	// ConfigMapNamespace is the namespace of the ConfigMap
	ConfigMapNamespace string

	// ConfigMapName is the name for the ConfigMap
	ConfigMapName string

	// TokenSecretNamespace string is the namespace for token Secrets.
	TokenSecretNamespace string

	// ConfigMapResynce is the time.Duration at which to fully re-list configmaps.
	// If zero, re-list will be delayed as long as possible
	ConfigMapResync time.Duration

	// SecretResync is the time.Duration at which to fully re-list secrets.
	// If zero, re-list will be delayed as long as possible
	SecretResync time.Duration
}

// DefaultBootstrapSignerOptions returns a set of default options for creating a
// BootstrapSigner
func DefaultBootstrapSignerOptions() BootstrapSignerOptions {
	return BootstrapSignerOptions{
		ConfigMapNamespace:   api.NamespacePublic,
		ConfigMapName:        bootstrapapi.ConfigMapClusterInfo,
		TokenSecretNamespace: api.NamespaceSystem,
	}
}

// BootstrapSigner is a controller that signs a ConfigMap with a set of tokens.
type BootstrapSigner struct {
	client             clientset.Interface
	configMapKey       string
	configMapName      string
	configMapNamespace string
	secretNamespace    string

	// syncQueue handles synchronizing updates to the ConfigMap.  We'll only ever
	// have one item (Named <ConfigMapName>) in this queue. We are using it
	// serializes and collapses updates as they can come from both the ConfigMap
	// and Secrets controllers.
	syncQueue workqueue.RateLimitingInterface

	secretLister corelisters.SecretLister
	secretSynced cache.InformerSynced

	configMapLister corelisters.ConfigMapLister
	configMapSynced cache.InformerSynced
}

// NewBootstrapSigner returns a new *BootstrapSigner.
func NewBootstrapSigner(cl clientset.Interface, secrets informers.SecretInformer, configMaps informers.ConfigMapInformer, options BootstrapSignerOptions) (*BootstrapSigner, error) {
	e := &BootstrapSigner{
		client:             cl,
		configMapKey:       options.ConfigMapNamespace + "/" + options.ConfigMapName,
		configMapName:      options.ConfigMapName,
		configMapNamespace: options.ConfigMapNamespace,
		secretNamespace:    options.TokenSecretNamespace,
		secretLister:       secrets.Lister(),
		secretSynced:       secrets.Informer().HasSynced,
		configMapLister:    configMaps.Lister(),
		configMapSynced:    configMaps.Informer().HasSynced,
		syncQueue:          workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "bootstrap_signer_queue"),
	}
	if cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		if err := metrics.RegisterMetricAndTrackRateLimiterUsage("bootstrap_signer", cl.CoreV1().RESTClient().GetRateLimiter()); err != nil {
			return nil, err
		}
	}

	configMaps.Informer().AddEventHandlerWithResyncPeriod(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.ConfigMap:
					return t.Name == options.ConfigMapName && t.Namespace == options.ConfigMapNamespace
				default:
					utilruntime.HandleError(fmt.Errorf("object passed to %T that is not expected: %T", e, obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    func(_ interface{}) { e.pokeConfigMapSync() },
				UpdateFunc: func(_, _ interface{}) { e.pokeConfigMapSync() },
			},
		},
		options.ConfigMapResync,
	)

	secrets.Informer().AddEventHandlerWithResyncPeriod(
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Secret:
					return t.Type == bootstrapapi.SecretTypeBootstrapToken && t.Namespace == e.secretNamespace
				default:
					utilruntime.HandleError(fmt.Errorf("object passed to %T that is not expected: %T", e, obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    func(_ interface{}) { e.pokeConfigMapSync() },
				UpdateFunc: func(_, _ interface{}) { e.pokeConfigMapSync() },
				DeleteFunc: func(_ interface{}) { e.pokeConfigMapSync() },
			},
		},
		options.SecretResync,
	)

	return e, nil
}

// Run runs controller loops and returns when they are done
func (e *BootstrapSigner) Run(stopCh <-chan struct{}) {
	// Shut down queues
	defer utilruntime.HandleCrash()
	defer e.syncQueue.ShutDown()

	if !controller.WaitForCacheSync("bootstrap_signer", stopCh, e.configMapSynced, e.secretSynced) {
		return
	}

	klog.V(5).Infof("Starting workers")
	go wait.Until(e.serviceConfigMapQueue, 0, stopCh)
	<-stopCh
	klog.V(1).Infof("Shutting down")
}

func (e *BootstrapSigner) pokeConfigMapSync() {
	e.syncQueue.Add(e.configMapKey)
}

func (e *BootstrapSigner) serviceConfigMapQueue() {
	key, quit := e.syncQueue.Get()
	if quit {
		return
	}
	defer e.syncQueue.Done(key)

	e.signConfigMap()
}

// signConfigMap computes the signatures on our latest cached objects and writes
// back if necessary.
func (e *BootstrapSigner) signConfigMap() {
	origCM := e.getConfigMap()

	if origCM == nil {
		return
	}

	var needUpdate = false

	newCM := origCM.DeepCopy()

	// First capture the config we are signing
	content, ok := newCM.Data[bootstrapapi.KubeConfigKey]
	if !ok {
		klog.V(3).Infof("No %s key in %s/%s ConfigMap", bootstrapapi.KubeConfigKey, origCM.Namespace, origCM.Name)
		return
	}

	// Next remove and save all existing signatures
	sigs := map[string]string{}
	for key, value := range newCM.Data {
		if strings.HasPrefix(key, bootstrapapi.JWSSignatureKeyPrefix) {
			tokenID := strings.TrimPrefix(key, bootstrapapi.JWSSignatureKeyPrefix)
			sigs[tokenID] = value
			delete(newCM.Data, key)
		}
	}

	// Now recompute signatures and store them on the new map
	tokens := e.getTokens()
	for tokenID, tokenValue := range tokens {
		sig, err := computeDetachedSig(content, tokenID, tokenValue)
		if err != nil {
			utilruntime.HandleError(err)
		}

		// Check to see if this signature is changed or new.
		oldSig, _ := sigs[tokenID]
		if sig != oldSig {
			needUpdate = true
		}
		delete(sigs, tokenID)

		newCM.Data[bootstrapapi.JWSSignatureKeyPrefix+tokenID] = sig
	}

	// If we have signatures left over we know that some signatures were
	// removed.  We now need to update the ConfigMap
	if len(sigs) != 0 {
		needUpdate = true
	}

	if needUpdate {
		e.updateConfigMap(newCM)
	}
}

func (e *BootstrapSigner) updateConfigMap(cm *v1.ConfigMap) {
	_, err := e.client.CoreV1().ConfigMaps(cm.Namespace).Update(cm)
	if err != nil && !apierrors.IsConflict(err) && !apierrors.IsNotFound(err) {
		klog.V(3).Infof("Error updating ConfigMap: %v", err)
	}
}

// getConfigMap gets the ConfigMap we are interested in
func (e *BootstrapSigner) getConfigMap() *v1.ConfigMap {
	configMap, err := e.configMapLister.ConfigMaps(e.configMapNamespace).Get(e.configMapName)

	// If we can't get the configmap just return nil. The resync will eventually
	// sync things up.
	if err != nil {
		if !apierrors.IsNotFound(err) {
			utilruntime.HandleError(err)
		}
		return nil
	}

	return configMap
}

func (e *BootstrapSigner) listSecrets() []*v1.Secret {
	secrets, err := e.secretLister.Secrets(e.secretNamespace).List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(err)
		return nil
	}

	items := []*v1.Secret{}
	for _, secret := range secrets {
		if secret.Type == bootstrapapi.SecretTypeBootstrapToken {
			items = append(items, secret)
		}
	}
	return items
}

// getTokens returns a map of tokenID->tokenSecret. It ensures the token is
// valid for signing.
func (e *BootstrapSigner) getTokens() map[string]string {
	ret := map[string]string{}
	secretObjs := e.listSecrets()
	for _, secret := range secretObjs {
		tokenID, tokenSecret, ok := validateSecretForSigning(secret)
		if !ok {
			continue
		}

		// Check and warn for duplicate secrets. Behavior here will be undefined.
		if _, ok := ret[tokenID]; ok {
			// This should never happen as we ensure a consistent secret name.
			// But leave this in here just in case.
			klog.V(1).Infof("Duplicate bootstrap tokens found for id %s, ignoring on in %s/%s", tokenID, secret.Namespace, secret.Name)
			continue
		}

		// This secret looks good, add it to the list.
		ret[tokenID] = tokenSecret
	}

	return ret
}
