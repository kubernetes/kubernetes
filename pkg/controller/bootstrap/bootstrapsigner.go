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

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api"
	bootstrapapi "k8s.io/kubernetes/pkg/bootstrap/api"
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
	client          clientset.Interface
	configMapKey    string
	secretNamespace string

	configMaps cache.Store
	secrets    cache.Store

	// syncQueue handles synchronizing updates to the ConfigMap.  We'll only ever
	// have one item (Named <ConfigMapName>) in this queue. We are using it
	// serializes and collapses updates as they can come from both the ConfigMap
	// and Secrets controllers.
	syncQueue workqueue.Interface

	// Since we join two objects, we'll watch both of them with controllers.
	configMapsController cache.Controller
	secretsController    cache.Controller
}

// NewBootstrapSigner returns a new *BootstrapSigner.
//
// TODO: Switch to shared informers
func NewBootstrapSigner(cl clientset.Interface, options BootstrapSignerOptions) *BootstrapSigner {
	e := &BootstrapSigner{
		client:          cl,
		configMapKey:    options.ConfigMapNamespace + "/" + options.ConfigMapName,
		secretNamespace: options.TokenSecretNamespace,
		syncQueue:       workqueue.NewNamed("bootstrap_signer_queue"),
	}
	if cl.CoreV1().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("bootstrap_signer", cl.CoreV1().RESTClient().GetRateLimiter())
	}
	configMapSelector := fields.SelectorFromSet(map[string]string{api.ObjectNameField: options.ConfigMapName})
	e.configMaps, e.configMapsController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(lo metav1.ListOptions) (runtime.Object, error) {
				lo.FieldSelector = configMapSelector.String()
				return e.client.CoreV1().ConfigMaps(options.ConfigMapNamespace).List(lo)
			},
			WatchFunc: func(lo metav1.ListOptions) (watch.Interface, error) {
				lo.FieldSelector = configMapSelector.String()
				return e.client.CoreV1().ConfigMaps(options.ConfigMapNamespace).Watch(lo)
			},
		},
		&v1.ConfigMap{},
		options.ConfigMapResync,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    func(_ interface{}) { e.pokeConfigMapSync() },
			UpdateFunc: func(_, _ interface{}) { e.pokeConfigMapSync() },
		},
	)

	secretSelector := fields.SelectorFromSet(map[string]string{api.SecretTypeField: string(bootstrapapi.SecretTypeBootstrapToken)})
	e.secrets, e.secretsController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(lo metav1.ListOptions) (runtime.Object, error) {
				lo.FieldSelector = secretSelector.String()
				return e.client.CoreV1().Secrets(e.secretNamespace).List(lo)
			},
			WatchFunc: func(lo metav1.ListOptions) (watch.Interface, error) {
				lo.FieldSelector = secretSelector.String()
				return e.client.CoreV1().Secrets(e.secretNamespace).Watch(lo)
			},
		},
		&v1.Secret{},
		options.SecretResync,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    func(_ interface{}) { e.pokeConfigMapSync() },
			UpdateFunc: func(_, _ interface{}) { e.pokeConfigMapSync() },
			DeleteFunc: func(_ interface{}) { e.pokeConfigMapSync() },
		},
	)
	return e
}

// Run runs controller loops and returns when they are done
func (e *BootstrapSigner) Run(stopCh <-chan struct{}) {
	go e.configMapsController.Run(stopCh)
	go e.secretsController.Run(stopCh)
	go wait.Until(e.serviceConfigMapQueue, 0, stopCh)
	<-stopCh
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
		glog.V(3).Infof("No %s key in %s/%s ConfigMap", bootstrapapi.KubeConfigKey, origCM.Namespace, origCM.Name)
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
		glog.V(3).Infof("Error updating ConfigMap: %v", err)
	}
}

// getConfigMap gets the ConfigMap we are interested in
func (e *BootstrapSigner) getConfigMap() *v1.ConfigMap {
	configMap, exists, err := e.configMaps.GetByKey(e.configMapKey)

	// If we can't get the configmap just return nil. The resync will eventually
	// sync things up.
	if err != nil {
		utilruntime.HandleError(err)
		return nil
	}

	if exists {
		return configMap.(*v1.ConfigMap)
	}
	return nil
}

func (e *BootstrapSigner) listSecrets() []*v1.Secret {
	secrets := e.secrets.List()

	items := []*v1.Secret{}
	for _, obj := range secrets {
		items = append(items, obj.(*v1.Secret))
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
			glog.V(1).Infof("Duplicate bootstrap tokens found for id %s, ignoring on in %s/%s", tokenID, secret.Namespace, secret.Name)
			continue
		}

		// This secret looks good, add it to the list.
		ret[tokenID] = tokenSecret
	}

	return ret
}
