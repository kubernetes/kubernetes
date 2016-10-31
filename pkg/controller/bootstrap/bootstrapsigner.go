/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/watch"
)

const (
	configMapClusterInfo = "cluster-info"
	kubeConfigKey        = "kubeconfig"
	signaturePrefix      = "jws-kubeconfig-"
)

// BootstrapSignerOptions contains options for the BootstrapSigner
type BootstrapSignerOptions struct {

	// ConfigMapNamespace is the namespace of the ConfigMap
	ConfigMapNamespace string

	// ConfigMapName is the name fo the ConfigMap
	ConfigMapName string

	// ConfigMapResynce is the time.Duration at which to fully re-list configmaps.
	// If zero, re-list will be delayed as long as possible
	ConfigMapResync time.Duration

	// SecretResync is the time.Duration at which to fully re-list secrets.
	// If zero, re-list will be delayed as long as possible
	SecretResync time.Duration

	// MaxRetries controls the maximum number of times a particular key is retried before giving up
	// If zero, a default max is used
	MaxRetries int
}

// DefaultBootstrapSignerOptions returns a set of default options for creating a
// BootstrapSigner
func DefaultBootstrapSignerOptions() BootstrapSignerOptions {
	return BootstrapSignerOptions{
		ConfigMapNamespace: api.NamespacePublic,
		ConfigMapName:      configMapClusterInfo,
	}
}

// BootstrapSigner is a controller that signs a ConfigMap with a set of tokens.
type BootstrapSigner struct {
	client       clientset.Interface
	configMapKey string

	configMaps cache.Store
	secrets    cache.Store

	// Since we join two objects, we'll watch both of them with controllers.
	configMapsController *cache.Controller
	secretsController    *cache.Controller

	maxRetries int
}

// NewBootstrapSigner returns a new *ServiceAccountsController.
func NewBootstrapSigner(cl clientset.Interface, options BootstrapSignerOptions) *BootstrapSigner {
	maxRetries := options.MaxRetries
	if maxRetries == 0 {
		maxRetries = 10
	}

	e := &BootstrapSigner{
		client:       cl,
		configMapKey: options.ConfigMapNamespace + "/" + options.ConfigMapName,
		maxRetries:   maxRetries,
	}
	if cl != nil && cl.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("bootstrap_signer", cl.Core().RESTClient().GetRateLimiter())
	}
	configMapSelector := fields.SelectorFromSet(map[string]string{api.ObjectNameField: options.ConfigMapName})
	e.configMaps, e.configMapsController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(lo v1.ListOptions) (runtime.Object, error) {
				lo.FieldSelector = configMapSelector.String()
				return e.client.Core().ConfigMaps(options.ConfigMapNamespace).List(lo)
			},
			WatchFunc: func(lo v1.ListOptions) (watch.Interface, error) {
				lo.FieldSelector = configMapSelector.String()
				return e.client.Core().ConfigMaps(options.ConfigMapNamespace).Watch(lo)
			},
		},
		&v1.ConfigMap{},
		options.ConfigMapResync,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    func(_ interface{}) { e.signConfigMap() },
			UpdateFunc: func(_, _ interface{}) { e.signConfigMap() },
		},
	)

	secretSelector := fields.SelectorFromSet(map[string]string{api.SecretTypeField: string(api.SecretTypeBootstrapToken)})
	e.secrets, e.secretsController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(lo v1.ListOptions) (runtime.Object, error) {
				lo.FieldSelector = secretSelector.String()
				return e.client.Core().Secrets(api.NamespaceSystem).List(lo)
			},
			WatchFunc: func(lo v1.ListOptions) (watch.Interface, error) {
				lo.FieldSelector = secretSelector.String()
				return e.client.Core().Secrets(api.NamespaceSystem).Watch(lo)
			},
		},
		&v1.Secret{},
		options.SecretResync,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    func(_ interface{}) { e.signConfigMap() },
			UpdateFunc: func(_, _ interface{}) { e.signConfigMap() },
			DeleteFunc: func(_ interface{}) { e.signConfigMap() },
		},
	)
	return e
}

// Run runs controller loops and returns immediately
func (e *BootstrapSigner) Run(stopCh <-chan struct{}) {
	go e.configMapsController.Run(stopCh)
	go e.secretsController.Run(stopCh)
	<-stopCh
}

// signConfigMap computes the signatures on our latest cached objects and writes
// back if necessary.
func (e *BootstrapSigner) signConfigMap() {
	origCM := e.getConfigMap()

	if origCM != nil {
		var needUpdate = false

		newCM, err := copyConfigMap(origCM)
		if err != nil {
			utilruntime.HandleError(err)
			return
		}

		// First capture the config we are signing
		content, ok := newCM.Data[kubeConfigKey]
		if !ok {
			glog.V(3).Infof("No %s key in %s/%s ConfigMap", kubeConfigKey, origCM.Namespace, origCM.Name)
			return
		}

		// Next remove and save all existing signatures
		sigs := map[string]string{}
		for key, value := range newCM.Data {
			if strings.HasPrefix(key, signaturePrefix) {
				tokenID := strings.TrimPrefix(key, signaturePrefix)
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

			newCM.Data[signaturePrefix+tokenID] = sig
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
}

func (e *BootstrapSigner) updateConfigMap(cm *v1.ConfigMap) {
	for i := 0; i < e.maxRetries; i++ {
		_, err := e.client.Core().ConfigMaps(cm.Namespace).Update(cm)
		if apierrors.IsConflict(err) || apierrors.IsNotFound(err) {
			// if we got a Conflict error, the ConfigMap was updated by someone else,
			// and we'll get an update notification later. if we got a NotFound error,
			// the ConfigMap no longer exists, and we don't need to update it.
			return
		}
	}
}

// getConfigMap gets the ConfigMap we are interested in
func (e *BootstrapSigner) getConfigMap() *v1.ConfigMap {
	configMap, exists, _ := e.configMaps.GetByKey(e.configMapKey)

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
			glog.V(3).Infof("Duplicate bootstrap tokens found for id %s, ignoring on in %s/%s", tokenID, secret.Namespace, secret.Name)
			continue
		}

		// This secret looks good, add it to the list.
		ret[tokenID] = tokenSecret
	}

	return ret
}

func copyConfigMap(orig *v1.ConfigMap) (*v1.ConfigMap, error) {
	newCMObj, err := api.Scheme.DeepCopy(orig)
	if err != nil {
		return nil, err
	}
	return newCMObj.(*v1.ConfigMap), nil
}
