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
	"context"
	"strings"
	"time"

	"k8s.io/klog/v2"

	"fmt"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/util/workqueue"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	jws "k8s.io/cluster-bootstrap/token/jws"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// SignerOptions contains options for the Signer
type SignerOptions struct {
	// ConfigMapNamespace is the namespace of the ConfigMap
	ConfigMapNamespace string

	// ConfigMapName is the name for the ConfigMap
	ConfigMapName string

	// TokenSecretNamespace string is the namespace for token Secrets.
	TokenSecretNamespace string

	// ConfigMapResync is the time.Duration at which to fully re-list configmaps.
	// If zero, re-list will be delayed as long as possible
	ConfigMapResync time.Duration

	// SecretResync is the time.Duration at which to fully re-list secrets.
	// If zero, re-list will be delayed as long as possible
	SecretResync time.Duration
}

// DefaultSignerOptions returns a set of default options for creating a Signer.
func DefaultSignerOptions() SignerOptions {
	return SignerOptions{
		ConfigMapNamespace:   api.NamespacePublic,
		ConfigMapName:        bootstrapapi.ConfigMapClusterInfo,
		TokenSecretNamespace: api.NamespaceSystem,
	}
}

// Signer is a controller that signs a ConfigMap with a set of tokens.
type Signer struct {
	client             clientset.Interface
	configMapKey       string
	configMapName      string
	configMapNamespace string
	secretNamespace    string

	// syncQueue handles synchronizing updates to the ConfigMap.  We'll only ever
	// have one item (Named <ConfigMapName>) in this queue. We are using it
	// serializes and collapses updates as they can come from both the ConfigMap
	// and Secrets controllers.
	syncQueue workqueue.TypedRateLimitingInterface[string]

	secretLister corelisters.SecretLister
	secretSynced cache.InformerSynced

	configMapLister corelisters.ConfigMapLister
	configMapSynced cache.InformerSynced
}

// NewSigner returns a new *Signer.
func NewSigner(ctx context.Context, cl clientset.Interface, secrets informers.SecretInformer, configMaps informers.ConfigMapInformer, options SignerOptions) (*Signer, error) {
	e := &Signer{
		client:             cl,
		configMapKey:       options.ConfigMapNamespace + "/" + options.ConfigMapName,
		configMapName:      options.ConfigMapName,
		configMapNamespace: options.ConfigMapNamespace,
		secretNamespace:    options.TokenSecretNamespace,
		secretLister:       secrets.Lister(),
		secretSynced:       secrets.Informer().HasSynced,
		configMapLister:    configMaps.Lister(),
		configMapSynced:    configMaps.Informer().HasSynced,
		syncQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[string](),
			workqueue.TypedRateLimitingQueueConfig[string]{
				Name: "bootstrap_signer_queue",
			},
		),
	}

	configMaps.Informer().AddEventHandlerWithConfig(
		ctx,
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.ConfigMap:
					return t.Name == options.ConfigMapName && t.Namespace == options.ConfigMapNamespace
				default:
					utilruntime.HandleErrorWithContext(ctx, nil, "Object passed to Signer that is not expected", "objectType", fmt.Sprintf("%T", obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    func(_ interface{}) { e.pokeConfigMapSync() },
				UpdateFunc: func(_, _ interface{}) { e.pokeConfigMapSync() },
			},
		},
		cache.HandlerConfig{
			ResyncPeriod: options.ConfigMapResync,
		},
	)

	secrets.Informer().AddEventHandlerWithConfig(
		ctx,
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Secret:
					return t.Type == bootstrapapi.SecretTypeBootstrapToken && t.Namespace == e.secretNamespace
				default:
					utilruntime.HandleErrorWithContext(ctx, nil, "Object passed to Signer that is not expected", "objectType", fmt.Sprintf("%T", obj))
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    func(_ interface{}) { e.pokeConfigMapSync() },
				UpdateFunc: func(_, _ interface{}) { e.pokeConfigMapSync() },
				DeleteFunc: func(_ interface{}) { e.pokeConfigMapSync() },
			},
		},
		cache.HandlerConfig{
			ResyncPeriod: options.SecretResync,
		},
	)

	return e, nil
}

// Run runs controller loops and returns when they are done
func (e *Signer) Run(ctx context.Context) {
	// Shut down queues
	defer utilruntime.HandleCrashWithContext(ctx)
	defer e.syncQueue.ShutDown()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, e.configMapSynced, e.secretSynced) {
		return
	}

	logger := klog.FromContext(ctx)
	logger.V(5).Info("Starting workers")
	go wait.UntilWithContext(ctx, e.serviceConfigMapQueue, 0)
	<-ctx.Done()
	logger.V(1).Info("Shutting down")
}

func (e *Signer) pokeConfigMapSync() {
	e.syncQueue.Add(e.configMapKey)
}

func (e *Signer) serviceConfigMapQueue(ctx context.Context) {
	key, quit := e.syncQueue.Get()
	if quit {
		return
	}
	defer e.syncQueue.Done(key)

	e.signConfigMap(ctx)
}

// signConfigMap computes the signatures on our latest cached objects and writes
// back if necessary.
func (e *Signer) signConfigMap(ctx context.Context) {
	origCM := e.getConfigMap()

	if origCM == nil {
		return
	}

	var needUpdate = false

	newCM := origCM.DeepCopy()

	logger := klog.FromContext(ctx)

	// First capture the config we are signing
	content, ok := newCM.Data[bootstrapapi.KubeConfigKey]
	if !ok {
		logger.V(3).Info("No key in ConfigMap", "key", bootstrapapi.KubeConfigKey, "configMap", klog.KObj(origCM))
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
	tokens := e.getTokens(ctx)
	for tokenID, tokenValue := range tokens {
		sig, err := jws.ComputeDetachedSignature(content, tokenID, tokenValue)
		if err != nil {
			utilruntime.HandleErrorWithContext(ctx, err, "Computing detached signature failed")
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
		e.updateConfigMap(ctx, newCM)
	}
}

func (e *Signer) updateConfigMap(ctx context.Context, cm *v1.ConfigMap) {
	_, err := e.client.CoreV1().ConfigMaps(cm.Namespace).Update(ctx, cm, metav1.UpdateOptions{})
	if err != nil && !apierrors.IsConflict(err) && !apierrors.IsNotFound(err) {
		klog.FromContext(ctx).V(3).Info("Error updating ConfigMap", "err", err)
	}
}

// getConfigMap gets the ConfigMap we are interested in
func (e *Signer) getConfigMap() *v1.ConfigMap {
	configMap, err := e.configMapLister.ConfigMaps(e.configMapNamespace).Get(e.configMapName)

	// If we can't get the configmap just return nil. The resync will eventually
	// sync things up.
	if err != nil {
		if !apierrors.IsNotFound(err) {
			utilruntime.HandleError(err) //nolint:logcheck // Local cache should not fail.
		}
		return nil
	}

	return configMap
}

func (e *Signer) listSecrets() []*v1.Secret {
	secrets, err := e.secretLister.Secrets(e.secretNamespace).List(labels.Everything())
	if err != nil {
		utilruntime.HandleError(err) //nolint:logcheck // Local cache should not fail.
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
func (e *Signer) getTokens(ctx context.Context) map[string]string {
	ret := map[string]string{}
	secretObjs := e.listSecrets()
	for _, secret := range secretObjs {
		tokenID, tokenSecret, ok := validateSecretForSigning(ctx, secret)
		if !ok {
			continue
		}

		// Check and warn for duplicate secrets. Behavior here will be undefined.
		if _, ok := ret[tokenID]; ok {
			// This should never happen as we ensure a consistent secret name.
			// But leave this in here just in case.
			klog.FromContext(ctx).V(1).Info("Duplicate bootstrap tokens found for id, ignoring on the duplicate secret", "tokenID", tokenID, "ignoredSecret", klog.KObj(secret))
			continue
		}

		// This secret looks good, add it to the list.
		ret[tokenID] = tokenSecret
	}

	return ret
}
