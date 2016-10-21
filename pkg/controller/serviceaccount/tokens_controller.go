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

package serviceaccount

import (
	"bytes"
	"fmt"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	apierrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	clientretry "k8s.io/kubernetes/pkg/client/retry"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/registry/core/secret"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/serviceaccount"
	"k8s.io/kubernetes/pkg/types"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/metrics"
	utilruntime "k8s.io/kubernetes/pkg/util/runtime"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/util/workqueue"
	"k8s.io/kubernetes/pkg/watch"
)

// RemoveTokenBackoff is the recommended (empirical) retry interval for removing
// a secret reference from a service account when the secret is deleted. It is
// exported for use by custom secret controllers.
var RemoveTokenBackoff = wait.Backoff{
	Steps:    10,
	Duration: 100 * time.Millisecond,
	Jitter:   1.0,
}

// TokensControllerOptions contains options for the TokensController
type TokensControllerOptions struct {
	// TokenGenerator is the generator to use to create new tokens
	TokenGenerator serviceaccount.TokenGenerator
	// ServiceAccountResync is the time.Duration at which to fully re-list service accounts.
	// If zero, re-list will be delayed as long as possible
	ServiceAccountResync time.Duration
	// SecretResync is the time.Duration at which to fully re-list secrets.
	// If zero, re-list will be delayed as long as possible
	SecretResync time.Duration
	// This CA will be added in the secrets of service accounts
	RootCA []byte

	// MaxRetries controls the maximum number of times a particular key is retried before giving up
	// If zero, a default max is used
	MaxRetries int
}

// NewTokensController returns a new *TokensController.
func NewTokensController(cl clientset.Interface, options TokensControllerOptions) *TokensController {
	maxRetries := options.MaxRetries
	if maxRetries == 0 {
		maxRetries = 10
	}

	e := &TokensController{
		client: cl,
		token:  options.TokenGenerator,
		rootCA: options.RootCA,

		syncServiceAccountQueue: workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "serviceaccount_tokens_service"),
		syncSecretQueue:         workqueue.NewNamedRateLimitingQueue(workqueue.DefaultControllerRateLimiter(), "serviceaccount_tokens_secret"),

		maxRetries: maxRetries,
	}
	if cl != nil && cl.Core().RESTClient().GetRateLimiter() != nil {
		metrics.RegisterMetricAndTrackRateLimiterUsage("serviceaccount_controller", cl.Core().RESTClient().GetRateLimiter())
	}

	e.serviceAccounts, e.serviceAccountController = cache.NewInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				return e.client.Core().ServiceAccounts(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				return e.client.Core().ServiceAccounts(api.NamespaceAll).Watch(options)
			},
		},
		&api.ServiceAccount{},
		options.ServiceAccountResync,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    e.queueServiceAccountSync,
			UpdateFunc: e.queueServiceAccountUpdateSync,
			DeleteFunc: e.queueServiceAccountSync,
		},
	)

	tokenSelector := fields.SelectorFromSet(map[string]string{api.SecretTypeField: string(api.SecretTypeServiceAccountToken)})
	e.secrets, e.secretController = cache.NewIndexerInformer(
		&cache.ListWatch{
			ListFunc: func(options api.ListOptions) (runtime.Object, error) {
				options.FieldSelector = tokenSelector
				return e.client.Core().Secrets(api.NamespaceAll).List(options)
			},
			WatchFunc: func(options api.ListOptions) (watch.Interface, error) {
				options.FieldSelector = tokenSelector
				return e.client.Core().Secrets(api.NamespaceAll).Watch(options)
			},
		},
		&api.Secret{},
		options.SecretResync,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    e.queueSecretSync,
			UpdateFunc: e.queueSecretUpdateSync,
			DeleteFunc: e.queueSecretSync,
		},
		cache.Indexers{"namespace": cache.MetaNamespaceIndexFunc},
	)

	return e
}

// TokensController manages ServiceAccountToken secrets for ServiceAccount objects
type TokensController struct {
	client clientset.Interface
	token  serviceaccount.TokenGenerator

	rootCA []byte

	serviceAccounts cache.Store
	secrets         cache.Indexer

	// Since we join two objects, we'll watch both of them with controllers.
	serviceAccountController *cache.Controller
	secretController         *cache.Controller

	// syncServiceAccountQueue handles service account events:
	//   * ensures a referenced token exists for service accounts which still exist
	//   * ensures tokens are removed for service accounts which no longer exist
	// key is "<namespace>/<name>/<uid>"
	syncServiceAccountQueue workqueue.RateLimitingInterface

	// syncSecretQueue handles secret events:
	//   * deletes tokens whose service account no longer exists
	//   * updates tokens with missing token or namespace data, or mismatched ca data
	//   * ensures service account secret references are removed for tokens which are deleted
	// key is a secretQueueKey{}
	syncSecretQueue workqueue.RateLimitingInterface

	maxRetries int
}

// Runs controller blocks until stopCh is closed
func (e *TokensController) Run(workers int, stopCh <-chan struct{}) {
	defer utilruntime.HandleCrash()

	// Start controllers (to fill stores, call informers, fill work queues)
	go e.serviceAccountController.Run(stopCh)
	go e.secretController.Run(stopCh)

	// Wait for stores to fill
	for !e.serviceAccountController.HasSynced() || !e.secretController.HasSynced() {
		time.Sleep(100 * time.Millisecond)
	}

	// Spawn workers to process work queues
	for i := 0; i < workers; i++ {
		go wait.Until(e.syncServiceAccount, 0, stopCh)
		go wait.Until(e.syncSecret, 0, stopCh)
	}

	// Block until stop channel is closed
	<-stopCh

	// Shut down queues
	e.syncServiceAccountQueue.ShutDown()
	e.syncSecretQueue.ShutDown()
}

func (e *TokensController) queueServiceAccountSync(obj interface{}) {
	if serviceAccount, ok := obj.(*api.ServiceAccount); ok {
		e.syncServiceAccountQueue.Add(makeServiceAccountKey(serviceAccount))
	}
}

func (e *TokensController) queueServiceAccountUpdateSync(oldObj interface{}, newObj interface{}) {
	if serviceAccount, ok := newObj.(*api.ServiceAccount); ok {
		e.syncServiceAccountQueue.Add(makeServiceAccountKey(serviceAccount))
	}
}

// complete optionally requeues key, then calls queue.Done(key)
func (e *TokensController) retryOrForget(queue workqueue.RateLimitingInterface, key interface{}, requeue bool) {
	if !requeue {
		queue.Forget(key)
		return
	}

	requeueCount := queue.NumRequeues(key)
	if requeueCount < e.maxRetries {
		queue.AddRateLimited(key)
		return
	}

	glog.V(4).Infof("retried %d times: %#v", requeueCount, key)
	queue.Forget(key)
}

func (e *TokensController) queueSecretSync(obj interface{}) {
	if secret, ok := obj.(*api.Secret); ok {
		e.syncSecretQueue.Add(makeSecretQueueKey(secret))
	}
}

func (e *TokensController) queueSecretUpdateSync(oldObj interface{}, newObj interface{}) {
	if secret, ok := newObj.(*api.Secret); ok {
		e.syncSecretQueue.Add(makeSecretQueueKey(secret))
	}
}

func (e *TokensController) syncServiceAccount() {
	key, quit := e.syncServiceAccountQueue.Get()
	if quit {
		return
	}
	defer e.syncServiceAccountQueue.Done(key)

	retry := false
	defer func() {
		e.retryOrForget(e.syncServiceAccountQueue, key, retry)
	}()

	saInfo, err := parseServiceAccountKey(key)
	if err != nil {
		glog.Error(err)
		return
	}

	sa, err := e.getServiceAccount(saInfo.namespace, saInfo.name, saInfo.uid, false)
	switch {
	case err != nil:
		glog.Error(err)
		retry = true
	case sa == nil:
		// service account no longer exists, so delete related tokens
		glog.V(4).Infof("syncServiceAccount(%s/%s), service account deleted, removing tokens", saInfo.namespace, saInfo.name)
		sa = &api.ServiceAccount{ObjectMeta: api.ObjectMeta{Namespace: saInfo.namespace, Name: saInfo.name, UID: saInfo.uid}}
		if retriable, err := e.deleteTokens(sa); err != nil {
			glog.Errorf("error deleting serviceaccount tokens for %s/%s: %v", saInfo.namespace, saInfo.name, err)
			retry = retriable
		}
	default:
		// ensure a token exists and is referenced by this service account
		if retriable, err := e.ensureReferencedToken(sa); err != nil {
			glog.Errorf("error synchronizing serviceaccount %s/%s: %v", saInfo.namespace, saInfo.name, err)
			retry = retriable
		}
	}
}

func (e *TokensController) syncSecret() {
	key, quit := e.syncSecretQueue.Get()
	if quit {
		return
	}
	defer e.syncSecretQueue.Done(key)

	// Track whether or not we should retry this sync
	retry := false
	defer func() {
		e.retryOrForget(e.syncSecretQueue, key, retry)
	}()

	secretInfo, err := parseSecretQueueKey(key)
	if err != nil {
		glog.Error(err)
		return
	}

	secret, err := e.getSecret(secretInfo.namespace, secretInfo.name, secretInfo.uid, false)
	switch {
	case err != nil:
		glog.Error(err)
		retry = true
	case secret == nil:
		// If the service account exists
		if sa, saErr := e.getServiceAccount(secretInfo.namespace, secretInfo.saName, secretInfo.saUID, false); saErr == nil && sa != nil {
			// secret no longer exists, so delete references to this secret from the service account
			if err := clientretry.RetryOnConflict(RemoveTokenBackoff, func() error {
				return e.removeSecretReference(secretInfo.namespace, secretInfo.saName, secretInfo.saUID, secretInfo.name)
			}); err != nil {
				glog.Error(err)
			}
		}
	default:
		// Ensure service account exists
		sa, saErr := e.getServiceAccount(secretInfo.namespace, secretInfo.saName, secretInfo.saUID, true)
		switch {
		case saErr != nil:
			glog.Error(saErr)
			retry = true
		case sa == nil:
			// Delete token
			glog.V(4).Infof("syncSecret(%s/%s), service account does not exist, deleting token", secretInfo.namespace, secretInfo.name)
			if retriable, err := e.deleteToken(secretInfo.namespace, secretInfo.name, secretInfo.uid); err != nil {
				glog.Errorf("error deleting serviceaccount token %s/%s for service account %s: %v", secretInfo.namespace, secretInfo.name, secretInfo.saName, err)
				retry = retriable
			}
		default:
			// Update token if needed
			if retriable, err := e.generateTokenIfNeeded(sa, secret); err != nil {
				glog.Errorf("error populating serviceaccount token %s/%s for service account %s: %v", secretInfo.namespace, secretInfo.name, secretInfo.saName, err)
				retry = retriable
			}
		}
	}
}

func (e *TokensController) deleteTokens(serviceAccount *api.ServiceAccount) ( /*retry*/ bool, error) {
	tokens, err := e.listTokenSecrets(serviceAccount)
	if err != nil {
		// don't retry on cache lookup errors
		return false, err
	}
	retry := false
	errs := []error{}
	for _, token := range tokens {
		r, err := e.deleteToken(token.Namespace, token.Name, token.UID)
		if err != nil {
			errs = append(errs, err)
		}
		if r {
			retry = true
		}
	}
	return retry, utilerrors.NewAggregate(errs)
}

func (e *TokensController) deleteToken(ns, name string, uid types.UID) ( /*retry*/ bool, error) {
	var opts *api.DeleteOptions
	if len(uid) > 0 {
		opts = &api.DeleteOptions{Preconditions: &api.Preconditions{UID: &uid}}
	}
	err := e.client.Core().Secrets(ns).Delete(name, opts)
	// NotFound doesn't need a retry (it's already been deleted)
	// Conflict doesn't need a retry (the UID precondition failed)
	if err == nil || apierrors.IsNotFound(err) || apierrors.IsConflict(err) {
		return false, nil
	}
	// Retry for any other error
	return true, err
}

// ensureReferencedToken makes sure at least one ServiceAccountToken secret exists, and is included in the serviceAccount's Secrets list
func (e *TokensController) ensureReferencedToken(serviceAccount *api.ServiceAccount) ( /* retry */ bool, error) {
	if len(serviceAccount.Secrets) > 0 {
		allSecrets, err := e.listTokenSecrets(serviceAccount)
		if err != nil {
			// Don't retry cache lookup errors
			return false, err
		}
		referencedSecrets := getSecretReferences(serviceAccount)
		for _, secret := range allSecrets {
			if referencedSecrets.Has(secret.Name) {
				// A service account token already exists, and is referenced, short-circuit
				return false, nil
			}
		}
	}

	// We don't want to update the cache's copy of the service account
	// so add the secret to a freshly retrieved copy of the service account
	serviceAccounts := e.client.Core().ServiceAccounts(serviceAccount.Namespace)
	liveServiceAccount, err := serviceAccounts.Get(serviceAccount.Name)
	if err != nil {
		// Retry for any error other than a NotFound
		return !apierrors.IsNotFound(err), err
	}
	if liveServiceAccount.ResourceVersion != serviceAccount.ResourceVersion {
		// our view of the service account is not up to date
		// we'll get notified of an update event later and get to try again
		glog.V(2).Infof("serviceaccount %s/%s is not up to date, skipping token creation", serviceAccount.Namespace, serviceAccount.Name)
		return false, nil
	}

	// Build the secret
	secret := &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name:      secret.Strategy.GenerateName(fmt.Sprintf("%s-token-", serviceAccount.Name)),
			Namespace: serviceAccount.Namespace,
			Annotations: map[string]string{
				api.ServiceAccountNameKey: serviceAccount.Name,
				api.ServiceAccountUIDKey:  string(serviceAccount.UID),
			},
		},
		Type: api.SecretTypeServiceAccountToken,
		Data: map[string][]byte{},
	}

	// Generate the token
	token, err := e.token.GenerateToken(*serviceAccount, *secret)
	if err != nil {
		// retriable error
		return true, err
	}
	secret.Data[api.ServiceAccountTokenKey] = []byte(token)
	secret.Data[api.ServiceAccountNamespaceKey] = []byte(serviceAccount.Namespace)
	if e.rootCA != nil && len(e.rootCA) > 0 {
		secret.Data[api.ServiceAccountRootCAKey] = e.rootCA
	}

	// Save the secret
	createdToken, err := e.client.Core().Secrets(serviceAccount.Namespace).Create(secret)
	if err != nil {
		// retriable error
		return true, err
	}
	// Manually add the new token to the cache store.
	// This prevents the service account update (below) triggering another token creation, if the referenced token couldn't be found in the store
	e.secrets.Add(createdToken)

	liveServiceAccount.Secrets = append(liveServiceAccount.Secrets, api.ObjectReference{Name: secret.Name})

	if _, err = serviceAccounts.Update(liveServiceAccount); err != nil {
		// we weren't able to use the token, try to clean it up.
		glog.V(2).Infof("deleting secret %s/%s because reference couldn't be added (%v)", secret.Namespace, secret.Name, err)
		deleteOpts := &api.DeleteOptions{Preconditions: &api.Preconditions{UID: &createdToken.UID}}
		if deleteErr := e.client.Core().Secrets(createdToken.Namespace).Delete(createdToken.Name, deleteOpts); deleteErr != nil {
			glog.Error(deleteErr) // if we fail, just log it
		}

		if apierrors.IsConflict(err) || apierrors.IsNotFound(err) {
			// if we got a Conflict error, the service account was updated by someone else, and we'll get an update notification later
			// if we got a NotFound error, the service account no longer exists, and we don't need to create a token for it
			return false, nil
		}
		// retry in all other cases
		return true, err
	}

	// success!
	return false, nil
}

func (e *TokensController) secretUpdateNeeded(secret *api.Secret) (bool, bool, bool) {
	caData := secret.Data[api.ServiceAccountRootCAKey]
	needsCA := len(e.rootCA) > 0 && bytes.Compare(caData, e.rootCA) != 0

	needsNamespace := len(secret.Data[api.ServiceAccountNamespaceKey]) == 0

	tokenData := secret.Data[api.ServiceAccountTokenKey]
	needsToken := len(tokenData) == 0

	return needsCA, needsNamespace, needsToken
}

// generateTokenIfNeeded populates the token data for the given Secret if not already set
func (e *TokensController) generateTokenIfNeeded(serviceAccount *api.ServiceAccount, cachedSecret *api.Secret) ( /* retry */ bool, error) {
	// Check the cached secret to see if changes are needed
	if needsCA, needsNamespace, needsToken := e.secretUpdateNeeded(cachedSecret); !needsCA && !needsToken && !needsNamespace {
		return false, nil
	}

	// We don't want to update the cache's copy of the secret
	// so add the token to a freshly retrieved copy of the secret
	secrets := e.client.Core().Secrets(cachedSecret.Namespace)
	liveSecret, err := secrets.Get(cachedSecret.Name)
	if err != nil {
		// Retry for any error other than a NotFound
		return !apierrors.IsNotFound(err), err
	}
	if liveSecret.ResourceVersion != cachedSecret.ResourceVersion {
		// our view of the secret is not up to date
		// we'll get notified of an update event later and get to try again
		glog.V(2).Infof("secret %s/%s is not up to date, skipping token population", liveSecret.Namespace, liveSecret.Name)
		return false, nil
	}

	needsCA, needsNamespace, needsToken := e.secretUpdateNeeded(liveSecret)
	if !needsCA && !needsToken && !needsNamespace {
		return false, nil
	}

	if liveSecret.Annotations == nil {
		liveSecret.Annotations = map[string]string{}
	}
	if liveSecret.Data == nil {
		liveSecret.Data = map[string][]byte{}
	}

	// Set the CA
	if needsCA {
		liveSecret.Data[api.ServiceAccountRootCAKey] = e.rootCA
	}
	// Set the namespace
	if needsNamespace {
		liveSecret.Data[api.ServiceAccountNamespaceKey] = []byte(liveSecret.Namespace)
	}

	// Generate the token
	if needsToken {
		token, err := e.token.GenerateToken(*serviceAccount, *liveSecret)
		if err != nil {
			return false, err
		}
		liveSecret.Data[api.ServiceAccountTokenKey] = []byte(token)
	}

	// Set annotations
	liveSecret.Annotations[api.ServiceAccountNameKey] = serviceAccount.Name
	liveSecret.Annotations[api.ServiceAccountUIDKey] = string(serviceAccount.UID)

	// Save the secret
	_, err = secrets.Update(liveSecret)
	if apierrors.IsConflict(err) || apierrors.IsNotFound(err) {
		// if we got a Conflict error, the secret was updated by someone else, and we'll get an update notification later
		// if we got a NotFound error, the secret no longer exists, and we don't need to populate a token
		return false, nil
	}
	if err != nil {
		return true, err
	}
	return false, nil
}

// removeSecretReference updates the given ServiceAccount to remove a reference to the given secretName if needed.
func (e *TokensController) removeSecretReference(saNamespace string, saName string, saUID types.UID, secretName string) error {
	// We don't want to update the cache's copy of the service account
	// so remove the secret from a freshly retrieved copy of the service account
	serviceAccounts := e.client.Core().ServiceAccounts(saNamespace)
	serviceAccount, err := serviceAccounts.Get(saName)
	// Ignore NotFound errors when attempting to remove a reference
	if apierrors.IsNotFound(err) {
		return nil
	}
	if err != nil {
		return err
	}

	// Short-circuit if the UID doesn't match
	if len(saUID) > 0 && saUID != serviceAccount.UID {
		return nil
	}

	// Short-circuit if the secret is no longer referenced
	if !getSecretReferences(serviceAccount).Has(secretName) {
		return nil
	}

	// Remove the secret
	secrets := []api.ObjectReference{}
	for _, s := range serviceAccount.Secrets {
		if s.Name != secretName {
			secrets = append(secrets, s)
		}
	}
	serviceAccount.Secrets = secrets
	_, err = serviceAccounts.Update(serviceAccount)
	// Ignore NotFound errors when attempting to remove a reference
	if apierrors.IsNotFound(err) {
		return nil
	}
	return err
}

func (e *TokensController) getServiceAccount(ns string, name string, uid types.UID, fetchOnCacheMiss bool) (*api.ServiceAccount, error) {
	// Look up in cache
	obj, exists, err := e.serviceAccounts.GetByKey(makeCacheKey(ns, name))
	if err != nil {
		return nil, err
	}
	if exists {
		sa, ok := obj.(*api.ServiceAccount)
		if !ok {
			return nil, fmt.Errorf("expected *api.ServiceAccount, got %#v", sa)
		}
		// Ensure UID matches if given
		if len(uid) == 0 || uid == sa.UID {
			return sa, nil
		}
	}

	if !fetchOnCacheMiss {
		return nil, nil
	}

	// Live lookup
	sa, err := e.client.Core().ServiceAccounts(ns).Get(name)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	// Ensure UID matches if given
	if len(uid) == 0 || uid == sa.UID {
		return sa, nil
	}
	return nil, nil
}

func (e *TokensController) getSecret(ns string, name string, uid types.UID, fetchOnCacheMiss bool) (*api.Secret, error) {
	// Look up in cache
	obj, exists, err := e.secrets.GetByKey(makeCacheKey(ns, name))
	if err != nil {
		return nil, err
	}
	if exists {
		secret, ok := obj.(*api.Secret)
		if !ok {
			return nil, fmt.Errorf("expected *api.Secret, got %#v", secret)
		}
		// Ensure UID matches if given
		if len(uid) == 0 || uid == secret.UID {
			return secret, nil
		}
	}

	if !fetchOnCacheMiss {
		return nil, nil
	}

	// Live lookup
	secret, err := e.client.Core().Secrets(ns).Get(name)
	if apierrors.IsNotFound(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	// Ensure UID matches if given
	if len(uid) == 0 || uid == secret.UID {
		return secret, nil
	}
	return nil, nil
}

// listTokenSecrets returns a list of all of the ServiceAccountToken secrets that
// reference the given service account's name and uid
func (e *TokensController) listTokenSecrets(serviceAccount *api.ServiceAccount) ([]*api.Secret, error) {
	namespaceSecrets, err := e.secrets.ByIndex("namespace", serviceAccount.Namespace)
	if err != nil {
		return nil, err
	}

	items := []*api.Secret{}
	for _, obj := range namespaceSecrets {
		secret := obj.(*api.Secret)

		if serviceaccount.IsServiceAccountToken(secret, serviceAccount) {
			items = append(items, secret)
		}
	}
	return items, nil
}

// serviceAccountNameAndUID is a helper method to get the ServiceAccount Name and UID from the given secret
// Returns "","" if the secret is not a ServiceAccountToken secret
// If the name or uid annotation is missing, "" is returned instead
func serviceAccountNameAndUID(secret *api.Secret) (string, string) {
	if secret.Type != api.SecretTypeServiceAccountToken {
		return "", ""
	}
	return secret.Annotations[api.ServiceAccountNameKey], secret.Annotations[api.ServiceAccountUIDKey]
}

func getSecretReferences(serviceAccount *api.ServiceAccount) sets.String {
	references := sets.NewString()
	for _, secret := range serviceAccount.Secrets {
		references.Insert(secret.Name)
	}
	return references
}

// serviceAccountQueueKey holds information we need to sync a service account.
// It contains enough information to look up the cached service account,
// or delete owned tokens if the service account no longer exists.
type serviceAccountQueueKey struct {
	namespace string
	name      string
	uid       types.UID
}

func makeServiceAccountKey(sa *api.ServiceAccount) interface{} {
	return serviceAccountQueueKey{
		namespace: sa.Namespace,
		name:      sa.Name,
		uid:       sa.UID,
	}
}

func parseServiceAccountKey(key interface{}) (serviceAccountQueueKey, error) {
	queueKey, ok := key.(serviceAccountQueueKey)
	if !ok || len(queueKey.namespace) == 0 || len(queueKey.name) == 0 || len(queueKey.uid) == 0 {
		return serviceAccountQueueKey{}, fmt.Errorf("invalid serviceaccount key: %#v", key)
	}
	return queueKey, nil
}

// secretQueueKey holds information we need to sync a service account token secret.
// It contains enough information to look up the cached service account,
// or delete the secret reference if the secret no longer exists.
type secretQueueKey struct {
	namespace string
	name      string
	uid       types.UID
	saName    string
	// optional, will be blank when syncing tokens missing the service account uid annotation
	saUID types.UID
}

func makeSecretQueueKey(secret *api.Secret) interface{} {
	return secretQueueKey{
		namespace: secret.Namespace,
		name:      secret.Name,
		uid:       secret.UID,
		saName:    secret.Annotations[api.ServiceAccountNameKey],
		saUID:     types.UID(secret.Annotations[api.ServiceAccountUIDKey]),
	}
}

func parseSecretQueueKey(key interface{}) (secretQueueKey, error) {
	queueKey, ok := key.(secretQueueKey)
	if !ok || len(queueKey.namespace) == 0 || len(queueKey.name) == 0 || len(queueKey.uid) == 0 || len(queueKey.saName) == 0 {
		return secretQueueKey{}, fmt.Errorf("invalid secret key: %#v", key)
	}
	return queueKey, nil
}

// produce the same key format as cache.MetaNamespaceKeyFunc
func makeCacheKey(namespace, name string) string {
	return namespace + "/" + name
}
