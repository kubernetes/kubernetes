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
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
	informers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	listersv1 "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	clientretry "k8s.io/client-go/util/retry"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/serviceaccount"
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
func NewTokensController(ctx context.Context, serviceAccounts informers.ServiceAccountInformer, secrets informers.SecretInformer, cl clientset.Interface, options TokensControllerOptions) (*TokensController, error) {
	maxRetries := options.MaxRetries
	if maxRetries == 0 {
		maxRetries = 10
	}

	e := &TokensController{
		client: cl,
		token:  options.TokenGenerator,
		rootCA: options.RootCA,

		syncServiceAccountQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[serviceAccountQueueKey](),
			workqueue.TypedRateLimitingQueueConfig[serviceAccountQueueKey]{Name: "serviceaccount_tokens_service"},
		),
		syncSecretQueue: workqueue.NewTypedRateLimitingQueueWithConfig(
			workqueue.DefaultTypedControllerRateLimiter[secretQueueKey](),
			workqueue.TypedRateLimitingQueueConfig[secretQueueKey]{Name: "serviceaccount_tokens_service"},
		),

		maxRetries: maxRetries,
	}

	e.serviceAccounts = serviceAccounts.Lister()
	e.serviceAccountSynced = serviceAccounts.Informer().HasSynced
	serviceAccounts.Informer().AddEventHandlerWithConfig(
		ctx,
		cache.ResourceEventHandlerFuncs{
			AddFunc:    e.queueServiceAccountSync,
			UpdateFunc: e.queueServiceAccountUpdateSync,
			DeleteFunc: e.queueServiceAccountSync,
		},
		cache.HandlerConfig{
			ResyncPeriod: options.ServiceAccountResync,
		},
	)

	secretCache := secrets.Informer().GetIndexer()
	e.updatedSecrets = cache.NewIntegerResourceVersionMutationCache(ctx, secretCache, secretCache, 60*time.Second, true)
	e.secretSynced = secrets.Informer().HasSynced
	secrets.Informer().AddEventHandlerWithConfig(
		ctx,
		cache.FilteringResourceEventHandler{
			FilterFunc: func(obj interface{}) bool {
				switch t := obj.(type) {
				case *v1.Secret:
					return t.Type == v1.SecretTypeServiceAccountToken
				default:
					utilruntime.HandleError(fmt.Errorf("object passed to %T that is not expected: %T", e, obj)) //nolint:logcheck // Not reached, shouldn't have unknown objects.
					return false
				}
			},
			Handler: cache.ResourceEventHandlerFuncs{
				AddFunc:    e.queueSecretSync,
				UpdateFunc: e.queueSecretUpdateSync,
				DeleteFunc: e.queueSecretSync,
			},
		},
		cache.HandlerConfig{
			ResyncPeriod: options.SecretResync,
		},
	)

	return e, nil
}

// TokensController manages ServiceAccountToken secrets for ServiceAccount objects
type TokensController struct {
	client clientset.Interface
	token  serviceaccount.TokenGenerator

	rootCA []byte

	serviceAccounts listersv1.ServiceAccountLister
	// updatedSecrets is a wrapper around the shared cache which allows us to record
	// and return our local mutations (since we're very likely to act on an updated
	// secret before the watch reports it).
	updatedSecrets cache.MutationCache

	// Since we join two objects, we'll watch both of them with controllers.
	serviceAccountSynced cache.InformerSynced
	secretSynced         cache.InformerSynced

	// syncServiceAccountQueue handles service account events:
	//   * ensures tokens are removed for service accounts which no longer exist
	// key is "<namespace>/<name>/<uid>"
	syncServiceAccountQueue workqueue.TypedRateLimitingInterface[serviceAccountQueueKey]

	// syncSecretQueue handles secret events:
	//   * deletes tokens whose service account no longer exists
	//   * updates tokens with missing token or namespace data, or mismatched ca data
	//   * ensures service account secret references are removed for tokens which are deleted
	// key is a secretQueueKey{}
	syncSecretQueue workqueue.TypedRateLimitingInterface[secretQueueKey]

	maxRetries int
}

// Run runs controller blocks until stopCh is closed
func (e *TokensController) Run(ctx context.Context, workers int) {
	// Shut down queues
	defer utilruntime.HandleCrashWithContext(ctx)
	defer e.syncServiceAccountQueue.ShutDown()
	defer e.syncSecretQueue.ShutDown()

	if !cache.WaitForNamedCacheSyncWithContext(ctx, e.serviceAccountSynced, e.secretSynced) {
		return
	}

	logger := klog.FromContext(ctx)
	logger.V(5).Info("Starting workers")
	for i := 0; i < workers; i++ {
		go wait.UntilWithContext(ctx, e.syncServiceAccount, 0)
		go wait.UntilWithContext(ctx, e.syncSecret, 0)
	}
	<-ctx.Done()
	logger.V(1).Info("Shutting down")
}

func (e *TokensController) queueServiceAccountSync(obj interface{}) {
	if serviceAccount, ok := obj.(*v1.ServiceAccount); ok {
		e.syncServiceAccountQueue.Add(makeServiceAccountKey(serviceAccount))
	}
}

func (e *TokensController) queueServiceAccountUpdateSync(oldObj interface{}, newObj interface{}) {
	if serviceAccount, ok := newObj.(*v1.ServiceAccount); ok {
		e.syncServiceAccountQueue.Add(makeServiceAccountKey(serviceAccount))
	}
}

// complete optionally requeues key, then calls queue.Done(key)
func retryOrForget[T comparable](logger klog.Logger, queue workqueue.TypedRateLimitingInterface[T], key T, requeue bool, maxRetries int) {
	if !requeue {
		queue.Forget(key)
		return
	}

	requeueCount := queue.NumRequeues(key)
	if requeueCount < maxRetries {
		queue.AddRateLimited(key)
		return
	}

	logger.V(4).Info("retried several times", "key", key, "count", requeueCount)
	queue.Forget(key)
}

func (e *TokensController) queueSecretSync(obj interface{}) {
	if secret, ok := obj.(*v1.Secret); ok {
		e.syncSecretQueue.Add(makeSecretQueueKey(secret))
	}
}

func (e *TokensController) queueSecretUpdateSync(oldObj interface{}, newObj interface{}) {
	if secret, ok := newObj.(*v1.Secret); ok {
		e.syncSecretQueue.Add(makeSecretQueueKey(secret))
	}
}

func (e *TokensController) syncServiceAccount(ctx context.Context) {
	logger := klog.FromContext(ctx)
	key, quit := e.syncServiceAccountQueue.Get()
	if quit {
		return
	}
	defer e.syncServiceAccountQueue.Done(key)

	retry := false
	defer func() {
		retryOrForget(logger, e.syncServiceAccountQueue, key, retry, e.maxRetries)
	}()

	saInfo, err := parseServiceAccountKey(key)
	if err != nil {
		logger.Error(err, "Parsing service account key")
		return
	}

	sa, err := e.getServiceAccount(saInfo.namespace, saInfo.name, saInfo.uid, false)
	switch {
	case err != nil:
		logger.Error(err, "Getting service account")
		retry = true
	case sa == nil:
		// service account no longer exists, so delete related tokens
		logger.V(4).Info("Service account deleted, removing tokens", "namespace", saInfo.namespace, "serviceaccount", saInfo.name)
		sa = &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Namespace: saInfo.namespace, Name: saInfo.name, UID: saInfo.uid}}
		retry, err = e.deleteTokens(sa)
		if err != nil {
			logger.Error(err, "Error deleting serviceaccount tokens", "namespace", saInfo.namespace, "serviceaccount", saInfo.name)
		}
	}
}

func (e *TokensController) syncSecret(ctx context.Context) {
	key, quit := e.syncSecretQueue.Get()
	if quit {
		return
	}
	defer e.syncSecretQueue.Done(key)

	logger := klog.FromContext(ctx)
	// Track whether or not we should retry this sync
	retry := false
	defer func() {
		retryOrForget(logger, e.syncSecretQueue, key, retry, e.maxRetries)
	}()

	secretInfo, err := parseSecretQueueKey(key)
	if err != nil {
		logger.Error(err, "Parsing secret queue key")
		return
	}

	secret, err := e.getSecret(secretInfo.namespace, secretInfo.name, secretInfo.uid, false)
	switch {
	case err != nil:
		logger.Error(err, "Getting secret")
		retry = true
	case secret == nil:
		// If the service account exists
		if sa, saErr := e.getServiceAccount(secretInfo.namespace, secretInfo.saName, secretInfo.saUID, false); saErr == nil && sa != nil {
			// secret no longer exists, so delete references to this secret from the service account
			if err := clientretry.RetryOnConflict(RemoveTokenBackoff, func() error {
				return e.removeSecretReference(secretInfo.namespace, secretInfo.saName, secretInfo.saUID, secretInfo.name)
			}); err != nil {
				logger.Error(err, "Removing secret reference")
			}
		}
	default:
		// Ensure service account exists
		sa, saErr := e.getServiceAccount(secretInfo.namespace, secretInfo.saName, secretInfo.saUID, true)
		switch {
		case saErr != nil:
			logger.Error(saErr, "Getting service account")
			retry = true
		case sa == nil:
			// Delete token
			logger.V(4).Info("Service account does not exist, deleting token", "secret", klog.KRef(secretInfo.namespace, secretInfo.name))
			if retriable, err := e.deleteToken(secretInfo.namespace, secretInfo.name, secretInfo.uid); err != nil {
				logger.Error(err, "Deleting serviceaccount token", "secret", klog.KRef(secretInfo.namespace, secretInfo.name), "serviceAccount", klog.KRef(secretInfo.namespace, secretInfo.saName))
				retry = retriable
			}
		default:
			// Update token if needed
			if retriable, err := e.generateTokenIfNeeded(logger, sa, secret); err != nil {
				logger.Error(err, "Populating serviceaccount token", "secret", klog.KRef(secretInfo.namespace, secretInfo.name), "serviceAccount", klog.KRef(secretInfo.namespace, secretInfo.saName))
				retry = retriable
			}
		}
	}
}

func (e *TokensController) deleteTokens(serviceAccount *v1.ServiceAccount) ( /*retry*/ bool, error) {
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
	var opts metav1.DeleteOptions
	if len(uid) > 0 {
		opts.Preconditions = &metav1.Preconditions{UID: &uid}
	}
	err := e.client.CoreV1().Secrets(ns).Delete(context.TODO(), name, opts)
	// NotFound doesn't need a retry (it's already been deleted)
	// Conflict doesn't need a retry (the UID precondition failed)
	if err == nil || apierrors.IsNotFound(err) || apierrors.IsConflict(err) {
		return false, nil
	}
	// Retry for any other error
	return true, err
}

func (e *TokensController) secretUpdateNeeded(secret *v1.Secret) (bool, bool, bool) {
	caData := secret.Data[v1.ServiceAccountRootCAKey]
	needsCA := len(e.rootCA) > 0 && !bytes.Equal(caData, e.rootCA)

	needsNamespace := len(secret.Data[v1.ServiceAccountNamespaceKey]) == 0

	tokenData := secret.Data[v1.ServiceAccountTokenKey]
	needsToken := len(tokenData) == 0

	return needsCA, needsNamespace, needsToken
}

// generateTokenIfNeeded populates the token data for the given Secret if not already set
func (e *TokensController) generateTokenIfNeeded(logger klog.Logger, serviceAccount *v1.ServiceAccount, cachedSecret *v1.Secret) ( /* retry */ bool, error) {
	// Check the cached secret to see if changes are needed
	if needsCA, needsNamespace, needsToken := e.secretUpdateNeeded(cachedSecret); !needsCA && !needsToken && !needsNamespace {
		return false, nil
	}

	// We don't want to update the cache's copy of the secret
	// so add the token to a freshly retrieved copy of the secret
	secrets := e.client.CoreV1().Secrets(cachedSecret.Namespace)
	liveSecret, err := secrets.Get(context.TODO(), cachedSecret.Name, metav1.GetOptions{})
	if err != nil {
		// Retry for any error other than a NotFound
		return !apierrors.IsNotFound(err), err
	}
	if liveSecret.ResourceVersion != cachedSecret.ResourceVersion {
		// our view of the secret is not up to date
		// we'll get notified of an update event later and get to try again
		logger.V(2).Info("Secret is not up to date, skipping token population", "secret", klog.KRef(liveSecret.Namespace, liveSecret.Name))
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
		liveSecret.Data[v1.ServiceAccountRootCAKey] = e.rootCA
	}
	// Set the namespace
	if needsNamespace {
		liveSecret.Data[v1.ServiceAccountNamespaceKey] = []byte(liveSecret.Namespace)
	}

	// Generate the token
	if needsToken {
		token, err := e.token.GenerateToken(serviceaccount.LegacyClaims(*serviceAccount, *liveSecret))
		if err != nil {
			return false, err
		}
		liveSecret.Data[v1.ServiceAccountTokenKey] = []byte(token)
	}

	// Set annotations
	liveSecret.Annotations[v1.ServiceAccountNameKey] = serviceAccount.Name
	liveSecret.Annotations[v1.ServiceAccountUIDKey] = string(serviceAccount.UID)

	// Save the secret
	_, err = secrets.Update(context.TODO(), liveSecret, metav1.UpdateOptions{})
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
	serviceAccounts := e.client.CoreV1().ServiceAccounts(saNamespace)
	serviceAccount, err := serviceAccounts.Get(context.TODO(), saName, metav1.GetOptions{})
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
	secrets := []v1.ObjectReference{}
	for _, s := range serviceAccount.Secrets {
		if s.Name != secretName {
			secrets = append(secrets, s)
		}
	}
	serviceAccount.Secrets = secrets
	_, err = serviceAccounts.Update(context.TODO(), serviceAccount, metav1.UpdateOptions{})
	// Ignore NotFound errors when attempting to remove a reference
	if apierrors.IsNotFound(err) {
		return nil
	}
	return err
}

func (e *TokensController) getServiceAccount(ns string, name string, uid types.UID, fetchOnCacheMiss bool) (*v1.ServiceAccount, error) {
	// Look up in cache
	sa, err := e.serviceAccounts.ServiceAccounts(ns).Get(name)
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	}
	if sa != nil {
		// Ensure UID matches if given
		if len(uid) == 0 || uid == sa.UID {
			return sa, nil
		}
	}

	if !fetchOnCacheMiss {
		return nil, nil
	}

	// Live lookup
	sa, err = e.client.CoreV1().ServiceAccounts(ns).Get(context.TODO(), name, metav1.GetOptions{})
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

func (e *TokensController) getSecret(ns string, name string, uid types.UID, fetchOnCacheMiss bool) (*v1.Secret, error) {
	// Look up in cache
	obj, exists, err := e.updatedSecrets.GetByKey(makeCacheKey(ns, name))
	if err != nil {
		return nil, err
	}
	if exists {
		secret, ok := obj.(*v1.Secret)
		if !ok {
			return nil, fmt.Errorf("expected *v1.Secret, got %#v", secret)
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
	secret, err := e.client.CoreV1().Secrets(ns).Get(context.TODO(), name, metav1.GetOptions{})
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
func (e *TokensController) listTokenSecrets(serviceAccount *v1.ServiceAccount) ([]*v1.Secret, error) {
	namespaceSecrets, err := e.updatedSecrets.ByIndex("namespace", serviceAccount.Namespace)
	if err != nil {
		return nil, err
	}

	items := []*v1.Secret{}
	for _, obj := range namespaceSecrets {
		secret := obj.(*v1.Secret)

		if apiserverserviceaccount.IsServiceAccountToken(secret, serviceAccount) {
			items = append(items, secret)
		}
	}
	return items, nil
}

func getSecretReferences(serviceAccount *v1.ServiceAccount) sets.String {
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

func makeServiceAccountKey(sa *v1.ServiceAccount) serviceAccountQueueKey {
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

func makeSecretQueueKey(secret *v1.Secret) secretQueueKey {
	return secretQueueKey{
		namespace: secret.Namespace,
		name:      secret.Name,
		uid:       secret.UID,
		saName:    secret.Annotations[v1.ServiceAccountNameKey],
		saUID:     types.UID(secret.Annotations[v1.ServiceAccountUIDKey]),
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
