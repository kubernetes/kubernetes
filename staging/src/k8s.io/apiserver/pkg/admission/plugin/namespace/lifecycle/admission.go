/*
Copyright 2015 The Kubernetes Authors.

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

package lifecycle

import (
	"fmt"
	"io"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilcache "k8s.io/apimachinery/pkg/util/cache"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/pkg/api/v1"
)

const (
	// Name of admission plug-in
	PluginName = "NamespaceLifecycle"
	// how long a namespace stays in the force live lookup cache before expiration.
	forceLiveLookupTTL = 30 * time.Second
	// how long to wait for a missing namespace before re-checking the cache (and then doing a live lookup)
	// this accomplishes two things:
	// 1. It allows a watch-fed cache time to observe a namespace creation event
	// 2. It allows time for a namespace creation to distribute to members of a storage cluster,
	//    so the live lookup has a better chance of succeeding even if it isn't performed against the leader.
	missingNamespaceWait = 50 * time.Millisecond
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewLifecycle(sets.NewString(metav1.NamespaceDefault, metav1.NamespaceSystem, metav1.NamespacePublic))
	})
}

// lifecycle is an implementation of admission.Interface.
// It enforces life-cycle constraints around a Namespace depending on its Phase
type lifecycle struct {
	*admission.Handler
	client             kubernetes.Interface
	immortalNamespaces sets.String
	namespaceLister    corelisters.NamespaceLister
	// forceLiveLookupCache holds a list of entries for namespaces that we have a strong reason to believe are stale in our local cache.
	// if a namespace is in this cache, then we will ignore our local state and always fetch latest from api server.
	forceLiveLookupCache *utilcache.LRUExpireCache
}

type forceLiveLookupEntry struct {
	expiry time.Time
}

var _ = initializer.WantsExternalKubeInformerFactory(&lifecycle{})
var _ = initializer.WantsExternalKubeClientSet(&lifecycle{})

func makeNamespaceKey(namespace string) *v1.Namespace {
	return &v1.Namespace{
		ObjectMeta: metav1.ObjectMeta{
			Name:      namespace,
			Namespace: "",
		},
	}
}

func (l *lifecycle) Admit(a admission.Attributes) error {
	// prevent deletion of immortal namespaces
	if a.GetOperation() == admission.Delete && a.GetKind().GroupKind() == v1.SchemeGroupVersion.WithKind("Namespace").GroupKind() && l.immortalNamespaces.Has(a.GetName()) {
		return errors.NewForbidden(a.GetResource().GroupResource(), a.GetName(), fmt.Errorf("this namespace may not be deleted"))
	}

	// if we're here, then we've already passed authentication, so we're allowed to do what we're trying to do
	// if we're here, then the API server has found a route, which means that if we have a non-empty namespace
	// its a namespaced resource.
	if len(a.GetNamespace()) == 0 || a.GetKind().GroupKind() == v1.SchemeGroupVersion.WithKind("Namespace").GroupKind() {
		// if a namespace is deleted, we want to prevent all further creates into it
		// while it is undergoing termination.  to reduce incidences where the cache
		// is slow to update, we add the namespace into a force live lookup list to ensure
		// we are not looking at stale state.
		if a.GetOperation() == admission.Delete {
			l.forceLiveLookupCache.Add(a.GetName(), true, forceLiveLookupTTL)
		}
		return nil
	}

	// always allow deletion of other resources
	if a.GetOperation() == admission.Delete {
		return nil
	}

	// always allow access review checks.  Returning status about the namespace would be leaking information
	if isAccessReview(a) {
		return nil
	}

	// we need to wait for our caches to warm
	if !l.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	var (
		exists bool
		err    error
	)

	namespace, err := l.namespaceLister.Get(a.GetNamespace())
	if err != nil {
		if !errors.IsNotFound(err) {
			return errors.NewInternalError(err)
		}
	} else {
		exists = true
	}

	if !exists && a.GetOperation() == admission.Create {
		// give the cache time to observe the namespace before rejecting a create.
		// this helps when creating a namespace and immediately creating objects within it.
		time.Sleep(missingNamespaceWait)
		namespace, err = l.namespaceLister.Get(a.GetNamespace())
		switch {
		case errors.IsNotFound(err):
			// no-op
		case err != nil:
			return errors.NewInternalError(err)
		default:
			exists = true
		}
		if exists {
			glog.V(4).Infof("found %s in cache after waiting", a.GetNamespace())
		}
	}

	// forceLiveLookup if true will skip looking at local cache state and instead always make a live call to server.
	forceLiveLookup := false
	if _, ok := l.forceLiveLookupCache.Get(a.GetNamespace()); ok {
		// we think the namespace was marked for deletion, but our current local cache says otherwise, we will force a live lookup.
		forceLiveLookup = exists && namespace.Status.Phase == v1.NamespaceActive
	}

	// refuse to operate on non-existent namespaces
	if !exists || forceLiveLookup {
		// as a last resort, make a call directly to storage
		namespace, err = l.client.Core().Namespaces().Get(a.GetNamespace(), metav1.GetOptions{})
		switch {
		case errors.IsNotFound(err):
			return err
		case err != nil:
			return errors.NewInternalError(err)
		}
		glog.V(4).Infof("found %s via storage lookup", a.GetNamespace())
	}

	// ensure that we're not trying to create objects in terminating namespaces
	if a.GetOperation() == admission.Create {
		if namespace.Status.Phase != v1.NamespaceTerminating {
			return nil
		}

		// TODO: This should probably not be a 403
		return admission.NewForbidden(a, fmt.Errorf("unable to create new content in namespace %s because it is being terminated.", a.GetNamespace()))
	}

	return nil
}

// NewLifecycle creates a new namespace lifecycle admission control handler
func NewLifecycle(immortalNamespaces sets.String) (admission.Interface, error) {
	return newLifecycleWithClock(immortalNamespaces, clock.RealClock{})
}

func newLifecycleWithClock(immortalNamespaces sets.String, clock utilcache.Clock) (admission.Interface, error) {
	forceLiveLookupCache := utilcache.NewLRUExpireCacheWithClock(100, clock)
	return &lifecycle{
		Handler:              admission.NewHandler(admission.Create, admission.Update, admission.Delete),
		immortalNamespaces:   immortalNamespaces,
		forceLiveLookupCache: forceLiveLookupCache,
	}, nil
}

func (l *lifecycle) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().V1().Namespaces()
	l.namespaceLister = namespaceInformer.Lister()
	l.SetReadyFunc(namespaceInformer.Informer().HasSynced)
}

func (l *lifecycle) SetExternalKubeClientSet(client kubernetes.Interface) {
	l.client = client
}

func (l *lifecycle) Validate() error {
	if l.namespaceLister == nil {
		return fmt.Errorf("missing namespaceLister")
	}
	if l.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}

// accessReviewResources are resources which give a view into permissions in a namespace.  Users must be allowed to create these
// resources because returning "not found" errors allows someone to search for the "people I'm going to fire in 2017" namespace.
var accessReviewResources = map[schema.GroupResource]bool{
	{Group: "authorization.k8s.io", Resource: "localsubjectaccessreviews"}: true,
}

func isAccessReview(a admission.Attributes) bool {
	return accessReviewResources[a.GetResource().GroupResource()]
}
