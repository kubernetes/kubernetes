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

	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller/informers"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	utilcache "k8s.io/kubernetes/pkg/util/cache"
	"k8s.io/kubernetes/pkg/util/clock"
	"k8s.io/kubernetes/pkg/util/sets"
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

func init() {
	admission.RegisterPlugin(PluginName, func(client clientset.Interface, config io.Reader) (admission.Interface, error) {
		return NewLifecycle(client, sets.NewString(api.NamespaceDefault, api.NamespaceSystem))
	})
}

// lifecycle is an implementation of admission.Interface.
// It enforces life-cycle constraints around a Namespace depending on its Phase
type lifecycle struct {
	*admission.Handler
	client             clientset.Interface
	immortalNamespaces sets.String
	namespaceInformer  cache.SharedIndexInformer
	// forceLiveLookupCache holds a list of entries for namespaces that we have a strong reason to believe are stale in our local cache.
	// if a namespace is in this cache, then we will ignore our local state and always fetch latest from api server.
	forceLiveLookupCache *utilcache.LRUExpireCache
}

type forceLiveLookupEntry struct {
	expiry time.Time
}

var _ = admission.WantsInformerFactory(&lifecycle{})

func makeNamespaceKey(namespace string) *api.Namespace {
	return &api.Namespace{
		ObjectMeta: api.ObjectMeta{
			Name:      namespace,
			Namespace: "",
		},
	}
}

func (l *lifecycle) Admit(a admission.Attributes) error {
	// prevent deletion of immortal namespaces
	if a.GetOperation() == admission.Delete && a.GetKind().GroupKind() == api.Kind("Namespace") && l.immortalNamespaces.Has(a.GetName()) {
		return errors.NewForbidden(a.GetResource().GroupResource(), a.GetName(), fmt.Errorf("this namespace may not be deleted"))
	}

	// if we're here, then we've already passed authentication, so we're allowed to do what we're trying to do
	// if we're here, then the API server has found a route, which means that if we have a non-empty namespace
	// its a namespaced resource.
	if len(a.GetNamespace()) == 0 || a.GetKind().GroupKind() == api.Kind("Namespace") {
		// if a namespace is deleted, we want to prevent all further creates into it
		// while it is undergoing termination.  to reduce incidences where the cache
		// is slow to update, we add the namespace into a force live lookup list to ensure
		// we are not looking at stale state.
		if a.GetOperation() == admission.Delete {
			l.forceLiveLookupCache.Add(a.GetName(), true, forceLiveLookupTTL)
		}
		return nil
	}

	// we need to wait for our caches to warm
	if !l.WaitForReady() {
		return admission.NewForbidden(a, fmt.Errorf("not yet ready to handle request"))
	}

	var (
		namespaceObj interface{}
		exists       bool
		err          error
	)

	key := makeNamespaceKey(a.GetNamespace())
	namespaceObj, exists, err = l.namespaceInformer.GetStore().Get(key)
	if err != nil {
		return errors.NewInternalError(err)
	}

	if !exists && a.GetOperation() == admission.Create {
		// give the cache time to observe the namespace before rejecting a create.
		// this helps when creating a namespace and immediately creating objects within it.
		time.Sleep(missingNamespaceWait)
		namespaceObj, exists, err = l.namespaceInformer.GetStore().Get(key)
		if err != nil {
			return errors.NewInternalError(err)
		}
		if exists {
			glog.V(4).Infof("found %s in cache after waiting", a.GetNamespace())
		}
	}

	// forceLiveLookup if true will skip looking at local cache state and instead always make a live call to server.
	forceLiveLookup := false
	if _, ok := l.forceLiveLookupCache.Get(a.GetNamespace()); ok {
		// we think the namespace was marked for deletion, but our current local cache says otherwise, we will force a live lookup.
		forceLiveLookup = exists && namespaceObj.(*api.Namespace).Status.Phase == api.NamespaceActive
	}

	// refuse to operate on non-existent namespaces
	if !exists || forceLiveLookup {
		// as a last resort, make a call directly to storage
		namespaceObj, err = l.client.Core().Namespaces().Get(a.GetNamespace())
		if err != nil {
			if errors.IsNotFound(err) {
				return err
			}
			return errors.NewInternalError(err)
		}
		glog.V(4).Infof("found %s via storage lookup", a.GetNamespace())
	}

	// ensure that we're not trying to create objects in terminating namespaces
	if a.GetOperation() == admission.Create {
		namespace := namespaceObj.(*api.Namespace)
		if namespace.Status.Phase != api.NamespaceTerminating {
			return nil
		}

		// TODO: This should probably not be a 403
		return admission.NewForbidden(a, fmt.Errorf("unable to create new content in namespace %s because it is being terminated.", a.GetNamespace()))
	}

	return nil
}

// NewLifecycle creates a new namespace lifecycle admission control handler
func NewLifecycle(c clientset.Interface, immortalNamespaces sets.String) (admission.Interface, error) {
	return newLifecycleWithClock(c, immortalNamespaces, clock.RealClock{})
}

func newLifecycleWithClock(c clientset.Interface, immortalNamespaces sets.String, clock utilcache.Clock) (admission.Interface, error) {
	forceLiveLookupCache := utilcache.NewLRUExpireCacheWithClock(100, clock)
	return &lifecycle{
		Handler:              admission.NewHandler(admission.Create, admission.Update, admission.Delete),
		client:               c,
		immortalNamespaces:   immortalNamespaces,
		forceLiveLookupCache: forceLiveLookupCache,
	}, nil
}

func (l *lifecycle) SetInformerFactory(f informers.SharedInformerFactory) {
	l.namespaceInformer = f.InternalNamespaces().Informer()
	l.SetReadyFunc(l.namespaceInformer.HasSynced)
}

func (l *lifecycle) Validate() error {
	if l.namespaceInformer == nil {
		return fmt.Errorf("missing namespaceInformer")
	}
	return nil
}
