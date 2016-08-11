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

	lru "github.com/hashicorp/golang-lru"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	"k8s.io/kubernetes/pkg/controller/framework"
	"k8s.io/kubernetes/pkg/controller/framework/informers"

	"k8s.io/kubernetes/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	// Name of admission plug-in
	PluginName = "NamespaceLifecycle"
	// how long a namespace stays in the force live lookup cache before expiration.
	forceLiveLookupTTL = 30 * time.Second
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
	namespaceInformer  framework.SharedIndexInformer
	// forceLiveLookupCache holds a list of entries for namespaces that we have a strong reason to believe are stale in our local cache.
	// if a namespace is in this cache, then we will ignore our local state and always fetch latest from api server.
	forceLiveLookupCache *lru.Cache
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
			newEntry := forceLiveLookupEntry{
				expiry: time.Now().Add(forceLiveLookupTTL),
			}
			l.forceLiveLookupCache.Add(a.GetName(), newEntry)
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

	// forceLiveLookup if true will skip looking at local cache state and instead always make a live call to server.
	forceLiveLookup := false
	lruItemObj, ok := l.forceLiveLookupCache.Get(a.GetNamespace())
	if ok && lruItemObj.(forceLiveLookupEntry).expiry.Before(time.Now()) {
		// we think the namespace was marked for deletion, but our current local cache says otherwise, we will force a live lookup.
		forceLiveLookup = exists && namespaceObj.(*api.Namespace).Status.Phase == api.NamespaceActive
	}

	// refuse to operate on non-existent namespaces
	if !exists || forceLiveLookup {
		// in case of latency in our caches, make a call direct to storage to verify that it truly exists or not
		namespaceObj, err = l.client.Core().Namespaces().Get(a.GetNamespace())
		if err != nil {
			if errors.IsNotFound(err) {
				return err
			}
			return errors.NewInternalError(err)
		}
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
	forceLiveLookupCache, err := lru.New(100)
	if err != nil {
		panic(err)
	}
	return &lifecycle{
		Handler:              admission.NewHandler(admission.Create, admission.Update, admission.Delete),
		client:               c,
		immortalNamespaces:   immortalNamespaces,
		forceLiveLookupCache: forceLiveLookupCache,
	}, nil
}

func (l *lifecycle) SetInformerFactory(f informers.SharedInformerFactory) {
	l.namespaceInformer = f.Namespaces().Informer()
	l.SetReadyFunc(l.namespaceInformer.HasSynced)
}

func (l *lifecycle) Validate() error {
	if l.namespaceInformer == nil {
		return fmt.Errorf("missing namespaceInformer")
	}
	return nil
}
