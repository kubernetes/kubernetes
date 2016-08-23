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

package etcd

import (
	"fmt"
	"net/http"
	"net/url"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	storeerr "k8s.io/kubernetes/pkg/api/errors/storage"
	"k8s.io/kubernetes/pkg/api/rest"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/api/validation"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/kubelet/client"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/cachesize"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/pod"
	podrest "k8s.io/kubernetes/pkg/registry/pod/rest"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
)

// PodStorage includes storage for pods and all sub resources
type PodStorage struct {
	Pod         *REST
	Binding     *BindingREST
	Eviction    *EvictionREST
	Status      *StatusREST
	Log         *podrest.LogREST
	Proxy       *podrest.ProxyREST
	Exec        *podrest.ExecREST
	Attach      *podrest.AttachREST
	PortForward *podrest.PortForwardREST
}

// REST implements a RESTStorage for pods against etcd
type REST struct {
	*registry.Store
	proxyTransport http.RoundTripper
}

// NewStorage returns a RESTStorage object that will work against pods.
func NewStorage(opts generic.RESTOptions, k client.ConnectionInfoGetter, proxyTransport http.RoundTripper) PodStorage {
	prefix := "/" + opts.ResourcePrefix

	newListFunc := func() runtime.Object { return &api.PodList{} }
	storageInterface, _ := opts.Decorator(
		opts.StorageConfig,
		cachesize.GetWatchCacheSizeByResource(cachesize.Pods),
		&api.Pod{},
		prefix,
		pod.Strategy,
		newListFunc,
		pod.NodeNameTriggerFunc,
	)

	store := &registry.Store{
		NewFunc:     func() runtime.Object { return &api.Pod{} },
		NewListFunc: newListFunc,
		KeyRootFunc: func(ctx api.Context) string {
			return registry.NamespaceKeyRootFunc(ctx, prefix)
		},
		KeyFunc: func(ctx api.Context, name string) (string, error) {
			return registry.NamespaceKeyFunc(ctx, prefix, name)
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) {
			return obj.(*api.Pod).Name, nil
		},
		PredicateFunc:           pod.MatchPod,
		QualifiedResource:       api.Resource("pods"),
		DeleteCollectionWorkers: opts.DeleteCollectionWorkers,

		CreateStrategy:      pod.Strategy,
		UpdateStrategy:      pod.Strategy,
		DeleteStrategy:      pod.Strategy,
		ReturnDeletedObject: true,

		Storage: storageInterface,
	}

	statusStore := *store
	statusStore.UpdateStrategy = pod.StatusStrategy

	return PodStorage{
		Pod:         &REST{store, proxyTransport},
		Binding:     &BindingREST{store: store},
		Eviction:    &EvictionREST{store: store},
		Status:      &StatusREST{store: &statusStore},
		Log:         &podrest.LogREST{Store: store, KubeletConn: k},
		Proxy:       &podrest.ProxyREST{Store: store, ProxyTransport: proxyTransport},
		Exec:        &podrest.ExecREST{Store: store, KubeletConn: k},
		Attach:      &podrest.AttachREST{Store: store, KubeletConn: k},
		PortForward: &podrest.PortForwardREST{Store: store, KubeletConn: k},
	}
}

// Implement Redirector.
var _ = rest.Redirector(&REST{})

// ResourceLocation returns a pods location from its HostIP
func (r *REST) ResourceLocation(ctx api.Context, name string) (*url.URL, http.RoundTripper, error) {
	return pod.ResourceLocation(r, r.proxyTransport, ctx, name)
}

// EvictionREST implements the REST endpoint for evicting pods from nodes when etcd is in use.
type EvictionREST struct {
	store                      *registry.Store
	PodDisruptionBudgetLister  rest.Lister
	PodDisruptionBudgetUpdater rest.Updater
}

var _ = rest.Creater(&EvictionREST{})

// New creates a new eviction resource
func (r *EvictionREST) New() runtime.Object {
	return &policy.Eviction{}
}

// Create attempts to create a new eviction.  That is, it tries to evict a pod.
func (r *EvictionREST) Create(ctx api.Context, obj runtime.Object) (runtime.Object, error) {
	eviction := obj.(*policy.Eviction)

	obj, err := r.store.Get(ctx, eviction.Name)
	if err != nil {
		return nil, err
	}
	pod := obj.(*api.Pod)
	pdbs, err := r.getPodDisruptionBudgets(ctx, pod)
	if err != nil {
		return nil, err
	}

	if len(pdbs) > 1 {
		return &unversioned.Status{
			Status:  unversioned.StatusFailure,
			Message: "This pod has more than one PodDisruptionBudget, which the eviction subresource does not support.",
			Code:    500,
		}, nil
	} else if len(pdbs) == 1 {
		pdb := pdbs[0]
		// Try to verify-and-decrement

		// If it was false already, or if it becomes false during the course of our retries,
		// raise an error marked as a 429.
		ok, err := r.checkAndDecrement(ctx, pdb)
		if err != nil {
			return nil, err
		}

		if !ok {
			return &unversioned.Status{
				Status: unversioned.StatusFailure,
				// TODO(mml): Include some more details about why the eviction is disallowed.
				// Ideally any such text is generated by the DisruptionController (offline).
				Message: "Cannot evict pod as it would violate the pod's disruption budget.",
				Code:    429,
				// TODO(mml): Add a Retry-After header.  Once there are time-based
				// budgets, we can sometimes compute a sensible suggested value.  But
				// even without that, we can give a suggestion (10 minutes?) that
				// prevents well-behaved clients from hammering us.
			}, nil
		}
	}

	// At this point there was either no PDB or we succeded in decrementing

	// Try the delete
	_, err = r.store.Delete(ctx, eviction.Name, eviction.DeleteOptions)
	if err != nil {
		return nil, err
	}

	// Success!
	return &unversioned.Status{Status: unversioned.StatusSuccess}, nil
}

// UpdatedObjectInfo is a simple interface for attempting updates to
// runtime.Objects.  EvictionREST implements it directly.
var _ = rest.UpdatedObjectInfo(&EvictionREST{})

// Preconditions returns any preconditions required prior to updating the
// PDB.  None currently.
func (r *EvictionREST) Preconditions() *api.Preconditions {
	return nil
}

// UpdatedObject returns the updated PDB if it is able to update
// PodDisruptionAllowed from true->false.
func (r *EvictionREST) UpdatedObject(ctx api.Context, oldObj runtime.Object) (newObj runtime.Object, err error) {
	copy, err := api.Scheme.DeepCopy(oldObj)
	if err != nil {
		return
	}
	newObj = copy.(runtime.Object)
	pdb := oldObj.(*policy.PodDisruptionBudget)
	if !pdb.Status.PodDisruptionAllowed {
		return nil, fmt.Errorf("PodDisruptionAllowed is already false")
	}
	pdb.Status.PodDisruptionAllowed = false

	return
}

func (r *EvictionREST) checkAndDecrement(ctx api.Context, pdb policy.PodDisruptionBudget) (ok bool, err error) {
	if !pdb.Status.PodDisruptionAllowed {
		return false, nil
	}
	newObj, _, err := r.PodDisruptionBudgetUpdater.Update(ctx, pdb.Name, r)
	if err != nil {
		return false, err
	}

	newPdb := newObj.(*policy.PodDisruptionBudget)
	if newPdb.Status.PodDisruptionAllowed {
		return false, fmt.Errorf("update did not succeed")
	}

	return true, nil
}

// Returns any PDBs that match the pod.
// err is set if there's an error.
func (r *EvictionREST) getPodDisruptionBudgets(ctx api.Context, pod *api.Pod) (pdbs []policy.PodDisruptionBudget, err error) {
	if len(pod.Labels) == 0 {
		return
	}

	l, err := r.PodDisruptionBudgetLister.List(ctx, nil)
	if err != nil {
		return
	}

	pdbList := l.(*policy.PodDisruptionBudgetList)

	for _, pdb := range pdbList.Items {
		if pdb.Namespace != pod.Namespace {
			continue
		}
		selector, err := unversioned.LabelSelectorAsSelector(pdb.Spec.Selector)
		if err != nil {
			continue
		}
		// If a PDB with a nil or empty selector creeps in, it should match nothing, not everything.
		if selector.Empty() || !selector.Matches(labels.Set(pod.Labels)) {
			continue
		}

		pdbs = append(pdbs, pdb)
	}

	return pdbs, nil
}

// BindingREST implements the REST endpoint for binding pods to nodes when etcd is in use.
type BindingREST struct {
	store *registry.Store
}

// New creates a new binding resource
func (r *BindingREST) New() runtime.Object {
	return &api.Binding{}
}

var _ = rest.Creater(&BindingREST{})

// Create ensures a pod is bound to a specific host.
func (r *BindingREST) Create(ctx api.Context, obj runtime.Object) (out runtime.Object, err error) {
	binding := obj.(*api.Binding)

	// TODO: move me to a binding strategy
	if errs := validation.ValidatePodBinding(binding); len(errs) != 0 {
		return nil, errs.ToAggregate()
	}

	err = r.assignPod(ctx, binding.Name, binding.Target.Name, binding.Annotations)
	out = &unversioned.Status{Status: unversioned.StatusSuccess}
	return
}

// setPodHostAndAnnotations sets the given pod's host to 'machine' if and only if it was
// previously 'oldMachine' and merges the provided annotations with those of the pod.
// Returns the current state of the pod, or an error.
func (r *BindingREST) setPodHostAndAnnotations(ctx api.Context, podID, oldMachine, machine string, annotations map[string]string) (finalPod *api.Pod, err error) {
	podKey, err := r.store.KeyFunc(ctx, podID)
	if err != nil {
		return nil, err
	}
	err = r.store.Storage.GuaranteedUpdate(ctx, podKey, &api.Pod{}, false, nil, storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
		pod, ok := obj.(*api.Pod)
		if !ok {
			return nil, fmt.Errorf("unexpected object: %#v", obj)
		}
		if pod.DeletionTimestamp != nil {
			return nil, fmt.Errorf("pod %s is being deleted, cannot be assigned to a host", pod.Name)
		}
		if pod.Spec.NodeName != oldMachine {
			return nil, fmt.Errorf("pod %v is already assigned to node %q", pod.Name, pod.Spec.NodeName)
		}
		pod.Spec.NodeName = machine
		if pod.Annotations == nil {
			pod.Annotations = make(map[string]string)
		}
		for k, v := range annotations {
			pod.Annotations[k] = v
		}
		api.UpdatePodCondition(&pod.Status, &api.PodCondition{
			Type:   api.PodScheduled,
			Status: api.ConditionTrue,
		})
		finalPod = pod
		return pod, nil
	}))
	return finalPod, err
}

// assignPod assigns the given pod to the given machine.
func (r *BindingREST) assignPod(ctx api.Context, podID string, machine string, annotations map[string]string) (err error) {
	if _, err = r.setPodHostAndAnnotations(ctx, podID, "", machine, annotations); err != nil {
		err = storeerr.InterpretGetError(err, api.Resource("pods"), podID)
		err = storeerr.InterpretUpdateError(err, api.Resource("pods"), podID)
		if _, ok := err.(*errors.StatusError); !ok {
			err = errors.NewConflict(api.Resource("pods/binding"), podID, err)
		}
	}
	return
}

// StatusREST implements the REST endpoint for changing the status of a pod.
type StatusREST struct {
	store *registry.Store
}

// New creates a new pod resource
func (r *StatusREST) New() runtime.Object {
	return &api.Pod{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx api.Context, name string) (runtime.Object, error) {
	return r.store.Get(ctx, name)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx api.Context, name string, objInfo rest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return r.store.Update(ctx, name, objInfo)
}
