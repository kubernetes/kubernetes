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

package storage

import (
	"context"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"net/url"
	"strconv"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/util/dryrun"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/core/service"
	svcreg "k8s.io/kubernetes/pkg/registry/core/service"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"

	netutil "k8s.io/utils/net"
)

type EndpointsStorage interface {
	rest.Getter
	rest.GracefulDeleter
}

type PodStorage interface {
	rest.Getter
}

type GenericREST struct {
	*genericregistry.Store
	primaryIPFamily   api.IPFamily
	secondaryIPFamily api.IPFamily
	alloc             RESTAllocStuff
	endpoints         EndpointsStorage
	pods              PodStorage
	proxyTransport    http.RoundTripper
}

var (
	_ rest.CategoriesProvider     = &GenericREST{}
	_ rest.ShortNamesProvider     = &GenericREST{}
	_ rest.StorageVersionProvider = &GenericREST{}
	_ rest.ResetFieldsStrategy    = &GenericREST{}
	_ rest.Redirector             = &GenericREST{}
)

// NewREST returns a RESTStorage object that will work against services.
func NewREST(
	optsGetter generic.RESTOptionsGetter,
	serviceIPFamily api.IPFamily,
	ipAllocs map[api.IPFamily]ipallocator.Interface,
	portAlloc portallocator.Interface,
	endpoints EndpointsStorage,
	pods PodStorage,
	proxyTransport http.RoundTripper) (*GenericREST, *StatusREST, *svcreg.ProxyREST, error) {

	strategy, _ := svcreg.StrategyForServiceCIDRs(ipAllocs[serviceIPFamily].CIDR(), len(ipAllocs) > 1)

	store := &genericregistry.Store{
		NewFunc:                  func() runtime.Object { return &api.Service{} },
		NewListFunc:              func() runtime.Object { return &api.ServiceList{} },
		DefaultQualifiedResource: api.Resource("services"),
		ReturnDeletedObject:      true,

		CreateStrategy:      strategy,
		UpdateStrategy:      strategy,
		DeleteStrategy:      strategy,
		ResetFieldsStrategy: strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, nil, err
	}

	statusStore := *store
	statusStrategy := service.NewServiceStatusStrategy(strategy)
	statusStore.UpdateStrategy = statusStrategy
	statusStore.ResetFieldsStrategy = statusStrategy

	var primaryIPFamily api.IPFamily = serviceIPFamily
	var secondaryIPFamily api.IPFamily = "" // sentinel value
	if len(ipAllocs) > 1 {
		secondaryIPFamily = otherFamily(serviceIPFamily)
	}
	genericStore := &GenericREST{
		Store:             store,
		primaryIPFamily:   primaryIPFamily,
		secondaryIPFamily: secondaryIPFamily,
		alloc:             makeAlloc(serviceIPFamily, ipAllocs, portAlloc),
		endpoints:         endpoints,
		pods:              pods,
		proxyTransport:    proxyTransport,
	}
	store.Decorator = genericStore.defaultOnRead
	store.AfterDelete = genericStore.afterDelete
	store.BeginCreate = genericStore.beginCreate
	store.BeginUpdate = genericStore.beginUpdate

	return genericStore, &StatusREST{store: &statusStore}, &svcreg.ProxyREST{Redirector: genericStore, ProxyTransport: proxyTransport}, nil
}

// otherFamily returns the non-selected IPFamily.  This assumes the input is
// valid.
func otherFamily(fam api.IPFamily) api.IPFamily {
	if fam == api.IPv4Protocol {
		return api.IPv6Protocol
	}
	return api.IPv4Protocol
}

var (
	_ rest.ShortNamesProvider = &GenericREST{}
	_ rest.CategoriesProvider = &GenericREST{}
)

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *GenericREST) ShortNames() []string {
	return []string{"svc"}
}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *GenericREST) Categories() []string {
	return []string{"all"}
}

// StatusREST implements the GenericREST endpoint for changing the status of a service.
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &api.Service{}
}

// Get retrieves the object from the storage. It is required to support Patch.
func (r *StatusREST) Get(ctx context.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return r.store.Get(ctx, name, options)
}

// Update alters the status subset of an object.
func (r *StatusREST) Update(ctx context.Context, name string, objInfo rest.UpdatedObjectInfo, createValidation rest.ValidateObjectFunc, updateValidation rest.ValidateObjectUpdateFunc, forceAllowCreate bool, options *metav1.UpdateOptions) (runtime.Object, bool, error) {
	// We are explicitly setting forceAllowCreate to false in the call to the underlying storage because
	// subresources should never allow create on update.
	return r.store.Update(ctx, name, objInfo, createValidation, updateValidation, false, options)
}

// GetResetFields implements rest.ResetFieldsStrategy
func (r *StatusREST) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return r.store.GetResetFields()
}

// defaultOnRead sets interlinked fields that were not previously set on read.
// We can't do this in the normal defaulting path because that same logic
// applies on Get, Create, and Update, but we need to distinguish between them.
//
// This will be called on both Service and ServiceList types.
func (r *GenericREST) defaultOnRead(obj runtime.Object) {
	switch s := obj.(type) {
	case *api.Service:
		r.defaultOnReadService(s)
	case *api.ServiceList:
		r.defaultOnReadServiceList(s)
	default:
		// This was not an object we can default.  This is not an error, as the
		// caching layer can pass through here, too.
	}
}

// defaultOnReadServiceList defaults a ServiceList.
func (r *GenericREST) defaultOnReadServiceList(serviceList *api.ServiceList) {
	if serviceList == nil {
		return
	}

	for i := range serviceList.Items {
		r.defaultOnReadService(&serviceList.Items[i])
	}
}

// defaultOnReadService defaults a single Service.
func (r *GenericREST) defaultOnReadService(service *api.Service) {
	if service == nil {
		return
	}

	// We might find Services that were written before ClusterIP became plural.
	// We still want to present a consistent view of them.
	// NOTE: the args are (old, new)
	svcreg.NormalizeClusterIPs(nil, service)

	// The rest of this does not apply unless dual-stack is enabled.
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
		return
	}

	if len(service.Spec.IPFamilies) > 0 {
		return // already defaulted
	}

	// set clusterIPs based on ClusterIP
	if len(service.Spec.ClusterIPs) == 0 {
		if len(service.Spec.ClusterIP) > 0 {
			service.Spec.ClusterIPs = []string{service.Spec.ClusterIP}
		}
	}

	requireDualStack := api.IPFamilyPolicyRequireDualStack
	singleStack := api.IPFamilyPolicySingleStack
	preferDualStack := api.IPFamilyPolicyPreferDualStack
	// headless services
	if len(service.Spec.ClusterIPs) == 1 && service.Spec.ClusterIPs[0] == api.ClusterIPNone {
		service.Spec.IPFamilies = []api.IPFamily{r.primaryIPFamily}

		// headless+selectorless
		// headless+selectorless takes both families. Why?
		// at this stage we don't know what kind of endpoints (specifically their IPFamilies) the
		// user has assigned to this selectorless service. We assume it has dualstack and we default
		// it to PreferDualStack on any cluster (single or dualstack configured).
		if len(service.Spec.Selector) == 0 {
			service.Spec.IPFamilyPolicy = &preferDualStack
			if r.primaryIPFamily == api.IPv4Protocol {
				service.Spec.IPFamilies = append(service.Spec.IPFamilies, api.IPv6Protocol)
			} else {
				service.Spec.IPFamilies = append(service.Spec.IPFamilies, api.IPv4Protocol)
			}
		} else {
			// headless w/ selector
			// this service type follows cluster configuration. this service (selector based) uses a
			// selector and will have to follow how the cluster is configured. If the cluster is
			// configured to dual stack then the service defaults to PreferDualStack. Otherwise we
			// default it to SingleStack.
			if r.secondaryIPFamily != "" {
				service.Spec.IPFamilies = append(service.Spec.IPFamilies, r.secondaryIPFamily)
				service.Spec.IPFamilyPolicy = &preferDualStack
			} else {
				service.Spec.IPFamilyPolicy = &singleStack
			}
		}
	} else {
		// headful
		// make sure a slice exists to receive the families
		service.Spec.IPFamilies = make([]api.IPFamily, len(service.Spec.ClusterIPs), len(service.Spec.ClusterIPs))
		for idx, ip := range service.Spec.ClusterIPs {
			if netutil.IsIPv6String(ip) {
				service.Spec.IPFamilies[idx] = api.IPv6Protocol
			} else {
				service.Spec.IPFamilies[idx] = api.IPv4Protocol
			}

			if len(service.Spec.IPFamilies) == 1 {
				service.Spec.IPFamilyPolicy = &singleStack
			} else if len(service.Spec.IPFamilies) == 2 {
				service.Spec.IPFamilyPolicy = &requireDualStack
			}
		}
	}
}

func (r *GenericREST) afterDelete(obj runtime.Object, options *metav1.DeleteOptions) {
	svc := obj.(*api.Service)

	// Normally this defaulting is done automatically, but the hook (Decorator)
	// is called at the end of this process, and we want the fully-formed
	// object.
	r.defaultOnReadService(svc)

	// Only perform the cleanup if this is a non-dryrun deletion
	if !dryrun.IsDryRun(options.DryRun) {
		// It would be better if we had the caller context, but that changes
		// this hook signature.
		ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), svc.Namespace)
		// TODO: This is clumsy.  It was added for fear that the endpoints
		// controller might lag, and we could end up rusing the service name
		// with old endpoints.  We should solve that better and remove this, or
		// else we should do this for EndpointSlice, too.
		_, _, err := r.endpoints.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
		if err != nil && !errors.IsNotFound(err) {
			klog.Errorf("delete service endpoints %s/%s failed: %v", svc.Name, svc.Namespace, err)
		}

		r.alloc.releaseAllocatedResources(svc)
	}
}

func (r *GenericREST) beginCreate(ctx context.Context, obj runtime.Object, options *metav1.CreateOptions) (genericregistry.FinishFunc, error) {
	svc := obj.(*api.Service)

	// Make sure ClusterIP and ClusterIPs are in sync.  This has to happen
	// early, before anyone looks at them.
	// NOTE: the args are (old, new)
	svcreg.NormalizeClusterIPs(nil, svc)

	// Allocate IPs and ports. If we had a transactional store, this would just
	// be part of the larger transaction.  We don't have that, so we have to do
	// it manually. This has to happen here and not in any earlier hooks (e.g.
	// defaulting) because it needs to be aware of flags and be able to access
	// API storage.
	txn, err := r.alloc.allocateCreate(svc, dryrun.IsDryRun(options.DryRun))
	if err != nil {
		return nil, err
	}

	// Our cleanup callback
	finish := func(_ context.Context, success bool) {
		if success {
			txn.Commit()
		} else {
			txn.Revert()
		}
	}

	return finish, nil
}

func (r *GenericREST) beginUpdate(ctx context.Context, obj, oldObj runtime.Object, options *metav1.UpdateOptions) (genericregistry.FinishFunc, error) {
	newSvc := obj.(*api.Service)
	oldSvc := oldObj.(*api.Service)

	// Fix up allocated values that the client may have not specified (for
	// idempotence).
	svcreg.PatchAllocatedValues(newSvc, oldSvc)

	// Make sure ClusterIP and ClusterIPs are in sync.  This has to happen
	// early, before anyone looks at them.
	// NOTE: the args are (old, new)
	svcreg.NormalizeClusterIPs(oldSvc, newSvc)

	// Allocate and initialize fields.
	txn, err := r.alloc.allocateUpdate(newSvc, oldSvc, dryrun.IsDryRun(options.DryRun))
	if err != nil {
		return nil, err
	}

	// Our cleanup callback
	finish := func(_ context.Context, success bool) {
		if success {
			txn.Commit()
		} else {
			txn.Revert()
		}
	}

	return finish, nil
}

// ResourceLocation returns a URL to which one can send traffic for the specified service.
func (r *GenericREST) ResourceLocation(ctx context.Context, id string) (*url.URL, http.RoundTripper, error) {
	// Allow ID as "svcname", "svcname:port", or "scheme:svcname:port".
	svcScheme, svcName, portStr, valid := utilnet.SplitSchemeNamePort(id)
	if !valid {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("invalid service request %q", id))
	}

	// If a port *number* was specified, find the corresponding service port name
	if portNum, err := strconv.ParseInt(portStr, 10, 64); err == nil {
		obj, err := r.Get(ctx, svcName, &metav1.GetOptions{})
		if err != nil {
			return nil, nil, err
		}
		svc := obj.(*api.Service)
		found := false
		for _, svcPort := range svc.Spec.Ports {
			if int64(svcPort.Port) == portNum {
				// use the declared port's name
				portStr = svcPort.Name
				found = true
				break
			}
		}
		if !found {
			return nil, nil, errors.NewServiceUnavailable(fmt.Sprintf("no service port %d found for service %q", portNum, svcName))
		}
	}

	obj, err := r.endpoints.Get(ctx, svcName, &metav1.GetOptions{})
	if err != nil {
		return nil, nil, err
	}
	eps := obj.(*api.Endpoints)
	if len(eps.Subsets) == 0 {
		return nil, nil, errors.NewServiceUnavailable(fmt.Sprintf("no endpoints available for service %q", svcName))
	}
	// Pick a random Subset to start searching from.
	ssSeed := rand.Intn(len(eps.Subsets))
	// Find a Subset that has the port.
	for ssi := 0; ssi < len(eps.Subsets); ssi++ {
		ss := &eps.Subsets[(ssSeed+ssi)%len(eps.Subsets)]
		if len(ss.Addresses) == 0 {
			continue
		}
		for i := range ss.Ports {
			if ss.Ports[i].Name == portStr {
				addrSeed := rand.Intn(len(ss.Addresses))
				// This is a little wonky, but it's expensive to test for the presence of a Pod
				// So we repeatedly try at random and validate it, this means that for an invalid
				// service with a lot of endpoints we're going to potentially make a lot of calls,
				// but in the expected case we'll only make one.
				for try := 0; try < len(ss.Addresses); try++ {
					addr := ss.Addresses[(addrSeed+try)%len(ss.Addresses)]
					// TODO(thockin): do we really need this check?
					if err := isValidAddress(ctx, &addr, r.pods); err != nil {
						utilruntime.HandleError(fmt.Errorf("Address %v isn't valid (%v)", addr, err))
						continue
					}
					ip := addr.IP
					port := int(ss.Ports[i].Port)
					return &url.URL{
						Scheme: svcScheme,
						Host:   net.JoinHostPort(ip, strconv.Itoa(port)),
					}, r.proxyTransport, nil
				}
				utilruntime.HandleError(fmt.Errorf("Failed to find a valid address, skipping subset: %v", ss))
			}
		}
	}
	return nil, nil, errors.NewServiceUnavailable(fmt.Sprintf("no endpoints available for service %q", id))
}
