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
	"k8s.io/apimachinery/pkg/util/sets"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/util/dryrun"
	"k8s.io/klog/v2"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	svcreg "k8s.io/kubernetes/pkg/registry/core/service"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	netutil "k8s.io/utils/net"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

type EndpointsStorage interface {
	rest.Getter
	rest.GracefulDeleter
}

type PodStorage interface {
	rest.Getter
}

type REST struct {
	*genericregistry.Store
	primaryIPFamily   api.IPFamily
	secondaryIPFamily api.IPFamily
	alloc             Allocators
	endpoints         EndpointsStorage
	pods              PodStorage
	proxyTransport    http.RoundTripper
}

var (
	_ rest.CategoriesProvider     = &REST{}
	_ rest.ShortNamesProvider     = &REST{}
	_ rest.StorageVersionProvider = &REST{}
	_ rest.ResetFieldsStrategy    = &REST{}
	_ rest.Redirector             = &REST{}
)

// NewREST returns a REST object that will work against services.
func NewREST(
	optsGetter generic.RESTOptionsGetter,
	serviceIPFamily api.IPFamily,
	ipAllocs map[api.IPFamily]ipallocator.Interface,
	portAlloc portallocator.Interface,
	endpoints EndpointsStorage,
	pods PodStorage,
	proxyTransport http.RoundTripper) (*REST, *StatusREST, *svcreg.ProxyREST, error) {

	store := &genericregistry.Store{
		NewFunc:                  func() runtime.Object { return &api.Service{} },
		NewListFunc:              func() runtime.Object { return &api.ServiceList{} },
		DefaultQualifiedResource: api.Resource("services"),
		ReturnDeletedObject:      true,

		CreateStrategy:      svcreg.Strategy,
		UpdateStrategy:      svcreg.Strategy,
		DeleteStrategy:      svcreg.Strategy,
		ResetFieldsStrategy: svcreg.Strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = svcreg.StatusStrategy
	statusStore.ResetFieldsStrategy = svcreg.StatusStrategy

	var primaryIPFamily api.IPFamily = serviceIPFamily
	var secondaryIPFamily api.IPFamily = "" // sentinel value
	if len(ipAllocs) > 1 {
		secondaryIPFamily = otherFamily(serviceIPFamily)
	}
	genericStore := &REST{
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
	_ rest.ShortNamesProvider = &REST{}
	_ rest.CategoriesProvider = &REST{}
)

// ShortNames implements the ShortNamesProvider interface. Returns a list of short names for a resource.
func (r *REST) ShortNames() []string {
	return []string{"svc"}
}

// Categories implements the CategoriesProvider interface. Returns a list of categories a resource is part of.
func (r *REST) Categories() []string {
	return []string{"all"}
}

// Destroy cleans up everything on shutdown.
func (r *REST) Destroy() {
	r.Store.Destroy()
	r.alloc.Destroy()
}

// StatusREST implements the REST endpoint for changing the status of a service.
type StatusREST struct {
	store *genericregistry.Store
}

func (r *StatusREST) New() runtime.Object {
	return &api.Service{}
}

// Destroy cleans up resources on shutdown.
func (r *StatusREST) Destroy() {
	// Given that underlying store is shared with REST,
	// we don't destroy it here explicitly.
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

func (r *StatusREST) ConvertToTable(ctx context.Context, object runtime.Object, tableOptions runtime.Object) (*metav1.Table, error) {
	return r.store.ConvertToTable(ctx, object, tableOptions)
}

// GetResetFields implements rest.ResetFieldsStrategy
func (r *StatusREST) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	return r.store.GetResetFields()
}

// We have a lot of functions that take a pair of "before" and "after" or
// "oldSvc" and "newSvc" args.  Convention across the codebase is to pass them
// as (new, old), but it's easy to screw up when they are the same type.
//
// These types force us to pay attention.  If the order of the arguments
// matters, please receive them as:
//    func something(after After, before Before) {
//        oldSvc, newSvc := before.Service, after.Service
//
// If the order of arguments DOES NOT matter, please receive them as:
//    func something(lhs, rhs *api.Service) {

type Before struct {
	*api.Service
}
type After struct {
	*api.Service
}

// defaultOnRead sets interlinked fields that were not previously set on read.
// We can't do this in the normal defaulting path because that same logic
// applies on Get, Create, and Update, but we need to distinguish between them.
//
// This will be called on both Service and ServiceList types.
func (r *REST) defaultOnRead(obj runtime.Object) {
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
func (r *REST) defaultOnReadServiceList(serviceList *api.ServiceList) {
	if serviceList == nil {
		return
	}

	for i := range serviceList.Items {
		r.defaultOnReadService(&serviceList.Items[i])
	}
}

// defaultOnReadService defaults a single Service.
func (r *REST) defaultOnReadService(service *api.Service) {
	if service == nil {
		return
	}

	// We might find Services that were written before ClusterIP became plural.
	// We still want to present a consistent view of them.
	normalizeClusterIPs(After{service}, Before{nil})

	// Set ipFamilies and ipFamilyPolicy if needed.
	r.defaultOnReadIPFamilies(service)

	// We unintentionally defaulted internalTrafficPolicy when it's not needed
	// for the ExternalName type. It's too late to change the field in storage,
	// but we can drop the field when read.
	defaultOnReadInternalTrafficPolicy(service)
}

func defaultOnReadInternalTrafficPolicy(service *api.Service) {
	if service.Spec.Type == api.ServiceTypeExternalName {
		service.Spec.InternalTrafficPolicy = nil
	}
}

func (r *REST) defaultOnReadIPFamilies(service *api.Service) {
	// ExternalName does not need this.
	if !needsClusterIP(service) {
		return
	}

	// If IPFamilies is set, we assume IPFamilyPolicy is also set (it should
	// not be possible to have one and not the other), and therefore we don't
	// need further defaulting.  Likewise, if IPFamilies is *not* set, we
	// assume IPFamilyPolicy can't be set either.
	if len(service.Spec.IPFamilies) > 0 {
		return
	}

	singleStack := api.IPFamilyPolicySingleStack
	requireDualStack := api.IPFamilyPolicyRequireDualStack

	if service.Spec.ClusterIP == api.ClusterIPNone {
		// Headless.
		if len(service.Spec.Selector) == 0 {
			// Headless + selectorless is a special-case.
			//
			// At this stage we don't know what kind of endpoints (specifically
			// their IPFamilies) the user has assigned to this selectorless
			// service. We assume it has dual-stack and we default it to
			// RequireDualStack on any cluster (single- or dual-stack
			// configured).
			service.Spec.IPFamilyPolicy = &requireDualStack
			service.Spec.IPFamilies = []api.IPFamily{r.primaryIPFamily, otherFamily(r.primaryIPFamily)}
		} else {
			// Headless + selector - default to single.
			service.Spec.IPFamilyPolicy = &singleStack
			service.Spec.IPFamilies = []api.IPFamily{r.primaryIPFamily}
		}
	} else {
		// Headful: init ipFamilies from clusterIPs.
		service.Spec.IPFamilies = make([]api.IPFamily, len(service.Spec.ClusterIPs))
		for idx, ip := range service.Spec.ClusterIPs {
			if netutil.IsIPv6String(ip) {
				service.Spec.IPFamilies[idx] = api.IPv6Protocol
			} else {
				service.Spec.IPFamilies[idx] = api.IPv4Protocol
			}
		}
		if len(service.Spec.IPFamilies) == 1 {
			service.Spec.IPFamilyPolicy = &singleStack
		} else if len(service.Spec.IPFamilies) == 2 {
			// It shouldn't be possible to get here, but just in case.
			service.Spec.IPFamilyPolicy = &requireDualStack
		}
	}
}

func (r *REST) afterDelete(obj runtime.Object, options *metav1.DeleteOptions) {
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

func (r *REST) beginCreate(ctx context.Context, obj runtime.Object, options *metav1.CreateOptions) (genericregistry.FinishFunc, error) {
	svc := obj.(*api.Service)

	// Make sure ClusterIP and ClusterIPs are in sync.  This has to happen
	// early, before anyone looks at them.
	normalizeClusterIPs(After{svc}, Before{nil})

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

func (r *REST) beginUpdate(ctx context.Context, obj, oldObj runtime.Object, options *metav1.UpdateOptions) (genericregistry.FinishFunc, error) {
	newSvc := obj.(*api.Service)
	oldSvc := oldObj.(*api.Service)

	// Make sure the existing object has all fields we expect to be defaulted.
	// This might not be true if the saved object predates these fields (the
	// Decorator hook is not called on 'old' in the update path.
	r.defaultOnReadService(oldSvc)

	// Fix up allocated values that the client may have not specified (for
	// idempotence).
	patchAllocatedValues(After{newSvc}, Before{oldSvc})

	// Make sure ClusterIP and ClusterIPs are in sync.  This has to happen
	// early, before anyone looks at them.
	normalizeClusterIPs(After{newSvc}, Before{oldSvc})

	// Allocate and initialize fields.
	txn, err := r.alloc.allocateUpdate(After{newSvc}, Before{oldSvc}, dryrun.IsDryRun(options.DryRun))
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
func (r *REST) ResourceLocation(ctx context.Context, id string) (*url.URL, http.RoundTripper, error) {
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

func isValidAddress(ctx context.Context, addr *api.EndpointAddress, pods rest.Getter) error {
	if addr.TargetRef == nil {
		return fmt.Errorf("Address has no target ref, skipping: %v", addr)
	}
	if genericapirequest.NamespaceValue(ctx) != addr.TargetRef.Namespace {
		return fmt.Errorf("Address namespace doesn't match context namespace")
	}
	obj, err := pods.Get(ctx, addr.TargetRef.Name, &metav1.GetOptions{})
	if err != nil {
		return err
	}
	pod, ok := obj.(*api.Pod)
	if !ok {
		return fmt.Errorf("failed to cast to pod: %v", obj)
	}
	if pod == nil {
		return fmt.Errorf("pod is missing, skipping (%s/%s)", addr.TargetRef.Namespace, addr.TargetRef.Name)
	}
	for _, podIP := range pod.Status.PodIPs {
		if podIP.IP == addr.IP {
			return nil
		}
	}
	return fmt.Errorf("pod ip(s) doesn't match endpoint ip, skipping: %v vs %s (%s/%s)", pod.Status.PodIPs, addr.IP, addr.TargetRef.Namespace, addr.TargetRef.Name)
}

// normalizeClusterIPs adjust clusterIPs based on ClusterIP.  This must not
// consider any other fields.
func normalizeClusterIPs(after After, before Before) {
	oldSvc, newSvc := before.Service, after.Service

	// In all cases here, we don't need to over-think the inputs.  Validation
	// will be called on the new object soon enough.  All this needs to do is
	// try to divine what user meant with these linked fields. The below
	// is verbosely written for clarity.

	// **** IMPORTANT *****
	// as a governing rule. User must (either)
	// -- Use singular only (old client)
	// -- singular and plural fields (new clients)

	if oldSvc == nil {
		// This was a create operation.
		// User specified singular and not plural (e.g. an old client), so init
		// plural for them.
		if len(newSvc.Spec.ClusterIP) > 0 && len(newSvc.Spec.ClusterIPs) == 0 {
			newSvc.Spec.ClusterIPs = []string{newSvc.Spec.ClusterIP}
			return
		}

		// we don't init singular based on plural because
		// new client must use both fields

		// Either both were not specified (will be allocated) or both were
		// specified (will be validated).
		return
	}

	// This was an update operation

	// ClusterIPs were cleared by an old client which was trying to patch
	// some field and didn't provide ClusterIPs
	if len(oldSvc.Spec.ClusterIPs) > 0 && len(newSvc.Spec.ClusterIPs) == 0 {
		// if ClusterIP is the same, then it is an old client trying to
		// patch service and didn't provide ClusterIPs
		if oldSvc.Spec.ClusterIP == newSvc.Spec.ClusterIP {
			newSvc.Spec.ClusterIPs = oldSvc.Spec.ClusterIPs
		}
	}

	// clusterIP is not the same
	if oldSvc.Spec.ClusterIP != newSvc.Spec.ClusterIP {
		// this is a client trying to clear it
		if len(oldSvc.Spec.ClusterIP) > 0 && len(newSvc.Spec.ClusterIP) == 0 {
			// if clusterIPs are the same, then clear on their behalf
			if sameClusterIPs(oldSvc, newSvc) {
				newSvc.Spec.ClusterIPs = nil
			}

			// if they provided nil, then we are fine (handled by patching case above)
			// if they changed it then validation will catch it
		} else {
			// ClusterIP has changed but not cleared *and* ClusterIPs are the same
			// then we set ClusterIPs based on ClusterIP
			if sameClusterIPs(oldSvc, newSvc) {
				newSvc.Spec.ClusterIPs = []string{newSvc.Spec.ClusterIP}
			}
		}
	}
}

// patchAllocatedValues allows clients to avoid a read-modify-write cycle while
// preserving values that we allocated on their behalf.  For example, they
// might create a Service without specifying the ClusterIP, in which case we
// allocate one.  If they resubmit that same YAML, we want it to succeed.
func patchAllocatedValues(after After, before Before) {
	oldSvc, newSvc := before.Service, after.Service

	if needsClusterIP(oldSvc) && needsClusterIP(newSvc) {
		if newSvc.Spec.ClusterIP == "" {
			newSvc.Spec.ClusterIP = oldSvc.Spec.ClusterIP
		}
		if len(newSvc.Spec.ClusterIPs) == 0 && len(oldSvc.Spec.ClusterIPs) > 0 {
			newSvc.Spec.ClusterIPs = oldSvc.Spec.ClusterIPs
		}
	}

	if needsNodePort(oldSvc) && needsNodePort(newSvc) {
		nodePortsUsed := func(svc *api.Service) sets.Int32 {
			used := sets.NewInt32()
			for _, p := range svc.Spec.Ports {
				if p.NodePort != 0 {
					used.Insert(p.NodePort)
				}
			}
			return used
		}

		// Build a set of all the ports in oldSvc that are also in newSvc.  We know
		// we can't patch these values.
		used := nodePortsUsed(oldSvc).Intersection(nodePortsUsed(newSvc))

		// Map NodePorts by name.  The user may have changed other properties
		// of the port, but we won't see that here.
		np := map[string]int32{}
		for i := range oldSvc.Spec.Ports {
			p := &oldSvc.Spec.Ports[i]
			np[p.Name] = p.NodePort
		}

		// If newSvc is missing values, try to patch them in when we know them and
		// they haven't been used for another port.

		for i := range newSvc.Spec.Ports {
			p := &newSvc.Spec.Ports[i]
			if p.NodePort == 0 {
				oldVal := np[p.Name]
				if !used.Has(oldVal) {
					p.NodePort = oldVal
				}
			}
		}
	}

	if needsHCNodePort(oldSvc) && needsHCNodePort(newSvc) {
		if newSvc.Spec.HealthCheckNodePort == 0 {
			newSvc.Spec.HealthCheckNodePort = oldSvc.Spec.HealthCheckNodePort
		}
	}
}

func needsClusterIP(svc *api.Service) bool {
	if svc.Spec.Type == api.ServiceTypeExternalName {
		return false
	}
	return true
}

func needsNodePort(svc *api.Service) bool {
	if svc.Spec.Type == api.ServiceTypeNodePort || svc.Spec.Type == api.ServiceTypeLoadBalancer {
		return true
	}
	return false
}

func needsHCNodePort(svc *api.Service) bool {
	if svc.Spec.Type != api.ServiceTypeLoadBalancer {
		return false
	}
	if svc.Spec.ExternalTrafficPolicy != api.ServiceExternalTrafficPolicyTypeLocal {
		return false
	}
	return true
}
