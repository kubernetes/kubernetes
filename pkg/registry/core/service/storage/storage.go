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
	"net"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/klog/v2"
	apiservice "k8s.io/kubernetes/pkg/api/service"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/printers"
	printersinternal "k8s.io/kubernetes/pkg/printers/internalversion"
	printerstorage "k8s.io/kubernetes/pkg/printers/storage"
	"k8s.io/kubernetes/pkg/registry/core/service"
	registry "k8s.io/kubernetes/pkg/registry/core/service"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"

	netutil "k8s.io/utils/net"

	"k8s.io/apiserver/pkg/util/dryrun"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

type GenericREST struct {
	*genericregistry.Store
	primaryIPFamily *api.IPFamily
	secondaryFamily *api.IPFamily
}

// NewREST returns a RESTStorage object that will work against services.
func NewGenericREST(optsGetter generic.RESTOptionsGetter, serviceCIDR net.IPNet, hasSecondary bool) (*GenericREST, *StatusREST, error) {
	strategy, _ := registry.StrategyForServiceCIDRs(serviceCIDR, hasSecondary)

	store := &genericregistry.Store{
		NewFunc:                  func() runtime.Object { return &api.Service{} },
		NewListFunc:              func() runtime.Object { return &api.ServiceList{} },
		DefaultQualifiedResource: api.Resource("services"),
		ReturnDeletedObject:      true,

		CreateStrategy: strategy,
		UpdateStrategy: strategy,
		DeleteStrategy: strategy,
		ExportStrategy: strategy,

		TableConvertor: printerstorage.TableConvertor{TableGenerator: printers.NewTableGenerator().With(printersinternal.AddHandlers)},
	}
	options := &generic.StoreOptions{RESTOptions: optsGetter}
	if err := store.CompleteWithOptions(options); err != nil {
		return nil, nil, err
	}

	statusStore := *store
	statusStore.UpdateStrategy = service.NewServiceStatusStrategy(strategy)

	ipv4 := api.IPv4Protocol
	ipv6 := api.IPv6Protocol
	var primaryIPFamily *api.IPFamily = nil
	var secondaryFamily *api.IPFamily = nil
	if netutil.IsIPv6CIDR(&serviceCIDR) {
		primaryIPFamily = &ipv6
		if hasSecondary {
			secondaryFamily = &ipv4
		}
	} else {
		primaryIPFamily = &ipv4
		if hasSecondary {
			secondaryFamily = &ipv6
		}
	}
	genericStore := &GenericREST{store, primaryIPFamily, secondaryFamily}
	store.Decorator = genericStore.defaultServiceOnRead // default on read
	store.BeginCreate = genericStore.beginCreate
	store.BeginUpdate = genericStore.beginUpdate

	return genericStore, &StatusREST{store: &statusStore}, nil
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

// defaults fields that were not previously set on read. becomes an
// essential part of upgrading a service
func (r *GenericREST) defaultServiceOnRead(obj runtime.Object) error {
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
		return nil
	}

	service, ok := obj.(*api.Service)
	if ok {
		return r.defaultAServiceOnRead(service)
	}

	serviceList, ok := obj.(*api.ServiceList)
	if ok {
		return r.defaultServiceList(serviceList)
	}

	// this was not an object we can default
	return nil
}

// defaults a service list
func (r *GenericREST) defaultServiceList(serviceList *api.ServiceList) error {
	if serviceList == nil {
		return nil
	}

	for i := range serviceList.Items {
		err := r.defaultAServiceOnRead(&serviceList.Items[i])
		if err != nil {
			return err
		}
	}

	return nil
}

// defaults a single service
func (r *GenericREST) defaultAServiceOnRead(service *api.Service) error {
	if service == nil {
		return nil
	}

	if len(service.Spec.IPFamilies) > 0 {
		return nil // already defaulted
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
		service.Spec.IPFamilies = []api.IPFamily{*r.primaryIPFamily}

		// headless+selectorless
		// headless+selectorless takes both families. Why?
		// at this stage we don't know what kind of endpoints (specifically their IPFamilies) the
		// user has assigned to this selectorless service. We assume it has dualstack and we default
		// it to PreferDualStack on any cluster (single or dualstack configured).
		if len(service.Spec.Selector) == 0 {
			service.Spec.IPFamilyPolicy = &preferDualStack
			if *r.primaryIPFamily == api.IPv4Protocol {
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
			if r.secondaryFamily != nil {
				service.Spec.IPFamilies = append(service.Spec.IPFamilies, *r.secondaryFamily)
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

	return nil
}

func (r *GenericREST) beginCreate(obj runtime.Object, options *metav1.CreateOptions) (func(runtime.Object, error), error) {
	service := obj.(*api.Service)

	// try set ip families (for missing ip families)
	// we do it here, since we want this to be visible
	// even when dryRun == true
	if err := r.tryDefaultValidateServiceClusterIPFields(service); err != nil {
		return nil, err
	}

	// bag of clusterIPs allocated in the process of creation
	// failed allocation will automatically trigger release
	var toReleaseClusterIPs map[api.IPFamily]string

	// node-port state to clean up
	nodePortOp := portallocator.StartOperation(r.serviceNodePorts, dryrun.IsDryRun(options.DryRun))

	// Our cleanup callback
	finish := func(_ runtime.Object, result error) {
		if result == nil {
			el := nodePortOp.Commit()
			if el != nil {
				// these should be caught by an eventual reconciliation / restart
				utilruntime.HandleError(fmt.Errorf("error(s) committing service node-ports changes: %v", el))
			}
		} else {
			released, err := r.releaseClusterIPs(toReleaseClusterIPs)
			if err != nil {
				klog.Warningf("failed to release clusterIPs for failed new service:%v allocated:%v released:%v error:%v",
					service.Name, toReleaseClusterIPs, released, err)
			}
		}
		nodePortOp.Finish()
	}

	//
	// Do allocations
	//

	var err error
	if !dryrun.IsDryRun(options.DryRun) {
		toReleaseClusterIPs, err = r.allocServiceClusterIPs(service)
		if err != nil {
			return finish, err
		}
	}

	if service.Spec.Type == api.ServiceTypeNodePort || service.Spec.Type == api.ServiceTypeLoadBalancer {
		if err := initNodePorts(service, nodePortOp); err != nil {
			return finish, err
		}
	}

	// Handle ExternalTraffic related fields during service creation.
	if apiservice.NeedsHealthCheck(service) {
		if err := allocateHealthCheckNodePort(service, nodePortOp); err != nil {
			return finish, errors.NewInternalError(err)
		}
	}

	//FIXME: move this somewhere more appropriate
	if errs := validation.ValidateServiceExternalTrafficFieldsCombination(service); len(errs) > 0 {
		return finish, errors.NewInvalid(api.Kind("Service"), service.Name, errs)
	}

	return finish, nil
}

func (r *GenericREST) beginUpdate(obj, oldObj runtime.Object, options *metav1.CreateOptions) (func(runtime.Object, runtime.Object, error), error) {
	service := obj.(*api.Service)
	oldService := oldObj.(*api.Service)

	var allocated map[api.IPFamily]string
	var toReleaseIPs map[api.IPFamily]string
	nodePortOp := portallocator.StartOperation(r.serviceNodePorts, dryrun.IsDryRun(options.DryRun))

	// Our cleanup callback
	finish := func(_, _ runtime.Object, result error) {
		if result == nil {
			toReleaseIPsReleased, err := r.releaseClusterIPs(toReleaseIPs)
			if err != nil {
				klog.V(4).Infof("service %v/%v failed to clean up after failed service update error:%v. ShouldRelease/Released:%v/%v", service.Namespace, service.Name, err, toReleaseIPs, toReleaseIPsReleased)
			}
			el := nodePortOp.Commit()
			if el != nil {
				// these should be caught by an eventual reconciliation / restart
				utilruntime.HandleError(fmt.Errorf("error(s) committing service node-ports changes: %v", el))
			}
		} else {
			allocatedReleased, err := r.releaseClusterIPs(allocated)
			if err != nil {
				klog.V(4).Infof("service %v/%v failed to clean up after failed service update error:%v. Allocated/Released:%v/%v", service.Namespace, service.Name, err, allocated, allocatedReleased)
			}
			nodePortOp.Finish()
		}
	}

	//
	// Do allocations
	//

	// try set ip families (for missing ip families)
	if err := r.tryDefaultValidateServiceClusterIPFields(service); err != nil {
		return finish, err
	}

	if !dryrun.IsDryRun(options.DryRun) {
		var err error
		allocated, toReleaseIPs, err = r.handleClusterIPsForUpdatedService(oldService, service)
		if err != nil {
			return finish, err
		}
	}
	// Update service from NodePort or LoadBalancer to ExternalName or ClusterIP, should release NodePort if exists.
	if (oldService.Spec.Type == api.ServiceTypeNodePort || oldService.Spec.Type == api.ServiceTypeLoadBalancer) &&
		(service.Spec.Type == api.ServiceTypeExternalName || service.Spec.Type == api.ServiceTypeClusterIP) {
		releaseNodePorts(oldService, nodePortOp)
	}
	// Update service from any type to NodePort or LoadBalancer, should update NodePort.
	if service.Spec.Type == api.ServiceTypeNodePort || service.Spec.Type == api.ServiceTypeLoadBalancer {
		if err := updateNodePorts(oldService, service, nodePortOp); err != nil {
			return finish, err
		}
	}
	// Update service from LoadBalancer to non-LoadBalancer, should remove any LoadBalancerStatus.
	if service.Spec.Type != api.ServiceTypeLoadBalancer {
		// Although loadbalancer delete is actually asynchronous, we don't need to expose the user to that complexity.
		service.Status.LoadBalancer = api.LoadBalancerStatus{}
	}

	// Handle ExternalTraffic related updates.
	success, err := r.healthCheckNodePortUpdate(oldService, service, nodePortOp)
	if !success || err != nil {
		return finish, err
	}
	externalTrafficPolicyUpdate(oldService, service)

	//FIXME: move this somewhere more appropriate
	if errs := validation.ValidateServiceExternalTrafficFieldsCombination(service); len(errs) > 0 {
		return finish, errors.NewInvalid(api.Kind("Service"), service.Name, errs)
	}

	return finish, nil
}

func (r *GenericREST) afterDelete(obj runtime.Object) error {
	svc := obj.(*api.Service)

	//FIXME: this is weird.  I am not sure we should do it here, but other
	// options are not obvious.
	// TODO: can leave dangling endpoints, and potentially return incorrect
	// endpoints if a new service is created with the same name
	_, _, err = rs.endpoints.Delete(ctx, id, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
	if err != nil && !errors.IsNotFound(err) {
		return nil, false, err
	}

	r.releaseAllocatedResources(svc)
}
