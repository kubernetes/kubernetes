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

package service

import (
	"context"
	"fmt"
	"net"
	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/features"
	netutil "k8s.io/utils/net"
)

type Strategy interface {
	rest.RESTCreateUpdateStrategy
	rest.RESTExportStrategy
}

// svcStrategy implements behavior for Services
type svcStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator

	ipFamilies []api.IPFamily
}

// StrategyForServiceCIDRs returns the appropriate service strategy for the given configuration.
func StrategyForServiceCIDRs(primaryCIDR net.IPNet, hasSecondary bool) (Strategy, api.IPFamily) {
	// detect this cluster default Service IPFamily (ipfamily of --service-cluster-ip-range)
	// we do it once here, to avoid having to do it over and over during ipfamily assignment
	serviceIPFamily := api.IPv4Protocol
	if netutil.IsIPv6CIDR(&primaryCIDR) {
		serviceIPFamily = api.IPv6Protocol
	}

	var strategy Strategy
	switch {
	case hasSecondary && serviceIPFamily == api.IPv4Protocol:
		strategy = svcStrategy{
			ObjectTyper:   legacyscheme.Scheme,
			NameGenerator: names.SimpleNameGenerator,
			ipFamilies:    []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		}
	case hasSecondary && serviceIPFamily == api.IPv6Protocol:
		strategy = svcStrategy{
			ObjectTyper:   legacyscheme.Scheme,
			NameGenerator: names.SimpleNameGenerator,
			ipFamilies:    []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
		}
	case serviceIPFamily == api.IPv6Protocol:
		strategy = svcStrategy{
			ObjectTyper:   legacyscheme.Scheme,
			NameGenerator: names.SimpleNameGenerator,
			ipFamilies:    []api.IPFamily{api.IPv6Protocol},
		}
	default:
		strategy = svcStrategy{
			ObjectTyper:   legacyscheme.Scheme,
			NameGenerator: names.SimpleNameGenerator,
			ipFamilies:    []api.IPFamily{api.IPv4Protocol},
		}
	}
	return strategy, serviceIPFamily
}

// NamespaceScoped is true for services.
func (svcStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate sets contextual defaults and clears fields that are not allowed to be set by end users on creation.
func (strategy svcStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	service := obj.(*api.Service)
	service.Status = api.ServiceStatus{}

	NormalizeClusterIPs(nil, service)
	dropServiceDisabledFields(service, nil)
}

// PrepareForUpdate sets contextual defaults and clears fields that are not allowed to be set by end users on update.
func (strategy svcStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newService := obj.(*api.Service)
	oldService := old.(*api.Service)
	newService.Status = oldService.Status

	NormalizeClusterIPs(oldService, newService)
	dropServiceDisabledFields(newService, oldService)
	dropTypeDependentFields(newService, oldService)
	trimFieldsForDualStackDowngrade(newService, oldService)
}

// Validate validates a new service.
func (strategy svcStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	service := obj.(*api.Service)
	allErrs := validation.ValidateServiceCreate(service)
	return allErrs
}

// Canonicalize normalizes the object after validation.
func (svcStrategy) Canonicalize(obj runtime.Object) {
}

func (svcStrategy) AllowCreateOnUpdate() bool {
	return true
}

func (strategy svcStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	allErrs := validation.ValidateServiceUpdate(obj.(*api.Service), old.(*api.Service))
	return allErrs
}

func (svcStrategy) AllowUnconditionalUpdate() bool {
	return true
}

func (svcStrategy) Export(ctx context.Context, obj runtime.Object, exact bool) error {
	t, ok := obj.(*api.Service)
	if !ok {
		// unexpected programmer error
		return fmt.Errorf("unexpected object: %v", obj)
	}
	// TODO: service does not yet have a prepare create strategy (see above)
	t.Status = api.ServiceStatus{}
	if exact {
		return nil
	}
	//set ClusterIPs as nil - if ClusterIPs[0] != None
	if len(t.Spec.ClusterIPs) > 0 && t.Spec.ClusterIPs[0] != api.ClusterIPNone {
		t.Spec.ClusterIP = ""
		t.Spec.ClusterIPs = nil
	}
	if t.Spec.Type == api.ServiceTypeNodePort {
		for i := range t.Spec.Ports {
			t.Spec.Ports[i].NodePort = 0
		}
	}
	return nil
}

// dropServiceDisabledFields drops fields that are not used if their associated feature gates
// are not enabled.  The typical pattern is:
//     if !utilfeature.DefaultFeatureGate.Enabled(features.MyFeature) && !myFeatureInUse(oldSvc) {
//         newSvc.Spec.MyFeature = nil
//     }
func dropServiceDisabledFields(newSvc *api.Service, oldSvc *api.Service) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) && !serviceDualStackFieldsInUse(oldSvc) {
		newSvc.Spec.IPFamilies = nil
		newSvc.Spec.IPFamilyPolicy = nil
		if len(newSvc.Spec.ClusterIPs) > 1 {
			newSvc.Spec.ClusterIPs = newSvc.Spec.ClusterIPs[0:1]
		}
	}

	// Drop TopologyKeys if ServiceTopology is not enabled
	if !utilfeature.DefaultFeatureGate.Enabled(features.ServiceTopology) && !topologyKeysInUse(oldSvc) {
		newSvc.Spec.TopologyKeys = nil
	}

	if !utilfeature.DefaultFeatureGate.Enabled(features.LoadBalancerIPMode) && !loadbalancerIPModeInUse(oldSvc) {
		for _, ing := range newSvc.Status.LoadBalancer.Ingress {
			ing.IPMode = nil
		}
	}
}

// returns true if svc.Spec.ServiceIPFamily field is in use
func serviceDualStackFieldsInUse(svc *api.Service) bool {
	if svc == nil {
		return false
	}

	ipFamilyPolicyInUse := svc.Spec.IPFamilyPolicy != nil
	ipFamiliesInUse := len(svc.Spec.IPFamilies) > 0
	ClusterIPsInUse := len(svc.Spec.ClusterIPs) > 1

	return ipFamilyPolicyInUse || ipFamiliesInUse || ClusterIPsInUse
}

// returns true if svc.Spec.TopologyKeys field is in use
func topologyKeysInUse(svc *api.Service) bool {
	if svc == nil {
		return false
	}
	return len(svc.Spec.TopologyKeys) > 0
}

// returns true when the LoadBalancer Ingress IPMode fields are in use.
func loadbalancerIPModeInUse(svc *api.Service) bool {
	if svc == nil {
		return false
	}
	for _, ing := range svc.Status.LoadBalancer.Ingress {
		if ing.IPMode != nil {
			return true
		}
	}
	return false
}

type serviceStatusStrategy struct {
	Strategy
}

// NewServiceStatusStrategy creates a status strategy for the provided base strategy.
func NewServiceStatusStrategy(strategy Strategy) Strategy {
	return serviceStatusStrategy{strategy}
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update of status
func (serviceStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newService := obj.(*api.Service)
	oldService := old.(*api.Service)
	// status changes are not allowed to update spec
	newService.Spec = oldService.Spec
}

// ValidateUpdate is the default update validation for an end user updating status
func (serviceStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidateServiceStatusUpdate(obj.(*api.Service), old.(*api.Service))
}

// NormalizeClusterIPs adjust clusterIPs based on ClusterIP.  This must not
// consider any other fields.
func NormalizeClusterIPs(oldSvc, newSvc *api.Service) {
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
			if sameStringSlice(oldSvc.Spec.ClusterIPs, newSvc.Spec.ClusterIPs) {
				newSvc.Spec.ClusterIPs = nil
			}

			// if they provided nil, then we are fine (handled by patching case above)
			// if they changed it then validation will catch it
		} else {
			// ClusterIP has changed but not cleared *and* ClusterIPs are the same
			// then we set ClusterIPs based on ClusterIP
			if sameStringSlice(oldSvc.Spec.ClusterIPs, newSvc.Spec.ClusterIPs) {
				newSvc.Spec.ClusterIPs = []string{newSvc.Spec.ClusterIP}
			}
		}
	}
}

func sameStringSlice(a []string, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// This is an unusual case.  Service has a number of inter-related fields and
// in order to avoid breaking clients we try really hard to infer what users
// mean when they change them.
//
// Services are effectively a discriminated union, where `type` is the
// discriminator. Some fields just don't make sense with some types, so we
// clear them.
//
// As a rule, we almost never change user input.  This can get tricky when APIs
// evolve and new dependent fields are added.  This specific case includes
// fields that are allocated from a pool and need to be released.  Anyone who
// is contemplating copying this pattern should think REALLY hard about almost
// any other option.
func dropTypeDependentFields(newSvc *api.Service, oldSvc *api.Service) {
	// For now we are only wiping on updates.  This minimizes potential
	// confusion since many of the cases we are handling here are pretty niche.
	if oldSvc == nil {
		return
	}

	// In all of these cases we only want to wipe a field if we a) know it no
	// longer applies; b) might have initialized it automatically; c) know the
	// user did not ALSO try to change it (in which case it should fail in
	// validation).

	// If the user is switching to a type that does not need a value in
	// clusterIP/clusterIPs (even "None" counts as a value), we might be able
	// to wipe some fields.
	if needsClusterIP(oldSvc) && !needsClusterIP(newSvc) {
		if sameClusterIPs(oldSvc, newSvc) {
			// These will be deallocated later.
			newSvc.Spec.ClusterIP = ""
			newSvc.Spec.ClusterIPs = nil
		}
		if sameIPFamilies(oldSvc, newSvc) {
			newSvc.Spec.IPFamilies = nil
		}
		if sameIPFamilyPolicy(oldSvc, newSvc) {
			newSvc.Spec.IPFamilyPolicy = nil
		}
	}

	// If the user is switching to a type that doesn't use NodePorts AND they
	// did not change any NodePort values, we can wipe them.  They will be
	// deallocated later.
	if needsNodePort(oldSvc) && !needsNodePort(newSvc) && sameNodePorts(oldSvc, newSvc) {
		for i := range newSvc.Spec.Ports {
			newSvc.Spec.Ports[i].NodePort = 0
		}
	}

	// If the user is switching to a case that doesn't use HealthCheckNodePort AND they
	// did not change the HealthCheckNodePort value, we can wipe it.  It will
	// be deallocated later.
	if needsHCNodePort(oldSvc) && !needsHCNodePort(newSvc) && sameHCNodePort(oldSvc, newSvc) {
		newSvc.Spec.HealthCheckNodePort = 0
	}

	// NOTE: there are other fields like `selector` which we could wipe.
	// Historically we did not wipe them and they are not allocated from
	// finite pools, so we are (currently) choosing to leave them alone.
}

func needsClusterIP(svc *api.Service) bool {
	if svc.Spec.Type == api.ServiceTypeExternalName {
		return false
	}
	return true
}

func sameClusterIPs(oldSvc, newSvc *api.Service) bool {
	sameSingular := oldSvc.Spec.ClusterIP == newSvc.Spec.ClusterIP
	samePlural := sameStringSlice(oldSvc.Spec.ClusterIPs, newSvc.Spec.ClusterIPs)
	return sameSingular && samePlural
}

func sameIPFamilies(oldSvc, newSvc *api.Service) bool {
	return reflect.DeepEqual(oldSvc.Spec.IPFamilies, newSvc.Spec.IPFamilies)
}

func getIPFamilyPolicy(svc *api.Service) string {
	if svc.Spec.IPFamilyPolicy == nil {
		return ""
	}
	return string(*svc.Spec.IPFamilyPolicy)
}

func sameIPFamilyPolicy(oldSvc, newSvc *api.Service) bool {
	return getIPFamilyPolicy(oldSvc) == getIPFamilyPolicy(newSvc)
}

func needsNodePort(svc *api.Service) bool {
	if svc.Spec.Type == api.ServiceTypeNodePort || svc.Spec.Type == api.ServiceTypeLoadBalancer {
		return true
	}
	return false
}

func sameNodePorts(oldSvc, newSvc *api.Service) bool {
	// Helper to make a set of NodePort values.
	allNodePorts := func(svc *api.Service) sets.Int {
		out := sets.NewInt()
		for i := range svc.Spec.Ports {
			if svc.Spec.Ports[i].NodePort != 0 {
				out.Insert(int(svc.Spec.Ports[i].NodePort))
			}
		}
		return out
	}

	oldPorts := allNodePorts(oldSvc)
	newPorts := allNodePorts(newSvc)

	// Users can add, remove, or modify ports, as long as they don't add any
	// net-new NodePorts.
	return oldPorts.IsSuperset(newPorts)
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

func sameHCNodePort(oldSvc, newSvc *api.Service) bool {
	return oldSvc.Spec.HealthCheckNodePort == newSvc.Spec.HealthCheckNodePort
}

// this func allows user to downgrade a service by just changing
// IPFamilyPolicy to SingleStack
func trimFieldsForDualStackDowngrade(newService, oldService *api.Service) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) {
		return
	}

	// not an update
	if oldService == nil {
		return
	}

	oldIsDualStack := oldService.Spec.IPFamilyPolicy != nil &&
		(*oldService.Spec.IPFamilyPolicy == api.IPFamilyPolicyRequireDualStack ||
			*oldService.Spec.IPFamilyPolicy == api.IPFamilyPolicyPreferDualStack)

	newIsNotDualStack := newService.Spec.IPFamilyPolicy != nil && *newService.Spec.IPFamilyPolicy == api.IPFamilyPolicySingleStack

	// if user want to downgrade then we auto remove secondary ip and family
	if oldIsDualStack && newIsNotDualStack {
		if len(newService.Spec.ClusterIPs) > 1 {
			newService.Spec.ClusterIPs = newService.Spec.ClusterIPs[0:1]
		}

		if len(newService.Spec.IPFamilies) > 1 {
			newService.Spec.IPFamilies = newService.Spec.IPFamilies[0:1]
		}
	}
}
