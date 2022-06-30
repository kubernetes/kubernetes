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
	"reflect"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/features"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
)

// svcStrategy implements behavior for Services
type svcStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Services
// objects via the REST API.
var Strategy = svcStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped is true for services.
func (svcStrategy) NamespaceScoped() bool {
	return true
}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (svcStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("status"),
		),
	}

	return fields
}

// PrepareForCreate sets contextual defaults and clears fields that are not allowed to be set by end users on creation.
func (svcStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	service := obj.(*api.Service)
	service.Status = api.ServiceStatus{}

	dropServiceDisabledFields(service, nil)
}

// PrepareForUpdate sets contextual defaults and clears fields that are not allowed to be set by end users on update.
func (svcStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newService := obj.(*api.Service)
	oldService := old.(*api.Service)
	newService.Status = oldService.Status

	dropServiceDisabledFields(newService, oldService)
	dropTypeDependentFields(newService, oldService)
}

// Validate validates a new service.
func (svcStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	service := obj.(*api.Service)
	allErrs := validation.ValidateServiceCreate(service)
	allErrs = append(allErrs, validation.ValidateConditionalService(service, nil)...)
	return allErrs
}

// WarningsOnCreate returns warnings for the creation of the given object.
func (svcStrategy) WarningsOnCreate(ctx context.Context, obj runtime.Object) []string { return nil }

// Canonicalize normalizes the object after validation.
func (svcStrategy) Canonicalize(obj runtime.Object) {
}

func (svcStrategy) AllowCreateOnUpdate() bool {
	return true
}

func (strategy svcStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	allErrs := validation.ValidateServiceUpdate(obj.(*api.Service), old.(*api.Service))
	allErrs = append(allErrs, validation.ValidateConditionalService(obj.(*api.Service), old.(*api.Service))...)
	return allErrs
}

// WarningsOnUpdate returns warnings for the given update.
func (svcStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
}

func (svcStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// dropServiceDisabledFields drops fields that are not used if their associated feature gates
// are not enabled.  The typical pattern is:
//     if !utilfeature.DefaultFeatureGate.Enabled(features.MyFeature) && !myFeatureInUse(oldSvc) {
//         newSvc.Spec.MyFeature = nil
//     }
func dropServiceDisabledFields(newSvc *api.Service, oldSvc *api.Service) {

	if !utilfeature.DefaultFeatureGate.Enabled(features.MixedProtocolLBService) {
		if !serviceConditionsInUse(oldSvc) {
			newSvc.Status.Conditions = nil
		}
		if !loadBalancerPortsInUse(oldSvc) {
			for i := range newSvc.Status.LoadBalancer.Ingress {
				newSvc.Status.LoadBalancer.Ingress[i].Ports = nil
			}
		}
	}

	// Clear InternalTrafficPolicy if not enabled
	if !utilfeature.DefaultFeatureGate.Enabled(features.ServiceInternalTrafficPolicy) {
		if !serviceInternalTrafficPolicyInUse(oldSvc) {
			newSvc.Spec.InternalTrafficPolicy = nil
		}
	}
}

// returns true when the svc.Status.Conditions field is in use.
func serviceConditionsInUse(svc *api.Service) bool {
	if svc == nil {
		return false
	}
	return svc.Status.Conditions != nil
}

// returns true when the svc.Status.LoadBalancer.Ingress.Ports field is in use.
func loadBalancerPortsInUse(svc *api.Service) bool {
	if svc == nil {
		return false
	}
	for _, ing := range svc.Status.LoadBalancer.Ingress {
		if ing.Ports != nil {
			return true
		}
	}
	return false
}

func serviceInternalTrafficPolicyInUse(svc *api.Service) bool {
	if svc == nil {
		return false
	}
	return svc.Spec.InternalTrafficPolicy != nil
}

type serviceStatusStrategy struct {
	svcStrategy
}

// StatusStrategy wraps and exports the used svcStrategy for the storage package.
var StatusStrategy = serviceStatusStrategy{Strategy}

// GetResetFields returns the set of fields that get reset by the strategy
// and should not be modified by the user.
func (serviceStatusStrategy) GetResetFields() map[fieldpath.APIVersion]*fieldpath.Set {
	fields := map[fieldpath.APIVersion]*fieldpath.Set{
		"v1": fieldpath.NewSet(
			fieldpath.MakePathOrDie("spec"),
		),
	}

	return fields
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

// WarningsOnUpdate returns warnings for the given update.
func (serviceStatusStrategy) WarningsOnUpdate(ctx context.Context, obj, old runtime.Object) []string {
	return nil
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

	// If a user is switching to a type that doesn't need allocatedLoadBalancerNodePorts AND they did not change
	// this field, it is safe to drop it.
	if oldSvc.Spec.Type == api.ServiceTypeLoadBalancer && newSvc.Spec.Type != api.ServiceTypeLoadBalancer {
		if newSvc.Spec.AllocateLoadBalancerNodePorts != nil && oldSvc.Spec.AllocateLoadBalancerNodePorts != nil {
			if *oldSvc.Spec.AllocateLoadBalancerNodePorts == *newSvc.Spec.AllocateLoadBalancerNodePorts {
				newSvc.Spec.AllocateLoadBalancerNodePorts = nil
			}
		}
	}

	// If a user is switching to a type that doesn't need LoadBalancerClass AND they did not change
	// this field, it is safe to drop it.
	if canSetLoadBalancerClass(oldSvc) && !canSetLoadBalancerClass(newSvc) && sameLoadBalancerClass(oldSvc, newSvc) {
		newSvc.Spec.LoadBalancerClass = nil
	}

	// If a user is switching to a type that doesn't need ExternalTrafficPolicy
	// AND they did not change this field, it is safe to drop it.
	if needsExternalTrafficPolicy(oldSvc) && !needsExternalTrafficPolicy(newSvc) && sameExternalTrafficPolicy(oldSvc, newSvc) {
		newSvc.Spec.ExternalTrafficPolicy = api.ServiceExternalTrafficPolicyType("")
	}

	// NOTE: there are other fields like `selector` which we could wipe.
	// Historically we did not wipe them and they are not allocated from
	// finite pools, so we are (currently) choosing to leave them alone.

	// Clear the load-balancer status if it is no longer appropriate.  Although
	// LB de-provisioning is actually asynchronous, we don't need to expose the
	// user to that complexity.
	if newSvc.Spec.Type != api.ServiceTypeLoadBalancer {
		newSvc.Status.LoadBalancer = api.LoadBalancerStatus{}
	}
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

func canSetLoadBalancerClass(svc *api.Service) bool {
	return svc.Spec.Type == api.ServiceTypeLoadBalancer
}

func sameLoadBalancerClass(oldSvc, newSvc *api.Service) bool {
	if (oldSvc.Spec.LoadBalancerClass == nil) != (newSvc.Spec.LoadBalancerClass == nil) {
		return false
	}
	if oldSvc.Spec.LoadBalancerClass == nil {
		return true // both are nil
	}
	return *oldSvc.Spec.LoadBalancerClass == *newSvc.Spec.LoadBalancerClass
}

func needsExternalTrafficPolicy(svc *api.Service) bool {
	return svc.Spec.Type == api.ServiceTypeNodePort || svc.Spec.Type == api.ServiceTypeLoadBalancer
}

func sameExternalTrafficPolicy(oldSvc, newSvc *api.Service) bool {
	return oldSvc.Spec.ExternalTrafficPolicy == newSvc.Spec.ExternalTrafficPolicy
}
