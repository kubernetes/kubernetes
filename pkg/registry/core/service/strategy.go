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

	"k8s.io/apimachinery/pkg/runtime"
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

	if utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) && service.Spec.IPFamily == nil {
		family := strategy.ipFamilies[0]
		service.Spec.IPFamily = &family
	}

	dropServiceDisabledFields(service, nil)
}

// PrepareForUpdate sets contextual defaults and clears fields that are not allowed to be set by end users on update.
func (strategy svcStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newService := obj.(*api.Service)
	oldService := old.(*api.Service)
	newService.Status = oldService.Status

	if utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) && newService.Spec.IPFamily == nil {
		if oldService.Spec.IPFamily != nil {
			newService.Spec.IPFamily = oldService.Spec.IPFamily
		} else {
			family := strategy.ipFamilies[0]
			newService.Spec.IPFamily = &family
		}
	}

	dropServiceDisabledFields(newService, oldService)
}

// Validate validates a new service.
func (strategy svcStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	service := obj.(*api.Service)
	allErrs := validation.ValidateServiceCreate(service)
	allErrs = append(allErrs, validation.ValidateConditionalService(service, nil, strategy.ipFamilies)...)
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
	allErrs = append(allErrs, validation.ValidateConditionalService(obj.(*api.Service), old.(*api.Service), strategy.ipFamilies)...)
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
	if t.Spec.ClusterIP != api.ClusterIPNone {
		t.Spec.ClusterIP = ""
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
	// Drop IPFamily if DualStack is not enabled
	if !utilfeature.DefaultFeatureGate.Enabled(features.IPv6DualStack) && !serviceIPFamilyInUse(oldSvc) {
		newSvc.Spec.IPFamily = nil
	}

	// Drop TopologyKeys if ServiceTopology is not enabled
	if !utilfeature.DefaultFeatureGate.Enabled(features.ServiceTopology) && !topologyKeysInUse(oldSvc) {
		newSvc.Spec.TopologyKeys = nil
	}
}

// returns true if svc.Spec.ServiceIPFamily field is in use
func serviceIPFamilyInUse(svc *api.Service) bool {
	if svc == nil {
		return false
	}
	if svc.Spec.IPFamily != nil {
		return true
	}
	return false
}

// returns true if svc.Spec.TopologyKeys field is in use
func topologyKeysInUse(svc *api.Service) bool {
	if svc == nil {
		return false
	}
	return len(svc.Spec.TopologyKeys) > 0
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
