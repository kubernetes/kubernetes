/*
Copyright 2025 The Kubernetes Authors.

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

package resource

import (
	"context"
	"fmt"

	resourcev1 "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/endpoints/filters"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	v1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/apis/resource"
)

// AuthorizedForAdmin checks if the request is authorized to get admin access to devices
// based on namespace label
func AuthorizedForAdmin(ctx context.Context, deviceRequests []resource.DeviceRequest, namespaceName string, nsClient v1.NamespaceInterface) field.ErrorList {
	var allErrs field.ErrorList
	adminRequested := false
	var adminAccessPath *field.Path

	// no need to check old request since spec is immutable

	for i := range deviceRequests {
		// AdminAccess can not be set on subrequests, so it can
		// only be used when the Exactly field is set.
		if deviceRequests[i].Exactly == nil {
			continue
		}
		value := deviceRequests[i].Exactly.AdminAccess
		if value != nil && *value {
			adminRequested = true
			adminAccessPath = field.NewPath("spec", "devices", "requests").Index(i).Child("adminAccess")
			break
		}
	}
	if !adminRequested {
		// No need to validate unless admin access is requested
		return allErrs
	}

	// Retrieve the namespace object from the store
	ns, err := nsClient.Get(ctx, namespaceName, metav1.GetOptions{})
	if err != nil {
		return append(allErrs, field.InternalError(adminAccessPath, fmt.Errorf("could not retrieve namespace to verify admin access: %w", err)))
	}
	if ns.Labels[resource.DRAAdminNamespaceLabelKey] != "true" {
		return append(allErrs, field.Forbidden(adminAccessPath, fmt.Sprintf("admin access to devices requires the `%s: true` label on the containing namespace", resource.DRAAdminNamespaceLabelKey)))
	}

	return allErrs
}

// AuthorizedForAdminStatus checks if the request status is authorized to get admin access to devices
// based on namespace label
func AuthorizedForAdminStatus(ctx context.Context, newAllocationResult, oldAllocationResult []resource.DeviceRequestAllocationResult, namespaceName string, nsClient v1.NamespaceInterface) field.ErrorList {
	var allErrs field.ErrorList
	var adminAccessPath *field.Path

	if wasGranted, _ := adminRequested(oldAllocationResult); wasGranted {
		// No need to validate if old status has admin access granted, since status.Allocation is immutable
		return allErrs
	}
	isRequested, adminAccessPath := adminRequested(newAllocationResult)
	if !isRequested {
		// No need to validate unless admin access is requested
		return allErrs
	}

	// Retrieve the namespace object from the store
	ns, err := nsClient.Get(ctx, namespaceName, metav1.GetOptions{})
	if err != nil {
		return append(allErrs, field.InternalError(adminAccessPath, fmt.Errorf("could not retrieve namespace to verify admin access: %w", err)))
	}
	if ns.Labels[resource.DRAAdminNamespaceLabelKey] != "true" {
		return append(allErrs, field.Forbidden(adminAccessPath, fmt.Sprintf("admin access to devices requires the `%s: true` label on the containing namespace", resource.DRAAdminNamespaceLabelKey)))
	}

	return allErrs
}

func adminRequested(deviceRequestResults []resource.DeviceRequestAllocationResult) (bool, *field.Path) {
	for i := range deviceRequestResults {
		value := deviceRequestResults[i].AdminAccess
		if value != nil && *value {
			return true, field.NewPath("status", "allocation", "devices", "results").Index(i).Child("adminAccess")
		}
	}
	return false, nil
}

// AuthorizedForBinding checks if the caller is authorized to update
// status.allocation and status.reservedFor by verifying permission on the
// synthetic resourceclaims/binding subresource.
func AuthorizedForBinding(ctx context.Context, fieldPath *field.Path, authz authorizer.Authorizer, newStatus, oldStatus resource.ResourceClaimStatus) field.ErrorList {
	var allErrs field.ErrorList

	if equality.Semantic.DeepEqual(newStatus.Allocation, oldStatus.Allocation) &&
		equality.Semantic.DeepEqual(newStatus.ReservedFor, oldStatus.ReservedFor) {
		return allErrs
	}

	baseAttrs, err := filters.GetAuthorizerAttributes(ctx)
	if err != nil {
		return append(allErrs, field.InternalError(fieldPath, fmt.Errorf("cannot build authorizer attributes: %w", err)))
	}

	attrs := &syntheticSubresourceAttrs{
		Attributes:  baseAttrs,
		verb:        baseAttrs.GetVerb(),           // verb is unchanged but must be specified
		subresource: resourcev1.SubresourceBinding, // the scheduler calls this "bind claim"
		namespace:   "",                            // cluster-wide
		name:        "",                            // all names
	}

	if err := checkAuthorization(ctx, authz, attrs); err != nil {
		return append(allErrs, field.Forbidden(fieldPath, fmt.Sprintf(`changing status.allocation or status.reservedFor requires resource="resourceclaims/binding", verb="%s" permission: %s`, attrs.verb, err)))
	}
	return allErrs
}

// AuthorizedForDeviceStatus checks if the caller is authorized to update
// status.devices by performing per-driver authorization checks using the
// associated-node / arbitrary-node verb prefix pattern on the synthetic
// resourceclaims/driver subresource.
func AuthorizedForDeviceStatus(ctx context.Context, fieldPath *field.Path, a authorizer.Authorizer, newStatus, oldStatus resource.ResourceClaimStatus) field.ErrorList {
	var allErrs field.ErrorList

	driversToAuthz := getModifiedDrivers(newStatus, oldStatus)
	if len(driversToAuthz) == 0 {
		return allErrs
	}

	baseAttrs, err := filters.GetAuthorizerAttributes(ctx)
	if err != nil {
		return append(allErrs, field.InternalError(fieldPath, fmt.Errorf("cannot build authorizer attributes: %w", err)))
	}

	// if service account is on the same node as the claim, check associated-node verb first, fall back to arbitrary-node
	// Otherwise, only try arbitrary-node
	requestVerb := baseAttrs.GetVerb()
	var verbs []string
	if saAssociatedWithAllocatedNode(baseAttrs.GetUser(), nodeNameFromAllocation(newStatus.Allocation)) {
		verbs = []string{resourcev1.VerbPrefixAssociatedNode + requestVerb, resourcev1.VerbPrefixArbitraryNode + requestVerb}
	} else {
		verbs = []string{resourcev1.VerbPrefixArbitraryNode + requestVerb}
	}

	for _, driverName := range sets.List(driversToAuthz) {
		if err := checkDriverAuthorization(ctx, baseAttrs, verbs, driverName, a); err != nil {
			allErrs = append(allErrs, field.Forbidden(fieldPath, fmt.Sprintf(`changing status.devices requires resource="resourceclaims/driver", verb="%s" permission: %s`, verbs, err)))
		}

	}

	return allErrs
}

func checkDriverAuthorization(ctx context.Context, baseAttrs authorizer.Attributes, verbs []string, driverName string, a authorizer.Authorizer) error {
	if len(verbs) == 0 {
		return fmt.Errorf("no verbs set for driver %s", driverName) // impossible for all inputs today
	}

	var firstErr error
	for _, verb := range verbs { // verbs are OR'd with each other
		attrs := &syntheticSubresourceAttrs{
			Attributes:  baseAttrs,
			verb:        verb,
			subresource: resourcev1.SubresourceDriver,
			namespace:   baseAttrs.GetNamespace(),
			name:        driverName,
		}
		err := checkAuthorization(ctx, a, attrs)
		if err == nil {
			return nil
		}
		if firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// getModifiedDrivers identifies all drivers whose status entries were added,
// removed, or changed between the old and new ResourceClaim objects.
func getModifiedDrivers(newAllocatedDeviceStatus, oldAllocatedDeviceStatus resource.ResourceClaimStatus) sets.Set[string] {
	driversToAuthz := sets.Set[string]{}

	oldDevices := make(map[deviceKey]resource.AllocatedDeviceStatus)
	for _, d := range oldAllocatedDeviceStatus.Devices {
		oldDevices[makeDeviceKey(d)] = d
	}

	// Check for new or modified device entries
	for _, d := range newAllocatedDeviceStatus.Devices {
		key := makeDeviceKey(d)
		oldDevice, ok := oldDevices[key]
		delete(oldDevices, key) // Remove from map to track processed devices

		// If entry is new or changed, we need to authorize this driver.
		if !ok || !equality.Semantic.DeepEqual(oldDevice, d) {
			driversToAuthz.Insert(d.Driver)
		}
	}

	// Check for removed device entries
	for _, d := range oldDevices {
		// Any remaining device in oldDevices was removed in rcNew.
		driversToAuthz.Insert(d.Driver)
	}

	return driversToAuthz
}

type deviceKey struct {
	driver  string
	pool    string
	device  string
	shareID string
}

func makeDeviceKey(d resource.AllocatedDeviceStatus) deviceKey {
	key := deviceKey{
		driver: d.Driver,
		pool:   d.Pool,
		device: d.Device,
	}
	if d.ShareID != nil {
		key.shareID = *d.ShareID
	}
	return key
}

func nodeNameFromAllocation(allocation *resource.AllocationResult) string {
	if allocation == nil || allocation.NodeSelector == nil {
		return ""
	}
	ns := allocation.NodeSelector
	if len(ns.NodeSelectorTerms) != 1 {
		return ""
	}
	term := ns.NodeSelectorTerms[0]
	if len(term.MatchExpressions) != 0 || len(term.MatchFields) != 1 {
		return ""
	}
	f := term.MatchFields[0]
	if f.Key != "metadata.name" || f.Operator != core.NodeSelectorOpIn || len(f.Values) != 1 {
		return ""
	}
	return f.Values[0]
}

func saAssociatedWithAllocatedNode(u user.Info, allocatedNodeName string) bool {
	if len(allocatedNodeName) == 0 {
		return false
	}

	// Must be a ServiceAccount
	if _, _, err := serviceaccount.SplitUsername(u.GetName()); err != nil {
		return false
	}

	// Must have exactly one node-name extra attribute
	nodeNames := u.GetExtra()[serviceaccount.NodeNameKey]
	if len(nodeNames) != 1 {
		return false
	}
	nodeName := nodeNames[0]

	// Must be a valid node name format
	if len(validation.ValidateNodeName(nodeName, false)) != 0 {
		return false
	}

	return nodeName == allocatedNodeName
}

func checkAuthorization(ctx context.Context, a authorizer.Authorizer, attributes authorizer.Attributes) error {
	authorized, reason, err := a.Authorize(ctx, attributes)

	// an authorizer like RBAC could encounter evaluation errors and still allow the request, so authorizer decision is checked before error here.
	if authorized == authorizer.DecisionAllow {
		return nil
	}

	msg := reason
	switch {
	case err != nil && len(reason) > 0:
		msg = fmt.Sprintf("%v: %s", err, reason)
	case err != nil:
		msg = err.Error()
	}

	return responsewriters.ForbiddenStatusError(attributes, msg)
}

type syntheticSubresourceAttrs struct {
	authorizer.Attributes
	verb        string
	subresource string
	namespace   string
	name        string
}

func (a *syntheticSubresourceAttrs) GetVerb() string        { return a.verb }
func (a *syntheticSubresourceAttrs) GetSubresource() string { return a.subresource }
func (a *syntheticSubresourceAttrs) GetNamespace() string   { return a.namespace }
func (a *syntheticSubresourceAttrs) GetName() string        { return a.name }
