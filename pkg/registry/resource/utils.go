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
	"reflect"

	resourcev1 "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	v1 "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/resource"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
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

// AuthorizedForDeviceStatus checks if the request status is authorized update the device status
func AuthorizedForDeviceStatus(ctx context.Context, fieldPath *field.Path, authz authorizer.Authorizer, newAllocatedDeviceStatus, oldAllocatedDeviceStatus resource.ResourceClaimStatus, namespaceName string) field.ErrorList {
	var allErrs field.ErrorList
	if fieldPath == nil {
		return append(allErrs, field.InternalError(fieldPath, fmt.Errorf("fieldPath is required for authorization errors")))
	}
	// check the drivers that have changes in status
	driversToAuthz := getModifiedDrivers(newAllocatedDeviceStatus, oldAllocatedDeviceStatus)
	if len(driversToAuthz) == 0 {
		return allErrs
	}

	user, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		return append(allErrs, field.InternalError(fieldPath, fmt.Errorf("cannot determine calling user to check driver status update authorization")))
	}

	nodeName, isNode := nodeidentifier.NewDefaultNodeAssociater().AssociatedNode(user)
	// If the request is from a node, check if the claim is allocated to that node,
	// drivers on nodes are not authorized to update device status for allocations on different nodes.
	if isNode &&
		!isClaimAllocatedNode(newAllocatedDeviceStatus, nodeName) &&
		!isClaimAllocatedNode(oldAllocatedDeviceStatus, nodeName) {
		return append(allErrs, field.Forbidden(fieldPath, fmt.Sprintf("user %q on node %q is not authorized to update device status for drivers on ResourceClaim not allocated to this node", user.GetName(), nodeName)))
	}

	// Check authorization for the specific driver name
	for _, driverName := range driversToAuthz.UnsortedList() {
		// First try the specific permission (on 'drivers/driverName').
		authzAttrs := authorizer.AttributesRecord{
			User:            user,
			Verb:            resourcev1.VerbUpdateDriverStatus,
			Name:            driverName,
			APIGroup:        "resource.k8s.io",
			APIVersion:      "*",
			Resource:        resourcev1.ResourceUpdateDriverStatus,
			ResourceRequest: true,
		}
		decision, _, err := authz.Authorize(ctx, authzAttrs)
		if err != nil {
			allErrs = append(allErrs, field.InternalError(fieldPath, fmt.Errorf("authorization check failed for driver %q: %w", driverName, err)))
			continue
		}

		if decision == authorizer.DecisionAllow {
			continue
		}

		// If the request is from a node, do not fall back to wildcard permission,
		// nodes must have explicit permission for each driver they report status for.
		if isNode {
			msg := fmt.Sprintf("user %q on node %q is not authorized to update device status for driver %q, requires explicit permission for %q on resource %q", user.GetName(), nodeName, driverName, resourcev1.VerbUpdateDriverStatus, resourcev1.ResourceUpdateDriverStatus)
			allErrs = append(allErrs, field.Forbidden(fieldPath, msg))
			continue
		}

		// Next try the wildcard permission (on 'drivers/*'), this is only for non-node requests.
		authzAttrsWildcard := authorizer.AttributesRecord{
			User:            user,
			Verb:            resourcev1.VerbUpdateDriverStatus,
			Name:            "*",
			APIGroup:        "resource.k8s.io",
			APIVersion:      "*",
			Resource:        resourcev1.ResourceUpdateDriverStatus,
			ResourceRequest: true,
		}

		decision, _, err = authz.Authorize(ctx, authzAttrsWildcard)
		if err != nil {
			allErrs = append(allErrs, field.InternalError(fieldPath, fmt.Errorf("authorization check failed for driver %q with wildcard: %w", driverName, err)))
			continue
		}

		if decision != authorizer.DecisionAllow {
			msg := fmt.Sprintf("user %q is not authorized to update device status for driver %q, requires permission for %q on resource %q", user.GetName(), driverName, resourcev1.VerbUpdateDriverStatus, resourcev1.ResourceUpdateDriverStatus)
			allErrs = append(allErrs, field.Forbidden(fieldPath, msg))
			continue
		}
	}

	return allErrs
}

// getModifiedDrivers identifies all drivers whose status entries were added,
// removed, or changed between the old and new ResourceClaim objects.
func getModifiedDrivers(newAllocatedDeviceStatus, oldAllocatedDeviceStatus resource.ResourceClaimStatus) sets.Set[string] {
	driversToAuthz := sets.Set[string]{}

	oldDevices := make(map[string]resource.AllocatedDeviceStatus)
	for _, d := range oldAllocatedDeviceStatus.Devices {
		oldDevices[deviceKey(d)] = d
	}

	// Check for new or modified device entries
	for _, d := range newAllocatedDeviceStatus.Devices {
		key := deviceKey(d)
		oldDevice, ok := oldDevices[key]
		delete(oldDevices, key) // Remove from map to track processed devices

		// If entry is new or changed, we need to authorize this driver.
		if !ok || !reflect.DeepEqual(oldDevice, d) {
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

func isClaimAllocatedNode(status resource.ResourceClaimStatus, nodeName string) bool {
	if status.Allocation == nil || status.Allocation.NodeSelector == nil {
		return false
	}
	return nodeSelectorMatches(*status.Allocation.NodeSelector, nodeName)
}

// nodeSelectorMatches checks if NodeSelector matches the given nodeName in metadata.name.
// This is a convention over the NodeSelector structure applied by the scheduler.
func nodeSelectorMatches(nodeSelector core.NodeSelector, nodeName string) bool {
	for _, term := range nodeSelector.NodeSelectorTerms {
		for _, field := range term.MatchFields {
			if field.Key == "metadata.name" && field.Operator == "In" {
				for _, value := range field.Values {
					if value == nodeName {
						return true
					}
				}
			}
		}
		for _, expr := range term.MatchExpressions {
			if expr.Key == "kubernetes.io/hostname" && expr.Operator == "In" {
				for _, value := range expr.Values {
					if value == nodeName {
						return true
					}
				}
			}
		}
	}
	return false
}

func deviceKey(d resource.AllocatedDeviceStatus) string {
	return d.Driver + "/" + d.Pool + "/" + d.Device
}
