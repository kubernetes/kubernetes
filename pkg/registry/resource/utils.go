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

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	v1 "k8s.io/client-go/kubernetes/typed/core/v1"
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
		value := deviceRequests[i].AdminAccess
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
