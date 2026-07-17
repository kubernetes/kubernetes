/*
Copyright The Kubernetes Authors.

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

package v1alpha1

import (
	"context"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/validate"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func ValidateCustom_Device_Attributes(
	ctx context.Context, op operation.Operation, fldPath *field.Path,
	obj, oldObj map[resourceapi.QualifiedName]resourceapi.DeviceAttribute,
) field.ErrorList {
	return validate.EachMapVal(ctx, op, fldPath, obj, oldObj, validate.SemanticDeepEqual, resourceapi.Validate_DeviceAttribute)
}

func ValidateCustom_Device_NetworkData(
	ctx context.Context, op operation.Operation, fldPath *field.Path,
	obj, oldObj *resourceapi.NetworkDeviceData,
) field.ErrorList {
	return resourceapi.Validate_NetworkDeviceData(ctx, op, fldPath, obj, oldObj)
}
