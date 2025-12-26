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

package validation

import (
	"context"

	extensionsv1beta1 "k8s.io/api/extensions/v1beta1"
	"k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/operation"
	"k8s.io/apimachinery/pkg/api/safe"
	"k8s.io/apimachinery/pkg/api/validate"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

// ValidateExtensionsIPBlock validates an instance of IPBlock according
// to declarative validation rules in the API schema.
func ValidateExtensionsIPBlock(ctx context.Context, op operation.Operation, fldPath *field.Path, obj, oldObj *extensionsv1beta1.IPBlock) (errs field.ErrorList) {
	// field extensionsv1beta1.IPBlock.CIDR
	errs = append(errs,
		func(fldPath *field.Path, obj, oldObj *string, oldValueCorrelated bool) (errs field.ErrorList) {
			// don't revalidate unchanged data
			if oldValueCorrelated && op.Type == operation.Update && (obj == oldObj || (obj != nil && oldObj != nil && *obj == *oldObj)) {
				return nil
			}
			// call field-attached validations
			earlyReturn := false
			if e := validate.RequiredValue(ctx, op, fldPath, obj, oldObj); len(e) != 0 {
				errs = append(errs, e...)
				earlyReturn = true
			}
			if earlyReturn {
				return // do not proceed
			}
			return
		}(fldPath.Child("cidr"), &obj.CIDR, safe.Field(oldObj, func(oldObj *extensionsv1beta1.IPBlock) *string { return &oldObj.CIDR }), oldObj != nil)...)

	// field extensionsv1beta1.IPBlock.Except has no validation
	return errs
}

// ValidateExtensionsNetworkPolicy validates an instance of NetworkPolicy according
// to declarative validation rules in the API schema.
func ValidateExtensionsNetworkPolicy(ctx context.Context, op operation.Operation, fldPath *field.Path, obj, oldObj *extensionsv1beta1.NetworkPolicy) (errs field.ErrorList) {
	// field extensionsv1beta1.NetworkPolicy.TypeMeta has no validation
	// field extensionsv1beta1.NetworkPolicy.ObjectMeta has no validation

	// field extensionsv1beta1.NetworkPolicy.Spec
	errs = append(errs,
		func(fldPath *field.Path, obj, oldObj *extensionsv1beta1.NetworkPolicySpec, oldValueCorrelated bool) (errs field.ErrorList) {
			// don't revalidate unchanged data
			if oldValueCorrelated && op.Type == operation.Update && equality.Semantic.DeepEqual(obj, oldObj) {
				return nil
			}
			// call the type's validation function
			errs = append(errs, ValidateExtensionsNetworkPolicySpec(ctx, op, fldPath, obj, oldObj)...)
			return
		}(fldPath.Child("spec"), &obj.Spec, safe.Field(oldObj, func(oldObj *extensionsv1beta1.NetworkPolicy) *extensionsv1beta1.NetworkPolicySpec {
			return &oldObj.Spec
		}), oldObj != nil)...)

	return errs
}

// ValidateExtensionsNetworkPolicyEgressRule validates an instance of NetworkPolicyEgressRule according
// to declarative validation rules in the API schema.
func ValidateExtensionsNetworkPolicyEgressRule(ctx context.Context, op operation.Operation, fldPath *field.Path, obj, oldObj *extensionsv1beta1.NetworkPolicyEgressRule) (errs field.ErrorList) {
	// field extensionsv1beta1.NetworkPolicyEgressRule.Ports has no validation

	// field extensionsv1beta1.NetworkPolicyEgressRule.To
	errs = append(errs,
		func(fldPath *field.Path, obj, oldObj []extensionsv1beta1.NetworkPolicyPeer, oldValueCorrelated bool) (errs field.ErrorList) {
			// don't revalidate unchanged data
			if oldValueCorrelated && op.Type == operation.Update && equality.Semantic.DeepEqual(obj, oldObj) {
				return nil
			}
			// iterate the list and call the type's validation function
			errs = append(errs, validate.EachSliceVal(ctx, op, fldPath, obj, oldObj, nil, nil, ValidateExtensionsNetworkPolicyPeer)...)
			return
		}(fldPath.Child("to"), obj.To, safe.Field(oldObj, func(oldObj *extensionsv1beta1.NetworkPolicyEgressRule) []extensionsv1beta1.NetworkPolicyPeer {
			return oldObj.To
		}), oldObj != nil)...)

	return errs
}

// ValidateExtensionsNetworkPolicyIngressRule validates an instance of NetworkPolicyIngressRule according
// to declarative validation rules in the API schema.
func ValidateExtensionsNetworkPolicyIngressRule(ctx context.Context, op operation.Operation, fldPath *field.Path, obj, oldObj *extensionsv1beta1.NetworkPolicyIngressRule) (errs field.ErrorList) {
	// field extensionsv1beta1.NetworkPolicyIngressRule.Ports has no validation

	// field extensionsv1beta1.NetworkPolicyIngressRule.From
	errs = append(errs,
		func(fldPath *field.Path, obj, oldObj []extensionsv1beta1.NetworkPolicyPeer, oldValueCorrelated bool) (errs field.ErrorList) {
			// don't revalidate unchanged data
			if oldValueCorrelated && op.Type == operation.Update && equality.Semantic.DeepEqual(obj, oldObj) {
				return nil
			}
			// iterate the list and call the type's validation function
			errs = append(errs, validate.EachSliceVal(ctx, op, fldPath, obj, oldObj, nil, nil, ValidateExtensionsNetworkPolicyPeer)...)
			return
		}(fldPath.Child("from"), obj.From, safe.Field(oldObj, func(oldObj *extensionsv1beta1.NetworkPolicyIngressRule) []extensionsv1beta1.NetworkPolicyPeer {
			return oldObj.From
		}), oldObj != nil)...)

	return errs
}

// ValidateExtensionsNetworkPolicyPeer validates an instance of NetworkPolicyPeer according
// to declarative validation rules in the API schema.
func ValidateExtensionsNetworkPolicyPeer(ctx context.Context, op operation.Operation, fldPath *field.Path, obj, oldObj *extensionsv1beta1.NetworkPolicyPeer) (errs field.ErrorList) {
	// field extensionsv1beta1.NetworkPolicyPeer.PodSelector has no validation
	// field extensionsv1beta1.NetworkPolicyPeer.NamespaceSelector has no validation

	// field extensionsv1beta1.NetworkPolicyPeer.IPBlock
	errs = append(errs,
		func(fldPath *field.Path, obj, oldObj *extensionsv1beta1.IPBlock, oldValueCorrelated bool) (errs field.ErrorList) {
			// don't revalidate unchanged data
			if oldValueCorrelated && op.Type == operation.Update && equality.Semantic.DeepEqual(obj, oldObj) {
				return nil
			}
			// call the type's validation function
			errs = append(errs, ValidateExtensionsIPBlock(ctx, op, fldPath, obj, oldObj)...)
			return
		}(fldPath.Child("ipBlock"), obj.IPBlock, safe.Field(oldObj, func(oldObj *extensionsv1beta1.NetworkPolicyPeer) *extensionsv1beta1.IPBlock {
			return oldObj.IPBlock
		}), oldObj != nil)...)

	return errs
}

// ValidateExtensionsNetworkPolicySpec validates an instance of NetworkPolicySpec according
// to declarative validation rules in the API schema.
func ValidateExtensionsNetworkPolicySpec(ctx context.Context, op operation.Operation, fldPath *field.Path, obj, oldObj *extensionsv1beta1.NetworkPolicySpec) (errs field.ErrorList) {
	// field extensionsv1beta1.NetworkPolicySpec.PodSelector has no validation

	// field extensionsv1beta1.NetworkPolicySpec.Ingress
	errs = append(errs,
		func(fldPath *field.Path, obj, oldObj []extensionsv1beta1.NetworkPolicyIngressRule, oldValueCorrelated bool) (errs field.ErrorList) {
			// don't revalidate unchanged data
			if oldValueCorrelated && op.Type == operation.Update && equality.Semantic.DeepEqual(obj, oldObj) {
				return nil
			}
			// iterate the list and call the type's validation function
			errs = append(errs, validate.EachSliceVal(ctx, op, fldPath, obj, oldObj, nil, nil, ValidateExtensionsNetworkPolicyIngressRule)...)
			return
		}(fldPath.Child("ingress"), obj.Ingress, safe.Field(oldObj, func(oldObj *extensionsv1beta1.NetworkPolicySpec) []extensionsv1beta1.NetworkPolicyIngressRule {
			return oldObj.Ingress
		}), oldObj != nil)...)

	// field extensionsv1beta1.NetworkPolicySpec.Egress
	errs = append(errs,
		func(fldPath *field.Path, obj, oldObj []extensionsv1beta1.NetworkPolicyEgressRule, oldValueCorrelated bool) (errs field.ErrorList) {
			// don't revalidate unchanged data
			if oldValueCorrelated && op.Type == operation.Update && equality.Semantic.DeepEqual(obj, oldObj) {
				return nil
			}
			// iterate the list and call the type's validation function
			errs = append(errs, validate.EachSliceVal(ctx, op, fldPath, obj, oldObj, nil, nil, ValidateExtensionsNetworkPolicyEgressRule)...)
			return
		}(fldPath.Child("egress"), obj.Egress, safe.Field(oldObj, func(oldObj *extensionsv1beta1.NetworkPolicySpec) []extensionsv1beta1.NetworkPolicyEgressRule {
			return oldObj.Egress
		}), oldObj != nil)...)

	// field extensionsv1beta1.NetworkPolicySpec.PolicyTypes has no validation
	return errs
}
