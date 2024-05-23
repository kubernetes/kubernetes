/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"fmt"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
)

// ResourceAttributesFrom combines the API object information and the user.Info from the context to build a full authorizer.AttributesRecord for resource access
func ResourceAttributesFrom(user user.Info, in authorizationapi.ResourceAttributes) authorizer.AttributesRecord {
	ret := authorizer.AttributesRecord{
		User:            user,
		Verb:            in.Verb,
		Namespace:       in.Namespace,
		APIGroup:        in.Group,
		APIVersion:      matchAllVersionIfEmpty(in.Version),
		Resource:        in.Resource,
		Subresource:     in.Subresource,
		Name:            in.Name,
		ResourceRequest: true,
	}

	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.AuthorizeWithSelectors) {
		if in.LabelSelector != nil {
			if len(in.LabelSelector.RawSelector) > 0 {
				labelSelector, err := labels.Parse(in.LabelSelector.RawSelector)
				if err != nil {
					ret.LabelSelectorRequirements, ret.LabelSelectorParsingErr = nil, err
				} else {
					requirements, _ /*selectable*/ := labelSelector.Requirements()
					ret.LabelSelectorRequirements, ret.LabelSelectorParsingErr = requirements, nil
				}
			}
			if len(in.LabelSelector.Requirements) > 0 {
				selector := &metav1.LabelSelector{MatchExpressions: in.LabelSelector.Requirements}
				labelSelector, err := metav1.LabelSelectorAsSelector(selector)
				if err != nil {
					ret.LabelSelectorRequirements, ret.LabelSelectorParsingErr = nil, err
				} else {
					requirements, _ /*selectable*/ := labelSelector.Requirements()
					ret.LabelSelectorRequirements, ret.LabelSelectorParsingErr = requirements, nil
				}
			}
		}

		if in.FieldSelector != nil {
			if len(in.FieldSelector.RawSelector) > 0 {
				fieldSelector, err := fields.ParseSelector(in.FieldSelector.RawSelector)
				if err != nil {
					ret.FieldSelectorRequirements, ret.FieldSelectorParsingErr = nil, err
				} else {
					ret.FieldSelectorRequirements, ret.FieldSelectorParsingErr = fieldSelector.Requirements(), nil
				}
			}
			if len(in.FieldSelector.Requirements) > 0 {
				ret.FieldSelectorRequirements, ret.FieldSelectorParsingErr = fieldSelectorAsSelector(in.FieldSelector.Requirements)
			}
		}
	}

	return ret
}

func fieldSelectorAsSelector(requirements []authorizationapi.FieldSelectorRequirement) (fields.Requirements, error) {
	if requirements == nil {
		return nil, nil
	}

	selectors := make([]fields.Selector, 0, len(requirements))
	for _, expr := range requirements {
		if len(expr.Values) > 1 {
			// this fails here instead of validation so that a SAR with an invalid field selector can be tried without the selector
			return nil, fmt.Errorf("fieldSelectors do not yet support multiple values")
		}

		switch expr.Operator {
		case metav1.LabelSelectorOpIn:
			if len(expr.Values) != 1 {
				// this fails here instead of validation so that a SAR with an invalid field selector can be tried without the selector
				return nil, fmt.Errorf("fieldSelectors in must have one value")
			}

			selectors = append(selectors, fields.OneTermEqualSelector(expr.Key, expr.Values[0]))

		case metav1.LabelSelectorOpNotIn:
			if len(expr.Values) != 1 {
				// this fails here instead of validation so that a SAR with an invalid field selector can be tried without the selector
				return nil, fmt.Errorf("fieldSelectors not in must have one value")
			}

			selectors = append(selectors, fields.OneTermNotEqualSelector(expr.Key, expr.Values[0]))

		case metav1.LabelSelectorOpExists:
			// this fails here instead of validation so that a SAR with an invalid field selector can be tried without the selector
			return nil, fmt.Errorf("fieldSelectors do not yet support %v", metav1.LabelSelectorOpExists)
		case metav1.LabelSelectorOpDoesNotExist:
			// this fails here instead of validation so that a SAR with an invalid field selector can be tried without the selector
			return nil, fmt.Errorf("fieldSelectors do not yet support %v", metav1.LabelSelectorOpDoesNotExist)
		default:
			return nil, fmt.Errorf("%q is not a valid field selector operator", expr.Operator)
		}
	}

	return fields.AndSelectors(selectors...).Requirements(), nil
}

// NonResourceAttributesFrom combines the API object information and the user.Info from the context to build a full authorizer.AttributesRecord for non resource access
func NonResourceAttributesFrom(user user.Info, in authorizationapi.NonResourceAttributes) authorizer.AttributesRecord {
	return authorizer.AttributesRecord{
		User:            user,
		ResourceRequest: false,
		Path:            in.Path,
		Verb:            in.Verb,
	}
}

func convertToUserInfoExtra(extra map[string]authorizationapi.ExtraValue) map[string][]string {
	if extra == nil {
		return nil
	}
	ret := map[string][]string{}
	for k, v := range extra {
		ret[k] = []string(v)
	}

	return ret
}

// AuthorizationAttributesFrom takes a spec and returns the proper authz attributes to check it.
func AuthorizationAttributesFrom(spec authorizationapi.SubjectAccessReviewSpec) authorizer.AttributesRecord {
	userToCheck := &user.DefaultInfo{
		Name:   spec.User,
		Groups: spec.Groups,
		UID:    spec.UID,
		Extra:  convertToUserInfoExtra(spec.Extra),
	}

	var authorizationAttributes authorizer.AttributesRecord
	if spec.ResourceAttributes != nil {
		authorizationAttributes = ResourceAttributesFrom(userToCheck, *spec.ResourceAttributes)
	} else {
		authorizationAttributes = NonResourceAttributesFrom(userToCheck, *spec.NonResourceAttributes)
	}

	return authorizationAttributes
}

// matchAllVersionIfEmpty returns a "*" if the version is unspecified
func matchAllVersionIfEmpty(version string) string {
	if len(version) == 0 {
		return "*"
	}
	return version
}
