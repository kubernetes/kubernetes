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
	"strings"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/selection"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
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
				ret.LabelSelectorRequirements, ret.LabelSelectorParsingErr = labelSelectorAsSelector(in.LabelSelector.Requirements)
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

var labelSelectorOpToSelectionOp = map[metav1.LabelSelectorOperator]selection.Operator{
	metav1.LabelSelectorOpIn:           selection.In,
	metav1.LabelSelectorOpNotIn:        selection.NotIn,
	metav1.LabelSelectorOpExists:       selection.Exists,
	metav1.LabelSelectorOpDoesNotExist: selection.DoesNotExist,
}

func labelSelectorAsSelector(requirements []metav1.LabelSelectorRequirement) (labels.Requirements, error) {
	if len(requirements) == 0 {
		return nil, nil
	}
	reqs := make([]labels.Requirement, 0, len(requirements))
	var errs []error
	for _, expr := range requirements {
		op, ok := labelSelectorOpToSelectionOp[expr.Operator]
		if !ok {
			errs = append(errs, fmt.Errorf("%q is not a valid label selector operator", expr.Operator))
			continue
		}
		values := expr.Values
		if len(values) == 0 {
			values = nil
		}
		req, err := labels.NewRequirement(expr.Key, op, values)
		if err != nil {
			errs = append(errs, err)
			continue
		}
		reqs = append(reqs, *req)
	}

	// If this happens, it means all requirements ended up getting skipped.
	// Return nil rather than [].
	if len(reqs) == 0 {
		reqs = nil
	}

	// Return any accumulated errors along with any accumulated requirements, so recognized / valid requirements can be considered by authorization.
	// This is safe because requirements are ANDed together so dropping unknown / invalid ones results in a strictly broader authorization check.
	return labels.Requirements(reqs), utilerrors.NewAggregate(errs)
}

func fieldSelectorAsSelector(requirements []metav1.FieldSelectorRequirement) (fields.Requirements, error) {
	if len(requirements) == 0 {
		return nil, nil
	}

	reqs := make([]fields.Requirement, 0, len(requirements))
	var errs []error
	for _, expr := range requirements {
		if len(expr.Values) > 1 {
			errs = append(errs, fmt.Errorf("fieldSelectors do not yet support multiple values"))
			continue
		}

		switch expr.Operator {
		case metav1.FieldSelectorOpIn:
			if len(expr.Values) != 1 {
				errs = append(errs, fmt.Errorf("fieldSelectors in must have one value"))
				continue
			}
			// when converting to fields.Requirement, use Equals to match how parsed field selectors behave
			reqs = append(reqs, fields.Requirement{Field: expr.Key, Operator: selection.Equals, Value: expr.Values[0]})
		case metav1.FieldSelectorOpNotIn:
			if len(expr.Values) != 1 {
				errs = append(errs, fmt.Errorf("fieldSelectors not in must have one value"))
				continue
			}
			// when converting to fields.Requirement, use NotEquals to match how parsed field selectors behave
			reqs = append(reqs, fields.Requirement{Field: expr.Key, Operator: selection.NotEquals, Value: expr.Values[0]})
		case metav1.FieldSelectorOpExists, metav1.FieldSelectorOpDoesNotExist:
			errs = append(errs, fmt.Errorf("fieldSelectors do not yet support %v", expr.Operator))
			continue
		default:
			errs = append(errs, fmt.Errorf("%q is not a valid field selector operator", expr.Operator))
			continue
		}
	}

	// If this happens, it means all requirements ended up getting skipped.
	// Return nil rather than [].
	if len(reqs) == 0 {
		reqs = nil
	}

	// Return any accumulated errors along with any accumulated requirements, so recognized / valid requirements can be considered by authorization.
	// This is safe because requirements are ANDed together so dropping unknown / invalid ones results in a strictly broader authorization check.
	return fields.Requirements(reqs), utilerrors.NewAggregate(errs)
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

// BuildEvaluationError constructs the evaluation error string to include in *SubjectAccessReview status
// based on the authorizer evaluation error and any field and label selector parse errors.
func BuildEvaluationError(evaluationError error, attrs authorizer.AttributesRecord) string {
	var evaluationErrors []string
	if evaluationError != nil {
		evaluationErrors = append(evaluationErrors, evaluationError.Error())
	}
	if reqs, err := attrs.GetFieldSelector(); err != nil {
		if len(reqs) > 0 {
			evaluationErrors = append(evaluationErrors, "spec.resourceAttributes.fieldSelector partially ignored due to parse error")
		} else {
			evaluationErrors = append(evaluationErrors, "spec.resourceAttributes.fieldSelector ignored due to parse error")
		}
	}
	if reqs, err := attrs.GetLabelSelector(); err != nil {
		if len(reqs) > 0 {
			evaluationErrors = append(evaluationErrors, "spec.resourceAttributes.labelSelector partially ignored due to parse error")
		} else {
			evaluationErrors = append(evaluationErrors, "spec.resourceAttributes.labelSelector ignored due to parse error")
		}
	}
	return strings.Join(evaluationErrors, "; ")
}
