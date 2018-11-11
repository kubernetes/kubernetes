/*
Copyright 2017 The Kubernetes Authors.

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

package selfsubjectrulesreview

import (
	"context"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	authorizationapi "k8s.io/kubernetes/pkg/apis/authorization"
)

// REST implements a RESTStorage for selfsubjectrulesreview.
type REST struct {
	ruleResolver authorizer.RuleResolver
}

// NewREST returns a RESTStorage object that will work against selfsubjectrulesreview.
func NewREST(ruleResolver authorizer.RuleResolver) *REST {
	return &REST{ruleResolver}
}

// NamespaceScoped fulfill rest.Scoper
func (r *REST) NamespaceScoped() bool {
	return false
}

// New creates a new selfsubjectrulesreview object.
func (r *REST) New() runtime.Object {
	return &authorizationapi.SelfSubjectRulesReview{}
}

// Create attempts to get self subject rules in specific namespace.
func (r *REST) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	selfSRR, ok := obj.(*authorizationapi.SelfSubjectRulesReview)
	if !ok {
		return nil, apierrors.NewBadRequest(fmt.Sprintf("not a SelfSubjectRulesReview: %#v", obj))
	}

	user, ok := genericapirequest.UserFrom(ctx)
	if !ok {
		return nil, apierrors.NewBadRequest("no user present on request")
	}

	namespace := selfSRR.Spec.Namespace
	if namespace == "" {
		return nil, apierrors.NewBadRequest("no namespace on request")
	}
	resourceInfo, nonResourceInfo, incomplete, err := r.ruleResolver.RulesFor(user, namespace)

	ret := &authorizationapi.SelfSubjectRulesReview{
		Status: authorizationapi.SubjectRulesReviewStatus{
			ResourceRules:    getResourceRules(resourceInfo),
			NonResourceRules: getNonResourceRules(nonResourceInfo),
			Incomplete:       incomplete,
		},
	}

	if err != nil {
		ret.Status.EvaluationError = err.Error()
	}

	return ret, nil
}

func getResourceRules(infos []authorizer.ResourceRuleInfo) []authorizationapi.ResourceRule {
	rules := make([]authorizationapi.ResourceRule, len(infos))
	for i, info := range infos {
		rules[i] = authorizationapi.ResourceRule{
			Verbs:         info.GetVerbs(),
			APIGroups:     info.GetAPIGroups(),
			Resources:     info.GetResources(),
			ResourceNames: info.GetResourceNames(),
		}
	}
	return rules
}

func getNonResourceRules(infos []authorizer.NonResourceRuleInfo) []authorizationapi.NonResourceRule {
	rules := make([]authorizationapi.NonResourceRule, len(infos))
	for i, info := range infos {
		rules[i] = authorizationapi.NonResourceRule{
			Verbs:           info.GetVerbs(),
			NonResourceURLs: info.GetNonResourceURLs(),
		}
	}
	return rules
}
