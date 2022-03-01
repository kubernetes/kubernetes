/*
Copyright 2021 The Kubernetes Authors.

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

package policy

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const (
	PDBV1beta1Label = "pdb.kubernetes.io/deprecated-v1beta1-empty-selector-match"
)

var (
	NonV1beta1MatchAllSelector  = &metav1.LabelSelector{}
	NonV1beta1MatchNoneSelector = &metav1.LabelSelector{
		MatchExpressions: []metav1.LabelSelectorRequirement{{Key: PDBV1beta1Label, Operator: metav1.LabelSelectorOpExists}},
	}

	V1beta1MatchNoneSelector = &metav1.LabelSelector{}
	V1beta1MatchAllSelector  = &metav1.LabelSelector{
		MatchExpressions: []metav1.LabelSelectorRequirement{{Key: PDBV1beta1Label, Operator: metav1.LabelSelectorOpDoesNotExist}},
	}
)

func StripPDBV1beta1Label(selector *metav1.LabelSelector) {
	if selector == nil {
		return
	}

	trimmedMatchExpressions := selector.MatchExpressions[:0]
	for _, exp := range selector.MatchExpressions {
		if exp.Key != PDBV1beta1Label {
			trimmedMatchExpressions = append(trimmedMatchExpressions, exp)
		}
	}
	selector.MatchExpressions = trimmedMatchExpressions
}
