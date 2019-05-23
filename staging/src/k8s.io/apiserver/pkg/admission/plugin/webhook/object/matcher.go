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

package object

import (
	"k8s.io/api/admissionregistration/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
)

// Matcher decides if a request selected by the ObjectSelector.
type Matcher struct {
}

// MatchObjectSelector decideds whether the request matches the ObjectSelector
// of the webhook. Only when they match, the webhook is called.
func (m *Matcher) MatchObjectSelector(h *v1beta1.Webhook, attr admission.Attributes) (bool, *apierrors.StatusError) {
	objectLabels := attr.GetLabels()
	selector, err := metav1.LabelSelectorAsSelector(h.ObjectSelector)
	if err != nil {
		return false, apierrors.NewInternalError(err)
	}
	return selector.Matches(labels.Set(objectLabels)), nil
}
