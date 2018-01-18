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

package configuration

import (
	"testing"

	"k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type disabledValidatingWebhookConfigLister struct{}

func (l *disabledValidatingWebhookConfigLister) List(options metav1.ListOptions) (*v1beta1.ValidatingWebhookConfigurationList, error) {
	return nil, errors.NewNotFound(schema.GroupResource{Group: "admissionregistration", Resource: "ValidatingWebhookConfigurations"}, "")
}
func TestWebhookConfigDisabled(t *testing.T) {
	manager := NewValidatingWebhookConfigurationManager(&disabledValidatingWebhookConfigLister{})
	manager.sync()
	_, err := manager.Webhooks()
	if err.Error() != ErrDisabled.Error() {
		t.Errorf("expected %v, got %v", ErrDisabled, err)
	}
}
