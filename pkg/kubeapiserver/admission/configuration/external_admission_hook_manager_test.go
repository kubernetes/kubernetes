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

	"k8s.io/api/admissionregistration/v1alpha1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

type disabledWebhookConfigLister struct{}

func (l *disabledWebhookConfigLister) List(options metav1.ListOptions) (*v1alpha1.ExternalAdmissionHookConfigurationList, error) {
	return nil, errors.NewNotFound(schema.GroupResource{Group: "admissionregistration", Resource: "externalAdmissionHookConfigurations"}, "")
}
func TestWebhookConfigDisabled(t *testing.T) {
	manager := NewExternalAdmissionHookConfigurationManager(&disabledWebhookConfigLister{})
	manager.sync()
	_, err := manager.ExternalAdmissionHooks()
	if err.Error() != ErrDisabled.Error() {
		t.Errorf("expected %v, got %v", ErrDisabled, err)
	}
}
