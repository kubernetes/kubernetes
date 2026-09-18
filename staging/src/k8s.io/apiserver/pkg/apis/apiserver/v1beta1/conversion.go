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

package v1beta1

import (
	"fmt"

	conversion "k8s.io/apimachinery/pkg/conversion"
	apiserver "k8s.io/apiserver/pkg/apis/apiserver"
)

func Convert_v1beta1_EgressSelection_To_apiserver_EgressSelection(in *EgressSelection, out *apiserver.EgressSelection, s conversion.Scope) error {
	if err := autoConvert_v1beta1_EgressSelection_To_apiserver_EgressSelection(in, out, s); err != nil {
		return err
	}
	if out.Name == "master" {
		out.Name = "controlplane"
	}
	return nil
}

func Convert_apiserver_WebhookConfiguration_To_v1beta1_WebhookConfiguration(in *apiserver.WebhookConfiguration, out *WebhookConfiguration, s conversion.Scope) error {
	if in.ConditionsReview != nil {
		return fmt.Errorf("WebhookConfiguration.ConditionsReview is not supported in v1beta1 AuthorizationConfiguration")
	}
	return autoConvert_apiserver_WebhookConfiguration_To_v1beta1_WebhookConfiguration(in, out, s)
}
