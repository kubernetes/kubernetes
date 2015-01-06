/*
Copyright 2014 Google Inc. All rights reserved.

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

package admission

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

type admissionController struct {
	client           client.Interface
	admissionHandler Interface
}

func NewAdmissionControl(client client.Interface, pluginNames []string, configFilePath string) AdmissionControl {
	return &admissionController{
		client:           client,
		admissionHandler: newInterface(pluginNames, configFilePath),
	}
}

func (ac *admissionController) AdmissionControl(operation, kind, namespace string, object runtime.Object) (err error) {
	return ac.admissionHandler.Admit(NewAttributesRecord(ac.client, object, namespace, kind, operation))
}
