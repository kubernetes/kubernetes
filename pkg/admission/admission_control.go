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
	"errors"

	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// stubAdmissionController is capable of either always admitting or always denying incoming requests
type stubAdmissionController struct {
	admit bool
}

func (ac *stubAdmissionController) AdmissionControl(operation, kind, namespace string, object runtime.Object) (err error) {
	if !ac.admit {
		err = apierrors.NewConflict(kind, "name", errors.New("No changes allowed"))
	}
	return err
}

func NewAlwaysAdmitController() AdmissionControl {
	return &stubAdmissionController{
		admit: true,
	}
}

func NewAlwaysDenyController() AdmissionControl {
	return &stubAdmissionController{
		admit: false,
	}
}

type admissionController struct {
	client           client.Interface
	admissionHandler Interface
}

func NewAdmissionControl(client client.Interface, pluginNames []string, configFilePath string) AdmissionControl {
	return NewAdmissionControlForHandler(client, newInterface(pluginNames, configFilePath))
}

func NewAdmissionControlForHandler(client client.Interface, handler Interface) AdmissionControl {
	return &admissionController{
		client:           client,
		admissionHandler: handler,
	}
}

func (ac *admissionController) AdmissionControl(operation, kind, namespace string, object runtime.Object) (err error) {
	return ac.admissionHandler.Admit(NewAttributesRecord(ac.client, object, namespace, kind, operation))
}
