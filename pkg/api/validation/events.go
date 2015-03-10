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

package validation

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	errs "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

// ValidateEvent makes sure that the event makes sense.
func ValidateEvent(event *api.Event) errs.ValidationErrorList {
	allErrs := errs.ValidationErrorList{}

	if !util.IsDNSSubdomain(event.Namespace) {
		allErrs = append(allErrs, errs.NewFieldInvalid("namespace", event.Namespace, ""))
	}
	// TODO: event.InvolvedObject is not versioned. References need to be normalizd to internal when converted from external and vice versa.
	if event.InvolvedObject.Kind == "Minion" || event.InvolvedObject.Kind == "Node" {
		// Do not check event Namespace with node namespace, they are likely different.
		if event.InvolvedObject.Namespace != "" {
			allErrs = append(allErrs, errs.NewFieldInvalid("involvedObject.namespace", event.InvolvedObject.Namespace, "involved node object namespace must be None"))
		}
	} else {
		if event.Namespace != event.InvolvedObject.Namespace {
			allErrs = append(allErrs, errs.NewFieldInvalid("involvedObject.namespace", event.InvolvedObject.Namespace, "namespace does not match involvedObject"))
		}
	}
	return allErrs
}
