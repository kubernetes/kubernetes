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

package kubelet

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func makeInvalidError(field string, value interface{}) api.ValidationError {
	return api.ValidationError{api.ErrTypeInvalid, field, value}
}

func ValidatePod(pod *Pod) (errors []error) {
	if !util.IsDNSSubdomain(pod.Name) {
		errors = append(errors, makeInvalidError("Pod.Name", pod.Name))
	}
	if errs := api.ValidateManifest(&pod.Manifest); len(errs) != 0 {
		errors = append(errors, errs...)
	}
	return errors
}
