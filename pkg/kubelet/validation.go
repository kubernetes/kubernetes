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
	apierrs "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/validation"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func ValidatePod(pod *Pod) (errors []error) {
	if !util.IsDNSSubdomain(pod.Name) {
		errors = append(errors, apierrs.NewFieldInvalid("name", pod.Name))
	}
	if errs := validation.ValidateManifest(&pod.Manifest); len(errs) != 0 {
		errors = append(errors, errs...)
	}
	return errors
}
