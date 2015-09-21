/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package runtime

import (
	"k8s.io/kubernetes/pkg/conversion"
)

// IsNotRegisteredError returns true if the error indicates the provided
// object or input data is not registered.
func IsNotRegisteredError(err error) bool {
	return conversion.IsNotRegisteredError(err)
}

// IsMissingKind returns true if the error indicates that the provided object
// is missing a 'Kind' field.
func IsMissingKind(err error) bool {
	return conversion.IsMissingKind(err)
}

// IsMissingVersion returns true if the error indicates that the provided object
// is missing a 'Versioj' field.
func IsMissingVersion(err error) bool {
	return conversion.IsMissingVersion(err)
}
