/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	apierrors "github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
)

// NewForbidden is a utility function to return a well-formatted admission control error response
func NewForbidden(a Attributes, internalError error) error {
	// do not double wrap an error of same type
	if apierrors.IsForbidden(internalError) {
		return internalError
	}

	name := "Unknown"
	kind := a.GetKind()
	obj := a.GetObject()
	if obj != nil {
		objectMeta, err := api.ObjectMetaFor(obj)
		if err != nil {
			return apierrors.NewForbidden(kind, name, internalError)
		}

		// this is necessary because name object name generation has not occurred yet
		if len(objectMeta.Name) > 0 {
			name = objectMeta.Name
		} else if len(objectMeta.GenerateName) > 0 {
			name = objectMeta.GenerateName
		}
	}
	return apierrors.NewForbidden(kind, name, internalError)
}
