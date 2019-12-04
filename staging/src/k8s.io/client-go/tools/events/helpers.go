/*
Copyright 2019 The Kubernetes Authors.

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

package events

import (
	"fmt"

	"k8s.io/api/core/v1"
	"k8s.io/api/events/v1beta1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GetFieldSelector returns the appropriate field selector based on the API version being used to communicate with the server.
// The returned field selector can be used with List and Watch to filter desired events.
func GetFieldSelector(groupVersion schema.GroupVersion, regardingName, regardingNamespace, regardingKind, regardingUID string) (fields.Selector, error) {
	field := fields.Set{}
	var err error

	switch groupVersion {
	case v1beta1.SchemeGroupVersion:
		if len(regardingName) > 0 {
			field["regarding.name"] = regardingName
		}
		if len(regardingNamespace) > 0 {
			field["regarding.namespace"] = regardingNamespace
		}
		if len(regardingKind) > 0 {
			field["regarding.kind"] = regardingKind
		}
		if len(regardingUID) > 0 {
			field["regarding.uid"] = regardingUID
		}
	case v1.SchemeGroupVersion:
		if len(regardingName) > 0 {
			field["involvedObject.name"] = regardingName
		}
		if len(regardingNamespace) > 0 {
			field["involvedObject.namespace"] = regardingNamespace
		}
		if len(regardingKind) > 0 {
			field["involvedObject.kind"] = regardingKind
		}
		if len(regardingUID) > 0 {
			field["involvedObject.uid"] = regardingUID
		}
	default:
		err = fmt.Errorf("unknown version %v", groupVersion)
	}

	return field.AsSelector(), err
}
