/*
Copyright 2016 The Kubernetes Authors.

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

package util

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
)

// TODO: remove this when Reflector takes an interface rather than a particular ListOptions as input parameter.
func VersionizeV1ListOptions(in api.ListOptions) (out v1.ListOptions) {
	if in.LabelSelector != nil {
		out.LabelSelector = in.LabelSelector.String()
	} else {
		out.LabelSelector = ""
	}
	if in.FieldSelector != nil {
		out.FieldSelector = in.FieldSelector.String()
	} else {
		out.FieldSelector = ""
	}
	out.Watch = in.Watch
	out.ResourceVersion = in.ResourceVersion
	out.TimeoutSeconds = in.TimeoutSeconds
	return out
}
