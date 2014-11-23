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

package v1beta1

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func init() {
	api.Scheme.AddDefaultingFuncs(
		func(obj *Volume) {
			if obj.Source == nil || util.AllPtrFieldsNil(obj.Source) {
				obj.Source = &VolumeSource{
					EmptyDir: &EmptyDir{},
				}
			}
		},
		func(obj *Port) {
			if obj.Protocol == "" {
				obj.Protocol = ProtocolTCP
			}
		},
		func(obj *Container) {
			// TODO: delete helper functions that touch this
			if obj.ImagePullPolicy == "" {
				obj.ImagePullPolicy = PullIfNotPresent
			}
			if obj.TerminationMessagePath == "" {
				// TODO: fix other code that sets this
				obj.TerminationMessagePath = api.TerminationMessagePathDefault
			}
		},
		func(obj *RestartPolicy) {
			if util.AllPtrFieldsNil(obj) {
				obj.Always = &RestartPolicyAlways{}
			}
		},
		func(obj *Service) {
			if obj.Protocol == "" {
				obj.Protocol = ProtocolTCP
			}
		},
	)
}

// TODO: remove redundant code in validation and conversion
