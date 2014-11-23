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
		func(in *Volume, out *api.Volume) {
			if in.Source == nil || util.AllFieldsNil(in.Source) {
				out.Source = &api.VolumeSource{
					EmptyDir: &api.EmptyDir{},
				}
			}
		},
		func(in *Port, out *api.Port) {
			if in.Protocol == "" {
				out.Protocol = api.ProtocolTCP
			}
		},
		func(in *Container, out *api.Container) {
			// TODO: delete helper functions that touch this
			if in.ImagePullPolicy == "" {
				out.ImagePullPolicy = api.PullIfNotPresent
			}
			if in.TerminationMessagePath == "" {
				// TODO: fix other code that sets this
				out.TerminationMessagePath = api.TerminationMessagePathDefault
			}
		},
		func(in *RestartPolicy, out *api.RestartPolicy) {
			if util.AllFieldsNil(in) {
				out.Always = &api.RestartPolicyAlways{}
			}
		},
		func(in *Service, out *api.Service) {
			if in.Protocol == "" {
				out.Spec.Protocol = api.ProtocolTCP
			}
		},
	)
}

// TODO: remove redundant code in validation and conversion
