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

package api

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
)

// Codec is the identity codec for this package - it can only convert itself
// to itself.
var Codec = runtime.CodecFor(Scheme, "")

func init() {
	Scheme.AddConversionFuncs(
		// Convert ContainerManifest to BoundPod
		func(in *ContainerManifest, out *BoundPod, s conversion.Scope) error {
			out.Spec.Containers = in.Containers
			out.Spec.Volumes = in.Volumes
			out.Spec.RestartPolicy = in.RestartPolicy
			out.Name = in.ID
			out.UID = in.UUID
			// TODO(dchen1107): Move this conversion to pkg/api/v1beta[123]/conversion.go
			// along with fixing #1502
			for i := range out.Spec.Containers {
				ctr := &out.Spec.Containers[i]
				if len(ctr.TerminationMessagePath) == 0 {
					ctr.TerminationMessagePath = TerminationMessagePathDefault
				}
			}
			return nil
		},
		func(in *BoundPod, out *ContainerManifest, s conversion.Scope) error {
			out.Containers = in.Spec.Containers
			out.Volumes = in.Spec.Volumes
			out.RestartPolicy = in.Spec.RestartPolicy
			out.Version = "v1beta2"
			out.ID = in.Name
			out.UUID = in.UID
			for i := range out.Containers {
				ctr := &out.Containers[i]
				if len(ctr.TerminationMessagePath) == 0 {
					ctr.TerminationMessagePath = TerminationMessagePathDefault
				}
			}
			return nil
		},
		func(in *ContainerManifestList, out *BoundPods, s conversion.Scope) error {
			if err := s.Convert(&in.Items, &out.Items, 0); err != nil {
				return err
			}
			for i := range out.Items {
				item := &out.Items[i]
				item.ResourceVersion = in.ResourceVersion
			}
			return nil
		},
		func(in *BoundPods, out *ContainerManifestList, s conversion.Scope) error {
			if err := s.Convert(&in.Items, &out.Items, 0); err != nil {
				return err
			}
			out.ResourceVersion = in.ResourceVersion
			return nil
		},

		// Convert Pod to BoundPod
		func(in *Pod, out *BoundPod, s conversion.Scope) error {
			if err := s.Convert(&in.DesiredState.Manifest, out, 0); err != nil {
				return err
			}
			// Only copy a subset of fields, and override manifest attributes with the pod
			// metadata
			out.UID = in.UID
			out.Name = in.Name
			out.Namespace = in.Namespace
			out.CreationTimestamp = in.CreationTimestamp
			return nil
		},
	)
}
