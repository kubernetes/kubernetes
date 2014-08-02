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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/internal"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

// AddTypes adds the types in this package to a conversion scheme
func AddTypes(scheme *conversion.Scheme) {
	internal.AddTypes(scheme)

	scheme.AddKnownTypes("v1beta1",
		PodList{},
		Pod{},
		ReplicationControllerList{},
		ReplicationController{},
		ServiceList{},
		Service{},
		MinionList{},
		Minion{},
		ContainerManifestList{},
		Endpoints{},
	)

	// TODO: when we get more of this stuff, move to its own file. This is not a
	// good home for lots of conversion functions.
	scheme.AddConversionFuncs(
		// EnvVar's Name is depricated in favor of Key.
		func(in *internal.EnvVar, out *EnvVar) error {
			out.Value = in.Value
			out.Key = in.Name
			out.Name = in.Name
			return nil
		},
		func(in *EnvVar, out *internal.EnvVar) error {
			out.Value = in.Value
			if in.Name != "" {
				out.Name = in.Name
			} else {
				out.Name = in.Key
			}
			return nil
		},
	)
}
