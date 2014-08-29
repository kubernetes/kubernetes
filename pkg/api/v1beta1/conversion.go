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
	// Alias this so it can be easily changed when we cut the next version.
	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	// Also import under original name for Convert and AddConversionFuncs.
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
)

func init() {
	// Shortcut for sub-conversions. TODO: This should possibly be refactored
	// such that this convert function is passed to each conversion func.
	Convert := api.Convert
	api.AddConversionFuncs(
		// EnvVar's Key is deprecated in favor of Name.
		func(in *newer.EnvVar, out *EnvVar) error {
			out.Value = in.Value
			out.Key = in.Name
			out.Name = in.Name
			return nil
		},
		func(in *EnvVar, out *newer.EnvVar) error {
			out.Value = in.Value
			if in.Name != "" {
				out.Name = in.Name
			} else {
				out.Name = in.Key
			}
			return nil
		},

		// Path & MountType are deprecated.
		func(in *newer.VolumeMount, out *VolumeMount) error {
			out.Name = in.Name
			out.ReadOnly = in.ReadOnly
			out.MountPath = in.MountPath
			out.Path = in.MountPath
			out.MountType = "" // MountType is ignored.
			return nil
		},
		func(in *VolumeMount, out *newer.VolumeMount) error {
			out.Name = in.Name
			out.ReadOnly = in.ReadOnly
			if in.MountPath == "" {
				out.MountPath = in.Path
			} else {
				out.MountPath = in.MountPath
			}
			return nil
		},

		// MinionList.Items had a wrong name in v1beta1
		func(in *newer.MinionList, out *MinionList) error {
			Convert(&in.JSONBase, &out.JSONBase)
			Convert(&in.Items, &out.Items)
			out.Minions = out.Items
			return nil
		},
		func(in *MinionList, out *newer.MinionList) error {
			Convert(&in.JSONBase, &out.JSONBase)
			if len(in.Items) == 0 {
				Convert(&in.Minions, &out.Items)
			} else {
				Convert(&in.Items, &out.Items)
			}
			return nil
		},
	)

}
