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
	"strconv"

	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

func init() {
	newer.Scheme.AddConversionFuncs(
		// TypeMeta has changed type of ResourceVersion internally
		func(in *newer.TypeMeta, out *TypeMeta, s conversion.Scope) error {
			out.APIVersion = in.APIVersion
			out.Kind = in.Kind
			out.Namespace = in.Namespace
			out.ID = in.Name
			out.CreationTimestamp = in.CreationTimestamp
			out.SelfLink = in.SelfLink
			out.Annotations = in.Annotations

			if len(in.ResourceVersion) > 0 {
				v, err := strconv.ParseUint(in.ResourceVersion, 10, 64)
				if err != nil {
					return err
				}
				out.ResourceVersion = v
			}
			return nil
		},
		func(in *TypeMeta, out *newer.TypeMeta, s conversion.Scope) error {
			out.APIVersion = in.APIVersion
			out.Kind = in.Kind
			out.Namespace = in.Namespace
			out.Name = in.ID
			out.CreationTimestamp = in.CreationTimestamp
			out.SelfLink = in.SelfLink
			out.Annotations = in.Annotations

			if in.ResourceVersion != 0 {
				out.ResourceVersion = strconv.FormatUint(in.ResourceVersion, 10)
			}
			return nil
		},

		// EnvVar's Key is deprecated in favor of Name.
		func(in *newer.EnvVar, out *EnvVar, s conversion.Scope) error {
			out.Value = in.Value
			out.Key = in.Name
			out.Name = in.Name
			return nil
		},
		func(in *EnvVar, out *newer.EnvVar, s conversion.Scope) error {
			out.Value = in.Value
			if in.Name != "" {
				out.Name = in.Name
			} else {
				out.Name = in.Key
			}
			return nil
		},

		// Path & MountType are deprecated.
		func(in *newer.VolumeMount, out *VolumeMount, s conversion.Scope) error {
			out.Name = in.Name
			out.ReadOnly = in.ReadOnly
			out.MountPath = in.MountPath
			out.Path = in.MountPath
			out.MountType = "" // MountType is ignored.
			return nil
		},
		func(in *VolumeMount, out *newer.VolumeMount, s conversion.Scope) error {
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
		func(in *newer.MinionList, out *MinionList, s conversion.Scope) error {
			s.Convert(&in.TypeMeta, &out.TypeMeta, 0)
			s.Convert(&in.Items, &out.Items, 0)
			out.Minions = out.Items
			return nil
		},
		func(in *MinionList, out *newer.MinionList, s conversion.Scope) error {
			s.Convert(&in.TypeMeta, &out.TypeMeta, 0)
			if len(in.Items) == 0 {
				s.Convert(&in.Minions, &out.Items, 0)
			} else {
				s.Convert(&in.Items, &out.Items, 0)
			}
			return nil
		},
	)

}
