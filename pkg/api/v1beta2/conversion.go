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

package v1beta2

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
	)
}
