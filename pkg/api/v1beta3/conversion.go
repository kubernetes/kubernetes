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

package v1beta3

import (
	newer "github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	//	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
)

func init() {
	newer.Scheme.AddConversionFuncs(
	/*		// Direct conversion
			func(in *newer.ListMeta, out *ListMeta, s conversion.Scope) error {
				out.SelfLink = in.SelfLink
				out.ResourceVersion = in.ResourceVersion
				return nil
			},
			func(in *ListMeta, out *newer.ListMeta, s conversion.Scope) error {
				out.SelfLink = in.SelfLink
				out.ResourceVersion = in.ResourceVersion
				return nil
			},
			func(in *newer.ObjectMeta, out *ObjectMeta, s conversion.Scope) error {
				out.UID = in.UID
				out.Name = in.Name
				out.Namespace = in.Namespace
				out.CreationTimestamp = in.CreationTimestamp
				out.SelfLink = in.SelfLink
				out.Labels = in.Labels
				out.Annotations = in.Annotations
				out.ResourceVersion = in.ResourceVersion
				return nil
			},
			func(in *ObjectMeta, out *newer.ObjectMeta, s conversion.Scope) error {
				out.UID = in.UID
				out.Name = in.Name
				out.Namespace = in.Namespace
				out.CreationTimestamp = in.CreationTimestamp
				out.SelfLink = in.SelfLink
				out.Labels = in.Labels
				out.Annotations = in.Annotations
				out.ResourceVersion = in.ResourceVersion
				return nil
			},*/
	)
}
