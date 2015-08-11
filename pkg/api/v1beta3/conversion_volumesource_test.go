/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/conversion"
)

func TestAPItoV1Beta3VolumeSourceConversion(t *testing.T) {
	c := conversion.NewConverter(conversion.DefaultNameFunc)
	c.Debug = t

	if err := c.RegisterConversionFunc(convert_api_VolumeSource_To_v1beta3_VolumeSource); err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	in := api.VolumeSource{
		DownwardAPI: &api.DownwardAPIVolumeSource{
			Items: []api.DownwardAPIVolumeFile{
				{
					Path: "./test/api-to-v1beta3/conversion",
				},
			},
		},
	}
	out := VolumeSource{}

	if err := c.Convert(&in, &out, 0, nil); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if e, a := in.DownwardAPI.Items[0].Path, out.Metadata.Items[0].Name; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}

func TestV1Beta3toAPIVolumeSourceConversion(t *testing.T) {
	c := conversion.NewConverter(conversion.DefaultNameFunc)
	c.Debug = t

	if err := c.RegisterConversionFunc(convert_v1beta3_VolumeSource_To_api_VolumeSource); err != nil {
		t.Fatalf("unexpected error %v", err)
	}

	in := VolumeSource{
		Metadata: &MetadataVolumeSource{
			Items: []MetadataFile{
				{
					Name: "./test/v1beta3-to-api/conversion",
				},
			},
		},
	}
	out := api.VolumeSource{}

	if err := c.Convert(&in, &out, 0, nil); err != nil {
		t.Fatalf("unexpected error %v", err)
	}
	if e, a := in.Metadata.Items[0].Name, out.DownwardAPI.Items[0].Path; e != a {
		t.Errorf("expected %v, got %v", e, a)
	}
}
