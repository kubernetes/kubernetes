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
	"strings"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
)

func TestValidateVolumes(t *testing.T) {
	successCase := []Volume{
		{Name: "abc"},
		{Name: "123"},
		{Name: "abc-123"},
	}
	names, err := validateVolumes(successCase)
	if err != nil {
		t.Errorf("expected success: %v", err)
	}
	if len(names) != 3 || !names.Has("abc") || !names.Has("123") || !names.Has("abc-123") {
		t.Errorf("wrong names result: %v", names)
	}

	errorCases := map[string][]Volume{
		"zero-length name":     {{Name: ""}},
		"name > 63 characters": {{Name: strings.Repeat("a", 64)}},
		"name not a DNS label": {{Name: "a.b.c"}},
		"name not unique":      {{Name: "abc"}, {Name: "abc"}},
	}
	for k, v := range errorCases {
		_, err := validateVolumes(v)
		if err == nil {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateContainers(t *testing.T) {
	volumes := util.StringSet{}

	successCase := []Container{
		{Name: "abc", Image: "image"},
		{Name: "123", Image: "image"},
		{Name: "abc-123", Image: "image"},
	}
	err := validateContainers(successCase, volumes)
	if err != nil {
		t.Errorf("expected success: %v", err)
	}

	errorCases := map[string][]Container{
		"zero-length name":     {{Name: "", Image: "image"}},
		"name > 63 characters": {{Name: strings.Repeat("a", 64), Image: "image"}},
		"name not a DNS label": {{Name: "a.b.c", Image: "image"}},
		"name not unique": {
			{Name: "abc", Image: "image"},
			{Name: "abc", Image: "image"},
		},
		"zero-length image": {{Name: "abc", Image: ""}},
	}
	for k, v := range errorCases {
		err := validateContainers(v, volumes)
		if err == nil {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateManifest(t *testing.T) {
	successCases := []ContainerManifest{
		{Version: "v1beta1", ID: "abc"},
		{Version: "v1beta1", ID: "123"},
		{Version: "v1beta1", ID: "abc.123.do-re-mi"},
		{
			Version: "v1beta1",
			ID:      "abc",
			Volumes: []Volume{{Name: "vol1"}, {Name: "vol2"}},
			Containers: []Container{
				{
					Name:       "abc",
					Image:      "image",
					Command:    []string{"foo", "bar"},
					WorkingDir: "/tmp",
					Memory:     1,
					CPU:        1,
				},
			},
		},
	}
	for _, manifest := range successCases {
		err := ValidateManifest(&manifest)
		if err != nil {
			t.Errorf("expected success: %v", err)
		}
	}

	errorCases := map[string]ContainerManifest{
		"empty version":          {Version: "", ID: "abc"},
		"invalid version":        {Version: "bogus", ID: "abc"},
		"zero-length id":         {Version: "v1beta1", ID: ""},
		"id > 255 characters":    {Version: "v1beta1", ID: strings.Repeat("a", 256)},
		"id not a DNS subdomain": {Version: "v1beta1", ID: "a.b.c."},
		"invalid volume name": {
			Version: "v1beta1",
			ID:      "abc",
			Volumes: []Volume{{Name: "vol.1"}},
		},
		"invalid container name": {
			Version:    "v1beta1",
			ID:         "abc",
			Containers: []Container{{Name: "ctr.1", Image: "image"}},
		},
	}
	for k, v := range errorCases {
		err := ValidateManifest(&v)
		if err == nil {
			t.Errorf("expected failure for %s", k)
		}
	}
}
