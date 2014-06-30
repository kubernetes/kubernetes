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

func TestValidateVolumeMounts(t *testing.T) {
	volumes := util.NewStringSet("abc", "123", "abc-123")

	successCase := []VolumeMount{
		{Name: "abc", MountPath: "/foo"},
		{Name: "123", MountPath: "/foo"},
		{Name: "abc-123", MountPath: "/bar"},
	}
	if err := validateVolumeMounts(successCase, volumes); err != nil {
		t.Errorf("expected success: %v", err)
	}

	nonCanonicalCase := []VolumeMount{
		{Name: "abc", Path: "/foo"},
	}
	err := validateVolumeMounts(nonCanonicalCase, volumes)
	if err != nil {
		t.Errorf("expected success: %v", err)
	}
	if nonCanonicalCase[0].MountPath != "/foo" {
		t.Errorf("expected canonicalized values: %+v", nonCanonicalCase[0])
	}

	errorCases := map[string][]VolumeMount{
		"empty name":      {{Name: "", MountPath: "/foo"}},
		"name not found":  {{Name: "", MountPath: "/foo"}},
		"empty mountpath": {{Name: "abc", MountPath: ""}},
	}
	for k, v := range errorCases {
		err := validateVolumeMounts(v, volumes)
		if err == nil {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidatePorts(t *testing.T) {
	successCase := []Port{
		{Name: "abc", ContainerPort: 80, HostPort: 80, Protocol: "TCP"},
		{Name: "123", ContainerPort: 81, HostPort: 81},
		{Name: "easy", ContainerPort: 82, Protocol: "TCP"},
		{Name: "as", ContainerPort: 83, Protocol: "UDP"},
		{Name: "do-re-me", ContainerPort: 84},
		{ContainerPort: 85},
	}
	err := validatePorts(successCase)
	if err != nil {
		t.Errorf("expected success: %v", err)
	}

	nonCanonicalCase := []Port{
		{ContainerPort: 80},
	}
	err = validatePorts(nonCanonicalCase)
	if err != nil {
		t.Errorf("expected success: %v", err)
	}
	if nonCanonicalCase[0].HostPort != 80 || nonCanonicalCase[0].Protocol != "TCP" {
		t.Errorf("expected default values: %+v", nonCanonicalCase[0])
	}

	errorCases := map[string][]Port{
		"name > 63 characters": {{Name: strings.Repeat("a", 64), ContainerPort: 80}},
		"name not a DNS label": {{Name: "a.b.c", ContainerPort: 80}},
		"name not unique": {
			{Name: "abc", ContainerPort: 80},
			{Name: "abc", ContainerPort: 81},
		},
		"zero container port":    {{ContainerPort: 0}},
		"invalid container port": {{ContainerPort: 65536}},
		"invalid host port":      {{ContainerPort: 80, HostPort: 65536}},
		"invalid protocol":       {{ContainerPort: 80, Protocol: "ICMP"}},
	}
	for k, v := range errorCases {
		err := validatePorts(v)
		if err == nil {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateEnv(t *testing.T) {
	successCase := []EnvVar{
		{Name: "abc", Value: "value"},
		{Name: "ABC", Value: "value"},
		{Name: "AbC_123", Value: "value"},
		{Name: "abc", Value: ""},
	}
	err := validateEnv(successCase)
	if err != nil {
		t.Errorf("expected success: %v", err)
	}

	//short, not ident
	errorCases := map[string][]EnvVar{
		"zero-length name":        {{Name: ""}},
		"name not a C identifier": {{Name: "a.b.c"}},
	}
	for k, v := range errorCases {
		err := validateEnv(v)
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
		"host port not unique": {
			{Name: "abc", Image: "image", Ports: []Port{{ContainerPort: 80, HostPort: 80}}},
			{Name: "def", Image: "image", Ports: []Port{{ContainerPort: 81, HostPort: 80}}},
		},
		"defaulted host port not unique in pod": {
			{Name: "abc", Image: "image", Ports: []Port{{ContainerPort: 80}}},
			{Name: "def", Image: "image", Ports: []Port{{ContainerPort: 81, HostPort: 80}}},
		},
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
		{Version: "v1beta1", Id: "abc"},
		{Version: "v1beta1", Id: "123"},
		{Version: "v1beta1", Id: "abc.123.do-re-mi"},
		{
			Version: "v1beta1",
			Id:      "abc",
			Volumes: []Volume{
				{Name: "vol1"},
				{Name: "vol2"},
			},
			Containers: []Container{
				{
					Name:       "abc",
					Image:      "image",
					Command:    []string{"sh", "-c", "date"},
					WorkingDir: "/tmp",
					Ports: []Port{
						{ContainerPort: 80},
						{ContainerPort: 81},
					},
					Env: []EnvVar{
						{Name: "var1"},
						{Name: "var2"},
					},
					VolumeMounts: []VolumeMount{
						{Name: "vol1", MountPath: "/tmp"},
						{Name: "vol2", MountPath: "/tmp"},
					},
					Memory: 1,
					CPU:    1,
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
		"empty version":          {Version: "", Id: "abc"},
		"invalid version":        {Version: "bogus", Id: "abc"},
		"zero-length id":         {Version: "v1beta1", Id: ""},
		"id > 255 characters":    {Version: "v1beta1", Id: strings.Repeat("a", 256)},
		"id not a DNS subdomain": {Version: "v1beta1", Id: "a.b.c."},
	}
	for k, v := range errorCases {
		err := ValidateManifest(&v)
		if err == nil {
			t.Errorf("expected failure for %s", k)
		}
	}
}
