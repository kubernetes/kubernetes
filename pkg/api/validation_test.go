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
		if _, err := validateVolumes(v); err == nil {
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
		{Name: "baby-you-and-me", ContainerPort: 82, Protocol: "tcp"},
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
	if err := validateEnv(successCase); err != nil {
		t.Errorf("expected success: %v", err)
	}

	nonCanonicalCase := []EnvVar{
		{Key: "EV"},
	}
	if err := validateEnv(nonCanonicalCase); err != nil {
		t.Errorf("expected success: %v", err)
	}
	if nonCanonicalCase[0].Name != "EV" || nonCanonicalCase[0].Value != "" {
		t.Errorf("expected default values: %+v", nonCanonicalCase[0])
	}

	errorCases := map[string][]EnvVar{
		"zero-length name":        {{Name: ""}},
		"name not a C identifier": {{Name: "a.b.c"}},
	}
	for k, v := range errorCases {
		if err := validateEnv(v); err == nil {
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

func TestValidateContainers(t *testing.T) {
	volumes := util.StringSet{}

	successCase := []Container{
		{Name: "abc", Image: "image"},
		{Name: "123", Image: "image"},
		{Name: "abc-123", Image: "image"},
	}
	if err := validateContainers(successCase, volumes); err != nil {
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
		"invalid env var name": {
			{Name: "abc", Image: "image", Env: []EnvVar{{Name: "ev.1"}}},
		},
		"unknown volume name": {
			{Name: "abc", Image: "image", VolumeMounts: []VolumeMount{{Name: "anything", MountPath: "/foo"}}},
		},
	}
	for k, v := range errorCases {
		if err := validateContainers(v, volumes); err == nil {
			t.Errorf("expected failure for %s", k)
		}
	}
}

func TestValidateManifest(t *testing.T) {
	successCases := []ContainerManifest{
		{Version: "v1beta1", ID: "abc"},
		{Version: "v1beta2", ID: "123"},
		{Version: "V1BETA1", ID: "abc.123.do-re-mi"},
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
					Ports: []Port{
						{Name: "p1", ContainerPort: 80, HostPort: 8080},
						{Name: "p2", ContainerPort: 81},
						{ContainerPort: 82},
					},
					Env: []EnvVar{
						{Name: "ev1", Value: "val1"},
						{Name: "ev2", Value: "val2"},
						{Key: "EV3", Value: "val3"},
					},
					VolumeMounts: []VolumeMount{
						{Name: "vol1", MountPath: "/foo"},
						{Name: "vol1", Path: "/bar"},
					},
				},
			},
		},
	}
	for _, manifest := range successCases {
		if err := ValidateManifest(&manifest); err != nil {
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
		if err := ValidateManifest(&v); err == nil {
			t.Errorf("expected failure for %s", k)
		}
	}
}
