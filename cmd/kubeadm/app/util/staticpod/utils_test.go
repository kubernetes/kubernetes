/*
Copyright 2017 The Kubernetes Authors.

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

package staticpod

import (
	"reflect"
	"sort"
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestComponentResources(t *testing.T) {
	a := ComponentResources("250m")
	if a.Requests == nil {
		t.Errorf(
			"failed componentResources, return value was nil",
		)
	}
}

func TestComponentProbe(t *testing.T) {
	var tests = []struct {
		port   int
		path   string
		scheme v1.URIScheme
	}{
		{
			port:   1,
			path:   "foo",
			scheme: v1.URISchemeHTTP,
		},
		{
			port:   2,
			path:   "bar",
			scheme: v1.URISchemeHTTPS,
		},
	}
	for _, rt := range tests {
		actual := ComponentProbe(rt.port, rt.path, rt.scheme)
		if actual.Handler.HTTPGet.Port != intstr.FromInt(rt.port) {
			t.Errorf(
				"failed componentProbe:\n\texpected: %v\n\t  actual: %v",
				rt.port,
				actual.Handler.HTTPGet.Port,
			)
		}
		if actual.Handler.HTTPGet.Path != rt.path {
			t.Errorf(
				"failed componentProbe:\n\texpected: %s\n\t  actual: %s",
				rt.path,
				actual.Handler.HTTPGet.Path,
			)
		}
		if actual.Handler.HTTPGet.Scheme != rt.scheme {
			t.Errorf(
				"failed componentProbe:\n\texpected: %v\n\t  actual: %v",
				rt.scheme,
				actual.Handler.HTTPGet.Scheme,
			)
		}
	}
}

func TestComponentPod(t *testing.T) {
	var tests = []struct {
		n string
	}{
		{
			n: "foo",
		},
	}

	for _, rt := range tests {
		c := v1.Container{Name: rt.n}
		v := []v1.Volume{}
		actual := ComponentPod(c, v)
		if actual.ObjectMeta.Name != rt.n {
			t.Errorf(
				"failed componentPod:\n\texpected: %s\n\t  actual: %s",
				rt.n,
				actual.ObjectMeta.Name,
			)
		}
	}
}

func TestNewVolume(t *testing.T) {
	var tests = []struct {
		name     string
		path     string
		expected v1.Volume
	}{
		{
			name: "foo",
			path: "/etc/foo",
			expected: v1.Volume{
				Name: "foo",
				VolumeSource: v1.VolumeSource{
					HostPath: &v1.HostPathVolumeSource{Path: "/etc/foo"},
				},
			},
		},
	}

	for _, rt := range tests {
		actual := NewVolume(rt.name, rt.path)
		if !reflect.DeepEqual(actual, rt.expected) {
			t.Errorf(
				"failed newVolume:\n\texpected: %v\n\t  actual: %v",
				rt.expected,
				actual,
			)
		}
	}
}

func TestNewVolumeMount(t *testing.T) {
	var tests = []struct {
		name     string
		path     string
		ro       bool
		expected v1.VolumeMount
	}{
		{
			name: "foo",
			path: "/etc/foo",
			ro:   false,
			expected: v1.VolumeMount{
				Name:      "foo",
				MountPath: "/etc/foo",
				ReadOnly:  false,
			},
		},
		{
			name: "bar",
			path: "/etc/foo/bar",
			ro:   true,
			expected: v1.VolumeMount{
				Name:      "bar",
				MountPath: "/etc/foo/bar",
				ReadOnly:  true,
			},
		},
	}

	for _, rt := range tests {
		actual := NewVolumeMount(rt.name, rt.path, rt.ro)
		if !reflect.DeepEqual(actual, rt.expected) {
			t.Errorf(
				"failed newVolumeMount:\n\texpected: %v\n\t  actual: %v",
				rt.expected,
				actual,
			)
		}
	}
}

func TestGetExtraParameters(t *testing.T) {
	var tests = []struct {
		overrides map[string]string
		defaults  map[string]string
		expected  []string
	}{
		{
			overrides: map[string]string{
				"admission-control": "NamespaceLifecycle,LimitRanger",
			},
			defaults: map[string]string{
				"admission-control":     "NamespaceLifecycle",
				"insecure-bind-address": "127.0.0.1",
				"allow-privileged":      "true",
			},
			expected: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
			},
		},
		{
			overrides: map[string]string{
				"admission-control": "NamespaceLifecycle,LimitRanger",
			},
			defaults: map[string]string{
				"insecure-bind-address": "127.0.0.1",
				"allow-privileged":      "true",
			},
			expected: []string{
				"--admission-control=NamespaceLifecycle,LimitRanger",
				"--insecure-bind-address=127.0.0.1",
				"--allow-privileged=true",
			},
		},
	}

	for _, rt := range tests {
		actual := GetExtraParameters(rt.overrides, rt.defaults)
		sort.Strings(actual)
		sort.Strings(rt.expected)
		if !reflect.DeepEqual(actual, rt.expected) {
			t.Errorf("failed getExtraParameters:\nexpected:\n%v\nsaw:\n%v", rt.expected, actual)
		}
	}
}
