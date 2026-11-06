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

package pods

import (
	"reflect"
	"slices"
	"testing"

	"github.com/stretchr/testify/assert"

	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
)

func TestVisitContainersWithPath(t *testing.T) {
	testCases := []struct {
		description string
		path        *field.Path
		haveSpec    *api.PodSpec
		wantNames   []string
	}{
		{
			"empty podspec",
			field.NewPath("spec"),
			&api.PodSpec{},
			[]string{},
		},
		{
			"regular containers",
			field.NewPath("spec"),
			&api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
			},
			[]string{"spec.containers[0]", "spec.containers[1]"},
		},
		{
			"init containers",
			field.NewPath("spec"),
			&api.PodSpec{
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2"},
				},
			},
			[]string{"spec.initContainers[0]", "spec.initContainers[1]"},
		},
		{
			"regular and init containers",
			field.NewPath("spec"),
			&api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2"},
				},
			},
			[]string{"spec.initContainers[0]", "spec.initContainers[1]", "spec.containers[0]", "spec.containers[1]"},
		},
		{
			"ephemeral containers",
			field.NewPath("spec"),
			&api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e1"}},
				},
			},
			[]string{"spec.containers[0]", "spec.containers[1]", "spec.ephemeralContainers[0]"},
		},
		{
			"all container types",
			field.NewPath("spec"),
			&api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2"},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e1"}},
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e2"}},
				},
			},
			[]string{"spec.initContainers[0]", "spec.initContainers[1]", "spec.containers[0]", "spec.containers[1]", "spec.ephemeralContainers[0]", "spec.ephemeralContainers[1]"},
		},
		{
			"all container types with template pod path",
			field.NewPath("template", "spec"),
			&api.PodSpec{
				Containers: []api.Container{
					{Name: "c1"},
					{Name: "c2"},
				},
				InitContainers: []api.Container{
					{Name: "i1"},
					{Name: "i2"},
				},
				EphemeralContainers: []api.EphemeralContainer{
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e1"}},
					{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: "e2"}},
				},
			},
			[]string{"template.spec.initContainers[0]", "template.spec.initContainers[1]", "template.spec.containers[0]", "template.spec.containers[1]", "template.spec.ephemeralContainers[0]", "template.spec.ephemeralContainers[1]"},
		},
	}

	for _, tc := range testCases {
		gotNames := []string{}
		VisitContainersWithPath(tc.haveSpec, tc.path, func(c *api.Container, p *field.Path) bool {
			gotNames = append(gotNames, p.String())
			return true
		})
		if !reflect.DeepEqual(gotNames, tc.wantNames) {
			t.Errorf("VisitContainersWithPath() for test case %q visited containers %q, wanted to visit %q", tc.description, gotNames, tc.wantNames)
		}
	}
}

func TestConvertDownwardAPIFieldLabel(t *testing.T) {
	testCases := []struct {
		version       string
		label         string
		value         string
		expectedErr   bool
		expectedLabel string
		expectedValue string
	}{
		{
			version:     "v2",
			label:       "metadata.name",
			value:       "test-pod",
			expectedErr: true,
		},
		{
			version:     "v1",
			label:       "invalid-label",
			value:       "value",
			expectedErr: true,
		},
		{
			version:       "v1",
			label:         "metadata.name",
			value:         "test-pod",
			expectedLabel: "metadata.name",
			expectedValue: "test-pod",
		},
		{
			version:       "v1",
			label:         "metadata.annotations",
			value:         "myValue",
			expectedLabel: "metadata.annotations",
			expectedValue: "myValue",
		},
		{
			version:       "v1",
			label:         "metadata.annotations['myKey']",
			value:         "myValue",
			expectedLabel: "metadata.annotations['myKey']",
			expectedValue: "myValue",
		},
		{
			version:       "v1",
			label:         "spec.host",
			value:         "127.0.0.1",
			expectedLabel: "spec.nodeName",
			expectedValue: "127.0.0.1",
		},
		{
			version:       "v1",
			label:         "status.podIPs",
			value:         "10.244.0.6,fd00::6",
			expectedLabel: "status.podIPs",
			expectedValue: "10.244.0.6,fd00::6",
		},
		{
			version:       "v1",
			label:         "status.podIPs",
			value:         "10.244.0.6",
			expectedLabel: "status.podIPs",
			expectedValue: "10.244.0.6",
		},
		{
			version:       "v1",
			label:         "status.hostIPs",
			value:         "10.244.0.6,fd00::6",
			expectedLabel: "status.hostIPs",
			expectedValue: "10.244.0.6,fd00::6",
		},
		{
			version:       "v1",
			label:         "status.hostIPs",
			value:         "10.244.0.6",
			expectedLabel: "status.hostIPs",
			expectedValue: "10.244.0.6",
		},
	}
	for _, tc := range testCases {
		label, value, err := ConvertDownwardAPIFieldLabel(tc.version, tc.label, tc.value)
		if err != nil {
			if tc.expectedErr {
				continue
			}
			t.Errorf("ConvertDownwardAPIFieldLabel(%s, %s, %s) failed: %s",
				tc.version, tc.label, tc.value, err)
		}
		if tc.expectedLabel != label || tc.expectedValue != value {
			t.Errorf("ConvertDownwardAPIFieldLabel(%s, %s, %s) = (%s, %s, nil), expected (%s, %s, nil)",
				tc.version, tc.label, tc.value, label, value, tc.expectedLabel, tc.expectedValue)
		}
	}
}

func TestContainerPairs(t *testing.T) {
	containers := func(names ...string) []api.Container {
		var containers []api.Container
		for _, name := range names {
			containers = append(containers, api.Container{Name: name})
		}
		return containers
	}
	ephemeralContainers := func(names ...string) []api.EphemeralContainer {
		var containers []api.EphemeralContainer
		for _, name := range names {
			containers = append(containers, api.EphemeralContainer{EphemeralContainerCommon: api.EphemeralContainerCommon{Name: name}})
		}
		return containers
	}
	// pairs returns each pair as "type:new/old", with "-" for a missing container.
	pairs := func(spec, oldSpec *api.PodSpec, mask ContainerType) []string {
		name := func(c *api.Container) string {
			if c == nil {
				return "-"
			}
			return c.Name
		}
		typeNames := map[ContainerType]string{InitContainers: "init", Containers: "main", EphemeralContainers: "ephemeral"}
		var pairs []string
		for pair, containerType := range ContainerPairs(&api.Pod{Spec: *spec}, &api.Pod{Spec: *oldSpec}, mask) {
			pairs = append(pairs, typeNames[containerType]+":"+name(pair.New)+"/"+name(pair.Old))
		}
		return pairs
	}

	t.Run("main containers", func(t *testing.T) {
		tests := []struct {
			name          string
			names, old    []string
			expectedPairs []string
		}{
			{name: "no containers"},
			{name: "unchanged", names: []string{"a", "b"}, old: []string{"a", "b"}, expectedPairs: []string{"a/a", "b/b"}},
			{name: "added to an empty list", names: []string{"a"}, expectedPairs: []string{"a/-"}},
			{name: "all removed", old: []string{"a", "b"}, expectedPairs: []string{"-/a", "-/b"}},
			{name: "added first", names: []string{"x", "a", "b"}, old: []string{"a", "b"}, expectedPairs: []string{"x/-", "a/a", "b/b"}},
			{name: "added in the middle", names: []string{"a", "x", "b"}, old: []string{"a", "b"}, expectedPairs: []string{"a/a", "x/-", "b/b"}},
			{name: "added last", names: []string{"a", "b", "x"}, old: []string{"a", "b"}, expectedPairs: []string{"a/a", "b/b", "x/-"}},
			{name: "removed first", names: []string{"b", "c"}, old: []string{"a", "b", "c"}, expectedPairs: []string{"-/a", "b/b", "c/c"}},
			{name: "removed in the middle", names: []string{"a", "c"}, old: []string{"a", "b", "c"}, expectedPairs: []string{"a/a", "-/b", "c/c"}},
			{name: "removed last", names: []string{"a", "b"}, old: []string{"a", "b", "c"}, expectedPairs: []string{"a/a", "b/b", "-/c"}},
			{name: "replaced", names: []string{"a", "x", "c"}, old: []string{"a", "b", "c"}, expectedPairs: []string{"a/a", "x/-", "-/b", "c/c"}},
			{
				name:          "added and removed in several places",
				names:         []string{"x", "b", "y", "d", "z"},
				old:           []string{"a", "b", "c", "d", "e"},
				expectedPairs: []string{"x/-", "-/a", "b/b", "y/-", "-/c", "d/d", "z/-", "-/e"},
			},
		}
		for _, tc := range tests {
			t.Run(tc.name, func(t *testing.T) {
				var expected []string
				for _, pair := range tc.expectedPairs {
					expected = append(expected, "main:"+pair)
				}
				assert.Equal(t, expected, pairs(&api.PodSpec{Containers: containers(tc.names...)}, &api.PodSpec{Containers: containers(tc.old...)}, Containers))
			})
		}
	})

	t.Run("mask", func(t *testing.T) {
		spec := &api.PodSpec{
			InitContainers:      containers("init"),
			Containers:          containers("a", "x"),
			EphemeralContainers: ephemeralContainers("debug", "debug2"),
		}
		oldSpec := &api.PodSpec{
			InitContainers:      containers("init"),
			Containers:          containers("a", "b"),
			EphemeralContainers: ephemeralContainers("debug"),
		}
		initPairs := []string{"init:init/init"}
		mainPairs := []string{"main:a/a", "main:x/-", "main:-/b"}
		ephemeralPairs := []string{"ephemeral:debug/debug", "ephemeral:debug2/-"}
		assert.Equal(t, initPairs, pairs(spec, oldSpec, InitContainers))
		assert.Equal(t, ephemeralPairs, pairs(spec, oldSpec, EphemeralContainers))
		assert.Equal(t, slices.Concat(initPairs, mainPairs, ephemeralPairs), pairs(spec, oldSpec, AllContainers))
	})

	t.Run("pairs point into the specs", func(t *testing.T) {
		pod := &api.Pod{Spec: api.PodSpec{
			InitContainers:      containers("init"),
			Containers:          containers("a"),
			EphemeralContainers: ephemeralContainers("debug"),
		}}
		oldPod := pod.DeepCopy()
		for pair, containerType := range ContainerPairs(pod, oldPod, AllContainers) {
			if containerType == Containers {
				assert.Same(t, &pod.Spec.Containers[0], pair.New)
				assert.Same(t, &oldPod.Spec.Containers[0], pair.Old)
			}
		}
		allocs := testing.AllocsPerRun(100, func() {
			for range ContainerPairs(pod, oldPod, AllContainers) {
			}
		})
		assert.Zero(t, allocs)
	})
}

func TestVolumePairs(t *testing.T) {
	volumes := func(names ...string) []api.Volume {
		var vols []api.Volume
		for _, name := range names {
			vols = append(vols, api.Volume{Name: name})
		}
		return vols
	}
	// pairs returns each pair as "new/old", with "-" for a missing volume.
	pairs := func(pod, oldPod *api.Pod) []string {
		name := func(v *api.Volume) string {
			if v == nil {
				return "-"
			}
			return v.Name
		}
		var out []string
		for pair := range VolumePairs(pod, oldPod) {
			out = append(out, name(pair.New)+"/"+name(pair.Old))
		}
		return out
	}

	tests := []struct {
		name          string
		names, old    []string
		expectedPairs []string
	}{
		{name: "no volumes"},
		{name: "unchanged", names: []string{"a", "b"}, old: []string{"a", "b"}, expectedPairs: []string{"a/a", "b/b"}},
		{name: "added to an empty list", names: []string{"a"}, expectedPairs: []string{"a/-"}},
		{name: "all removed", old: []string{"a", "b"}, expectedPairs: []string{"-/a", "-/b"}},
		{name: "added first", names: []string{"x", "a", "b"}, old: []string{"a", "b"}, expectedPairs: []string{"x/-", "a/a", "b/b"}},
		{name: "added in the middle", names: []string{"a", "x", "b"}, old: []string{"a", "b"}, expectedPairs: []string{"a/a", "x/-", "b/b"}},
		{name: "added last", names: []string{"a", "b", "x"}, old: []string{"a", "b"}, expectedPairs: []string{"a/a", "b/b", "x/-"}},
		{name: "removed first", names: []string{"b", "c"}, old: []string{"a", "b", "c"}, expectedPairs: []string{"-/a", "b/b", "c/c"}},
		{name: "removed in the middle", names: []string{"a", "c"}, old: []string{"a", "b", "c"}, expectedPairs: []string{"a/a", "-/b", "c/c"}},
		{name: "removed last", names: []string{"a", "b"}, old: []string{"a", "b", "c"}, expectedPairs: []string{"a/a", "b/b", "-/c"}},
		{name: "replaced", names: []string{"a", "x", "c"}, old: []string{"a", "b", "c"}, expectedPairs: []string{"a/a", "x/-", "-/b", "c/c"}},
		{
			name:          "added and removed in several places",
			names:         []string{"x", "b", "y", "d", "z"},
			old:           []string{"a", "b", "c", "d", "e"},
			expectedPairs: []string{"x/-", "-/a", "b/b", "y/-", "-/c", "d/d", "z/-", "-/e"},
		},
		{name: "reordered", names: []string{"b", "a"}, old: []string{"a", "b"}, expectedPairs: []string{"-/a", "b/b", "a/-"}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			pod := &api.Pod{Spec: api.PodSpec{Volumes: volumes(tc.names...)}}
			oldPod := &api.Pod{Spec: api.PodSpec{Volumes: volumes(tc.old...)}}
			assert.Equal(t, tc.expectedPairs, pairs(pod, oldPod))
		})
	}

	t.Run("pairs point into the specs and do not allocate", func(t *testing.T) {
		pod := &api.Pod{Spec: api.PodSpec{Volumes: volumes("a", "x", "c")}}
		oldPod := &api.Pod{Spec: api.PodSpec{Volumes: volumes("a", "b", "c")}}
		first := true
		for pair := range VolumePairs(pod, oldPod) {
			if first {
				assert.Same(t, &pod.Spec.Volumes[0], pair.New)
				assert.Same(t, &oldPod.Spec.Volumes[0], pair.Old)
				first = false
			}
		}
		allocs := testing.AllocsPerRun(100, func() {
			for range VolumePairs(pod, oldPod) {
			}
		})
		assert.Zero(t, allocs)
	})

	t.Run("early break", func(t *testing.T) {
		pod := &api.Pod{Spec: api.PodSpec{Volumes: volumes("x", "b", "z")}}
		oldPod := &api.Pod{Spec: api.PodSpec{Volumes: volumes("a", "b", "c")}}
		for stopAfter := 1; stopAfter <= 5; stopAfter++ {
			count := 0
			for range VolumePairs(pod, oldPod) {
				count++
				if count == stopAfter {
					break
				}
			}
			assert.Equal(t, stopAfter, count)
		}
	})
}
