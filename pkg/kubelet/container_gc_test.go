/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package kubelet

import (
	"fmt"
	"reflect"
	"sort"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/kubelet/dockertools"
	docker "github.com/fsouza/go-dockerclient"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func newTestContainerGC(t *testing.T, MinAge time.Duration, MaxPerPodContainer, MaxContainers int) (containerGC, *dockertools.FakeDockerClient) {
	fakeDocker := new(dockertools.FakeDockerClient)
	gc, err := newContainerGC(fakeDocker, ContainerGCPolicy{
		MinAge:             MinAge,
		MaxPerPodContainer: MaxPerPodContainer,
		MaxContainers:      MaxContainers,
	})
	require.Nil(t, err)
	return gc, fakeDocker
}

// Makes a stable time object, lower id is earlier time.
func makeTime(id int) time.Time {
	return zero.Add(time.Duration(id) * time.Second)
}

// Makes an API object with the specified Docker ID and pod UID.
func makeAPIContainer(uid, name, dockerID string) docker.APIContainers {
	return docker.APIContainers{
		Names: []string{fmt.Sprintf("/k8s_%s_bar_new_%s_42", name, uid)},
		ID:    dockerID,
	}
}

// Makes a function that adds to a map a detailed container with the specified properties.
func makeContainerDetail(id string, running bool, created time.Time) func(map[string]*docker.Container) {
	return func(m map[string]*docker.Container) {
		m[id] = &docker.Container{
			State: docker.State{
				Running: running,
			},
			ID:      id,
			Created: created,
		}
	}
}

// Makes a detailed container map from the specified functions.
func makeContainerDetailMap(funcs ...func(map[string]*docker.Container)) map[string]*docker.Container {
	m := make(map[string]*docker.Container, len(funcs))
	for _, f := range funcs {
		f(m)
	}
	return m
}

func verifyStringArrayEqualsAnyOrder(t *testing.T, actual, expected []string) {
	act := make([]string, len(actual))
	exp := make([]string, len(expected))
	copy(act, actual)
	copy(exp, expected)

	sort.StringSlice(act).Sort()
	sort.StringSlice(exp).Sort()

	if !reflect.DeepEqual(exp, act) {
		t.Errorf("Expected(sorted): %#v, Actual(sorted): %#v", exp, act)
	}
}

func TestGarbageCollectZeroMaxContainers(t *testing.T) {
	gc, fakeDocker := newTestContainerGC(t, time.Minute, 1, 0)
	fakeDocker.ContainerList = []docker.APIContainers{
		makeAPIContainer("foo", "POD", "1876"),
	}
	fakeDocker.ContainerMap = makeContainerDetailMap(
		makeContainerDetail("1876", false, makeTime(0)),
	)

	assert.Nil(t, gc.GarbageCollect())
	assert.Len(t, fakeDocker.Removed, 1)
}

func TestGarbageCollectNoMaxPerPodContainerLimit(t *testing.T) {
	gc, fakeDocker := newTestContainerGC(t, time.Minute, -1, 4)
	fakeDocker.ContainerList = []docker.APIContainers{
		makeAPIContainer("foo", "POD", "1876"),
		makeAPIContainer("foo1", "POD", "2876"),
		makeAPIContainer("foo2", "POD", "3876"),
		makeAPIContainer("foo3", "POD", "4876"),
		makeAPIContainer("foo4", "POD", "5876"),
	}
	fakeDocker.ContainerMap = makeContainerDetailMap(
		makeContainerDetail("1876", false, makeTime(0)),
		makeContainerDetail("2876", false, makeTime(1)),
		makeContainerDetail("3876", false, makeTime(2)),
		makeContainerDetail("4876", false, makeTime(3)),
		makeContainerDetail("5876", false, makeTime(4)),
	)

	assert.Nil(t, gc.GarbageCollect())
	assert.Len(t, fakeDocker.Removed, 1)
}

func TestGarbageCollectNoMaxLimit(t *testing.T) {
	gc, fakeDocker := newTestContainerGC(t, time.Minute, 1, -1)
	fakeDocker.ContainerList = []docker.APIContainers{
		makeAPIContainer("foo", "POD", "1876"),
		makeAPIContainer("foo1", "POD", "2876"),
		makeAPIContainer("foo2", "POD", "3876"),
		makeAPIContainer("foo3", "POD", "4876"),
		makeAPIContainer("foo4", "POD", "5876"),
	}
	fakeDocker.ContainerMap = makeContainerDetailMap(
		makeContainerDetail("1876", false, makeTime(0)),
		makeContainerDetail("2876", false, makeTime(0)),
		makeContainerDetail("3876", false, makeTime(0)),
		makeContainerDetail("4876", false, makeTime(0)),
		makeContainerDetail("5876", false, makeTime(0)),
	)

	assert.Nil(t, gc.GarbageCollect())
	assert.Len(t, fakeDocker.Removed, 0)
}

func TestGarbageCollect(t *testing.T) {
	tests := []struct {
		containers       []docker.APIContainers
		containerDetails map[string]*docker.Container
		expectedRemoved  []string
	}{
		// Don't remove containers started recently.
		{
			containers: []docker.APIContainers{
				makeAPIContainer("foo", "POD", "1876"),
				makeAPIContainer("foo", "POD", "2876"),
				makeAPIContainer("foo", "POD", "3876"),
			},
			containerDetails: makeContainerDetailMap(
				makeContainerDetail("1876", false, time.Now()),
				makeContainerDetail("2876", false, time.Now()),
				makeContainerDetail("3876", false, time.Now()),
			),
		},
		// Remove oldest containers.
		{
			containers: []docker.APIContainers{
				makeAPIContainer("foo", "POD", "1876"),
				makeAPIContainer("foo", "POD", "2876"),
				makeAPIContainer("foo", "POD", "3876"),
			},
			containerDetails: makeContainerDetailMap(
				makeContainerDetail("1876", false, makeTime(0)),
				makeContainerDetail("2876", false, makeTime(1)),
				makeContainerDetail("3876", false, makeTime(2)),
			),
			expectedRemoved: []string{"1876"},
		},
		// Only remove non-running containers.
		{
			containers: []docker.APIContainers{
				makeAPIContainer("foo", "POD", "1876"),
				makeAPIContainer("foo", "POD", "2876"),
				makeAPIContainer("foo", "POD", "3876"),
				makeAPIContainer("foo", "POD", "4876"),
			},
			containerDetails: makeContainerDetailMap(
				makeContainerDetail("1876", true, makeTime(0)),
				makeContainerDetail("2876", false, makeTime(1)),
				makeContainerDetail("3876", false, makeTime(2)),
				makeContainerDetail("4876", false, makeTime(3)),
			),
			expectedRemoved: []string{"2876"},
		},
		// Less than maxContainerCount doesn't delete any.
		{
			containers: []docker.APIContainers{
				makeAPIContainer("foo", "POD", "1876"),
			},
			containerDetails: makeContainerDetailMap(
				makeContainerDetail("1876", false, makeTime(0)),
			),
		},
		// maxContainerCount applies per (UID,container) pair.
		{
			containers: []docker.APIContainers{
				makeAPIContainer("foo", "POD", "1876"),
				makeAPIContainer("foo", "POD", "2876"),
				makeAPIContainer("foo", "POD", "3876"),
				makeAPIContainer("foo", "bar", "1076"),
				makeAPIContainer("foo", "bar", "2076"),
				makeAPIContainer("foo", "bar", "3076"),
				makeAPIContainer("foo2", "POD", "1176"),
				makeAPIContainer("foo2", "POD", "2176"),
				makeAPIContainer("foo2", "POD", "3176"),
			},
			containerDetails: makeContainerDetailMap(
				makeContainerDetail("1876", false, makeTime(0)),
				makeContainerDetail("2876", false, makeTime(1)),
				makeContainerDetail("3876", false, makeTime(2)),
				makeContainerDetail("1076", false, makeTime(0)),
				makeContainerDetail("2076", false, makeTime(1)),
				makeContainerDetail("3076", false, makeTime(2)),
				makeContainerDetail("1176", false, makeTime(0)),
				makeContainerDetail("2176", false, makeTime(1)),
				makeContainerDetail("3176", false, makeTime(2)),
			),
			expectedRemoved: []string{"1076", "1176", "1876"},
		},
		// Remove non-running unidentified Kubernetes containers.
		{
			containers: []docker.APIContainers{
				{
					// Unidentified Kubernetes container.
					Names: []string{"/k8s_unidentified"},
					ID:    "1876",
				},
				{
					// Unidentified (non-running) Kubernetes container.
					Names: []string{"/k8s_unidentified"},
					ID:    "2876",
				},
				makeAPIContainer("foo", "POD", "3876"),
			},
			containerDetails: makeContainerDetailMap(
				makeContainerDetail("1876", true, makeTime(0)),
				makeContainerDetail("2876", false, makeTime(0)),
				makeContainerDetail("3876", false, makeTime(0)),
			),
			expectedRemoved: []string{"2876"},
		},
		// Max limit applied and tries to keep from every pod.
		{
			containers: []docker.APIContainers{
				makeAPIContainer("foo", "POD", "1876"),
				makeAPIContainer("foo", "POD", "2876"),
				makeAPIContainer("foo1", "POD", "3876"),
				makeAPIContainer("foo1", "POD", "4876"),
				makeAPIContainer("foo2", "POD", "5876"),
				makeAPIContainer("foo2", "POD", "6876"),
				makeAPIContainer("foo3", "POD", "7876"),
				makeAPIContainer("foo3", "POD", "8876"),
				makeAPIContainer("foo4", "POD", "9876"),
				makeAPIContainer("foo4", "POD", "10876"),
			},
			containerDetails: makeContainerDetailMap(
				makeContainerDetail("1876", false, makeTime(0)),
				makeContainerDetail("2876", false, makeTime(1)),
				makeContainerDetail("3876", false, makeTime(0)),
				makeContainerDetail("4876", false, makeTime(1)),
				makeContainerDetail("5876", false, makeTime(0)),
				makeContainerDetail("6876", false, makeTime(1)),
				makeContainerDetail("7876", false, makeTime(0)),
				makeContainerDetail("8876", false, makeTime(1)),
				makeContainerDetail("9876", false, makeTime(0)),
				makeContainerDetail("10876", false, makeTime(1)),
			),
			expectedRemoved: []string{"1876", "3876", "5876", "7876", "9876"},
		},
		// If more pods than limit allows, evicts oldest pod.
		{
			containers: []docker.APIContainers{
				makeAPIContainer("foo", "POD", "1876"),
				makeAPIContainer("foo", "POD", "2876"),
				makeAPIContainer("foo1", "POD", "3876"),
				makeAPIContainer("foo1", "POD", "4876"),
				makeAPIContainer("foo2", "POD", "5876"),
				makeAPIContainer("foo3", "POD", "6876"),
				makeAPIContainer("foo4", "POD", "7876"),
				makeAPIContainer("foo5", "POD", "8876"),
				makeAPIContainer("foo6", "POD", "9876"),
				makeAPIContainer("foo7", "POD", "10876"),
			},
			containerDetails: makeContainerDetailMap(
				makeContainerDetail("1876", false, makeTime(1)),
				makeContainerDetail("2876", false, makeTime(2)),
				makeContainerDetail("3876", false, makeTime(1)),
				makeContainerDetail("4876", false, makeTime(2)),
				makeContainerDetail("5876", false, makeTime(0)),
				makeContainerDetail("6876", false, makeTime(1)),
				makeContainerDetail("7876", false, makeTime(0)),
				makeContainerDetail("8876", false, makeTime(1)),
				makeContainerDetail("9876", false, makeTime(2)),
				makeContainerDetail("10876", false, makeTime(1)),
			),
			expectedRemoved: []string{"1876", "3876", "5876", "7876"},
		},
	}
	for i, test := range tests {
		t.Logf("Running test case with index %d", i)
		gc, fakeDocker := newTestContainerGC(t, time.Hour, 2, 6)
		fakeDocker.ContainerList = test.containers
		fakeDocker.ContainerMap = test.containerDetails
		assert.Nil(t, gc.GarbageCollect())
		verifyStringArrayEqualsAnyOrder(t, fakeDocker.Removed, test.expectedRemoved)
	}
}
