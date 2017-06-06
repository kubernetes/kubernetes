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

package dockershim

import (
	"testing"

	dockercontainer "github.com/docker/engine-api/types/container"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	"k8s.io/kubernetes/pkg/kubelet/types"
)

func TestConvertLegacyNameAndLabels(t *testing.T) {
	for desc, test := range map[string]struct {
		names        []string
		labels       map[string]string
		expectNames  []string
		expectLabels map[string]string
		expectError  bool
	}{

		"legacy infra container": {
			names: []string{"k8s_POD.hash1_podname_podnamespace_poduid_randomid"},
			labels: map[string]string{
				types.KubernetesPodNameLabel:       "podname",
				types.KubernetesPodNamespaceLabel:  "podnamespace",
				types.KubernetesPodUIDLabel:        "poduid",
				types.KubernetesContainerNameLabel: "POD",
				containerHashLabel:                 "hash",
				containerRestartCountLabel:         "0",
			},
			expectNames: []string{"k8s_POD_podname_podnamespace_poduid_0"},
			expectLabels: map[string]string{
				types.KubernetesPodNameLabel:                  "podname",
				types.KubernetesPodNamespaceLabel:             "podnamespace",
				types.KubernetesPodUIDLabel:                   "poduid",
				types.KubernetesContainerNameLabel:            "POD",
				annotationPrefix + containerHashLabel:         "hash",
				annotationPrefix + containerRestartCountLabel: "0",
				containerTypeLabelKey:                         containerTypeLabelSandbox,
			},
		},
		"legacy application container": {
			names: []string{"k8s_containername.hash_podname_podnamespace_poduid_randomid"},
			labels: map[string]string{
				types.KubernetesPodNameLabel:           "podname",
				types.KubernetesPodNamespaceLabel:      "podnamespace",
				types.KubernetesPodUIDLabel:            "poduid",
				types.KubernetesContainerNameLabel:     "containername",
				containerHashLabel:                     "hash",
				containerRestartCountLabel:             "5",
				containerTerminationMessagePathLabel:   "terminationmessagepath",
				containerTerminationMessagePolicyLabel: "terminationmessagepolicy",
				containerPreStopHandlerLabel:           "prestophandler",
				containerPortsLabel:                    "ports",
			},
			expectNames: []string{"k8s_containername_podname_podnamespace_poduid_5"},
			expectLabels: map[string]string{
				types.KubernetesPodNameLabel:                              "podname",
				types.KubernetesPodNamespaceLabel:                         "podnamespace",
				types.KubernetesPodUIDLabel:                               "poduid",
				types.KubernetesContainerNameLabel:                        "containername",
				annotationPrefix + containerHashLabel:                     "hash",
				annotationPrefix + containerRestartCountLabel:             "5",
				annotationPrefix + containerTerminationMessagePathLabel:   "terminationmessagepath",
				annotationPrefix + containerTerminationMessagePolicyLabel: "terminationmessagepolicy",
				annotationPrefix + containerPreStopHandlerLabel:           "prestophandler",
				annotationPrefix + containerPortsLabel:                    "ports",
				containerTypeLabelKey:                                     containerTypeLabelContainer,
			},
			expectError: false,
		},
		"invalid sandbox name": {
			names:       []string{"POD_podname_podnamespace_poduid_0"},
			expectError: true,
		},
		"invalid dockershim container": {
			names:       []string{"containername_podname_podnamespace_poduid_5"},
			expectError: true,
		},
	} {
		t.Logf("TestCase %q", desc)
		names, labels, err := convertLegacyNameAndLabels(test.names, test.labels)
		require.Equal(t, test.expectError, err != nil)
		assert.Equal(t, test.expectNames, names)
		assert.Equal(t, test.expectLabels, labels)
	}
}

// getFakeLegacyContainers returns a list of fake legacy containers.
func getFakeLegacyContainers() []*libdocker.FakeContainer {
	return []*libdocker.FakeContainer{
		{
			ID:   "12",
			Name: "k8s_POD.hash1_podname_podnamespace_poduid_randomid",
			Config: &dockercontainer.Config{
				Labels: map[string]string{
					types.KubernetesPodNameLabel:       "podname",
					types.KubernetesPodNamespaceLabel:  "podnamespace",
					types.KubernetesPodUIDLabel:        "poduid",
					types.KubernetesContainerNameLabel: "POD",
					containerHashLabel:                 "hash1",
					containerRestartCountLabel:         "0",
				},
			},
		},
		{
			ID:   "34",
			Name: "k8s_legacycontainer.hash2_podname_podnamespace_poduid_randomid",
			Config: &dockercontainer.Config{
				Labels: map[string]string{
					types.KubernetesPodNameLabel:       "podname",
					types.KubernetesPodNamespaceLabel:  "podnamespace",
					types.KubernetesPodUIDLabel:        "poduid",
					types.KubernetesContainerNameLabel: "legacyContainer",
					containerHashLabel:                 "hash2",
					containerRestartCountLabel:         "5",
				},
			},
		},
	}
}

// getFakeNewContainers returns a list of fake new containers.
func getFakeNewContainers() []*libdocker.FakeContainer {
	return []*libdocker.FakeContainer{
		{
			ID:   "56",
			Name: "k8s_POD_podname_podnamespace_poduid_0",
			Config: &dockercontainer.Config{
				Labels: map[string]string{
					types.KubernetesPodNameLabel:       "podname",
					types.KubernetesPodNamespaceLabel:  "podnamespace",
					types.KubernetesPodUIDLabel:        "poduid",
					types.KubernetesContainerNameLabel: "POD",
					containerTypeLabelKey:              containerTypeLabelSandbox,
				},
			},
		},
		{
			ID:   "78",
			Name: "k8s_newcontainer_podname_podnamespace_poduid_3",
			Config: &dockercontainer.Config{
				Labels: map[string]string{
					types.KubernetesPodNameLabel:                  "podname",
					types.KubernetesPodNamespaceLabel:             "podnamespace",
					types.KubernetesPodUIDLabel:                   "poduid",
					types.KubernetesContainerNameLabel:            "newcontainer",
					annotationPrefix + containerHashLabel:         "hash4",
					annotationPrefix + containerRestartCountLabel: "3",
					containerTypeLabelKey:                         containerTypeLabelContainer,
				},
			},
		},
	}

}

func TestListLegacyContainers(t *testing.T) {
	ds, fDocker, _ := newTestDockerService()
	newContainers := getFakeLegacyContainers()
	legacyContainers := getFakeNewContainers()
	fDocker.SetFakeContainers(append(newContainers, legacyContainers...))

	// ListContainers should list only new containers when legacyCleanup is done.
	containers, err := ds.ListContainers(nil)
	assert.NoError(t, err)
	require.Len(t, containers, 1)
	assert.Equal(t, "78", containers[0].Id)

	// ListLegacyContainers should list only legacy containers.
	containers, err = ds.ListLegacyContainers(nil)
	assert.NoError(t, err)
	require.Len(t, containers, 1)
	assert.Equal(t, "34", containers[0].Id)

	// Mark legacyCleanup as not done.
	ds.legacyCleanup.done = 0

	// ListContainers should list all containers when legacyCleanup is not done.
	containers, err = ds.ListContainers(nil)
	assert.NoError(t, err)
	require.Len(t, containers, 2)
	assert.Contains(t, []string{containers[0].Id, containers[1].Id}, "34")
	assert.Contains(t, []string{containers[0].Id, containers[1].Id}, "78")
}

func TestListLegacyPodSandbox(t *testing.T) {
	ds, fDocker, _ := newTestDockerService()
	newContainers := getFakeLegacyContainers()
	legacyContainers := getFakeNewContainers()
	fDocker.SetFakeContainers(append(newContainers, legacyContainers...))

	// ListPodSandbox should list only new sandboxes when legacyCleanup is done.
	sandboxes, err := ds.ListPodSandbox(nil)
	assert.NoError(t, err)
	require.Len(t, sandboxes, 1)
	assert.Equal(t, "56", sandboxes[0].Id)

	// ListLegacyPodSandbox should list only legacy sandboxes.
	sandboxes, err = ds.ListLegacyPodSandbox(nil)
	assert.NoError(t, err)
	require.Len(t, sandboxes, 1)
	assert.Equal(t, "12", sandboxes[0].Id)

	// Mark legacyCleanup as not done.
	ds.legacyCleanup.done = 0

	// ListPodSandbox should list all sandboxes when legacyCleanup is not done.
	sandboxes, err = ds.ListPodSandbox(nil)
	assert.NoError(t, err)
	require.Len(t, sandboxes, 2)
	assert.Contains(t, []string{sandboxes[0].Id, sandboxes[1].Id}, "12")
	assert.Contains(t, []string{sandboxes[0].Id, sandboxes[1].Id}, "56")
}

func TestCheckLegacyCleanup(t *testing.T) {
	for desc, test := range map[string]struct {
		containers []*libdocker.FakeContainer
		done       bool
	}{
		"no containers": {
			containers: []*libdocker.FakeContainer{},
			done:       true,
		},
		"only new containers": {
			containers: getFakeNewContainers(),
			done:       true,
		},
		"only legacy containers": {
			containers: getFakeLegacyContainers(),
			done:       false,
		},
		"both legacy and new containers": {
			containers: append(getFakeNewContainers(), getFakeLegacyContainers()...),
			done:       false,
		},
	} {
		t.Logf("TestCase %q", desc)
		ds, fDocker, _ := newTestDockerService()
		fDocker.SetFakeContainers(test.containers)
		ds.legacyCleanup.done = 0

		clean, err := ds.checkLegacyCleanup()
		assert.NoError(t, err)
		assert.Equal(t, test.done, clean)
		assert.Equal(t, test.done, ds.legacyCleanup.Done())
	}
}
