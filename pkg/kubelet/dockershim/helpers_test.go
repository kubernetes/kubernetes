/*
Copyright 2016 The Kubernetes Authors.

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
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/blang/semver"
	dockertypes "github.com/docker/docker/api/types"
	dockernat "github.com/docker/go-connections/nat"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/kubelet/dockershim/libdocker"
	"k8s.io/kubernetes/pkg/security/apparmor"
)

func TestLabelsAndAnnotationsRoundTrip(t *testing.T) {
	expectedLabels := map[string]string{"foo.123.abc": "baz", "bar.456.xyz": "qwe"}
	expectedAnnotations := map[string]string{"uio.ert": "dfs", "jkl": "asd"}
	// Merge labels and annotations into docker labels.
	dockerLabels := makeLabels(expectedLabels, expectedAnnotations)
	// Extract labels and annotations from docker labels.
	actualLabels, actualAnnotations := extractLabels(dockerLabels)
	assert.Equal(t, expectedLabels, actualLabels)
	assert.Equal(t, expectedAnnotations, actualAnnotations)
}

// TestGetApparmorSecurityOpts tests the logic of generating container apparmor options from sandbox annotations.
func TestGetApparmorSecurityOpts(t *testing.T) {
	makeConfig := func(profile string) *runtimeapi.LinuxContainerSecurityContext {
		return &runtimeapi.LinuxContainerSecurityContext{
			ApparmorProfile: profile,
		}
	}

	tests := []struct {
		msg          string
		config       *runtimeapi.LinuxContainerSecurityContext
		expectedOpts []string
	}{{
		msg:          "No AppArmor options",
		config:       makeConfig(""),
		expectedOpts: nil,
	}, {
		msg:          "AppArmor runtime/default",
		config:       makeConfig("runtime/default"),
		expectedOpts: []string{},
	}, {
		msg:          "AppArmor local profile",
		config:       makeConfig(apparmor.ProfileNamePrefix + "foo"),
		expectedOpts: []string{"apparmor=foo"},
	}}

	for i, test := range tests {
		opts, err := getApparmorSecurityOpts(test.config, '=')
		assert.NoError(t, err, "TestCase[%d]: %s", i, test.msg)
		assert.Len(t, opts, len(test.expectedOpts), "TestCase[%d]: %s", i, test.msg)
		for _, opt := range test.expectedOpts {
			assert.Contains(t, opts, opt, "TestCase[%d]: %s", i, test.msg)
		}
	}
}

// TestGetUserFromImageUser tests the logic of getting image uid or user name of image user.
func TestGetUserFromImageUser(t *testing.T) {
	newI64 := func(i int64) *int64 { return &i }
	for c, test := range map[string]struct {
		user string
		uid  *int64
		name string
	}{
		"no gid": {
			user: "0",
			uid:  newI64(0),
		},
		"uid/gid": {
			user: "0:1",
			uid:  newI64(0),
		},
		"empty user": {
			user: "",
		},
		"multiple spearators": {
			user: "1:2:3",
			uid:  newI64(1),
		},
		"root username": {
			user: "root:root",
			name: "root",
		},
		"username": {
			user: "test:test",
			name: "test",
		},
	} {
		t.Logf("TestCase - %q", c)
		actualUID, actualName := getUserFromImageUser(test.user)
		assert.Equal(t, test.uid, actualUID)
		assert.Equal(t, test.name, actualName)
	}
}

func TestParsingCreationConflictError(t *testing.T) {
	// Expected error message from docker.
	msg := "Conflict. The name \"/k8s_POD_pfpod_e2e-tests-port-forwarding-dlxt2_81a3469e-99e1-11e6-89f2-42010af00002_0\" is already in use by container 24666ab8c814d16f986449e504ea0159468ddf8da01897144a770f66dce0e14e. You have to remove (or rename) that container to be able to reuse that name."

	matches := conflictRE.FindStringSubmatch(msg)
	require.Len(t, matches, 2)
	require.Equal(t, matches[1], "24666ab8c814d16f986449e504ea0159468ddf8da01897144a770f66dce0e14e")
}

func TestGetSecurityOptSeparator(t *testing.T) {
	for c, test := range map[string]struct {
		desc     string
		version  *semver.Version
		expected rune
	}{
		"older docker version": {
			version:  &semver.Version{Major: 1, Minor: 22, Patch: 0},
			expected: ':',
		},
		"changed docker version": {
			version:  &semver.Version{Major: 1, Minor: 23, Patch: 0},
			expected: '=',
		},
		"newer docker version": {
			version:  &semver.Version{Major: 1, Minor: 24, Patch: 0},
			expected: '=',
		},
	} {
		actual := getSecurityOptSeparator(test.version)
		assert.Equal(t, test.expected, actual, c)
	}
}

// writeDockerConfig will write a config file into a temporary dir, and return that dir.
// Caller is responsible for deleting the dir and its contents.
func writeDockerConfig(cfg string) (string, error) {
	tmpdir, err := ioutil.TempDir("", "dockershim=helpers_test.go=")
	if err != nil {
		return "", err
	}
	dir := filepath.Join(tmpdir, ".docker")
	if err := os.Mkdir(dir, 0755); err != nil {
		return "", err
	}
	return tmpdir, ioutil.WriteFile(filepath.Join(dir, "config.json"), []byte(cfg), 0644)
}

func TestEnsureSandboxImageExists(t *testing.T) {
	sandboxImage := "gcr.io/test/image"
	authConfig := dockertypes.AuthConfig{Username: "user", Password: "pass"}
	for desc, test := range map[string]struct {
		injectImage  bool
		imgNeedsAuth bool
		injectErr    error
		calls        []string
		err          bool
		configJSON   string
	}{
		"should not pull image when it already exists": {
			injectImage: true,
			injectErr:   nil,
			calls:       []string{"inspect_image"},
		},
		"should pull image when it doesn't exist": {
			injectImage: false,
			injectErr:   libdocker.ImageNotFoundError{ID: "image_id"},
			calls:       []string{"inspect_image", "pull"},
		},
		"should return error when inspect image fails": {
			injectImage: false,
			injectErr:   fmt.Errorf("arbitrary error"),
			calls:       []string{"inspect_image"},
			err:         true,
		},
		"should return error when image pull needs private auth, but none provided": {
			injectImage:  true,
			imgNeedsAuth: true,
			injectErr:    libdocker.ImageNotFoundError{ID: "image_id"},
			calls:        []string{"inspect_image", "pull"},
			err:          true,
		},
	} {
		t.Logf("TestCase: %q", desc)
		_, fakeDocker, _ := newTestDockerService()
		if test.injectImage {
			images := []dockertypes.ImageSummary{{ID: sandboxImage}}
			fakeDocker.InjectImages(images)
			if test.imgNeedsAuth {
				fakeDocker.MakeImagesPrivate(images, authConfig)
			}
		}
		fakeDocker.InjectError("inspect_image", test.injectErr)

		err := ensureSandboxImageExists(fakeDocker, sandboxImage)
		assert.NoError(t, fakeDocker.AssertCalls(test.calls))
		assert.Equal(t, test.err, err != nil)
	}
}

func TestMakePortsAndBindings(t *testing.T) {
	for desc, test := range map[string]struct {
		pm           []*runtimeapi.PortMapping
		exposedPorts dockernat.PortSet
		portmappings map[dockernat.Port][]dockernat.PortBinding
	}{
		"no port mapping": {
			pm:           nil,
			exposedPorts: map[dockernat.Port]struct{}{},
			portmappings: map[dockernat.Port][]dockernat.PortBinding{},
		},
		"tcp port mapping": {
			pm: []*runtimeapi.PortMapping{
				{
					Protocol:      runtimeapi.Protocol_TCP,
					ContainerPort: 80,
					HostPort:      80,
				},
			},
			exposedPorts: map[dockernat.Port]struct{}{
				"80/tcp": {},
			},
			portmappings: map[dockernat.Port][]dockernat.PortBinding{
				"80/tcp": {
					{
						HostPort: "80",
					},
				},
			},
		},
		"udp port mapping": {
			pm: []*runtimeapi.PortMapping{
				{
					Protocol:      runtimeapi.Protocol_UDP,
					ContainerPort: 80,
					HostPort:      80,
				},
			},
			exposedPorts: map[dockernat.Port]struct{}{
				"80/udp": {},
			},
			portmappings: map[dockernat.Port][]dockernat.PortBinding{
				"80/udp": {
					{
						HostPort: "80",
					},
				},
			},
		},
		"multipe port mappings": {
			pm: []*runtimeapi.PortMapping{
				{
					Protocol:      runtimeapi.Protocol_TCP,
					ContainerPort: 80,
					HostPort:      80,
				},
				{
					Protocol:      runtimeapi.Protocol_TCP,
					ContainerPort: 80,
					HostPort:      81,
				},
			},
			exposedPorts: map[dockernat.Port]struct{}{
				"80/tcp": {},
			},
			portmappings: map[dockernat.Port][]dockernat.PortBinding{
				"80/tcp": {
					{
						HostPort: "80",
					},
					{
						HostPort: "81",
					},
				},
			},
		},
	} {
		t.Logf("TestCase: %s", desc)
		actualExposedPorts, actualPortMappings := makePortsAndBindings(test.pm)
		assert.Equal(t, test.exposedPorts, actualExposedPorts)
		assert.Equal(t, test.portmappings, actualPortMappings)
	}
}

func TestGenerateMountBindings(t *testing.T) {
	mounts := []*runtimeapi.Mount{
		// everything default
		{
			HostPath:      "/mnt/1",
			ContainerPath: "/var/lib/mysql/1",
		},
		// readOnly
		{
			HostPath:      "/mnt/2",
			ContainerPath: "/var/lib/mysql/2",
			Readonly:      true,
		},
		// SELinux
		{
			HostPath:       "/mnt/3",
			ContainerPath:  "/var/lib/mysql/3",
			SelinuxRelabel: true,
		},
		// Propagation private
		{
			HostPath:      "/mnt/4",
			ContainerPath: "/var/lib/mysql/4",
			Propagation:   runtimeapi.MountPropagation_PROPAGATION_PRIVATE,
		},
		// Propagation rslave
		{
			HostPath:      "/mnt/5",
			ContainerPath: "/var/lib/mysql/5",
			Propagation:   runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
		},
		// Propagation rshared
		{
			HostPath:      "/mnt/6",
			ContainerPath: "/var/lib/mysql/6",
			Propagation:   runtimeapi.MountPropagation_PROPAGATION_BIDIRECTIONAL,
		},
		// Propagation unknown (falls back to private)
		{
			HostPath:      "/mnt/7",
			ContainerPath: "/var/lib/mysql/7",
			Propagation:   runtimeapi.MountPropagation(42),
		},
		// Everything
		{
			HostPath:       "/mnt/8",
			ContainerPath:  "/var/lib/mysql/8",
			Readonly:       true,
			SelinuxRelabel: true,
			Propagation:    runtimeapi.MountPropagation_PROPAGATION_BIDIRECTIONAL,
		},
	}
	expectedResult := []string{
		"/mnt/1:/var/lib/mysql/1",
		"/mnt/2:/var/lib/mysql/2:ro",
		"/mnt/3:/var/lib/mysql/3:Z",
		"/mnt/4:/var/lib/mysql/4",
		"/mnt/5:/var/lib/mysql/5:rslave",
		"/mnt/6:/var/lib/mysql/6:rshared",
		"/mnt/7:/var/lib/mysql/7",
		"/mnt/8:/var/lib/mysql/8:ro,Z,rshared",
	}
	result := generateMountBindings(mounts)

	assert.Equal(t, result, expectedResult)
}
