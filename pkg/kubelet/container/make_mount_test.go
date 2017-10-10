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

package container

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/api/core/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	// TODO: remove this import if
	// api.Registry.GroupOrDie(v1.GroupName).GroupVersion.String() is changed
	// to "v1"?
	_ "k8s.io/kubernetes/pkg/api/install"
	runtimeapi "k8s.io/kubernetes/pkg/kubelet/apis/cri/v1alpha1/runtime"
	"k8s.io/kubernetes/pkg/volume"
)

type stubVolume struct {
	path string
	volume.MetricsNil
}

func (f *stubVolume) GetPath() string {
	return f.path
}

func (f *stubVolume) GetAttributes() volume.Attributes {
	return volume.Attributes{}
}

func (f *stubVolume) CanMount() error {
	return nil
}

func (f *stubVolume) SetUp(fsGroup *int64) error {
	return nil
}

func (f *stubVolume) SetUpAt(dir string, fsGroup *int64) error {
	return nil
}

func TestMakeMounts(t *testing.T) {
	bTrue := true
	propagationHostToContainer := v1.MountPropagationHostToContainer
	propagationBidirectional := v1.MountPropagationBidirectional

	testCases := map[string]struct {
		container      v1.Container
		podVolumes     VolumeMap
		expectErr      bool
		expectedErrMsg string
		expectedMounts []Mount
	}{
		"valid mounts in unprivileged container": {
			podVolumes: VolumeMap{
				"disk":  VolumeInfo{Mounter: &stubVolume{path: "/mnt/disk"}},
				"disk4": VolumeInfo{Mounter: &stubVolume{path: "/mnt/host"}},
				"disk5": VolumeInfo{Mounter: &stubVolume{path: "/var/lib/kubelet/podID/volumes/empty/disk5"}},
			},
			container: v1.Container{
				Name: "container1",
				VolumeMounts: []v1.VolumeMount{
					{
						MountPath:        "/etc/hosts",
						Name:             "disk",
						ReadOnly:         false,
						MountPropagation: &propagationHostToContainer,
					},
					{
						MountPath: "/mnt/path3",
						Name:      "disk",
						ReadOnly:  true,
					},
					{
						MountPath: "/mnt/path4",
						Name:      "disk4",
						ReadOnly:  false,
					},
					{
						MountPath: "/mnt/path5",
						Name:      "disk5",
						ReadOnly:  false,
					},
				},
			},
			expectedMounts: []Mount{
				{
					Name:           "disk",
					ContainerPath:  "/etc/hosts",
					HostPath:       "/mnt/disk",
					ReadOnly:       false,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
				},
				{
					Name:           "disk",
					ContainerPath:  "/mnt/path3",
					HostPath:       "/mnt/disk",
					ReadOnly:       true,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
				},
				{
					Name:           "disk4",
					ContainerPath:  "/mnt/path4",
					HostPath:       "/mnt/host",
					ReadOnly:       false,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
				},
				{
					Name:           "disk5",
					ContainerPath:  "/mnt/path5",
					HostPath:       "/var/lib/kubelet/podID/volumes/empty/disk5",
					ReadOnly:       false,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
				},
			},
			expectErr: false,
		},
		"valid mounts in privileged container": {
			podVolumes: VolumeMap{
				"disk":  VolumeInfo{Mounter: &stubVolume{path: "/mnt/disk"}},
				"disk4": VolumeInfo{Mounter: &stubVolume{path: "/mnt/host"}},
				"disk5": VolumeInfo{Mounter: &stubVolume{path: "/var/lib/kubelet/podID/volumes/empty/disk5"}},
			},
			container: v1.Container{
				Name: "container1",
				VolumeMounts: []v1.VolumeMount{
					{
						MountPath:        "/etc/hosts",
						Name:             "disk",
						ReadOnly:         false,
						MountPropagation: &propagationBidirectional,
					},
					{
						MountPath:        "/mnt/path3",
						Name:             "disk",
						ReadOnly:         true,
						MountPropagation: &propagationHostToContainer,
					},
					{
						MountPath: "/mnt/path4",
						Name:      "disk4",
						ReadOnly:  false,
					},
				},
				SecurityContext: &v1.SecurityContext{
					Privileged: &bTrue,
				},
			},
			expectedMounts: []Mount{
				{
					Name:           "disk",
					ContainerPath:  "/etc/hosts",
					HostPath:       "/mnt/disk",
					ReadOnly:       false,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_BIDIRECTIONAL,
				},
				{
					Name:           "disk",
					ContainerPath:  "/mnt/path3",
					HostPath:       "/mnt/disk",
					ReadOnly:       true,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
				},
				{
					Name:           "disk4",
					ContainerPath:  "/mnt/path4",
					HostPath:       "/mnt/host",
					ReadOnly:       false,
					SELinuxRelabel: false,
					Propagation:    runtimeapi.MountPropagation_PROPAGATION_HOST_TO_CONTAINER,
				},
			},
			expectErr: false,
		},
		"invalid absolute SubPath": {
			podVolumes: VolumeMap{
				"disk": VolumeInfo{Mounter: &stubVolume{path: "/mnt/disk"}},
			},
			container: v1.Container{
				VolumeMounts: []v1.VolumeMount{
					{
						MountPath: "/mnt/path3",
						SubPath:   "/must/not/be/absolute",
						Name:      "disk",
						ReadOnly:  true,
					},
				},
			},
			expectErr:      true,
			expectedErrMsg: "error SubPath `/must/not/be/absolute` must not be an absolute path",
		},
		"invalid SubPath with backsteps": {
			podVolumes: VolumeMap{
				"disk": VolumeInfo{Mounter: &stubVolume{path: "/mnt/disk"}},
			},
			container: v1.Container{
				VolumeMounts: []v1.VolumeMount{
					{
						MountPath: "/mnt/path3",
						SubPath:   "no/backsteps/../allowed",
						Name:      "disk",
						ReadOnly:  true,
					},
				},
			},
			expectErr:      true,
			expectedErrMsg: "unable to provision SubPath `no/backsteps/../allowed`: must not contain '..'",
		},
	}

	for name, tc := range testCases {
		t.Run(name, func(t *testing.T) {
			pod := v1.Pod{
				Spec: v1.PodSpec{
					HostNetwork: true,
				},
			}
			// test MakeMounts with enabled mount propagation
			err := utilfeature.DefaultFeatureGate.Set("MountPropagation=true")
			if err != nil {
				t.Errorf("Failed to enable feature gate for MountPropagation: %v", err)
				return
			}

			mounts, err := MakeMounts(&pod, "/pod", &tc.container, "fakepodname", "", "", tc.podVolumes)

			// validate only the error if we expect an error
			if tc.expectErr {
				if err == nil || err.Error() != tc.expectedErrMsg {
					t.Fatalf("expected error message `%s` but got `%v`", tc.expectedErrMsg, err)
				}
				return
			}

			// otherwise validate the mounts
			if err != nil {
				t.Fatal(err)
			}

			assert.Equal(t, tc.expectedMounts, mounts, "mounts of container %+v", tc.container)

			// test MakeMounts with disabled mount propagation
			err = utilfeature.DefaultFeatureGate.Set("MountPropagation=false")
			if err != nil {
				t.Errorf("Failed to enable feature gate for MountPropagation: %v", err)
				return
			}
			mounts, err = MakeMounts(&pod, "/pod", &tc.container, "fakepodname", "", "", tc.podVolumes)
			if !tc.expectErr {
				expectedPrivateMounts := []Mount{}
				for _, mount := range tc.expectedMounts {
					// all mounts are expected to be private when mount
					// propagation is disabled
					mount.Propagation = runtimeapi.MountPropagation_PROPAGATION_PRIVATE
					expectedPrivateMounts = append(expectedPrivateMounts, mount)
				}
				assert.Equal(t, expectedPrivateMounts, mounts, "mounts of container %+v", tc.container)
			}
		})
	}
}

func TestManagedHostsFileContent(t *testing.T) {
	testCases := []struct {
		hostIP          string
		hostName        string
		hostDomainName  string
		hostAliases     []v1.HostAlias
		expectedContent string
	}{
		{
			"123.45.67.89",
			"podFoo",
			"",
			[]v1.HostAlias{},
			`# Kubernetes-managed hosts file.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
123.45.67.89	podFoo
`,
		},
		{
			"203.0.113.1",
			"podFoo",
			"domainFoo",
			[]v1.HostAlias{},
			`# Kubernetes-managed hosts file.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
203.0.113.1	podFoo.domainFoo	podFoo
`,
		},
		{
			"203.0.113.1",
			"podFoo",
			"domainFoo",
			[]v1.HostAlias{
				{IP: "123.45.67.89", Hostnames: []string{"foo", "bar", "baz"}},
			},
			`# Kubernetes-managed hosts file.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
203.0.113.1	podFoo.domainFoo	podFoo

# Entries added by HostAliases.
123.45.67.89	foo
123.45.67.89	bar
123.45.67.89	baz
`,
		},
		{
			"203.0.113.1",
			"podFoo",
			"domainFoo",
			[]v1.HostAlias{
				{IP: "123.45.67.89", Hostnames: []string{"foo", "bar", "baz"}},
				{IP: "456.78.90.123", Hostnames: []string{"park", "doo", "boo"}},
			},
			`# Kubernetes-managed hosts file.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
203.0.113.1	podFoo.domainFoo	podFoo

# Entries added by HostAliases.
123.45.67.89	foo
123.45.67.89	bar
123.45.67.89	baz
456.78.90.123	park
456.78.90.123	doo
456.78.90.123	boo
`,
		},
	}

	for _, testCase := range testCases {
		actualContent := managedHostsFileContent(testCase.hostIP, testCase.hostName, testCase.hostDomainName, testCase.hostAliases)
		assert.Equal(t, testCase.expectedContent, string(actualContent), "hosts file content not expected")
	}
}

func TestNodeHostsFileContent(t *testing.T) {
	testCases := []struct {
		hostsFileName            string
		hostAliases              []v1.HostAlias
		rawHostsFileContent      string
		expectedHostsFileContent string
	}{
		{
			"hosts_test_file1",
			[]v1.HostAlias{},
			`# hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
123.45.67.89	some.domain
`,
			`# hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
123.45.67.89	some.domain
`,
		},
		{
			"hosts_test_file2",
			[]v1.HostAlias{},
			`# another hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
12.34.56.78	another.domain
`,
			`# another hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
12.34.56.78	another.domain
`,
		},
		{
			"hosts_test_file1_with_host_aliases",
			[]v1.HostAlias{
				{IP: "123.45.67.89", Hostnames: []string{"foo", "bar", "baz"}},
			},
			`# hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
123.45.67.89	some.domain
`,
			`# hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
123.45.67.89	some.domain

# Entries added by HostAliases.
123.45.67.89	foo
123.45.67.89	bar
123.45.67.89	baz
`,
		},
		{
			"hosts_test_file2_with_host_aliases",
			[]v1.HostAlias{
				{IP: "123.45.67.89", Hostnames: []string{"foo", "bar", "baz"}},
				{IP: "456.78.90.123", Hostnames: []string{"park", "doo", "boo"}},
			},
			`# another hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
12.34.56.78	another.domain
`,
			`# another hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
12.34.56.78	another.domain

# Entries added by HostAliases.
123.45.67.89	foo
123.45.67.89	bar
123.45.67.89	baz
456.78.90.123	park
456.78.90.123	doo
456.78.90.123	boo
`,
		},
	}

	for _, testCase := range testCases {
		tmpdir, err := writeHostsFile(testCase.hostsFileName, testCase.rawHostsFileContent)
		require.NoError(t, err, "could not create a temp hosts file")
		defer os.RemoveAll(tmpdir)

		actualContent, fileReadErr := nodeHostsFileContent(filepath.Join(tmpdir, testCase.hostsFileName), testCase.hostAliases)
		require.NoError(t, fileReadErr, "could not create read hosts file")
		assert.Equal(t, testCase.expectedHostsFileContent, string(actualContent), "hosts file content not expected")
	}
}

// writeHostsFile will write a hosts file into a temporary dir, and return that dir.
// Caller is responsible for deleting the dir and its contents.
func writeHostsFile(filename string, cfg string) (string, error) {
	tmpdir, err := ioutil.TempDir("", "kubelet=kubelet_pods_test.go=")
	if err != nil {
		return "", err
	}
	return tmpdir, ioutil.WriteFile(filepath.Join(tmpdir, filename), []byte(cfg), 0644)
}
