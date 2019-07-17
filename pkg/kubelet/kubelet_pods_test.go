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

package kubelet

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	featuregatetesting "k8s.io/component-base/featuregate/testing"

	// TODO: remove this import if
	// api.Registry.GroupOrDie(v1.GroupName).GroupVersions[0].String() is changed
	// to "v1"?

	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/server/portforward"
	"k8s.io/kubernetes/pkg/kubelet/server/remotecommand"
	"k8s.io/kubernetes/pkg/util/mount"
	"k8s.io/kubernetes/pkg/volume/util/subpath"
)

func TestDisabledSubpath(t *testing.T) {
	fhu := &mount.FakeHostUtil{}
	fsp := &subpath.FakeSubpath{}
	pod := v1.Pod{
		Spec: v1.PodSpec{
			HostNetwork: true,
		},
	}
	podVolumes := kubecontainer.VolumeMap{
		"disk": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/disk"}},
	}

	cases := map[string]struct {
		container   v1.Container
		expectError bool
	}{
		"subpath not specified": {
			v1.Container{
				VolumeMounts: []v1.VolumeMount{
					{
						MountPath: "/mnt/path3",
						Name:      "disk",
						ReadOnly:  true,
					},
				},
			},
			false,
		},
		"subpath specified": {
			v1.Container{
				VolumeMounts: []v1.VolumeMount{
					{
						MountPath: "/mnt/path3",
						SubPath:   "/must/not/be/absolute",
						Name:      "disk",
						ReadOnly:  true,
					},
				},
			},
			true,
		},
	}

	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.VolumeSubpath, false)()
	for name, test := range cases {
		_, _, err := makeMounts(&pod, "/pod", &test.container, "fakepodname", "", "", podVolumes, fhu, fsp, nil)
		if err != nil && !test.expectError {
			t.Errorf("test %v failed: %v", name, err)
		}
		if err == nil && test.expectError {
			t.Errorf("test %v failed: expected error", name)
		}
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
			`# Kubernetes-managed hosts file (host network).
# hosts file for testing.
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
			`# Kubernetes-managed hosts file (host network).
# another hosts file for testing.
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
			`# Kubernetes-managed hosts file (host network).
# hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
123.45.67.89	some.domain

# Entries added by HostAliases.
123.45.67.89	foo	bar	baz
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
			`# Kubernetes-managed hosts file (host network).
# another hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
12.34.56.78	another.domain

# Entries added by HostAliases.
123.45.67.89	foo	bar	baz
456.78.90.123	park	doo	boo
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
123.45.67.89	foo	bar	baz
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
123.45.67.89	foo	bar	baz
456.78.90.123	park	doo	boo
`,
		},
	}

	for _, testCase := range testCases {
		actualContent := managedHostsFileContent(testCase.hostIP, testCase.hostName, testCase.hostDomainName, testCase.hostAliases)
		assert.Equal(t, testCase.expectedContent, string(actualContent), "hosts file content not expected")
	}
}

func TestRunInContainerNoSuchPod(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	defer testKubelet.Cleanup()
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
	fakeRuntime.PodList = []*containertest.FakePod{}

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerName := "containerFoo"
	output, err := kubelet.RunInContainer(
		kubecontainer.GetPodFullName(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: podName, Namespace: podNamespace}}),
		"",
		containerName,
		[]string{"ls"})
	assert.Error(t, err)
	assert.Nil(t, output, "output should be nil")
}

func TestRunInContainer(t *testing.T) {
	for _, testError := range []error{nil, errors.New("bar")} {
		testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
		defer testKubelet.Cleanup()
		kubelet := testKubelet.kubelet
		fakeRuntime := testKubelet.fakeRuntime
		fakeCommandRunner := containertest.FakeContainerCommandRunner{
			Err:    testError,
			Stdout: "foo",
		}
		kubelet.runner = &fakeCommandRunner

		containerID := kubecontainer.ContainerID{Type: "test", ID: "abc1234"}
		fakeRuntime.PodList = []*containertest.FakePod{
			{Pod: &kubecontainer.Pod{
				ID:        "12345678",
				Name:      "podFoo",
				Namespace: "nsFoo",
				Containers: []*kubecontainer.Container{
					{Name: "containerFoo",
						ID: containerID,
					},
				},
			}},
		}
		cmd := []string{"ls"}
		actualOutput, err := kubelet.RunInContainer("podFoo_nsFoo", "", "containerFoo", cmd)
		assert.Equal(t, containerID, fakeCommandRunner.ContainerID, "(testError=%v) ID", testError)
		assert.Equal(t, cmd, fakeCommandRunner.Cmd, "(testError=%v) command", testError)
		// this isn't 100% foolproof as a bug in a real ContainerCommandRunner where it fails to copy to stdout/stderr wouldn't be caught by this test
		assert.Equal(t, "foo", string(actualOutput), "(testError=%v) output", testError)
		assert.Equal(t, err, testError, "(testError=%v) err", testError)
	}
}

type testServiceLister struct {
	services []*v1.Service
}

func (ls testServiceLister) List(labels.Selector) ([]*v1.Service, error) {
	return ls.services, nil
}

type envs []kubecontainer.EnvVar

func (e envs) Len() int {
	return len(e)
}

func (e envs) Swap(i, j int) { e[i], e[j] = e[j], e[i] }

func (e envs) Less(i, j int) bool { return e[i].Name < e[j].Name }

func buildService(name, namespace, clusterIP, protocol string, port int) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: name, Namespace: namespace},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				Protocol: v1.Protocol(protocol),
				Port:     int32(port),
			}},
			ClusterIP: clusterIP,
		},
	}
}

func TestMakeEnvironmentVariables(t *testing.T) {
	trueVal := true
	services := []*v1.Service{
		buildService("kubernetes", metav1.NamespaceDefault, "1.2.3.1", "TCP", 8081),
		buildService("test", "test1", "1.2.3.3", "TCP", 8083),
		buildService("kubernetes", "test2", "1.2.3.4", "TCP", 8084),
		buildService("test", "test2", "1.2.3.5", "TCP", 8085),
		buildService("test", "test2", "None", "TCP", 8085),
		buildService("test", "test2", "", "TCP", 8085),
		buildService("kubernetes", "kubernetes", "1.2.3.6", "TCP", 8086),
		buildService("not-special", "kubernetes", "1.2.3.8", "TCP", 8088),
		buildService("not-special", "kubernetes", "None", "TCP", 8088),
		buildService("not-special", "kubernetes", "", "TCP", 8088),
	}

	trueValue := true
	falseValue := false
	testCases := []struct {
		name               string                 // the name of the test case
		ns                 string                 // the namespace to generate environment for
		enableServiceLinks *bool                  // enabling service links
		container          *v1.Container          // the container to use
		masterServiceNs    string                 // the namespace to read master service info from
		nilLister          bool                   // whether the lister should be nil
		configMap          *v1.ConfigMap          // an optional ConfigMap to pull from
		secret             *v1.Secret             // an optional Secret to pull from
		expectedEnvs       []kubecontainer.EnvVar // a set of expected environment vars
		expectedError      bool                   // does the test fail
		expectedEvent      string                 // does the test emit an event
	}{
		{
			name:               "api server = Y, kubelet = Y",
			ns:                 "test1",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{Name: "FOO", Value: "BAR"},
					{Name: "TEST_SERVICE_HOST", Value: "1.2.3.3"},
					{Name: "TEST_SERVICE_PORT", Value: "8083"},
					{Name: "TEST_PORT", Value: "tcp://1.2.3.3:8083"},
					{Name: "TEST_PORT_8083_TCP", Value: "tcp://1.2.3.3:8083"},
					{Name: "TEST_PORT_8083_TCP_PROTO", Value: "tcp"},
					{Name: "TEST_PORT_8083_TCP_PORT", Value: "8083"},
					{Name: "TEST_PORT_8083_TCP_ADDR", Value: "1.2.3.3"},
				},
			},
			masterServiceNs: metav1.NamespaceDefault,
			nilLister:       false,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "FOO", Value: "BAR"},
				{Name: "TEST_SERVICE_HOST", Value: "1.2.3.3"},
				{Name: "TEST_SERVICE_PORT", Value: "8083"},
				{Name: "TEST_PORT", Value: "tcp://1.2.3.3:8083"},
				{Name: "TEST_PORT_8083_TCP", Value: "tcp://1.2.3.3:8083"},
				{Name: "TEST_PORT_8083_TCP_PROTO", Value: "tcp"},
				{Name: "TEST_PORT_8083_TCP_PORT", Value: "8083"},
				{Name: "TEST_PORT_8083_TCP_ADDR", Value: "1.2.3.3"},
				{Name: "KUBERNETES_SERVICE_PORT", Value: "8081"},
				{Name: "KUBERNETES_SERVICE_HOST", Value: "1.2.3.1"},
				{Name: "KUBERNETES_PORT", Value: "tcp://1.2.3.1:8081"},
				{Name: "KUBERNETES_PORT_8081_TCP", Value: "tcp://1.2.3.1:8081"},
				{Name: "KUBERNETES_PORT_8081_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_PORT_8081_TCP_PORT", Value: "8081"},
				{Name: "KUBERNETES_PORT_8081_TCP_ADDR", Value: "1.2.3.1"},
			},
		},
		{
			name:               "api server = Y, kubelet = N",
			ns:                 "test1",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{Name: "FOO", Value: "BAR"},
					{Name: "TEST_SERVICE_HOST", Value: "1.2.3.3"},
					{Name: "TEST_SERVICE_PORT", Value: "8083"},
					{Name: "TEST_PORT", Value: "tcp://1.2.3.3:8083"},
					{Name: "TEST_PORT_8083_TCP", Value: "tcp://1.2.3.3:8083"},
					{Name: "TEST_PORT_8083_TCP_PROTO", Value: "tcp"},
					{Name: "TEST_PORT_8083_TCP_PORT", Value: "8083"},
					{Name: "TEST_PORT_8083_TCP_ADDR", Value: "1.2.3.3"},
				},
			},
			masterServiceNs: metav1.NamespaceDefault,
			nilLister:       true,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "FOO", Value: "BAR"},
				{Name: "TEST_SERVICE_HOST", Value: "1.2.3.3"},
				{Name: "TEST_SERVICE_PORT", Value: "8083"},
				{Name: "TEST_PORT", Value: "tcp://1.2.3.3:8083"},
				{Name: "TEST_PORT_8083_TCP", Value: "tcp://1.2.3.3:8083"},
				{Name: "TEST_PORT_8083_TCP_PROTO", Value: "tcp"},
				{Name: "TEST_PORT_8083_TCP_PORT", Value: "8083"},
				{Name: "TEST_PORT_8083_TCP_ADDR", Value: "1.2.3.3"},
			},
		},
		{
			name:               "api server = N; kubelet = Y",
			ns:                 "test1",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{Name: "FOO", Value: "BAZ"},
				},
			},
			masterServiceNs: metav1.NamespaceDefault,
			nilLister:       false,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "FOO", Value: "BAZ"},
				{Name: "KUBERNETES_SERVICE_HOST", Value: "1.2.3.1"},
				{Name: "KUBERNETES_SERVICE_PORT", Value: "8081"},
				{Name: "KUBERNETES_PORT", Value: "tcp://1.2.3.1:8081"},
				{Name: "KUBERNETES_PORT_8081_TCP", Value: "tcp://1.2.3.1:8081"},
				{Name: "KUBERNETES_PORT_8081_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_PORT_8081_TCP_PORT", Value: "8081"},
				{Name: "KUBERNETES_PORT_8081_TCP_ADDR", Value: "1.2.3.1"},
			},
		},
		{
			name:               "api server = N; kubelet = Y; service env vars",
			ns:                 "test1",
			enableServiceLinks: &trueValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{Name: "FOO", Value: "BAZ"},
				},
			},
			masterServiceNs: metav1.NamespaceDefault,
			nilLister:       false,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "FOO", Value: "BAZ"},
				{Name: "TEST_SERVICE_HOST", Value: "1.2.3.3"},
				{Name: "TEST_SERVICE_PORT", Value: "8083"},
				{Name: "TEST_PORT", Value: "tcp://1.2.3.3:8083"},
				{Name: "TEST_PORT_8083_TCP", Value: "tcp://1.2.3.3:8083"},
				{Name: "TEST_PORT_8083_TCP_PROTO", Value: "tcp"},
				{Name: "TEST_PORT_8083_TCP_PORT", Value: "8083"},
				{Name: "TEST_PORT_8083_TCP_ADDR", Value: "1.2.3.3"},
				{Name: "KUBERNETES_SERVICE_HOST", Value: "1.2.3.1"},
				{Name: "KUBERNETES_SERVICE_PORT", Value: "8081"},
				{Name: "KUBERNETES_PORT", Value: "tcp://1.2.3.1:8081"},
				{Name: "KUBERNETES_PORT_8081_TCP", Value: "tcp://1.2.3.1:8081"},
				{Name: "KUBERNETES_PORT_8081_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_PORT_8081_TCP_PORT", Value: "8081"},
				{Name: "KUBERNETES_PORT_8081_TCP_ADDR", Value: "1.2.3.1"},
			},
		},
		{
			name:               "master service in pod ns",
			ns:                 "test2",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{Name: "FOO", Value: "ZAP"},
				},
			},
			masterServiceNs: "kubernetes",
			nilLister:       false,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "FOO", Value: "ZAP"},
				{Name: "KUBERNETES_SERVICE_HOST", Value: "1.2.3.6"},
				{Name: "KUBERNETES_SERVICE_PORT", Value: "8086"},
				{Name: "KUBERNETES_PORT", Value: "tcp://1.2.3.6:8086"},
				{Name: "KUBERNETES_PORT_8086_TCP", Value: "tcp://1.2.3.6:8086"},
				{Name: "KUBERNETES_PORT_8086_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_PORT_8086_TCP_PORT", Value: "8086"},
				{Name: "KUBERNETES_PORT_8086_TCP_ADDR", Value: "1.2.3.6"},
			},
		},
		{
			name:               "master service in pod ns, service env vars",
			ns:                 "test2",
			enableServiceLinks: &trueValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{Name: "FOO", Value: "ZAP"},
				},
			},
			masterServiceNs: "kubernetes",
			nilLister:       false,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "FOO", Value: "ZAP"},
				{Name: "TEST_SERVICE_HOST", Value: "1.2.3.5"},
				{Name: "TEST_SERVICE_PORT", Value: "8085"},
				{Name: "TEST_PORT", Value: "tcp://1.2.3.5:8085"},
				{Name: "TEST_PORT_8085_TCP", Value: "tcp://1.2.3.5:8085"},
				{Name: "TEST_PORT_8085_TCP_PROTO", Value: "tcp"},
				{Name: "TEST_PORT_8085_TCP_PORT", Value: "8085"},
				{Name: "TEST_PORT_8085_TCP_ADDR", Value: "1.2.3.5"},
				{Name: "KUBERNETES_SERVICE_HOST", Value: "1.2.3.4"},
				{Name: "KUBERNETES_SERVICE_PORT", Value: "8084"},
				{Name: "KUBERNETES_PORT", Value: "tcp://1.2.3.4:8084"},
				{Name: "KUBERNETES_PORT_8084_TCP", Value: "tcp://1.2.3.4:8084"},
				{Name: "KUBERNETES_PORT_8084_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_PORT_8084_TCP_PORT", Value: "8084"},
				{Name: "KUBERNETES_PORT_8084_TCP_ADDR", Value: "1.2.3.4"},
			},
		},
		{
			name:               "pod in master service ns",
			ns:                 "kubernetes",
			enableServiceLinks: &falseValue,
			container:          &v1.Container{},
			masterServiceNs:    "kubernetes",
			nilLister:          false,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "KUBERNETES_SERVICE_HOST", Value: "1.2.3.6"},
				{Name: "KUBERNETES_SERVICE_PORT", Value: "8086"},
				{Name: "KUBERNETES_PORT", Value: "tcp://1.2.3.6:8086"},
				{Name: "KUBERNETES_PORT_8086_TCP", Value: "tcp://1.2.3.6:8086"},
				{Name: "KUBERNETES_PORT_8086_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_PORT_8086_TCP_PORT", Value: "8086"},
				{Name: "KUBERNETES_PORT_8086_TCP_ADDR", Value: "1.2.3.6"},
			},
		},
		{
			name:               "pod in master service ns, service env vars",
			ns:                 "kubernetes",
			enableServiceLinks: &trueValue,
			container:          &v1.Container{},
			masterServiceNs:    "kubernetes",
			nilLister:          false,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "NOT_SPECIAL_SERVICE_HOST", Value: "1.2.3.8"},
				{Name: "NOT_SPECIAL_SERVICE_PORT", Value: "8088"},
				{Name: "NOT_SPECIAL_PORT", Value: "tcp://1.2.3.8:8088"},
				{Name: "NOT_SPECIAL_PORT_8088_TCP", Value: "tcp://1.2.3.8:8088"},
				{Name: "NOT_SPECIAL_PORT_8088_TCP_PROTO", Value: "tcp"},
				{Name: "NOT_SPECIAL_PORT_8088_TCP_PORT", Value: "8088"},
				{Name: "NOT_SPECIAL_PORT_8088_TCP_ADDR", Value: "1.2.3.8"},
				{Name: "KUBERNETES_SERVICE_HOST", Value: "1.2.3.6"},
				{Name: "KUBERNETES_SERVICE_PORT", Value: "8086"},
				{Name: "KUBERNETES_PORT", Value: "tcp://1.2.3.6:8086"},
				{Name: "KUBERNETES_PORT_8086_TCP", Value: "tcp://1.2.3.6:8086"},
				{Name: "KUBERNETES_PORT_8086_TCP_PROTO", Value: "tcp"},
				{Name: "KUBERNETES_PORT_8086_TCP_PORT", Value: "8086"},
				{Name: "KUBERNETES_PORT_8086_TCP_ADDR", Value: "1.2.3.6"},
			},
		},
		{
			name:               "downward api pod",
			ns:                 "downward-api",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{
						Name: "POD_NAME",
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.name",
							},
						},
					},
					{
						Name: "POD_NAMESPACE",
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.namespace",
							},
						},
					},
					{
						Name: "POD_NODE_NAME",
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "spec.nodeName",
							},
						},
					},
					{
						Name: "POD_SERVICE_ACCOUNT_NAME",
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "spec.serviceAccountName",
							},
						},
					},
					{
						Name: "POD_IP",
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "status.podIP",
							},
						},
					},
					{
						Name: "HOST_IP",
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "status.hostIP",
							},
						},
					},
				},
			},
			masterServiceNs: "nothing",
			nilLister:       true,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "POD_NAME", Value: "dapi-test-pod-name"},
				{Name: "POD_NAMESPACE", Value: "downward-api"},
				{Name: "POD_NODE_NAME", Value: "node-name"},
				{Name: "POD_SERVICE_ACCOUNT_NAME", Value: "special"},
				{Name: "POD_IP", Value: "1.2.3.4"},
				{Name: "HOST_IP", Value: testKubeletHostIP},
			},
		},
		{
			name:               "env expansion",
			ns:                 "test1",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{
						Name:  "TEST_LITERAL",
						Value: "test-test-test",
					},
					{
						Name: "POD_NAME",
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{
								APIVersion: "v1", //legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion.String(),
								FieldPath:  "metadata.name",
							},
						},
					},
					{
						Name:  "OUT_OF_ORDER_TEST",
						Value: "$(OUT_OF_ORDER_TARGET)",
					},
					{
						Name:  "OUT_OF_ORDER_TARGET",
						Value: "FOO",
					},
					{
						Name: "EMPTY_VAR",
					},
					{
						Name:  "EMPTY_TEST",
						Value: "foo-$(EMPTY_VAR)",
					},
					{
						Name:  "POD_NAME_TEST2",
						Value: "test2-$(POD_NAME)",
					},
					{
						Name:  "POD_NAME_TEST3",
						Value: "$(POD_NAME_TEST2)-3",
					},
					{
						Name:  "LITERAL_TEST",
						Value: "literal-$(TEST_LITERAL)",
					},
					{
						Name:  "TEST_UNDEFINED",
						Value: "$(UNDEFINED_VAR)",
					},
				},
			},
			masterServiceNs: "nothing",
			nilLister:       false,
			expectedEnvs: []kubecontainer.EnvVar{
				{
					Name:  "TEST_LITERAL",
					Value: "test-test-test",
				},
				{
					Name:  "POD_NAME",
					Value: "dapi-test-pod-name",
				},
				{
					Name:  "POD_NAME_TEST2",
					Value: "test2-dapi-test-pod-name",
				},
				{
					Name:  "POD_NAME_TEST3",
					Value: "test2-dapi-test-pod-name-3",
				},
				{
					Name:  "LITERAL_TEST",
					Value: "literal-test-test-test",
				},
				{
					Name:  "OUT_OF_ORDER_TEST",
					Value: "$(OUT_OF_ORDER_TARGET)",
				},
				{
					Name:  "OUT_OF_ORDER_TARGET",
					Value: "FOO",
				},
				{
					Name:  "TEST_UNDEFINED",
					Value: "$(UNDEFINED_VAR)",
				},
				{
					Name: "EMPTY_VAR",
				},
				{
					Name:  "EMPTY_TEST",
					Value: "foo-",
				},
			},
		},
		{
			name:               "env expansion, service env vars",
			ns:                 "test1",
			enableServiceLinks: &trueValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{
						Name:  "TEST_LITERAL",
						Value: "test-test-test",
					},
					{
						Name: "POD_NAME",
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "metadata.name",
							},
						},
					},
					{
						Name:  "OUT_OF_ORDER_TEST",
						Value: "$(OUT_OF_ORDER_TARGET)",
					},
					{
						Name:  "OUT_OF_ORDER_TARGET",
						Value: "FOO",
					},
					{
						Name: "EMPTY_VAR",
					},
					{
						Name:  "EMPTY_TEST",
						Value: "foo-$(EMPTY_VAR)",
					},
					{
						Name:  "POD_NAME_TEST2",
						Value: "test2-$(POD_NAME)",
					},
					{
						Name:  "POD_NAME_TEST3",
						Value: "$(POD_NAME_TEST2)-3",
					},
					{
						Name:  "LITERAL_TEST",
						Value: "literal-$(TEST_LITERAL)",
					},
					{
						Name:  "SERVICE_VAR_TEST",
						Value: "$(TEST_SERVICE_HOST):$(TEST_SERVICE_PORT)",
					},
					{
						Name:  "TEST_UNDEFINED",
						Value: "$(UNDEFINED_VAR)",
					},
				},
			},
			masterServiceNs: "nothing",
			nilLister:       false,
			expectedEnvs: []kubecontainer.EnvVar{
				{
					Name:  "TEST_LITERAL",
					Value: "test-test-test",
				},
				{
					Name:  "POD_NAME",
					Value: "dapi-test-pod-name",
				},
				{
					Name:  "POD_NAME_TEST2",
					Value: "test2-dapi-test-pod-name",
				},
				{
					Name:  "POD_NAME_TEST3",
					Value: "test2-dapi-test-pod-name-3",
				},
				{
					Name:  "LITERAL_TEST",
					Value: "literal-test-test-test",
				},
				{
					Name:  "TEST_SERVICE_HOST",
					Value: "1.2.3.3",
				},
				{
					Name:  "TEST_SERVICE_PORT",
					Value: "8083",
				},
				{
					Name:  "TEST_PORT",
					Value: "tcp://1.2.3.3:8083",
				},
				{
					Name:  "TEST_PORT_8083_TCP",
					Value: "tcp://1.2.3.3:8083",
				},
				{
					Name:  "TEST_PORT_8083_TCP_PROTO",
					Value: "tcp",
				},
				{
					Name:  "TEST_PORT_8083_TCP_PORT",
					Value: "8083",
				},
				{
					Name:  "TEST_PORT_8083_TCP_ADDR",
					Value: "1.2.3.3",
				},
				{
					Name:  "SERVICE_VAR_TEST",
					Value: "1.2.3.3:8083",
				},
				{
					Name:  "OUT_OF_ORDER_TEST",
					Value: "$(OUT_OF_ORDER_TARGET)",
				},
				{
					Name:  "OUT_OF_ORDER_TARGET",
					Value: "FOO",
				},
				{
					Name:  "TEST_UNDEFINED",
					Value: "$(UNDEFINED_VAR)",
				},
				{
					Name: "EMPTY_VAR",
				},
				{
					Name:  "EMPTY_TEST",
					Value: "foo-",
				},
			},
		},
		{
			name:               "configmapkeyref_missing_optional",
			ns:                 "test",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{
						Name: "POD_NAME",
						ValueFrom: &v1.EnvVarSource{
							ConfigMapKeyRef: &v1.ConfigMapKeySelector{
								LocalObjectReference: v1.LocalObjectReference{Name: "missing-config-map"},
								Key:                  "key",
								Optional:             &trueVal,
							},
						},
					},
				},
			},
			masterServiceNs: "nothing",
			expectedEnvs:    nil,
		},
		{
			name:               "configmapkeyref_missing_key_optional",
			ns:                 "test",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{
						Name: "POD_NAME",
						ValueFrom: &v1.EnvVarSource{
							ConfigMapKeyRef: &v1.ConfigMapKeySelector{
								LocalObjectReference: v1.LocalObjectReference{Name: "test-config-map"},
								Key:                  "key",
								Optional:             &trueVal,
							},
						},
					},
				},
			},
			masterServiceNs: "nothing",
			nilLister:       true,
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test1",
					Name:      "test-configmap",
				},
				Data: map[string]string{
					"a": "b",
				},
			},
			expectedEnvs: nil,
		},
		{
			name:               "secretkeyref_missing_optional",
			ns:                 "test",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{
						Name: "POD_NAME",
						ValueFrom: &v1.EnvVarSource{
							SecretKeyRef: &v1.SecretKeySelector{
								LocalObjectReference: v1.LocalObjectReference{Name: "missing-secret"},
								Key:                  "key",
								Optional:             &trueVal,
							},
						},
					},
				},
			},
			masterServiceNs: "nothing",
			expectedEnvs:    nil,
		},
		{
			name:               "secretkeyref_missing_key_optional",
			ns:                 "test",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
					{
						Name: "POD_NAME",
						ValueFrom: &v1.EnvVarSource{
							SecretKeyRef: &v1.SecretKeySelector{
								LocalObjectReference: v1.LocalObjectReference{Name: "test-secret"},
								Key:                  "key",
								Optional:             &trueVal,
							},
						},
					},
				},
			},
			masterServiceNs: "nothing",
			nilLister:       true,
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test1",
					Name:      "test-secret",
				},
				Data: map[string][]byte{
					"a": []byte("b"),
				},
			},
			expectedEnvs: nil,
		},
		{
			name:               "configmap",
			ns:                 "test1",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{
						ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-config-map"}},
					},
					{
						Prefix:       "p_",
						ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-config-map"}},
					},
				},
				Env: []v1.EnvVar{
					{
						Name:  "TEST_LITERAL",
						Value: "test-test-test",
					},
					{
						Name:  "EXPANSION_TEST",
						Value: "$(REPLACE_ME)",
					},
					{
						Name:  "DUPE_TEST",
						Value: "ENV_VAR",
					},
				},
			},
			masterServiceNs: "nothing",
			nilLister:       false,
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test1",
					Name:      "test-configmap",
				},
				Data: map[string]string{
					"REPLACE_ME": "FROM_CONFIG_MAP",
					"DUPE_TEST":  "CONFIG_MAP",
				},
			},
			expectedEnvs: []kubecontainer.EnvVar{
				{
					Name:  "TEST_LITERAL",
					Value: "test-test-test",
				},
				{
					Name:  "REPLACE_ME",
					Value: "FROM_CONFIG_MAP",
				},
				{
					Name:  "EXPANSION_TEST",
					Value: "FROM_CONFIG_MAP",
				},
				{
					Name:  "DUPE_TEST",
					Value: "ENV_VAR",
				},
				{
					Name:  "p_REPLACE_ME",
					Value: "FROM_CONFIG_MAP",
				},
				{
					Name:  "p_DUPE_TEST",
					Value: "CONFIG_MAP",
				},
			},
		},
		{
			name:               "configmap, service env vars",
			ns:                 "test1",
			enableServiceLinks: &trueValue,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{
						ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-config-map"}},
					},
					{
						Prefix:       "p_",
						ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-config-map"}},
					},
				},
				Env: []v1.EnvVar{
					{
						Name:  "TEST_LITERAL",
						Value: "test-test-test",
					},
					{
						Name:  "EXPANSION_TEST",
						Value: "$(REPLACE_ME)",
					},
					{
						Name:  "DUPE_TEST",
						Value: "ENV_VAR",
					},
				},
			},
			masterServiceNs: "nothing",
			nilLister:       false,
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test1",
					Name:      "test-configmap",
				},
				Data: map[string]string{
					"REPLACE_ME": "FROM_CONFIG_MAP",
					"DUPE_TEST":  "CONFIG_MAP",
				},
			},
			expectedEnvs: []kubecontainer.EnvVar{
				{
					Name:  "TEST_LITERAL",
					Value: "test-test-test",
				},
				{
					Name:  "TEST_SERVICE_HOST",
					Value: "1.2.3.3",
				},
				{
					Name:  "TEST_SERVICE_PORT",
					Value: "8083",
				},
				{
					Name:  "TEST_PORT",
					Value: "tcp://1.2.3.3:8083",
				},
				{
					Name:  "TEST_PORT_8083_TCP",
					Value: "tcp://1.2.3.3:8083",
				},
				{
					Name:  "TEST_PORT_8083_TCP_PROTO",
					Value: "tcp",
				},
				{
					Name:  "TEST_PORT_8083_TCP_PORT",
					Value: "8083",
				},
				{
					Name:  "TEST_PORT_8083_TCP_ADDR",
					Value: "1.2.3.3",
				},
				{
					Name:  "REPLACE_ME",
					Value: "FROM_CONFIG_MAP",
				},
				{
					Name:  "EXPANSION_TEST",
					Value: "FROM_CONFIG_MAP",
				},
				{
					Name:  "DUPE_TEST",
					Value: "ENV_VAR",
				},
				{
					Name:  "p_REPLACE_ME",
					Value: "FROM_CONFIG_MAP",
				},
				{
					Name:  "p_DUPE_TEST",
					Value: "CONFIG_MAP",
				},
			},
		},
		{
			name:               "configmap_missing",
			ns:                 "test1",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-config-map"}}},
				},
			},
			masterServiceNs: "nothing",
			expectedError:   true,
		},
		{
			name:               "configmap_missing_optional",
			ns:                 "test",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{ConfigMapRef: &v1.ConfigMapEnvSource{
						Optional:             &trueVal,
						LocalObjectReference: v1.LocalObjectReference{Name: "missing-config-map"}}},
				},
			},
			masterServiceNs: "nothing",
			expectedEnvs:    nil,
		},
		{
			name:               "configmap_invalid_keys",
			ns:                 "test",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-config-map"}}},
				},
			},
			masterServiceNs: "nothing",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test1",
					Name:      "test-configmap",
				},
				Data: map[string]string{
					"1234": "abc",
					"1z":   "abc",
					"key":  "value",
				},
			},
			expectedEnvs: []kubecontainer.EnvVar{
				{
					Name:  "key",
					Value: "value",
				},
			},
			expectedEvent: "Warning InvalidEnvironmentVariableNames Keys [1234, 1z] from the EnvFrom configMap test/test-config-map were skipped since they are considered invalid environment variable names.",
		},
		{
			name:               "configmap_invalid_keys_valid",
			ns:                 "test",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{
						Prefix:       "p_",
						ConfigMapRef: &v1.ConfigMapEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-config-map"}},
					},
				},
			},
			masterServiceNs: "",
			configMap: &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test1",
					Name:      "test-configmap",
				},
				Data: map[string]string{
					"1234": "abc",
				},
			},
			expectedEnvs: []kubecontainer.EnvVar{
				{
					Name:  "p_1234",
					Value: "abc",
				},
			},
		},
		{
			name:               "secret",
			ns:                 "test1",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{
						SecretRef: &v1.SecretEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-secret"}},
					},
					{
						Prefix:    "p_",
						SecretRef: &v1.SecretEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-secret"}},
					},
				},
				Env: []v1.EnvVar{
					{
						Name:  "TEST_LITERAL",
						Value: "test-test-test",
					},
					{
						Name:  "EXPANSION_TEST",
						Value: "$(REPLACE_ME)",
					},
					{
						Name:  "DUPE_TEST",
						Value: "ENV_VAR",
					},
				},
			},
			masterServiceNs: "nothing",
			nilLister:       false,
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test1",
					Name:      "test-secret",
				},
				Data: map[string][]byte{
					"REPLACE_ME": []byte("FROM_SECRET"),
					"DUPE_TEST":  []byte("SECRET"),
				},
			},
			expectedEnvs: []kubecontainer.EnvVar{
				{
					Name:  "TEST_LITERAL",
					Value: "test-test-test",
				},
				{
					Name:  "REPLACE_ME",
					Value: "FROM_SECRET",
				},
				{
					Name:  "EXPANSION_TEST",
					Value: "FROM_SECRET",
				},
				{
					Name:  "DUPE_TEST",
					Value: "ENV_VAR",
				},
				{
					Name:  "p_REPLACE_ME",
					Value: "FROM_SECRET",
				},
				{
					Name:  "p_DUPE_TEST",
					Value: "SECRET",
				},
			},
		},
		{
			name:               "secret, service env vars",
			ns:                 "test1",
			enableServiceLinks: &trueValue,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{
						SecretRef: &v1.SecretEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-secret"}},
					},
					{
						Prefix:    "p_",
						SecretRef: &v1.SecretEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-secret"}},
					},
				},
				Env: []v1.EnvVar{
					{
						Name:  "TEST_LITERAL",
						Value: "test-test-test",
					},
					{
						Name:  "EXPANSION_TEST",
						Value: "$(REPLACE_ME)",
					},
					{
						Name:  "DUPE_TEST",
						Value: "ENV_VAR",
					},
				},
			},
			masterServiceNs: "nothing",
			nilLister:       false,
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test1",
					Name:      "test-secret",
				},
				Data: map[string][]byte{
					"REPLACE_ME": []byte("FROM_SECRET"),
					"DUPE_TEST":  []byte("SECRET"),
				},
			},
			expectedEnvs: []kubecontainer.EnvVar{
				{
					Name:  "TEST_LITERAL",
					Value: "test-test-test",
				},
				{
					Name:  "TEST_SERVICE_HOST",
					Value: "1.2.3.3",
				},
				{
					Name:  "TEST_SERVICE_PORT",
					Value: "8083",
				},
				{
					Name:  "TEST_PORT",
					Value: "tcp://1.2.3.3:8083",
				},
				{
					Name:  "TEST_PORT_8083_TCP",
					Value: "tcp://1.2.3.3:8083",
				},
				{
					Name:  "TEST_PORT_8083_TCP_PROTO",
					Value: "tcp",
				},
				{
					Name:  "TEST_PORT_8083_TCP_PORT",
					Value: "8083",
				},
				{
					Name:  "TEST_PORT_8083_TCP_ADDR",
					Value: "1.2.3.3",
				},
				{
					Name:  "REPLACE_ME",
					Value: "FROM_SECRET",
				},
				{
					Name:  "EXPANSION_TEST",
					Value: "FROM_SECRET",
				},
				{
					Name:  "DUPE_TEST",
					Value: "ENV_VAR",
				},
				{
					Name:  "p_REPLACE_ME",
					Value: "FROM_SECRET",
				},
				{
					Name:  "p_DUPE_TEST",
					Value: "SECRET",
				},
			},
		},
		{
			name:               "secret_missing",
			ns:                 "test1",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{SecretRef: &v1.SecretEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-secret"}}},
				},
			},
			masterServiceNs: "nothing",
			expectedError:   true,
		},
		{
			name:               "secret_missing_optional",
			ns:                 "test",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{SecretRef: &v1.SecretEnvSource{
						LocalObjectReference: v1.LocalObjectReference{Name: "missing-secret"},
						Optional:             &trueVal}},
				},
			},
			masterServiceNs: "nothing",
			expectedEnvs:    nil,
		},
		{
			name:               "secret_invalid_keys",
			ns:                 "test",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{SecretRef: &v1.SecretEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-secret"}}},
				},
			},
			masterServiceNs: "nothing",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test1",
					Name:      "test-secret",
				},
				Data: map[string][]byte{
					"1234":  []byte("abc"),
					"1z":    []byte("abc"),
					"key.1": []byte("value"),
				},
			},
			expectedEnvs: []kubecontainer.EnvVar{
				{
					Name:  "key.1",
					Value: "value",
				},
			},
			expectedEvent: "Warning InvalidEnvironmentVariableNames Keys [1234, 1z] from the EnvFrom secret test/test-secret were skipped since they are considered invalid environment variable names.",
		},
		{
			name:               "secret_invalid_keys_valid",
			ns:                 "test",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{
						Prefix:    "p_",
						SecretRef: &v1.SecretEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-secret"}},
					},
				},
			},
			masterServiceNs: "",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test1",
					Name:      "test-secret",
				},
				Data: map[string][]byte{
					"1234.name": []byte("abc"),
				},
			},
			expectedEnvs: []kubecontainer.EnvVar{
				{
					Name:  "p_1234.name",
					Value: "abc",
				},
			},
		},
		{
			name:               "nil_enableServiceLinks",
			ns:                 "test",
			enableServiceLinks: nil,
			container: &v1.Container{
				EnvFrom: []v1.EnvFromSource{
					{
						Prefix:    "p_",
						SecretRef: &v1.SecretEnvSource{LocalObjectReference: v1.LocalObjectReference{Name: "test-secret"}},
					},
				},
			},
			masterServiceNs: "",
			secret: &v1.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: "test1",
					Name:      "test-secret",
				},
				Data: map[string][]byte{
					"1234.name": []byte("abc"),
				},
			},
			expectedError: true,
		},
	}

	for _, tc := range testCases {
		fakeRecorder := record.NewFakeRecorder(1)
		testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
		testKubelet.kubelet.recorder = fakeRecorder
		defer testKubelet.Cleanup()
		kl := testKubelet.kubelet
		kl.masterServiceNamespace = tc.masterServiceNs
		if tc.nilLister {
			kl.serviceLister = nil
		} else {
			kl.serviceLister = testServiceLister{services}
		}

		testKubelet.fakeKubeClient.AddReactor("get", "configmaps", func(action core.Action) (bool, runtime.Object, error) {
			var err error
			if tc.configMap == nil {
				err = apierrors.NewNotFound(action.GetResource().GroupResource(), "configmap-name")
			}
			return true, tc.configMap, err
		})
		testKubelet.fakeKubeClient.AddReactor("get", "secrets", func(action core.Action) (bool, runtime.Object, error) {
			var err error
			if tc.secret == nil {
				err = apierrors.NewNotFound(action.GetResource().GroupResource(), "secret-name")
			}
			return true, tc.secret, err
		})

		testKubelet.fakeKubeClient.AddReactor("get", "secrets", func(action core.Action) (bool, runtime.Object, error) {
			var err error
			if tc.secret == nil {
				err = errors.New("no secret defined")
			}
			return true, tc.secret, err
		})

		testPod := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: tc.ns,
				Name:      "dapi-test-pod-name",
			},
			Spec: v1.PodSpec{
				ServiceAccountName: "special",
				NodeName:           "node-name",
				EnableServiceLinks: tc.enableServiceLinks,
			},
		}
		podIP := "1.2.3.4"

		result, err := kl.makeEnvironmentVariables(testPod, tc.container, podIP)
		select {
		case e := <-fakeRecorder.Events:
			assert.Equal(t, tc.expectedEvent, e)
		default:
			assert.Equal(t, "", tc.expectedEvent)
		}
		if tc.expectedError {
			assert.Error(t, err, tc.name)
		} else {
			assert.NoError(t, err, "[%s]", tc.name)

			sort.Sort(envs(result))
			sort.Sort(envs(tc.expectedEnvs))
			assert.Equal(t, tc.expectedEnvs, result, "[%s] env entries", tc.name)
		}
	}
}

func waitingState(cName string) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name: cName,
		State: v1.ContainerState{
			Waiting: &v1.ContainerStateWaiting{},
		},
	}
}
func waitingStateWithLastTermination(cName string) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name: cName,
		State: v1.ContainerState{
			Waiting: &v1.ContainerStateWaiting{},
		},
		LastTerminationState: v1.ContainerState{
			Terminated: &v1.ContainerStateTerminated{
				ExitCode: 0,
			},
		},
	}
}
func runningState(cName string) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name: cName,
		State: v1.ContainerState{
			Running: &v1.ContainerStateRunning{},
		},
	}
}
func stoppedState(cName string) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name: cName,
		State: v1.ContainerState{
			Terminated: &v1.ContainerStateTerminated{},
		},
	}
}
func succeededState(cName string) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name: cName,
		State: v1.ContainerState{
			Terminated: &v1.ContainerStateTerminated{
				ExitCode: 0,
			},
		},
	}
}
func failedState(cName string) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name: cName,
		State: v1.ContainerState{
			Terminated: &v1.ContainerStateTerminated{
				ExitCode: -1,
			},
		},
	}
}

func TestPodPhaseWithRestartAlways(t *testing.T) {
	desiredState := v1.PodSpec{
		NodeName: "machine",
		Containers: []v1.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: v1.RestartPolicyAlways,
	}

	tests := []struct {
		pod    *v1.Pod
		status v1.PodPhase
		test   string
	}{
		{&v1.Pod{Spec: desiredState, Status: v1.PodStatus{}}, v1.PodPending, "waiting"},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			v1.PodRunning,
			"all running",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						stoppedState("containerA"),
						stoppedState("containerB"),
					},
				},
			},
			v1.PodRunning,
			"all stopped with restart always",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						stoppedState("containerB"),
					},
				},
			},
			v1.PodRunning,
			"mixed state #1 with restart always",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
					},
				},
			},
			v1.PodPending,
			"mixed state #2 with restart always",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						waitingState("containerB"),
					},
				},
			},
			v1.PodPending,
			"mixed state #3 with restart always",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						waitingStateWithLastTermination("containerB"),
					},
				},
			},
			v1.PodRunning,
			"backoff crashloop container with restart always",
		},
	}
	for _, test := range tests {
		status := getPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses)
		assert.Equal(t, test.status, status, "[test %s]", test.test)
	}
}

func TestPodPhaseWithRestartNever(t *testing.T) {
	desiredState := v1.PodSpec{
		NodeName: "machine",
		Containers: []v1.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: v1.RestartPolicyNever,
	}

	tests := []struct {
		pod    *v1.Pod
		status v1.PodPhase
		test   string
	}{
		{&v1.Pod{Spec: desiredState, Status: v1.PodStatus{}}, v1.PodPending, "waiting"},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			v1.PodRunning,
			"all running with restart never",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						succeededState("containerA"),
						succeededState("containerB"),
					},
				},
			},
			v1.PodSucceeded,
			"all succeeded with restart never",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						failedState("containerA"),
						failedState("containerB"),
					},
				},
			},
			v1.PodFailed,
			"all failed with restart never",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						succeededState("containerB"),
					},
				},
			},
			v1.PodRunning,
			"mixed state #1 with restart never",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
					},
				},
			},
			v1.PodPending,
			"mixed state #2 with restart never",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						waitingState("containerB"),
					},
				},
			},
			v1.PodPending,
			"mixed state #3 with restart never",
		},
	}
	for _, test := range tests {
		status := getPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses)
		assert.Equal(t, test.status, status, "[test %s]", test.test)
	}
}

func TestPodPhaseWithRestartOnFailure(t *testing.T) {
	desiredState := v1.PodSpec{
		NodeName: "machine",
		Containers: []v1.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: v1.RestartPolicyOnFailure,
	}

	tests := []struct {
		pod    *v1.Pod
		status v1.PodPhase
		test   string
	}{
		{&v1.Pod{Spec: desiredState, Status: v1.PodStatus{}}, v1.PodPending, "waiting"},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			v1.PodRunning,
			"all running with restart onfailure",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						succeededState("containerA"),
						succeededState("containerB"),
					},
				},
			},
			v1.PodSucceeded,
			"all succeeded with restart onfailure",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						failedState("containerA"),
						failedState("containerB"),
					},
				},
			},
			v1.PodRunning,
			"all failed with restart never",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						succeededState("containerB"),
					},
				},
			},
			v1.PodRunning,
			"mixed state #1 with restart onfailure",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
					},
				},
			},
			v1.PodPending,
			"mixed state #2 with restart onfailure",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						waitingState("containerB"),
					},
				},
			},
			v1.PodPending,
			"mixed state #3 with restart onfailure",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						waitingStateWithLastTermination("containerB"),
					},
				},
			},
			v1.PodRunning,
			"backoff crashloop container with restart onfailure",
		},
	}
	for _, test := range tests {
		status := getPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses)
		assert.Equal(t, test.status, status, "[test %s]", test.test)
	}
}

func TestGetExec(t *testing.T) {
	const (
		podName                = "podFoo"
		podNamespace           = "nsFoo"
		podUID       types.UID = "12345678"
		containerID            = "containerFoo"
		tty                    = true
	)
	var (
		podFullName = kubecontainer.GetPodFullName(podWithUIDNameNs(podUID, podName, podNamespace))
		command     = []string{"ls"}
	)

	testcases := []struct {
		description string
		podFullName string
		container   string
		expectError bool
	}{{
		description: "success case",
		podFullName: podFullName,
		container:   containerID,
	}, {
		description: "no such pod",
		podFullName: "bar" + podFullName,
		container:   containerID,
		expectError: true,
	}, {
		description: "no such container",
		podFullName: podFullName,
		container:   "containerBar",
		expectError: true,
	}}

	for _, tc := range testcases {
		testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
		defer testKubelet.Cleanup()
		kubelet := testKubelet.kubelet
		testKubelet.fakeRuntime.PodList = []*containertest.FakePod{
			{Pod: &kubecontainer.Pod{
				ID:        podUID,
				Name:      podName,
				Namespace: podNamespace,
				Containers: []*kubecontainer.Container{
					{Name: containerID,
						ID: kubecontainer.ContainerID{Type: "test", ID: containerID},
					},
				},
			}},
		}

		description := "streaming - " + tc.description
		fakeRuntime := &containertest.FakeStreamingRuntime{FakeRuntime: testKubelet.fakeRuntime}
		kubelet.containerRuntime = fakeRuntime
		kubelet.streamingRuntime = fakeRuntime

		redirect, err := kubelet.GetExec(tc.podFullName, podUID, tc.container, command, remotecommand.Options{})
		if tc.expectError {
			assert.Error(t, err, description)
		} else {
			assert.NoError(t, err, description)
			assert.Equal(t, containertest.FakeHost, redirect.Host, description+": redirect")
		}
	}
}

func TestGetPortForward(t *testing.T) {
	const (
		podName                = "podFoo"
		podNamespace           = "nsFoo"
		podUID       types.UID = "12345678"
		port         int32     = 5000
	)

	testcases := []struct {
		description string
		podName     string
		expectError bool
	}{{
		description: "success case",
		podName:     podName,
	}, {
		description: "no such pod",
		podName:     "bar",
		expectError: true,
	}}

	for _, tc := range testcases {
		testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
		defer testKubelet.Cleanup()
		kubelet := testKubelet.kubelet
		testKubelet.fakeRuntime.PodList = []*containertest.FakePod{
			{Pod: &kubecontainer.Pod{
				ID:        podUID,
				Name:      podName,
				Namespace: podNamespace,
				Containers: []*kubecontainer.Container{
					{Name: "foo",
						ID: kubecontainer.ContainerID{Type: "test", ID: "foo"},
					},
				},
			}},
		}

		description := "streaming - " + tc.description
		fakeRuntime := &containertest.FakeStreamingRuntime{FakeRuntime: testKubelet.fakeRuntime}
		kubelet.containerRuntime = fakeRuntime
		kubelet.streamingRuntime = fakeRuntime

		redirect, err := kubelet.GetPortForward(tc.podName, podNamespace, podUID, portforward.V4Options{})
		if tc.expectError {
			assert.Error(t, err, description)
		} else {
			assert.NoError(t, err, description)
			assert.Equal(t, containertest.FakeHost, redirect.Host, description+": redirect")
		}
	}
}

func TestHasHostMountPVC(t *testing.T) {
	tests := map[string]struct {
		pvError       error
		pvcError      error
		expected      bool
		podHasPVC     bool
		pvcIsHostPath bool
	}{
		"no pvc": {podHasPVC: false, expected: false},
		"error fetching pvc": {
			podHasPVC: true,
			pvcError:  fmt.Errorf("foo"),
			expected:  false,
		},
		"error fetching pv": {
			podHasPVC: true,
			pvError:   fmt.Errorf("foo"),
			expected:  false,
		},
		"host path pvc": {
			podHasPVC:     true,
			pvcIsHostPath: true,
			expected:      true,
		},
		"non host path pvc": {
			podHasPVC:     true,
			pvcIsHostPath: false,
			expected:      false,
		},
	}

	for k, v := range tests {
		testKubelet := newTestKubelet(t, false)
		defer testKubelet.Cleanup()
		pod := &v1.Pod{
			Spec: v1.PodSpec{},
		}

		volumeToReturn := &v1.PersistentVolume{
			Spec: v1.PersistentVolumeSpec{},
		}

		if v.podHasPVC {
			pod.Spec.Volumes = []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{},
					},
				},
			}

			if v.pvcIsHostPath {
				volumeToReturn.Spec.PersistentVolumeSource = v1.PersistentVolumeSource{
					HostPath: &v1.HostPathVolumeSource{},
				}
			}

		}

		testKubelet.fakeKubeClient.AddReactor("get", "persistentvolumeclaims", func(action core.Action) (bool, runtime.Object, error) {
			return true, &v1.PersistentVolumeClaim{
				Spec: v1.PersistentVolumeClaimSpec{
					VolumeName: "foo",
				},
			}, v.pvcError
		})
		testKubelet.fakeKubeClient.AddReactor("get", "persistentvolumes", func(action core.Action) (bool, runtime.Object, error) {
			return true, volumeToReturn, v.pvError
		})

		actual := testKubelet.kubelet.hasHostMountPVC(pod)
		if actual != v.expected {
			t.Errorf("%s expected %t but got %t", k, v.expected, actual)
		}

	}
}

func TestHasNonNamespacedCapability(t *testing.T) {
	createPodWithCap := func(caps []v1.Capability) *v1.Pod {
		pod := &v1.Pod{
			Spec: v1.PodSpec{
				Containers: []v1.Container{{}},
			},
		}

		if len(caps) > 0 {
			pod.Spec.Containers[0].SecurityContext = &v1.SecurityContext{
				Capabilities: &v1.Capabilities{
					Add: caps,
				},
			}
		}
		return pod
	}

	nilCaps := createPodWithCap([]v1.Capability{v1.Capability("foo")})
	nilCaps.Spec.Containers[0].SecurityContext = nil

	tests := map[string]struct {
		pod      *v1.Pod
		expected bool
	}{
		"nil security contxt":           {createPodWithCap(nil), false},
		"nil caps":                      {nilCaps, false},
		"namespaced cap":                {createPodWithCap([]v1.Capability{v1.Capability("foo")}), false},
		"non-namespaced cap MKNOD":      {createPodWithCap([]v1.Capability{v1.Capability("MKNOD")}), true},
		"non-namespaced cap SYS_TIME":   {createPodWithCap([]v1.Capability{v1.Capability("SYS_TIME")}), true},
		"non-namespaced cap SYS_MODULE": {createPodWithCap([]v1.Capability{v1.Capability("SYS_MODULE")}), true},
	}

	for k, v := range tests {
		actual := hasNonNamespacedCapability(v.pod)
		if actual != v.expected {
			t.Errorf("%s failed, expected %t but got %t", k, v.expected, actual)
		}
	}
}

func TestHasHostVolume(t *testing.T) {
	pod := &v1.Pod{
		Spec: v1.PodSpec{
			Volumes: []v1.Volume{
				{
					VolumeSource: v1.VolumeSource{
						HostPath: &v1.HostPathVolumeSource{},
					},
				},
			},
		},
	}

	result := hasHostVolume(pod)
	if !result {
		t.Errorf("expected host volume to enable host user namespace")
	}

	pod.Spec.Volumes[0].VolumeSource.HostPath = nil
	result = hasHostVolume(pod)
	if result {
		t.Errorf("expected nil host volume to not enable host user namespace")
	}
}

func TestHasHostNamespace(t *testing.T) {
	tests := map[string]struct {
		ps       v1.PodSpec
		expected bool
	}{
		"nil psc": {
			ps:       v1.PodSpec{},
			expected: false},

		"host pid true": {
			ps: v1.PodSpec{
				HostPID:         true,
				SecurityContext: &v1.PodSecurityContext{},
			},
			expected: true,
		},
		"host ipc true": {
			ps: v1.PodSpec{
				HostIPC:         true,
				SecurityContext: &v1.PodSecurityContext{},
			},
			expected: true,
		},
		"host net true": {
			ps: v1.PodSpec{
				HostNetwork:     true,
				SecurityContext: &v1.PodSecurityContext{},
			},
			expected: true,
		},
		"no host ns": {
			ps: v1.PodSpec{
				SecurityContext: &v1.PodSecurityContext{},
			},
			expected: false,
		},
	}

	for k, v := range tests {
		pod := &v1.Pod{
			Spec: v.ps,
		}
		actual := hasHostNamespace(pod)
		if actual != v.expected {
			t.Errorf("%s failed, expected %t but got %t", k, v.expected, actual)
		}
	}
}

func TestTruncatePodHostname(t *testing.T) {
	for c, test := range map[string]struct {
		input  string
		output string
	}{
		"valid hostname": {
			input:  "test.pod.hostname",
			output: "test.pod.hostname",
		},
		"too long hostname": {
			input:  "1234567.1234567.1234567.1234567.1234567.1234567.1234567.1234567.1234567.", // 8*9=72 chars
			output: "1234567.1234567.1234567.1234567.1234567.1234567.1234567.1234567",          //8*8-1=63 chars
		},
		"hostname end with .": {
			input:  "1234567.1234567.1234567.1234567.1234567.1234567.1234567.123456.1234567.", // 8*9-1=71 chars
			output: "1234567.1234567.1234567.1234567.1234567.1234567.1234567.123456",          //8*8-2=62 chars
		},
		"hostname end with -": {
			input:  "1234567.1234567.1234567.1234567.1234567.1234567.1234567.123456-1234567.", // 8*9-1=71 chars
			output: "1234567.1234567.1234567.1234567.1234567.1234567.1234567.123456",          //8*8-2=62 chars
		},
	} {
		t.Logf("TestCase: %q", c)
		output, err := truncatePodHostnameIfNeeded("test-pod", test.input)
		assert.NoError(t, err)
		assert.Equal(t, test.output, output)
	}
}
