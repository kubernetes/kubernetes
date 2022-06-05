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
	"net"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/record"
	netutils "k8s.io/utils/net"

	// TODO: remove this import if
	// api.Registry.GroupOrDie(v1.GroupName).GroupVersions[0].String() is changed
	// to "v1"?

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	runtimeapi "k8s.io/cri-api/pkg/apis/runtime/v1"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/kubelet/cri/streaming/portforward"
	"k8s.io/kubernetes/pkg/kubelet/cri/streaming/remotecommand"
	"k8s.io/kubernetes/pkg/kubelet/prober/results"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

func TestNodeHostsFileContent(t *testing.T) {
	testCases := []struct {
		hostsFileName            string
		hostAliases              []v1.HostAlias
		rawHostsFileContent      string
		expectedHostsFileContent string
	}{
		{
			hostsFileName: "hosts_test_file1",
			hostAliases:   []v1.HostAlias{},
			rawHostsFileContent: `# hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
123.45.67.89	some.domain
`,
			expectedHostsFileContent: `# Kubernetes-managed hosts file (host network).
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
			hostsFileName: "hosts_test_file2",
			hostAliases:   []v1.HostAlias{},
			rawHostsFileContent: `# another hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
12.34.56.78	another.domain
`,
			expectedHostsFileContent: `# Kubernetes-managed hosts file (host network).
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
			hostsFileName: "hosts_test_file1_with_host_aliases",
			hostAliases: []v1.HostAlias{
				{IP: "123.45.67.89", Hostnames: []string{"foo", "bar", "baz"}},
			},
			rawHostsFileContent: `# hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
123.45.67.89	some.domain
`,
			expectedHostsFileContent: `# Kubernetes-managed hosts file (host network).
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
			hostsFileName: "hosts_test_file2_with_host_aliases",
			hostAliases: []v1.HostAlias{
				{IP: "123.45.67.89", Hostnames: []string{"foo", "bar", "baz"}},
				{IP: "456.78.90.123", Hostnames: []string{"park", "doo", "boo"}},
			},
			rawHostsFileContent: `# another hosts file for testing.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
12.34.56.78	another.domain
`,
			expectedHostsFileContent: `# Kubernetes-managed hosts file (host network).
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
		t.Run(testCase.hostsFileName, func(t *testing.T) {
			tmpdir, err := writeHostsFile(testCase.hostsFileName, testCase.rawHostsFileContent)
			require.NoError(t, err, "could not create a temp hosts file")
			defer os.RemoveAll(tmpdir)

			actualContent, fileReadErr := nodeHostsFileContent(filepath.Join(tmpdir, testCase.hostsFileName), testCase.hostAliases)
			require.NoError(t, fileReadErr, "could not create read hosts file")
			assert.Equal(t, testCase.expectedHostsFileContent, string(actualContent), "hosts file content not expected")
		})
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
		hostIPs         []string
		hostName        string
		hostDomainName  string
		hostAliases     []v1.HostAlias
		expectedContent string
	}{
		{
			hostIPs:     []string{"123.45.67.89"},
			hostName:    "podFoo",
			hostAliases: []v1.HostAlias{},
			expectedContent: `# Kubernetes-managed hosts file.
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
			hostIPs:        []string{"203.0.113.1"},
			hostName:       "podFoo",
			hostDomainName: "domainFoo",
			hostAliases:    []v1.HostAlias{},
			expectedContent: `# Kubernetes-managed hosts file.
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
			hostIPs:        []string{"203.0.113.1"},
			hostName:       "podFoo",
			hostDomainName: "domainFoo",
			hostAliases: []v1.HostAlias{
				{IP: "123.45.67.89", Hostnames: []string{"foo", "bar", "baz"}},
			},
			expectedContent: `# Kubernetes-managed hosts file.
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
			hostIPs:        []string{"203.0.113.1"},
			hostName:       "podFoo",
			hostDomainName: "domainFoo",
			hostAliases: []v1.HostAlias{
				{IP: "123.45.67.89", Hostnames: []string{"foo", "bar", "baz"}},
				{IP: "456.78.90.123", Hostnames: []string{"park", "doo", "boo"}},
			},
			expectedContent: `# Kubernetes-managed hosts file.
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
		{
			hostIPs:        []string{"203.0.113.1", "fd00::6"},
			hostName:       "podFoo",
			hostDomainName: "domainFoo",
			hostAliases:    []v1.HostAlias{},
			expectedContent: `# Kubernetes-managed hosts file.
127.0.0.1	localhost
::1	localhost ip6-localhost ip6-loopback
fe00::0	ip6-localnet
fe00::0	ip6-mcastprefix
fe00::1	ip6-allnodes
fe00::2	ip6-allrouters
203.0.113.1	podFoo.domainFoo	podFoo
fd00::6	podFoo.domainFoo	podFoo
`,
		},
	}

	for _, testCase := range testCases {
		actualContent := managedHostsFileContent(testCase.hostIPs, testCase.hostName, testCase.hostDomainName, testCase.hostAliases)
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
		// this isn't 100% foolproof as a bug in a real CommandRunner where it fails to copy to stdout/stderr wouldn't be caught by this test
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
		staticPod          bool                   // whether the pod should be a static pod (versus an API pod)
		unsyncedServices   bool                   // whether the services should NOT be synced
		configMap          *v1.ConfigMap          // an optional ConfigMap to pull from
		secret             *v1.Secret             // an optional Secret to pull from
		podIPs             []string               // the pod IPs
		expectedEnvs       []kubecontainer.EnvVar // a set of expected environment vars
		expectedError      bool                   // does the test fail
		expectedEvent      string                 // does the test emit an event
	}{
		{
			name:               "if services aren't synced, non-static pods should fail",
			ns:                 "test1",
			enableServiceLinks: &falseValue,
			container:          &v1.Container{Env: []v1.EnvVar{}},
			masterServiceNs:    metav1.NamespaceDefault,
			nilLister:          false,
			staticPod:          false,
			unsyncedServices:   true,
			expectedEnvs:       []kubecontainer.EnvVar{},
			expectedError:      true,
		},
		{
			name:               "if services aren't synced, static pods should succeed", // if there is no service
			ns:                 "test1",
			enableServiceLinks: &falseValue,
			container:          &v1.Container{Env: []v1.EnvVar{}},
			masterServiceNs:    metav1.NamespaceDefault,
			nilLister:          false,
			staticPod:          true,
			unsyncedServices:   true,
		},
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
						Name: "POD_IPS",
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "status.podIPs",
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
			podIPs:          []string{"1.2.3.4", "fd00::6"},
			masterServiceNs: "nothing",
			nilLister:       true,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "POD_NAME", Value: "dapi-test-pod-name"},
				{Name: "POD_NAMESPACE", Value: "downward-api"},
				{Name: "POD_NODE_NAME", Value: "node-name"},
				{Name: "POD_SERVICE_ACCOUNT_NAME", Value: "special"},
				{Name: "POD_IP", Value: "1.2.3.4"},
				{Name: "POD_IPS", Value: "1.2.3.4,fd00::6"},
				{Name: "HOST_IP", Value: testKubeletHostIP},
			},
		},
		{
			name:               "downward api pod ips reverse order",
			ns:                 "downward-api",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
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
						Name: "POD_IPS",
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "status.podIPs",
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
			podIPs:          []string{"fd00::6", "1.2.3.4"},
			masterServiceNs: "nothing",
			nilLister:       true,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "POD_IP", Value: "1.2.3.4"},
				{Name: "POD_IPS", Value: "1.2.3.4,fd00::6"},
				{Name: "HOST_IP", Value: testKubeletHostIP},
			},
		},
		{
			name:               "downward api pod ips multiple ips",
			ns:                 "downward-api",
			enableServiceLinks: &falseValue,
			container: &v1.Container{
				Env: []v1.EnvVar{
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
						Name: "POD_IPS",
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{
								APIVersion: "v1",
								FieldPath:  "status.podIPs",
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
			podIPs:          []string{"1.2.3.4", "192.168.1.1.", "fd00::6"},
			masterServiceNs: "nothing",
			nilLister:       true,
			expectedEnvs: []kubecontainer.EnvVar{
				{Name: "POD_IP", Value: "1.2.3.4"},
				{Name: "POD_IPS", Value: "1.2.3.4,fd00::6"},
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
		t.Run(tc.name, func(t *testing.T) {
			fakeRecorder := record.NewFakeRecorder(1)
			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			testKubelet.kubelet.recorder = fakeRecorder
			defer testKubelet.Cleanup()
			kl := testKubelet.kubelet
			kl.masterServiceNamespace = tc.masterServiceNs
			if tc.nilLister {
				kl.serviceLister = nil
			} else if tc.unsyncedServices {
				kl.serviceLister = testServiceLister{}
				kl.serviceHasSynced = func() bool { return false }
			} else {
				kl.serviceLister = testServiceLister{services}
				kl.serviceHasSynced = func() bool { return true }
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
					Namespace:   tc.ns,
					Name:        "dapi-test-pod-name",
					Annotations: map[string]string{},
				},
				Spec: v1.PodSpec{
					ServiceAccountName: "special",
					NodeName:           "node-name",
					EnableServiceLinks: tc.enableServiceLinks,
				},
			}
			podIP := ""
			if len(tc.podIPs) > 0 {
				podIP = tc.podIPs[0]
			}
			if tc.staticPod {
				testPod.Annotations[kubetypes.ConfigSourceAnnotationKey] = "file"
			}

			result, err := kl.makeEnvironmentVariables(testPod, tc.container, podIP, tc.podIPs)
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
		})

	}
}

func waitingState(cName string) v1.ContainerStatus {
	return waitingStateWithReason(cName, "")
}
func waitingStateWithReason(cName, reason string) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name: cName,
		State: v1.ContainerState{
			Waiting: &v1.ContainerStateWaiting{Reason: reason},
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
func waitingStateWithNonZeroTermination(cName string) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name: cName,
		State: v1.ContainerState{
			Waiting: &v1.ContainerStateWaiting{},
		},
		LastTerminationState: v1.ContainerState{
			Terminated: &v1.ContainerStateTerminated{
				ExitCode: -1,
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
func runningStateWithStartedAt(cName string, startedAt time.Time) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name: cName,
		State: v1.ContainerState{
			Running: &v1.ContainerStateRunning{StartedAt: metav1.Time{Time: startedAt}},
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
func waitingWithLastTerminationUnknown(cName string, restartCount int32) v1.ContainerStatus {
	return v1.ContainerStatus{
		Name: cName,
		State: v1.ContainerState{
			Waiting: &v1.ContainerStateWaiting{Reason: "ContainerCreating"},
		},
		LastTerminationState: v1.ContainerState{
			Terminated: &v1.ContainerStateTerminated{
				Reason:   "ContainerStatusUnknown",
				Message:  "The container could not be located when the pod was deleted.  The container used to be Running",
				ExitCode: 137,
			},
		},
		RestartCount: restartCount,
	}
}
func ready(status v1.ContainerStatus) v1.ContainerStatus {
	status.Ready = true
	return status
}
func withID(status v1.ContainerStatus, id string) v1.ContainerStatus {
	status.ContainerID = id
	return status
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

func TestPodPhaseWithRestartAlwaysInitContainers(t *testing.T) {
	desiredState := v1.PodSpec{
		NodeName: "machine",
		InitContainers: []v1.Container{
			{Name: "containerX"},
		},
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
		{&v1.Pod{Spec: desiredState, Status: v1.PodStatus{}}, v1.PodPending, "empty, waiting"},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						runningState("containerX"),
					},
				},
			},
			v1.PodPending,
			"init container running",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						failedState("containerX"),
					},
				},
			},
			v1.PodPending,
			"init container terminated non-zero",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						waitingStateWithLastTermination("containerX"),
					},
				},
			},
			v1.PodPending,
			"init container waiting, terminated zero",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						waitingStateWithNonZeroTermination("containerX"),
					},
				},
			},
			v1.PodPending,
			"init container waiting, terminated non-zero",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						waitingState("containerX"),
					},
				},
			},
			v1.PodPending,
			"init container waiting, not terminated",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						succeededState("containerX"),
					},
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			v1.PodRunning,
			"init container succeeded",
		},
	}
	for _, test := range tests {
		statusInfo := append(test.pod.Status.InitContainerStatuses[:], test.pod.Status.ContainerStatuses[:]...)
		status := getPhase(&test.pod.Spec, statusInfo)
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

func TestPodPhaseWithRestartNeverInitContainers(t *testing.T) {
	desiredState := v1.PodSpec{
		NodeName: "machine",
		InitContainers: []v1.Container{
			{Name: "containerX"},
		},
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
		{&v1.Pod{Spec: desiredState, Status: v1.PodStatus{}}, v1.PodPending, "empty, waiting"},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						runningState("containerX"),
					},
				},
			},
			v1.PodPending,
			"init container running",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						failedState("containerX"),
					},
				},
			},
			v1.PodFailed,
			"init container terminated non-zero",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						waitingStateWithLastTermination("containerX"),
					},
				},
			},
			v1.PodPending,
			"init container waiting, terminated zero",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						waitingStateWithNonZeroTermination("containerX"),
					},
				},
			},
			v1.PodFailed,
			"init container waiting, terminated non-zero",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						waitingState("containerX"),
					},
				},
			},
			v1.PodPending,
			"init container waiting, not terminated",
		},
		{
			&v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					InitContainerStatuses: []v1.ContainerStatus{
						succeededState("containerX"),
					},
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			v1.PodRunning,
			"init container succeeded",
		},
	}
	for _, test := range tests {
		statusInfo := append(test.pod.Status.InitContainerStatuses[:], test.pod.Status.ContainerStatuses[:]...)
		status := getPhase(&test.pod.Spec, statusInfo)
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

// No special init-specific logic for this, see RestartAlways case
// func TestPodPhaseWithRestartOnFailureInitContainers(t *testing.T) {
// }

func TestConvertToAPIContainerStatuses(t *testing.T) {
	desiredState := v1.PodSpec{
		NodeName: "machine",
		Containers: []v1.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: v1.RestartPolicyAlways,
	}
	now := metav1.Now()

	tests := []struct {
		name              string
		pod               *v1.Pod
		currentStatus     *kubecontainer.PodStatus
		previousStatus    []v1.ContainerStatus
		containers        []v1.Container
		hasInitContainers bool
		isInitContainer   bool
		expected          []v1.ContainerStatus
	}{
		{
			name: "no current status, with previous statuses and deletion",
			pod: &v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
				ObjectMeta: metav1.ObjectMeta{Name: "my-pod", DeletionTimestamp: &now},
			},
			currentStatus: &kubecontainer.PodStatus{},
			previousStatus: []v1.ContainerStatus{
				runningState("containerA"),
				runningState("containerB"),
			},
			containers: desiredState.Containers,
			// no init containers
			// is not an init container
			expected: []v1.ContainerStatus{
				waitingWithLastTerminationUnknown("containerA", 0),
				waitingWithLastTerminationUnknown("containerB", 0),
			},
		},
		{
			name: "no current status, with previous statuses and no deletion",
			pod: &v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			currentStatus: &kubecontainer.PodStatus{},
			previousStatus: []v1.ContainerStatus{
				runningState("containerA"),
				runningState("containerB"),
			},
			containers: desiredState.Containers,
			// no init containers
			// is not an init container
			expected: []v1.ContainerStatus{
				waitingWithLastTerminationUnknown("containerA", 1),
				waitingWithLastTerminationUnknown("containerB", 1),
			},
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			defer testKubelet.Cleanup()
			kl := testKubelet.kubelet
			containerStatuses := kl.convertToAPIContainerStatuses(
				test.pod,
				test.currentStatus,
				test.previousStatus,
				test.containers,
				test.hasInitContainers,
				test.isInitContainer,
			)
			for i, status := range containerStatuses {
				assert.Equal(t, test.expected[i], status, "[test %s]", test.name)
			}
		})
	}
}

func Test_generateAPIPodStatus(t *testing.T) {
	desiredState := v1.PodSpec{
		NodeName: "machine",
		Containers: []v1.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: v1.RestartPolicyAlways,
	}
	sandboxReadyStatus := &kubecontainer.PodStatus{
		SandboxStatuses: []*runtimeapi.PodSandboxStatus{
			{
				Network: &runtimeapi.PodSandboxNetworkStatus{
					Ip: "10.0.0.10",
				},
				Metadata: &runtimeapi.PodSandboxMetadata{Attempt: uint32(0)},
				State:    runtimeapi.PodSandboxState_SANDBOX_READY,
			},
		},
	}
	now := metav1.Now()

	tests := []struct {
		name                           string
		pod                            *v1.Pod
		currentStatus                  *kubecontainer.PodStatus
		unreadyContainer               []string
		previousStatus                 v1.PodStatus
		expected                       v1.PodStatus
		expectedPodHasNetworkCondition v1.PodCondition
	}{
		{
			name: "current status ready, with previous statuses and deletion",
			pod: &v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
				ObjectMeta: metav1.ObjectMeta{Name: "my-pod", DeletionTimestamp: &now},
			},
			currentStatus: sandboxReadyStatus,
			previousStatus: v1.PodStatus{
				ContainerStatuses: []v1.ContainerStatus{
					runningState("containerA"),
					runningState("containerB"),
				},
			},
			expected: v1.PodStatus{
				Phase:    v1.PodRunning,
				HostIP:   "127.0.0.1",
				QOSClass: v1.PodQOSBestEffort,
				Conditions: []v1.PodCondition{
					{Type: v1.PodInitialized, Status: v1.ConditionTrue},
					{Type: v1.PodReady, Status: v1.ConditionTrue},
					{Type: v1.ContainersReady, Status: v1.ConditionTrue},
					{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				},
				ContainerStatuses: []v1.ContainerStatus{
					ready(waitingWithLastTerminationUnknown("containerA", 0)),
					ready(waitingWithLastTerminationUnknown("containerB", 0)),
				},
			},
			expectedPodHasNetworkCondition: v1.PodCondition{
				Type:   kubetypes.PodHasNetwork,
				Status: v1.ConditionTrue,
			},
		},
		{
			name: "current status ready, with previous statuses and no deletion",
			pod: &v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			currentStatus: sandboxReadyStatus,
			previousStatus: v1.PodStatus{
				ContainerStatuses: []v1.ContainerStatus{
					runningState("containerA"),
					runningState("containerB"),
				},
			},
			expected: v1.PodStatus{
				Phase:    v1.PodRunning,
				HostIP:   "127.0.0.1",
				QOSClass: v1.PodQOSBestEffort,
				Conditions: []v1.PodCondition{
					{Type: v1.PodInitialized, Status: v1.ConditionTrue},
					{Type: v1.PodReady, Status: v1.ConditionTrue},
					{Type: v1.ContainersReady, Status: v1.ConditionTrue},
					{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				},
				ContainerStatuses: []v1.ContainerStatus{
					ready(waitingWithLastTerminationUnknown("containerA", 1)),
					ready(waitingWithLastTerminationUnknown("containerB", 1)),
				},
			},
			expectedPodHasNetworkCondition: v1.PodCondition{
				Type:   kubetypes.PodHasNetwork,
				Status: v1.ConditionTrue,
			},
		},
		{
			name: "terminal phase cannot be changed (apiserver previous is succeeded)",
			pod: &v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					Phase: v1.PodSucceeded,
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			currentStatus: &kubecontainer.PodStatus{},
			previousStatus: v1.PodStatus{
				ContainerStatuses: []v1.ContainerStatus{
					runningState("containerA"),
					runningState("containerB"),
				},
			},
			expected: v1.PodStatus{
				Phase:    v1.PodSucceeded,
				HostIP:   "127.0.0.1",
				QOSClass: v1.PodQOSBestEffort,
				Conditions: []v1.PodCondition{
					{Type: v1.PodInitialized, Status: v1.ConditionTrue, Reason: "PodCompleted"},
					{Type: v1.PodReady, Status: v1.ConditionFalse, Reason: "PodCompleted"},
					{Type: v1.ContainersReady, Status: v1.ConditionFalse, Reason: "PodCompleted"},
					{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				},
				ContainerStatuses: []v1.ContainerStatus{
					ready(waitingWithLastTerminationUnknown("containerA", 1)),
					ready(waitingWithLastTerminationUnknown("containerB", 1)),
				},
			},
			expectedPodHasNetworkCondition: v1.PodCondition{
				Type:   kubetypes.PodHasNetwork,
				Status: v1.ConditionFalse,
			},
		},
		{
			name: "terminal phase from previous status must remain terminal, restartAlways",
			pod: &v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			currentStatus: &kubecontainer.PodStatus{},
			previousStatus: v1.PodStatus{
				Phase: v1.PodSucceeded,
				ContainerStatuses: []v1.ContainerStatus{
					runningState("containerA"),
					runningState("containerB"),
				},
				// Reason and message should be preserved
				Reason:  "Test",
				Message: "test",
			},
			expected: v1.PodStatus{
				Phase:    v1.PodSucceeded,
				HostIP:   "127.0.0.1",
				QOSClass: v1.PodQOSBestEffort,
				Conditions: []v1.PodCondition{
					{Type: v1.PodInitialized, Status: v1.ConditionTrue, Reason: "PodCompleted"},
					{Type: v1.PodReady, Status: v1.ConditionFalse, Reason: "PodCompleted"},
					{Type: v1.ContainersReady, Status: v1.ConditionFalse, Reason: "PodCompleted"},
					{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				},
				ContainerStatuses: []v1.ContainerStatus{
					ready(waitingWithLastTerminationUnknown("containerA", 1)),
					ready(waitingWithLastTerminationUnknown("containerB", 1)),
				},
				Reason:  "Test",
				Message: "test",
			},
			expectedPodHasNetworkCondition: v1.PodCondition{
				Type:   kubetypes.PodHasNetwork,
				Status: v1.ConditionFalse,
			},
		},
		{
			name: "terminal phase from previous status must remain terminal, restartNever",
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					NodeName: "machine",
					Containers: []v1.Container{
						{Name: "containerA"},
						{Name: "containerB"},
					},
					RestartPolicy: v1.RestartPolicyNever,
				},
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			currentStatus: &kubecontainer.PodStatus{},
			previousStatus: v1.PodStatus{
				Phase: v1.PodSucceeded,
				ContainerStatuses: []v1.ContainerStatus{
					succeededState("containerA"),
					succeededState("containerB"),
				},
				// Reason and message should be preserved
				Reason:  "Test",
				Message: "test",
			},
			expected: v1.PodStatus{
				Phase:    v1.PodSucceeded,
				HostIP:   "127.0.0.1",
				QOSClass: v1.PodQOSBestEffort,
				Conditions: []v1.PodCondition{
					{Type: v1.PodInitialized, Status: v1.ConditionTrue, Reason: "PodCompleted"},
					{Type: v1.PodReady, Status: v1.ConditionFalse, Reason: "PodCompleted"},
					{Type: v1.ContainersReady, Status: v1.ConditionFalse, Reason: "PodCompleted"},
					{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				},
				ContainerStatuses: []v1.ContainerStatus{
					ready(succeededState("containerA")),
					ready(succeededState("containerB")),
				},
				Reason:  "Test",
				Message: "test",
			},
			expectedPodHasNetworkCondition: v1.PodCondition{
				Type:   kubetypes.PodHasNetwork,
				Status: v1.ConditionFalse,
			},
		},
		{
			name: "running can revert to pending",
			pod: &v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
					ContainerStatuses: []v1.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			currentStatus: sandboxReadyStatus,
			previousStatus: v1.PodStatus{
				ContainerStatuses: []v1.ContainerStatus{
					waitingState("containerA"),
					waitingState("containerB"),
				},
			},
			expected: v1.PodStatus{
				Phase:    v1.PodPending,
				HostIP:   "127.0.0.1",
				QOSClass: v1.PodQOSBestEffort,
				Conditions: []v1.PodCondition{
					{Type: v1.PodInitialized, Status: v1.ConditionTrue},
					{Type: v1.PodReady, Status: v1.ConditionTrue},
					{Type: v1.ContainersReady, Status: v1.ConditionTrue},
					{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				},
				ContainerStatuses: []v1.ContainerStatus{
					ready(waitingStateWithReason("containerA", "ContainerCreating")),
					ready(waitingStateWithReason("containerB", "ContainerCreating")),
				},
			},
			expectedPodHasNetworkCondition: v1.PodCondition{
				Type:   kubetypes.PodHasNetwork,
				Status: v1.ConditionTrue,
			},
		},
		{
			name: "reason and message are preserved when phase doesn't change",
			pod: &v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					Phase: v1.PodRunning,
					ContainerStatuses: []v1.ContainerStatus{
						waitingState("containerA"),
						waitingState("containerB"),
					},
				},
			},
			currentStatus: &kubecontainer.PodStatus{
				SandboxStatuses: sandboxReadyStatus.SandboxStatuses,
				ContainerStatuses: []*kubecontainer.Status{
					{
						ID:        kubecontainer.ContainerID{ID: "foo"},
						Name:      "containerB",
						StartedAt: time.Unix(1, 0).UTC(),
						State:     kubecontainer.ContainerStateRunning,
					},
				},
			},
			previousStatus: v1.PodStatus{
				Phase:   v1.PodPending,
				Reason:  "Test",
				Message: "test",
				ContainerStatuses: []v1.ContainerStatus{
					waitingState("containerA"),
					runningState("containerB"),
				},
			},
			expected: v1.PodStatus{
				Phase:    v1.PodPending,
				Reason:   "Test",
				Message:  "test",
				HostIP:   "127.0.0.1",
				QOSClass: v1.PodQOSBestEffort,
				Conditions: []v1.PodCondition{
					{Type: v1.PodInitialized, Status: v1.ConditionTrue},
					{Type: v1.PodReady, Status: v1.ConditionTrue},
					{Type: v1.ContainersReady, Status: v1.ConditionTrue},
					{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				},
				ContainerStatuses: []v1.ContainerStatus{
					ready(waitingStateWithReason("containerA", "ContainerCreating")),
					ready(withID(runningStateWithStartedAt("containerB", time.Unix(1, 0).UTC()), "://foo")),
				},
			},
			expectedPodHasNetworkCondition: v1.PodCondition{
				Type:   kubetypes.PodHasNetwork,
				Status: v1.ConditionTrue,
			},
		},
		{
			name: "reason and message are cleared when phase changes",
			pod: &v1.Pod{
				Spec: desiredState,
				Status: v1.PodStatus{
					Phase: v1.PodPending,
					ContainerStatuses: []v1.ContainerStatus{
						waitingState("containerA"),
						waitingState("containerB"),
					},
				},
			},
			currentStatus: &kubecontainer.PodStatus{
				SandboxStatuses: sandboxReadyStatus.SandboxStatuses,
				ContainerStatuses: []*kubecontainer.Status{
					{
						ID:        kubecontainer.ContainerID{ID: "c1"},
						Name:      "containerA",
						StartedAt: time.Unix(1, 0).UTC(),
						State:     kubecontainer.ContainerStateRunning,
					},
					{
						ID:        kubecontainer.ContainerID{ID: "c2"},
						Name:      "containerB",
						StartedAt: time.Unix(2, 0).UTC(),
						State:     kubecontainer.ContainerStateRunning,
					},
				},
			},
			previousStatus: v1.PodStatus{
				Phase:   v1.PodPending,
				Reason:  "Test",
				Message: "test",
				ContainerStatuses: []v1.ContainerStatus{
					runningState("containerA"),
					runningState("containerB"),
				},
			},
			expected: v1.PodStatus{
				Phase:    v1.PodRunning,
				HostIP:   "127.0.0.1",
				QOSClass: v1.PodQOSBestEffort,
				Conditions: []v1.PodCondition{
					{Type: v1.PodInitialized, Status: v1.ConditionTrue},
					{Type: v1.PodReady, Status: v1.ConditionTrue},
					{Type: v1.ContainersReady, Status: v1.ConditionTrue},
					{Type: v1.PodScheduled, Status: v1.ConditionTrue},
				},
				ContainerStatuses: []v1.ContainerStatus{
					ready(withID(runningStateWithStartedAt("containerA", time.Unix(1, 0).UTC()), "://c1")),
					ready(withID(runningStateWithStartedAt("containerB", time.Unix(2, 0).UTC()), "://c2")),
				},
			},
			expectedPodHasNetworkCondition: v1.PodCondition{
				Type:   kubetypes.PodHasNetwork,
				Status: v1.ConditionTrue,
			},
		},
	}
	for _, test := range tests {
		for _, enablePodHasNetworkCondition := range []bool{false, true} {
			t.Run(test.name, func(t *testing.T) {
				defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodHasNetworkCondition, enablePodHasNetworkCondition)()
				testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
				defer testKubelet.Cleanup()
				kl := testKubelet.kubelet
				kl.statusManager.SetPodStatus(test.pod, test.previousStatus)
				for _, name := range test.unreadyContainer {
					kl.readinessManager.Set(kubecontainer.BuildContainerID("", findContainerStatusByName(test.expected, name).ContainerID), results.Failure, test.pod)
				}
				actual := kl.generateAPIPodStatus(test.pod, test.currentStatus)
				if enablePodHasNetworkCondition {
					test.expected.Conditions = append([]v1.PodCondition{test.expectedPodHasNetworkCondition}, test.expected.Conditions...)
				}
				if !apiequality.Semantic.DeepEqual(test.expected, actual) {
					t.Fatalf("Unexpected status: %s", diff.ObjectReflectDiff(actual, test.expected))
				}
			})
		}
	}
}

func findContainerStatusByName(status v1.PodStatus, name string) *v1.ContainerStatus {
	for i, c := range status.InitContainerStatuses {
		if c.Name == name {
			return &status.InitContainerStatuses[i]
		}
	}
	for i, c := range status.ContainerStatuses {
		if c.Name == name {
			return &status.ContainerStatuses[i]
		}
	}
	for i, c := range status.EphemeralContainerStatuses {
		if c.Name == name {
			return &status.EphemeralContainerStatuses[i]
		}
	}
	return nil
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
	)

	testcases := []struct {
		description string
		podFullName string
		container   string
		command     []string
		expectError bool
	}{{
		description: "success case",
		podFullName: podFullName,
		container:   containerID,
		command:     []string{"ls"},
		expectError: false,
	}, {
		description: "no such pod",
		podFullName: "bar" + podFullName,
		container:   containerID,
		command:     []string{"ls"},
		expectError: true,
	}, {
		description: "no such container",
		podFullName: podFullName,
		container:   "containerBar",
		command:     []string{"ls"},
		expectError: true,
	}, {
		description: "null exec command",
		podFullName: podFullName,
		container:   containerID,
		expectError: false,
	}, {
		description: "multi exec commands",
		podFullName: podFullName,
		container:   containerID,
		command:     []string{"bash", "-c", "ls"},
		expectError: false,
	}}

	for _, tc := range testcases {
		t.Run(tc.description, func(t *testing.T) {
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

			redirect, err := kubelet.GetExec(tc.podFullName, podUID, tc.container, tc.command, remotecommand.Options{})
			if tc.expectError {
				assert.Error(t, err, description)
			} else {
				assert.NoError(t, err, description)
				assert.Equal(t, containertest.FakeHost, redirect.Host, description+": redirect")
			}
		})
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
	type testcase struct {
		pvError         error
		pvcError        error
		expected        bool
		podHasPVC       bool
		pvcIsHostPath   bool
		podHasEphemeral bool
	}
	tests := map[string]testcase{
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
		"enabled ephemeral host path": {
			podHasEphemeral: true,
			pvcIsHostPath:   true,
			expected:        true,
		},
		"non host path pvc": {
			podHasPVC:     true,
			pvcIsHostPath: false,
			expected:      false,
		},
	}

	run := func(t *testing.T, v testcase) {
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
		}

		if v.podHasEphemeral {
			pod.Spec.Volumes = []v1.Volume{
				{
					Name: "xyz",
					VolumeSource: v1.VolumeSource{
						Ephemeral: &v1.EphemeralVolumeSource{},
					},
				},
			}
		}

		if (v.podHasPVC || v.podHasEphemeral) && v.pvcIsHostPath {
			volumeToReturn.Spec.PersistentVolumeSource = v1.PersistentVolumeSource{
				HostPath: &v1.HostPathVolumeSource{},
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
			t.Errorf("expected %t but got %t", v.expected, actual)
		}
	}

	for k, v := range tests {
		t.Run(k, func(t *testing.T) {
			run(t, v)
		})
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

func TestGenerateAPIPodStatusHostNetworkPodIPs(t *testing.T) {
	testcases := []struct {
		name          string
		nodeAddresses []v1.NodeAddress
		criPodIPs     []string
		podIPs        []v1.PodIP
	}{
		{
			name: "Simple",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
			},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
			},
		},
		{
			name: "InternalIP is preferred over ExternalIP",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeExternalIP, Address: "192.168.0.1"},
				{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
			},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
			},
		},
		{
			name: "Single-stack addresses in dual-stack cluster",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
			},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
			},
		},
		{
			name: "Multiple single-stack addresses in dual-stack cluster",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
				{Type: v1.NodeInternalIP, Address: "10.0.0.2"},
				{Type: v1.NodeExternalIP, Address: "192.168.0.1"},
			},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
			},
		},
		{
			name: "Dual-stack addresses in dual-stack cluster",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
				{Type: v1.NodeInternalIP, Address: "fd01::1234"},
			},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
				{IP: "fd01::1234"},
			},
		},
		{
			name: "CRI PodIPs override NodeAddresses",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
				{Type: v1.NodeInternalIP, Address: "fd01::1234"},
			},
			criPodIPs: []string{"192.168.0.1"},
			podIPs: []v1.PodIP{
				{IP: "192.168.0.1"},
				{IP: "fd01::1234"},
			},
		},
		{
			name: "CRI dual-stack PodIPs override NodeAddresses",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
				{Type: v1.NodeInternalIP, Address: "fd01::1234"},
			},
			criPodIPs: []string{"192.168.0.1", "2001:db8::2"},
			podIPs: []v1.PodIP{
				{IP: "192.168.0.1"},
				{IP: "2001:db8::2"},
			},
		},
		{
			// by default the cluster prefers IPv4
			name: "CRI dual-stack PodIPs override NodeAddresses prefer IPv4",
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
				{Type: v1.NodeInternalIP, Address: "fd01::1234"},
			},
			criPodIPs: []string{"2001:db8::2", "192.168.0.1"},
			podIPs: []v1.PodIP{
				{IP: "192.168.0.1"},
				{IP: "2001:db8::2"},
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			defer testKubelet.Cleanup()
			kl := testKubelet.kubelet

			kl.nodeLister = testNodeLister{nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{Name: string(kl.nodeName)},
					Status: v1.NodeStatus{
						Addresses: tc.nodeAddresses,
					},
				},
			}}

			pod := podWithUIDNameNs("12345", "test-pod", "test-namespace")
			pod.Spec.HostNetwork = true

			criStatus := &kubecontainer.PodStatus{
				ID:        pod.UID,
				Name:      pod.Name,
				Namespace: pod.Namespace,
				IPs:       tc.criPodIPs,
			}

			status := kl.generateAPIPodStatus(pod, criStatus)
			if !reflect.DeepEqual(status.PodIPs, tc.podIPs) {
				t.Fatalf("Expected PodIPs %#v, got %#v", tc.podIPs, status.PodIPs)
			}
			if tc.criPodIPs == nil && status.HostIP != status.PodIPs[0].IP {
				t.Fatalf("Expected HostIP %q to equal PodIPs[0].IP %q", status.HostIP, status.PodIPs[0].IP)
			}
		})
	}
}

func TestNodeAddressUpdatesGenerateAPIPodStatusHostNetworkPodIPs(t *testing.T) {
	testcases := []struct {
		name           string
		nodeIPs        []string
		nodeAddresses  []v1.NodeAddress
		expectedPodIPs []v1.PodIP
	}{

		{
			name:    "Immutable after update node addresses single-stack",
			nodeIPs: []string{"10.0.0.1"},
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.1.1.1"},
			},
			expectedPodIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
			},
		},
		{
			name:    "Immutable after update node addresses dual-stack - primary address",
			nodeIPs: []string{"10.0.0.1", "2001:db8::2"},
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "2001:db8::2"},
			},
			expectedPodIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
				{IP: "2001:db8::2"},
			},
		},
		{
			name:    "Immutable after update node addresses dual-stack - secondary address",
			nodeIPs: []string{"10.0.0.1", "2001:db8::2"},
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
				{Type: v1.NodeInternalIP, Address: "2001:db8:1:2:3::2"},
			},
			expectedPodIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
				{IP: "2001:db8::2"},
			},
		},
		{
			name:    "Immutable after update node addresses dual-stack - primary and secondary address",
			nodeIPs: []string{"10.0.0.1", "2001:db8::2"},
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "1.1.1.1"},
				{Type: v1.NodeInternalIP, Address: "2001:db8:1:2:3::2"},
			},
			expectedPodIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
				{IP: "2001:db8::2"},
			},
		},
		{
			name:    "Update secondary after new secondary address dual-stack",
			nodeIPs: []string{"10.0.0.1"},
			nodeAddresses: []v1.NodeAddress{
				{Type: v1.NodeInternalIP, Address: "10.0.0.1"},
				{Type: v1.NodeInternalIP, Address: "2001:db8::2"},
			},
			expectedPodIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
				{IP: "2001:db8::2"},
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			defer testKubelet.Cleanup()
			kl := testKubelet.kubelet
			for _, ip := range tc.nodeIPs {
				kl.nodeIPs = append(kl.nodeIPs, netutils.ParseIPSloppy(ip))
			}
			kl.nodeLister = testNodeLister{nodes: []*v1.Node{
				{
					ObjectMeta: metav1.ObjectMeta{Name: string(kl.nodeName)},
					Status: v1.NodeStatus{
						Addresses: tc.nodeAddresses,
					},
				},
			}}

			pod := podWithUIDNameNs("12345", "test-pod", "test-namespace")
			pod.Spec.HostNetwork = true
			for _, ip := range tc.nodeIPs {
				pod.Status.PodIPs = append(pod.Status.PodIPs, v1.PodIP{IP: ip})
			}
			if len(pod.Status.PodIPs) > 0 {
				pod.Status.PodIP = pod.Status.PodIPs[0].IP
			}

			// set old status
			podStatus := &kubecontainer.PodStatus{
				ID:        pod.UID,
				Name:      pod.Name,
				Namespace: pod.Namespace,
			}
			podStatus.IPs = tc.nodeIPs

			status := kl.generateAPIPodStatus(pod, podStatus)
			if !reflect.DeepEqual(status.PodIPs, tc.expectedPodIPs) {
				t.Fatalf("Expected PodIPs %#v, got %#v", tc.expectedPodIPs, status.PodIPs)
			}
			if kl.nodeIPs[0].String() != status.PodIPs[0].IP {
				t.Fatalf("Expected HostIP %q to equal PodIPs[0].IP %q", status.HostIP, status.PodIPs[0].IP)
			}
		})
	}
}

func TestGenerateAPIPodStatusPodIPs(t *testing.T) {
	testcases := []struct {
		name      string
		nodeIP    string
		criPodIPs []string
		podIPs    []v1.PodIP
	}{
		{
			name:      "Simple",
			nodeIP:    "",
			criPodIPs: []string{"10.0.0.1"},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
			},
		},
		{
			name:      "Dual-stack",
			nodeIP:    "",
			criPodIPs: []string{"10.0.0.1", "fd01::1234"},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
				{IP: "fd01::1234"},
			},
		},
		{
			name:      "Dual-stack with explicit node IP",
			nodeIP:    "192.168.1.1",
			criPodIPs: []string{"10.0.0.1", "fd01::1234"},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
				{IP: "fd01::1234"},
			},
		},
		{
			name:      "Dual-stack with CRI returning wrong family first",
			nodeIP:    "",
			criPodIPs: []string{"fd01::1234", "10.0.0.1"},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
				{IP: "fd01::1234"},
			},
		},
		{
			name:      "Dual-stack with explicit node IP with CRI returning wrong family first",
			nodeIP:    "192.168.1.1",
			criPodIPs: []string{"fd01::1234", "10.0.0.1"},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
				{IP: "fd01::1234"},
			},
		},
		{
			name:      "Dual-stack with IPv6 node IP",
			nodeIP:    "fd00::5678",
			criPodIPs: []string{"10.0.0.1", "fd01::1234"},
			podIPs: []v1.PodIP{
				{IP: "fd01::1234"},
				{IP: "10.0.0.1"},
			},
		},
		{
			name:      "Dual-stack with IPv6 node IP, other CRI order",
			nodeIP:    "fd00::5678",
			criPodIPs: []string{"fd01::1234", "10.0.0.1"},
			podIPs: []v1.PodIP{
				{IP: "fd01::1234"},
				{IP: "10.0.0.1"},
			},
		},
		{
			name:      "No Pod IP matching Node IP",
			nodeIP:    "fd00::5678",
			criPodIPs: []string{"10.0.0.1"},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
			},
		},
		{
			name:      "No Pod IP matching (unspecified) Node IP",
			nodeIP:    "",
			criPodIPs: []string{"fd01::1234"},
			podIPs: []v1.PodIP{
				{IP: "fd01::1234"},
			},
		},
		{
			name:      "Multiple IPv4 IPs",
			nodeIP:    "",
			criPodIPs: []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
			},
		},
		{
			name:      "Multiple Dual-Stack IPs",
			nodeIP:    "",
			criPodIPs: []string{"10.0.0.1", "10.0.0.2", "fd01::1234", "10.0.0.3", "fd01::5678"},
			podIPs: []v1.PodIP{
				{IP: "10.0.0.1"},
				{IP: "fd01::1234"},
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			defer testKubelet.Cleanup()
			kl := testKubelet.kubelet
			if tc.nodeIP != "" {
				kl.nodeIPs = []net.IP{netutils.ParseIPSloppy(tc.nodeIP)}
			}

			pod := podWithUIDNameNs("12345", "test-pod", "test-namespace")

			criStatus := &kubecontainer.PodStatus{
				ID:        pod.UID,
				Name:      pod.Name,
				Namespace: pod.Namespace,
				IPs:       tc.criPodIPs,
			}

			status := kl.generateAPIPodStatus(pod, criStatus)
			if !reflect.DeepEqual(status.PodIPs, tc.podIPs) {
				t.Fatalf("Expected PodIPs %#v, got %#v", tc.podIPs, status.PodIPs)
			}
			if status.PodIP != status.PodIPs[0].IP {
				t.Fatalf("Expected PodIP %q to equal PodIPs[0].IP %q", status.PodIP, status.PodIPs[0].IP)
			}
		})
	}
}

func TestSortPodIPs(t *testing.T) {
	testcases := []struct {
		name        string
		nodeIP      string
		podIPs      []string
		expectedIPs []string
	}{
		{
			name:        "Simple",
			nodeIP:      "",
			podIPs:      []string{"10.0.0.1"},
			expectedIPs: []string{"10.0.0.1"},
		},
		{
			name:        "Dual-stack",
			nodeIP:      "",
			podIPs:      []string{"10.0.0.1", "fd01::1234"},
			expectedIPs: []string{"10.0.0.1", "fd01::1234"},
		},
		{
			name:        "Dual-stack with explicit node IP",
			nodeIP:      "192.168.1.1",
			podIPs:      []string{"10.0.0.1", "fd01::1234"},
			expectedIPs: []string{"10.0.0.1", "fd01::1234"},
		},
		{
			name:        "Dual-stack with CRI returning wrong family first",
			nodeIP:      "",
			podIPs:      []string{"fd01::1234", "10.0.0.1"},
			expectedIPs: []string{"10.0.0.1", "fd01::1234"},
		},
		{
			name:        "Dual-stack with explicit node IP with CRI returning wrong family first",
			nodeIP:      "192.168.1.1",
			podIPs:      []string{"fd01::1234", "10.0.0.1"},
			expectedIPs: []string{"10.0.0.1", "fd01::1234"},
		},
		{
			name:        "Dual-stack with IPv6 node IP",
			nodeIP:      "fd00::5678",
			podIPs:      []string{"10.0.0.1", "fd01::1234"},
			expectedIPs: []string{"fd01::1234", "10.0.0.1"},
		},
		{
			name:        "Dual-stack with IPv6 node IP, other CRI order",
			nodeIP:      "fd00::5678",
			podIPs:      []string{"fd01::1234", "10.0.0.1"},
			expectedIPs: []string{"fd01::1234", "10.0.0.1"},
		},
		{
			name:        "No Pod IP matching Node IP",
			nodeIP:      "fd00::5678",
			podIPs:      []string{"10.0.0.1"},
			expectedIPs: []string{"10.0.0.1"},
		},
		{
			name:        "No Pod IP matching (unspecified) Node IP",
			nodeIP:      "",
			podIPs:      []string{"fd01::1234"},
			expectedIPs: []string{"fd01::1234"},
		},
		{
			name:        "Multiple IPv4 IPs",
			nodeIP:      "",
			podIPs:      []string{"10.0.0.1", "10.0.0.2", "10.0.0.3"},
			expectedIPs: []string{"10.0.0.1"},
		},
		{
			name:        "Multiple Dual-Stack IPs",
			nodeIP:      "",
			podIPs:      []string{"10.0.0.1", "10.0.0.2", "fd01::1234", "10.0.0.3", "fd01::5678"},
			expectedIPs: []string{"10.0.0.1", "fd01::1234"},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
			defer testKubelet.Cleanup()
			kl := testKubelet.kubelet
			if tc.nodeIP != "" {
				kl.nodeIPs = []net.IP{netutils.ParseIPSloppy(tc.nodeIP)}
			}

			podIPs := kl.sortPodIPs(tc.podIPs)
			if !reflect.DeepEqual(podIPs, tc.expectedIPs) {
				t.Fatalf("Expected PodIPs %#v, got %#v", tc.expectedIPs, podIPs)
			}
		})
	}
}

func TestConvertToAPIContainerStatusesDataRace(t *testing.T) {
	pod := podWithUIDNameNs("12345", "test-pod", "test-namespace")

	testTimestamp := time.Unix(123456789, 987654321)

	criStatus := &kubecontainer.PodStatus{
		ID:        pod.UID,
		Name:      pod.Name,
		Namespace: pod.Namespace,
		ContainerStatuses: []*kubecontainer.Status{
			{Name: "containerA", CreatedAt: testTimestamp},
			{Name: "containerB", CreatedAt: testTimestamp.Add(1)},
		},
	}

	testKubelet := newTestKubelet(t, false)
	defer testKubelet.Cleanup()
	kl := testKubelet.kubelet

	// convertToAPIContainerStatuses is purely transformative and shouldn't alter the state of the kubelet
	// as there are no synchronisation events in that function (no locks, no channels, ...) each test routine
	// should have its own vector clock increased independently. Golang race detector uses pure happens-before
	// detection, so would catch a race condition consistently, despite only spawning 2 goroutines
	for i := 0; i < 2; i++ {
		go func() {
			kl.convertToAPIContainerStatuses(pod, criStatus, []v1.ContainerStatus{}, []v1.Container{}, false, false)
		}()
	}
}
