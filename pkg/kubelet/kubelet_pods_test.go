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
	"bytes"
	"errors"
	"fmt"
	"io"
	"net"
	"sort"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	containertest "k8s.io/kubernetes/pkg/kubelet/container/testing"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/term"
)

func TestMakeMounts(t *testing.T) {
	container := api.Container{
		VolumeMounts: []api.VolumeMount{
			{
				MountPath: "/etc/hosts",
				Name:      "disk",
				ReadOnly:  false,
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
	}

	podVolumes := kubecontainer.VolumeMap{
		"disk":  kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/disk"}},
		"disk4": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/mnt/host"}},
		"disk5": kubecontainer.VolumeInfo{Mounter: &stubVolume{path: "/var/lib/kubelet/podID/volumes/empty/disk5"}},
	}

	pod := api.Pod{
		Spec: api.PodSpec{
			SecurityContext: &api.PodSecurityContext{
				HostNetwork: true,
			},
		},
	}

	mounts, _ := makeMounts(&pod, "/pod", &container, "fakepodname", "", "", podVolumes)

	expectedMounts := []kubecontainer.Mount{
		{
			Name:           "disk",
			ContainerPath:  "/etc/hosts",
			HostPath:       "/mnt/disk",
			ReadOnly:       false,
			SELinuxRelabel: false,
		},
		{
			Name:           "disk",
			ContainerPath:  "/mnt/path3",
			HostPath:       "/mnt/disk",
			ReadOnly:       true,
			SELinuxRelabel: false,
		},
		{
			Name:           "disk4",
			ContainerPath:  "/mnt/path4",
			HostPath:       "/mnt/host",
			ReadOnly:       false,
			SELinuxRelabel: false,
		},
		{
			Name:           "disk5",
			ContainerPath:  "/mnt/path5",
			HostPath:       "/var/lib/kubelet/podID/volumes/empty/disk5",
			ReadOnly:       false,
			SELinuxRelabel: false,
		},
	}
	assert.Equal(t, expectedMounts, mounts, "mounts of container %+v", container)
}

type fakeContainerCommandRunner struct {
	// what was passed in
	Cmd    []string
	ID     kubecontainer.ContainerID
	PodID  types.UID
	E      error
	Stdin  io.Reader
	Stdout io.WriteCloser
	Stderr io.WriteCloser
	TTY    bool
	Port   uint16
	Stream io.ReadWriteCloser

	// what to return
	StdoutData string
	StderrData string
}

func (f *fakeContainerCommandRunner) ExecInContainer(id kubecontainer.ContainerID, cmd []string, in io.Reader, out, err io.WriteCloser, tty bool, resize <-chan term.Size) error {
	// record params
	f.Cmd = cmd
	f.ID = id
	f.Stdin = in
	f.Stdout = out
	f.Stderr = err
	f.TTY = tty

	// Copy stdout/stderr data
	fmt.Fprint(out, f.StdoutData)
	fmt.Fprint(out, f.StderrData)

	return f.E
}

func (f *fakeContainerCommandRunner) PortForward(pod *kubecontainer.Pod, port uint16, stream io.ReadWriteCloser) error {
	f.PodID = pod.ID
	f.Port = port
	f.Stream = stream
	return nil
}

func TestRunInContainerNoSuchPod(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
	fakeRuntime.PodList = []*containertest.FakePod{}

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerName := "containerFoo"
	output, err := kubelet.RunInContainer(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{Name: podName, Namespace: podNamespace}}),
		"",
		containerName,
		[]string{"ls"})
	assert.Error(t, err)
	assert.Nil(t, output, "output should be nil")
}

func TestRunInContainer(t *testing.T) {
	for _, testError := range []error{nil, errors.New("foo")} {
		testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
		kubelet := testKubelet.kubelet
		fakeRuntime := testKubelet.fakeRuntime
		fakeCommandRunner := fakeContainerCommandRunner{
			E:          testError,
			StdoutData: "foo",
			StderrData: "bar",
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
		assert.Equal(t, containerID, fakeCommandRunner.ID, "(testError=%v) ID", testError)
		assert.Equal(t, cmd, fakeCommandRunner.Cmd, "(testError=%v) command", testError)
		// this isn't 100% foolproof as a bug in a real ContainerCommandRunner where it fails to copy to stdout/stderr wouldn't be caught by this test
		assert.Equal(t, "foobar", string(actualOutput), "(testError=%v) output", testError)
		assert.Equal(t, fmt.Sprintf("%s", err), fmt.Sprintf("%s", testError), "(testError=%v) err", testError)
	}
}

func TestGenerateRunContainerOptions_DNSConfigurationParams(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet

	clusterNS := "203.0.113.1"
	kubelet.clusterDomain = "kubernetes.io"
	kubelet.clusterDNS = net.ParseIP(clusterNS)

	pods := newTestPods(2)
	pods[0].Spec.DNSPolicy = api.DNSClusterFirst
	pods[1].Spec.DNSPolicy = api.DNSDefault

	options := make([]*kubecontainer.RunContainerOptions, 2)
	for i, pod := range pods {
		var err error
		options[i], err = kubelet.GenerateRunContainerOptions(pod, &api.Container{}, "")
		if err != nil {
			t.Fatalf("failed to generate container options: %v", err)
		}
	}
	if len(options[0].DNS) != 1 || options[0].DNS[0] != clusterNS {
		t.Errorf("expected nameserver %s, got %+v", clusterNS, options[0].DNS)
	}
	if len(options[0].DNSSearch) == 0 || options[0].DNSSearch[0] != ".svc."+kubelet.clusterDomain {
		t.Errorf("expected search %s, got %+v", ".svc."+kubelet.clusterDomain, options[0].DNSSearch)
	}
	if len(options[1].DNS) != 1 || options[1].DNS[0] != "127.0.0.1" {
		t.Errorf("expected nameserver 127.0.0.1, got %+v", options[1].DNS)
	}
	if len(options[1].DNSSearch) != 1 || options[1].DNSSearch[0] != "." {
		t.Errorf("expected search \".\", got %+v", options[1].DNSSearch)
	}

	kubelet.resolverConfig = "/etc/resolv.conf"
	for i, pod := range pods {
		var err error
		options[i], err = kubelet.GenerateRunContainerOptions(pod, &api.Container{}, "")
		if err != nil {
			t.Fatalf("failed to generate container options: %v", err)
		}
	}
	t.Logf("nameservers %+v", options[1].DNS)
	if len(options[0].DNS) != 1 {
		t.Errorf("expected cluster nameserver only, got %+v", options[0].DNS)
	} else if options[0].DNS[0] != clusterNS {
		t.Errorf("expected nameserver %s, got %v", clusterNS, options[0].DNS[0])
	}
	if len(options[0].DNSSearch) != len(options[1].DNSSearch)+3 {
		t.Errorf("expected prepend of cluster domain, got %+v", options[0].DNSSearch)
	} else if options[0].DNSSearch[0] != ".svc."+kubelet.clusterDomain {
		t.Errorf("expected domain %s, got %s", ".svc."+kubelet.clusterDomain, options[0].DNSSearch)
	}
}

type testServiceLister struct {
	services []*api.Service
}

func (ls testServiceLister) List(labels.Selector) ([]*api.Service, error) {
	return ls.services, nil
}

type envs []kubecontainer.EnvVar

func (e envs) Len() int {
	return len(e)
}

func (e envs) Swap(i, j int) { e[i], e[j] = e[j], e[i] }

func (e envs) Less(i, j int) bool { return e[i].Name < e[j].Name }

func buildService(name, namespace, clusterIP, protocol string, port int) *api.Service {
	return &api.Service{
		ObjectMeta: api.ObjectMeta{Name: name, Namespace: namespace},
		Spec: api.ServiceSpec{
			Ports: []api.ServicePort{{
				Protocol: api.Protocol(protocol),
				Port:     int32(port),
			}},
			ClusterIP: clusterIP,
		},
	}
}

func TestMakeEnvironmentVariables(t *testing.T) {
	services := []*api.Service{
		buildService("kubernetes", api.NamespaceDefault, "1.2.3.1", "TCP", 8081),
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

	testCases := []struct {
		name            string                 // the name of the test case
		ns              string                 // the namespace to generate environment for
		container       *api.Container         // the container to use
		masterServiceNs string                 // the namespace to read master service info from
		nilLister       bool                   // whether the lister should be nil
		expectedEnvs    []kubecontainer.EnvVar // a set of expected environment vars
	}{
		{
			name: "api server = Y, kubelet = Y",
			ns:   "test1",
			container: &api.Container{
				Env: []api.EnvVar{
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
			masterServiceNs: api.NamespaceDefault,
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
			name: "api server = Y, kubelet = N",
			ns:   "test1",
			container: &api.Container{
				Env: []api.EnvVar{
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
			masterServiceNs: api.NamespaceDefault,
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
			name: "api server = N; kubelet = Y",
			ns:   "test1",
			container: &api.Container{
				Env: []api.EnvVar{
					{Name: "FOO", Value: "BAZ"},
				},
			},
			masterServiceNs: api.NamespaceDefault,
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
			name: "master service in pod ns",
			ns:   "test2",
			container: &api.Container{
				Env: []api.EnvVar{
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
			name:            "pod in master service ns",
			ns:              "kubernetes",
			container:       &api.Container{},
			masterServiceNs: "kubernetes",
			nilLister:       false,
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
			name: "downward api pod",
			ns:   "downward-api",
			container: &api.Container{
				Env: []api.EnvVar{
					{
						Name: "POD_NAME",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: testapi.Default.GroupVersion().String(),
								FieldPath:  "metadata.name",
							},
						},
					},
					{
						Name: "POD_NAMESPACE",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: testapi.Default.GroupVersion().String(),
								FieldPath:  "metadata.namespace",
							},
						},
					},
					{
						Name: "POD_NODE_NAME",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: testapi.Default.GroupVersion().String(),
								FieldPath:  "spec.nodeName",
							},
						},
					},
					{
						Name: "POD_SERVICE_ACCOUNT_NAME",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: testapi.Default.GroupVersion().String(),
								FieldPath:  "spec.serviceAccountName",
							},
						},
					},
					{
						Name: "POD_IP",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: testapi.Default.GroupVersion().String(),
								FieldPath:  "status.podIP",
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
			},
		},
		{
			name: "env expansion",
			ns:   "test1",
			container: &api.Container{
				Env: []api.EnvVar{
					{
						Name:  "TEST_LITERAL",
						Value: "test-test-test",
					},
					{
						Name: "POD_NAME",
						ValueFrom: &api.EnvVarSource{
							FieldRef: &api.ObjectFieldSelector{
								APIVersion: testapi.Default.GroupVersion().String(),
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
	}

	for _, tc := range testCases {
		testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
		kl := testKubelet.kubelet
		kl.masterServiceNamespace = tc.masterServiceNs
		if tc.nilLister {
			kl.serviceLister = nil
		} else {
			kl.serviceLister = testServiceLister{services}
		}

		testPod := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Namespace: tc.ns,
				Name:      "dapi-test-pod-name",
			},
			Spec: api.PodSpec{
				ServiceAccountName: "special",
				NodeName:           "node-name",
			},
		}
		podIP := "1.2.3.4"

		result, err := kl.makeEnvironmentVariables(testPod, tc.container, podIP)
		assert.NoError(t, err, "[%s]", tc.name)

		sort.Sort(envs(result))
		sort.Sort(envs(tc.expectedEnvs))
		assert.Equal(t, tc.expectedEnvs, result, "[%s] env entries", tc.name)
	}
}

func waitingState(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Waiting: &api.ContainerStateWaiting{},
		},
	}
}
func waitingStateWithLastTermination(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Waiting: &api.ContainerStateWaiting{},
		},
		LastTerminationState: api.ContainerState{
			Terminated: &api.ContainerStateTerminated{
				ExitCode: 0,
			},
		},
	}
}
func runningState(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Running: &api.ContainerStateRunning{},
		},
	}
}
func stoppedState(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Terminated: &api.ContainerStateTerminated{},
		},
	}
}
func succeededState(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Terminated: &api.ContainerStateTerminated{
				ExitCode: 0,
			},
		},
	}
}
func failedState(cName string) api.ContainerStatus {
	return api.ContainerStatus{
		Name: cName,
		State: api.ContainerState{
			Terminated: &api.ContainerStateTerminated{
				ExitCode: -1,
			},
		},
	}
}

func TestPodPhaseWithRestartAlways(t *testing.T) {
	desiredState := api.PodSpec{
		NodeName: "machine",
		Containers: []api.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: api.RestartPolicyAlways,
	}

	tests := []struct {
		pod    *api.Pod
		status api.PodPhase
		test   string
	}{
		{&api.Pod{Spec: desiredState, Status: api.PodStatus{}}, api.PodPending, "waiting"},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			api.PodRunning,
			"all running",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						stoppedState("containerA"),
						stoppedState("containerB"),
					},
				},
			},
			api.PodRunning,
			"all stopped with restart always",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						stoppedState("containerB"),
					},
				},
			},
			api.PodRunning,
			"mixed state #1 with restart always",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
					},
				},
			},
			api.PodPending,
			"mixed state #2 with restart always",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						waitingState("containerB"),
					},
				},
			},
			api.PodPending,
			"mixed state #3 with restart always",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						waitingStateWithLastTermination("containerB"),
					},
				},
			},
			api.PodRunning,
			"backoff crashloop container with restart always",
		},
	}
	for _, test := range tests {
		status := GetPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses)
		assert.Equal(t, test.status, status, "[test %s]", test.test)
	}
}

func TestPodPhaseWithRestartNever(t *testing.T) {
	desiredState := api.PodSpec{
		NodeName: "machine",
		Containers: []api.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: api.RestartPolicyNever,
	}

	tests := []struct {
		pod    *api.Pod
		status api.PodPhase
		test   string
	}{
		{&api.Pod{Spec: desiredState, Status: api.PodStatus{}}, api.PodPending, "waiting"},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			api.PodRunning,
			"all running with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						succeededState("containerA"),
						succeededState("containerB"),
					},
				},
			},
			api.PodSucceeded,
			"all succeeded with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						failedState("containerA"),
						failedState("containerB"),
					},
				},
			},
			api.PodFailed,
			"all failed with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						succeededState("containerB"),
					},
				},
			},
			api.PodRunning,
			"mixed state #1 with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
					},
				},
			},
			api.PodPending,
			"mixed state #2 with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						waitingState("containerB"),
					},
				},
			},
			api.PodPending,
			"mixed state #3 with restart never",
		},
	}
	for _, test := range tests {
		status := GetPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses)
		assert.Equal(t, test.status, status, "[test %s]", test.test)
	}
}

func TestPodPhaseWithRestartOnFailure(t *testing.T) {
	desiredState := api.PodSpec{
		NodeName: "machine",
		Containers: []api.Container{
			{Name: "containerA"},
			{Name: "containerB"},
		},
		RestartPolicy: api.RestartPolicyOnFailure,
	}

	tests := []struct {
		pod    *api.Pod
		status api.PodPhase
		test   string
	}{
		{&api.Pod{Spec: desiredState, Status: api.PodStatus{}}, api.PodPending, "waiting"},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						runningState("containerB"),
					},
				},
			},
			api.PodRunning,
			"all running with restart onfailure",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						succeededState("containerA"),
						succeededState("containerB"),
					},
				},
			},
			api.PodSucceeded,
			"all succeeded with restart onfailure",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						failedState("containerA"),
						failedState("containerB"),
					},
				},
			},
			api.PodRunning,
			"all failed with restart never",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						succeededState("containerB"),
					},
				},
			},
			api.PodRunning,
			"mixed state #1 with restart onfailure",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
					},
				},
			},
			api.PodPending,
			"mixed state #2 with restart onfailure",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						waitingState("containerB"),
					},
				},
			},
			api.PodPending,
			"mixed state #3 with restart onfailure",
		},
		{
			&api.Pod{
				Spec: desiredState,
				Status: api.PodStatus{
					ContainerStatuses: []api.ContainerStatus{
						runningState("containerA"),
						waitingStateWithLastTermination("containerB"),
					},
				},
			},
			api.PodRunning,
			"backoff crashloop container with restart onfailure",
		},
	}
	for _, test := range tests {
		status := GetPhase(&test.pod.Spec, test.pod.Status.ContainerStatuses)
		assert.Equal(t, test.status, status, "[test %s]", test.test)
	}
}

func TestExecInContainerNoSuchPod(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet.runner = &fakeCommandRunner
	fakeRuntime.PodList = []*containertest.FakePod{}

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerID := "containerFoo"
	err := kubelet.ExecInContainer(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{Name: podName, Namespace: podNamespace}}),
		"",
		containerID,
		[]string{"ls"},
		nil,
		nil,
		nil,
		false,
		nil,
	)
	require.Error(t, err)
	require.True(t, fakeCommandRunner.ID.IsEmpty(), "Unexpected invocation of runner.ExecInContainer")
}

func TestExecInContainerNoSuchContainer(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerID := "containerFoo"
	fakeRuntime.PodList = []*containertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID:        "12345678",
			Name:      podName,
			Namespace: podNamespace,
			Containers: []*kubecontainer.Container{
				{Name: "bar",
					ID: kubecontainer.ContainerID{Type: "test", ID: "barID"}},
			},
		}},
	}

	err := kubelet.ExecInContainer(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      podName,
			Namespace: podNamespace,
		}}),
		"",
		containerID,
		[]string{"ls"},
		nil,
		nil,
		nil,
		false,
		nil,
	)
	require.Error(t, err)
	require.True(t, fakeCommandRunner.ID.IsEmpty(), "Unexpected invocation of runner.ExecInContainer")
}

type fakeReadWriteCloser struct{}

func (f *fakeReadWriteCloser) Write(data []byte) (int, error) {
	return 0, nil
}

func (f *fakeReadWriteCloser) Read(data []byte) (int, error) {
	return 0, nil
}

func (f *fakeReadWriteCloser) Close() error {
	return nil
}

func TestExecInContainer(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "nsFoo"
	containerID := "containerFoo"
	command := []string{"ls"}
	stdin := &bytes.Buffer{}
	stdout := &fakeReadWriteCloser{}
	stderr := &fakeReadWriteCloser{}
	tty := true
	fakeRuntime.PodList = []*containertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID:        "12345678",
			Name:      podName,
			Namespace: podNamespace,
			Containers: []*kubecontainer.Container{
				{Name: containerID,
					ID: kubecontainer.ContainerID{Type: "test", ID: containerID},
				},
			},
		}},
	}

	err := kubelet.ExecInContainer(
		kubecontainer.GetPodFullName(podWithUidNameNs("12345678", podName, podNamespace)),
		"",
		containerID,
		[]string{"ls"},
		stdin,
		stdout,
		stderr,
		tty,
		nil,
	)
	require.NoError(t, err)
	require.Equal(t, fakeCommandRunner.ID.ID, containerID, "ID")
	require.Equal(t, fakeCommandRunner.Cmd, command, "Command")
	require.Equal(t, fakeCommandRunner.Stdin, stdin, "Stdin")
	require.Equal(t, fakeCommandRunner.Stdout, stdout, "Stdout")
	require.Equal(t, fakeCommandRunner.Stderr, stderr, "Stderr")
	require.Equal(t, fakeCommandRunner.TTY, tty, "TTY")
}

func TestPortForwardNoSuchPod(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime
	fakeRuntime.PodList = []*containertest.FakePod{}
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet.runner = &fakeCommandRunner

	podName := "podFoo"
	podNamespace := "nsFoo"
	var port uint16 = 5000

	err := kubelet.PortForward(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{Name: podName, Namespace: podNamespace}}),
		"",
		port,
		nil,
	)
	require.Error(t, err)
	require.True(t, fakeCommandRunner.ID.IsEmpty(), "unexpected invocation of runner.PortForward")
}

func TestPortForward(t *testing.T) {
	testKubelet := newTestKubelet(t, false /* controllerAttachDetachEnabled */)
	kubelet := testKubelet.kubelet
	fakeRuntime := testKubelet.fakeRuntime

	podName := "podFoo"
	podNamespace := "nsFoo"
	podID := types.UID("12345678")
	fakeRuntime.PodList = []*containertest.FakePod{
		{Pod: &kubecontainer.Pod{
			ID:        podID,
			Name:      podName,
			Namespace: podNamespace,
			Containers: []*kubecontainer.Container{
				{
					Name: "foo",
					ID:   kubecontainer.ContainerID{Type: "test", ID: "containerFoo"},
				},
			},
		}},
	}
	fakeCommandRunner := fakeContainerCommandRunner{}
	kubelet.runner = &fakeCommandRunner

	var port uint16 = 5000
	stream := &fakeReadWriteCloser{}
	err := kubelet.PortForward(
		kubecontainer.GetPodFullName(&api.Pod{ObjectMeta: api.ObjectMeta{
			UID:       "12345678",
			Name:      podName,
			Namespace: podNamespace,
		}}),
		"",
		port,
		stream,
	)
	require.NoError(t, err)
	require.Equal(t, fakeCommandRunner.PodID, podID, "Pod ID")
	require.Equal(t, fakeCommandRunner.Port, port, "Port")
	require.Equal(t, fakeCommandRunner.Stream, stream, "stream")
}

// Tests that identify the host port conflicts are detected correctly.
func TestGetHostPortConflicts(t *testing.T) {
	pods := []*api.Pod{
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 80}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 81}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 82}}}}}},
		{Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 83}}}}}},
	}
	// Pods should not cause any conflict.
	assert.False(t, hasHostPortConflicts(pods), "Should not have port conflicts")

	expected := &api.Pod{
		Spec: api.PodSpec{Containers: []api.Container{{Ports: []api.ContainerPort{{HostPort: 81}}}}},
	}
	// The new pod should cause conflict and be reported.
	pods = append(pods, expected)
	assert.True(t, hasHostPortConflicts(pods), "Should have port conflicts")
}
