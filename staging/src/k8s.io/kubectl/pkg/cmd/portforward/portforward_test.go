/*
Copyright 2014 The Kubernetes Authors.

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

package portforward

import (
	"context"
	"fmt"
	"net/http"
	"net/url"
	"reflect"
	"testing"

	"github.com/spf13/cobra"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/client-go/rest/fake"
	"k8s.io/client-go/tools/portforward"
	cmdtesting "k8s.io/kubectl/pkg/cmd/testing"
	cmdfeaturegate "k8s.io/kubectl/pkg/cmd/util/featuregate"
	"k8s.io/kubectl/pkg/scheme"
)

type fakePortForwarder struct {
	method string
	url    *url.URL
	pfErr  error
}

func (f *fakePortForwarder) ForwardPorts(method string, url *url.URL, opts PortForwardOptions) error {
	f.method = method
	f.url = url
	return f.pfErr
}

func testPortForward(t *testing.T, flags map[string]string, args []string) {
	version := "v1"

	tests := []struct {
		name            string
		podPath, pfPath string
		pod             *corev1.Pod
		pfErr           bool
	}{
		{
			name:    "pod portforward",
			podPath: "/api/" + version + "/namespaces/test/pods/foo",
			pfPath:  "/api/" + version + "/namespaces/test/pods/foo/portforward",
			pod:     execPod(),
		},
		{
			name:    "pod portforward error",
			podPath: "/api/" + version + "/namespaces/test/pods/foo",
			pfPath:  "/api/" + version + "/namespaces/test/pods/foo/portforward",
			pod:     execPod(),
			pfErr:   true,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			var err error
			tf := cmdtesting.NewTestFactory().WithNamespace("test")
			defer tf.Cleanup()

			codec := scheme.Codecs.LegacyCodec(scheme.Scheme.PrioritizedVersionsAllGroups()...)
			ns := scheme.Codecs.WithoutConversion()

			tf.Client = &fake.RESTClient{
				VersionedAPIPath:     "/api/v1",
				GroupVersion:         schema.GroupVersion{Group: "", Version: "v1"},
				NegotiatedSerializer: ns,
				Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {
					switch p, m := req.URL.Path, req.Method; {
					case p == test.podPath && m == "GET":
						body := cmdtesting.ObjBody(codec, test.pod)
						return &http.Response{StatusCode: http.StatusOK, Header: cmdtesting.DefaultHeader(), Body: body}, nil
					default:
						t.Errorf("%s: unexpected request: %#v\n%#v", test.name, req.URL, req)
						return nil, nil
					}
				}),
			}
			tf.ClientConfigVal = cmdtesting.DefaultClientConfig()
			ff := &fakePortForwarder{}
			if test.pfErr {
				ff.pfErr = fmt.Errorf("pf error")
			}

			opts := &PortForwardOptions{}
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			cmd := NewCmdPortForward(tf, genericiooptions.NewTestIOStreamsDiscard())
			cmd.Run = func(cmd *cobra.Command, args []string) {
				if err = opts.Complete(tf, cmd, args); err != nil {
					return
				}
				opts.PortForwarder = ff
				if err = opts.Validate(); err != nil {
					return
				}
				err = opts.RunPortForwardContext(ctx)
			}

			for name, value := range flags {
				cmd.Flags().Set(name, value)
			}
			cmd.Run(cmd, args)

			if test.pfErr && err != ff.pfErr {
				t.Errorf("%s: Unexpected port-forward error: %v", test.name, err)
			}
			if !test.pfErr && err != nil {
				t.Errorf("%s: Unexpected error: %v", test.name, err)
			}
			if test.pfErr {
				return
			}

			if ff.url == nil || ff.url.Path != test.pfPath {
				t.Errorf("%s: Did not get expected path for portforward request", test.name)
			}
			if ff.method != "POST" {
				t.Errorf("%s: Did not get method for attach request: %s", test.name, ff.method)
			}
		})
	}
}

func TestPortForward(t *testing.T) {
	testPortForward(t, nil, []string{"foo", ":5000", ":1000"})
}

func TestTranslateServicePortToTargetPort(t *testing.T) {
	cases := []struct {
		name       string
		svc        corev1.Service
		pod        corev1.Pod
		ports      []string
		translated []string
		err        bool
	}{
		{
			name: "test success 1 (int port)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromInt32(8080),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(8080)},
							},
						},
					},
				},
			},
			ports:      []string{"80"},
			translated: []string{"80:8080"},
			err:        false,
		},
		{
			name: "test success 1 (int port with random local port)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromInt32(8080),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(8080)},
							},
						},
					},
				},
			},
			ports:      []string{":80"},
			translated: []string{":8080"},
			err:        false,
		},
		{
			name: "test success 1 (int port with explicit local port)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       8080,
							TargetPort: intstr.FromInt32(8080),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(8080)},
							},
						},
					},
				},
			},
			ports:      []string{"8000:8080"},
			translated: []string{"8000:8080"},
			err:        false,
		},
		{
			name: "test success 2 (clusterIP: None)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					ClusterIP: "None",
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromInt32(8080),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(8080)},
							},
						},
					},
				},
			},
			ports:      []string{"80"},
			translated: []string{"80"},
			err:        false,
		},
		{
			name: "test success 2 (clusterIP: None with random local port)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					ClusterIP: "None",
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromInt32(8080),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(8080)},
							},
						},
					},
				},
			},
			ports:      []string{":80"},
			translated: []string{":80"},
			err:        false,
		},
		{
			name: "test success 3 (named target port)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
						{
							Port:       443,
							TargetPort: intstr.FromString("https"),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(8080)},
								{
									Name:          "https",
									ContainerPort: int32(8443)},
							},
						},
					},
				},
			},
			ports:      []string{"80", "443"},
			translated: []string{"80:8080", "443:8443"},
			err:        false,
		},
		{
			name: "test success 3 (named target port with random local port)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
						{
							Port:       443,
							TargetPort: intstr.FromString("https"),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(8080)},
								{
									Name:          "https",
									ContainerPort: int32(8443)},
							},
						},
					},
				},
			},
			ports:      []string{":80", ":443"},
			translated: []string{":8080", ":8443"},
			err:        false,
		},
		{
			name: "test success 4 (named service port)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Name:       "http",
							TargetPort: intstr.FromInt32(8080),
						},
						{
							Port:       443,
							Name:       "https",
							TargetPort: intstr.FromInt32(8443),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									ContainerPort: int32(8080)},
								{
									ContainerPort: int32(8443)},
							},
						},
					},
				},
			},
			ports:      []string{"http", "https"},
			translated: []string{"80:8080", "443:8443"},
			err:        false,
		},
		{
			name: "test success 4 (named service port with random local port)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Name:       "http",
							TargetPort: intstr.FromInt32(8080),
						},
						{
							Port:       443,
							Name:       "https",
							TargetPort: intstr.FromInt32(8443),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									ContainerPort: int32(8080)},
								{
									ContainerPort: int32(8443)},
							},
						},
					},
				},
			},
			ports:      []string{":http", ":https"},
			translated: []string{":8080", ":8443"},
			err:        false,
		},
		{
			name: "test success 4 (named service port and named pod container port)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							Name:       "http",
							TargetPort: intstr.FromString("http"),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
				},
			},
			ports:      []string{"http"},
			translated: []string{"80"},
			err:        false,
		},
		{
			name: "test success (targetPort omitted)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port: 80,
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
				},
			},
			ports:      []string{"80"},
			translated: []string{"80"},
			err:        false,
		},
		{
			name: "test success (targetPort omitted with random local port)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port: 80,
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
				},
			},
			ports:      []string{":80"},
			translated: []string{":80"},
			err:        false,
		},
		{
			name: "test failure 1 (named target port lookup failure)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "https",
									ContainerPort: int32(443)},
							},
						},
					},
				},
			},
			ports:      []string{"80"},
			translated: []string{},
			err:        true,
		},
		{
			name: "test failure 1 (named service port lookup failure)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(8080)},
							},
						},
					},
				},
			},
			ports:      []string{"https"},
			translated: []string{},
			err:        true,
		},
		{
			name: "test failure 2 (service port not declared)",
			svc: corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{
							Port:       80,
							TargetPort: intstr.FromString("http"),
						},
					},
				},
			},
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "https",
									ContainerPort: int32(443)},
							},
						},
					},
				},
			},
			ports:      []string{"443"},
			translated: []string{},
			err:        true,
		},
	}

	for _, tc := range cases {
		translated, err := translateServicePortToTargetPort(tc.ports, tc.svc, tc.pod)
		if err != nil {
			if tc.err {
				continue
			}

			t.Errorf("%v: unexpected error: %v", tc.name, err)
			continue
		}

		if tc.err {
			t.Errorf("%v: unexpected success", tc.name)
			continue
		}

		if !reflect.DeepEqual(translated, tc.translated) {
			t.Errorf("%v: expected %v; got %v", tc.name, tc.translated, translated)
		}
	}
}

func execPod() *corev1.Pod {
	return &corev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "10"},
		Spec: corev1.PodSpec{
			RestartPolicy: corev1.RestartPolicyAlways,
			DNSPolicy:     corev1.DNSClusterFirst,
			Containers: []corev1.Container{
				{
					Name: "bar",
				},
			},
		},
		Status: corev1.PodStatus{
			Phase: corev1.PodRunning,
		},
	}
}

func TestConvertPodNamedPortToNumber(t *testing.T) {
	cases := []struct {
		name      string
		pod       corev1.Pod
		ports     []string
		converted []string
		err       bool
	}{
		{
			name: "port number without local port",
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
				},
			},
			ports:     []string{"80"},
			converted: []string{"80"},
			err:       false,
		},
		{
			name: "port number with local port",
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
				},
			},
			ports:     []string{"8000:80"},
			converted: []string{"8000:80"},
			err:       false,
		},
		{
			name: "port number with random local port",
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
				},
			},
			ports:     []string{":80"},
			converted: []string{":80"},
			err:       false,
		},
		{
			name: "named port without local port",
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
				},
			},
			ports:     []string{"http"},
			converted: []string{"80"},
			err:       false,
		},
		{
			name: "named port with local port",
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
				},
			},
			ports:     []string{"8000:http"},
			converted: []string{"8000:80"},
			err:       false,
		},
		{
			name: "named port with random local port",
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "http",
									ContainerPort: int32(80)},
							},
						},
					},
				},
			},
			ports:     []string{":http"},
			converted: []string{":80"},
			err:       false,
		},
		{
			name: "named port can not be found",
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "https",
									ContainerPort: int32(443)},
							},
						},
					},
				},
			},
			ports: []string{"http"},
			err:   true,
		},
		{
			name: "one of the requested named ports can not be found",
			pod: corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{
									Name:          "https",
									ContainerPort: int32(443)},
							},
						},
					},
				},
			},
			ports: []string{"https", "http"},
			err:   true,
		},
	}

	for _, tc := range cases {
		converted, err := convertPodNamedPortToNumber(tc.ports, tc.pod)
		if err != nil {
			if tc.err {
				continue
			}

			t.Errorf("%v: unexpected error: %v", tc.name, err)
			continue
		}

		if tc.err {
			t.Errorf("%v: unexpected success", tc.name)
			continue
		}

		if !reflect.DeepEqual(converted, tc.converted) {
			t.Errorf("%v: expected %v; got %v", tc.name, tc.converted, converted)
		}
	}
}

func TestCheckUDPPort(t *testing.T) {
	tests := []struct {
		name        string
		pod         *corev1.Pod
		service     *corev1.Service
		ports       []string
		expectError bool
	}{
		{
			name: "forward to a UDP port in a Pod",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{Protocol: corev1.ProtocolUDP, ContainerPort: 53},
							},
						},
					},
				},
			},
			ports:       []string{"53"},
			expectError: true,
		},
		{
			name: "forward to a named UDP port in a Pod",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{Protocol: corev1.ProtocolUDP, ContainerPort: 53, Name: "dns"},
							},
						},
					},
				},
			},
			ports:       []string{"dns"},
			expectError: true,
		},
		{
			name: "Pod has ports with both TCP and UDP protocol (UDP first)",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{Protocol: corev1.ProtocolUDP, ContainerPort: 53},
								{Protocol: corev1.ProtocolTCP, ContainerPort: 53},
							},
						},
					},
				},
			},
			ports: []string{":53"},
		},
		{
			name: "Pod has ports with both TCP and UDP protocol (TCP first)",
			pod: &corev1.Pod{
				Spec: corev1.PodSpec{
					Containers: []corev1.Container{
						{
							Ports: []corev1.ContainerPort{
								{Protocol: corev1.ProtocolTCP, ContainerPort: 53},
								{Protocol: corev1.ProtocolUDP, ContainerPort: 53},
							},
						},
					},
				},
			},
			ports: []string{":53"},
		},

		{
			name: "forward to a UDP port in a Service",
			service: &corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Protocol: corev1.ProtocolUDP, Port: 53},
					},
				},
			},
			ports:       []string{"53"},
			expectError: true,
		},
		{
			name: "forward to a named UDP port in a Service",
			service: &corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Protocol: corev1.ProtocolUDP, Port: 53, Name: "dns"},
					},
				},
			},
			ports:       []string{"10053:dns"},
			expectError: true,
		},
		{
			name: "Service has ports with both TCP and UDP protocol (UDP first)",
			service: &corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Protocol: corev1.ProtocolUDP, Port: 53},
						{Protocol: corev1.ProtocolTCP, Port: 53},
					},
				},
			},
			ports: []string{"53"},
		},
		{
			name: "Service has ports with both TCP and UDP protocol (TCP first)",
			service: &corev1.Service{
				Spec: corev1.ServiceSpec{
					Ports: []corev1.ServicePort{
						{Protocol: corev1.ProtocolTCP, Port: 53},
						{Protocol: corev1.ProtocolUDP, Port: 53},
					},
				},
			},
			ports: []string{"53"},
		},
	}
	for _, tc := range tests {
		var err error
		if tc.pod != nil {
			err = checkUDPPortInPod(tc.ports, tc.pod)
		} else if tc.service != nil {
			err = checkUDPPortInService(tc.ports, tc.service)
		}
		if err != nil {
			if tc.expectError {
				continue
			}
			t.Errorf("%v: unexpected error: %v", tc.name, err)
			continue
		}
		if tc.expectError {
			t.Errorf("%v: unexpected success", tc.name)
		}
	}
}

func TestCreateDialer(t *testing.T) {
	url, err := url.Parse("http://localhost:8080/index.html")
	if err != nil {
		t.Fatalf("unable to parse test url: %v", err)
	}
	config := cmdtesting.DefaultClientConfig()
	opts := PortForwardOptions{Config: config}
	// First, ensure that no environment variable creates the fallback dialer.
	dialer, err := createDialer("GET", url, opts)
	if err != nil {
		t.Fatalf("unable to create dialer: %v", err)
	}
	if _, isFallback := dialer.(*portforward.FallbackDialer); !isFallback {
		t.Errorf("expected fallback dialer, got %#v", dialer)
	}
	// Next, check turning on feature flag explicitly also creates fallback dialer.
	t.Setenv(string(cmdfeaturegate.PortForwardWebsockets), "true")
	dialer, err = createDialer("GET", url, opts)
	if err != nil {
		t.Fatalf("unable to create dialer: %v", err)
	}
	if _, isFallback := dialer.(*portforward.FallbackDialer); !isFallback {
		t.Errorf("expected fallback dialer, got %#v", dialer)
	}
	// Finally, check explicit disabling does NOT create the fallback dialer.
	t.Setenv(string(cmdfeaturegate.PortForwardWebsockets), "false")
	dialer, err = createDialer("GET", url, opts)
	if err != nil {
		t.Fatalf("unable to create dialer: %v", err)
	}
	if _, isFallback := dialer.(*portforward.FallbackDialer); isFallback {
		t.Errorf("expected fallback dialer, got %#v", dialer)
	}
}
