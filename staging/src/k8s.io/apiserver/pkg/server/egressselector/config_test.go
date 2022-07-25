/*
Copyright 2019 The Kubernetes Authors.

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

package egressselector

import (
	"fmt"
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/apis/apiserver"
)

func strptr(s string) *string {
	return &s
}

func TestReadEgressSelectorConfiguration(t *testing.T) {
	testcases := []struct {
		name           string
		contents       string
		createFile     bool
		expectedResult *apiserver.EgressSelectorConfiguration
		expectedError  *string
	}{
		{
			name:           "empty",
			createFile:     true,
			contents:       ``,
			expectedResult: nil,
			expectedError:  strptr("invalid service configuration object \"\""),
		},
		{
			name:           "absent",
			createFile:     false,
			contents:       ``,
			expectedResult: nil,
			expectedError:  strptr("unable to read egress selector configuration from \"test-egress-selector-config-absent\" [open test-egress-selector-config-absent: no such file or directory]"),
		},
		{
			name:       "v1beta1",
			createFile: true,
			contents: `
apiVersion: apiserver.k8s.io/v1beta1
kind: EgressSelectorConfiguration
egressSelections:
- name: "cluster"
  connection:
    proxyProtocol: "HTTPConnect"
    transport:
      tcp:
        url: "https://127.0.0.1:8131"
        tlsConfig:
          caBundle: "/etc/srv/kubernetes/pki/konnectivity-server/ca.crt"
          clientKey: "/etc/srv/kubernetes/pki/konnectivity-server/client.key"
          clientCert: "/etc/srv/kubernetes/pki/konnectivity-server/client.crt"
- name: "controlplane"
  connection:
    proxyProtocol: "HTTPConnect"
    transport:
      tcp:
        url: "https://127.0.0.1:8132"
        tlsConfig:
          caBundle: "/etc/srv/kubernetes/pki/konnectivity-server-master/ca.crt"
          clientKey: "/etc/srv/kubernetes/pki/konnectivity-server-master/client.key"
          clientCert: "/etc/srv/kubernetes/pki/konnectivity-server-master/client.crt"
- name: "etcd"
  connection:
    proxyProtocol: "Direct"
`,
			expectedResult: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "cluster",
						Connection: apiserver.Connection{
							ProxyProtocol: "HTTPConnect",
							Transport: &apiserver.Transport{
								TCP: &apiserver.TCPTransport{
									URL: "https://127.0.0.1:8131",

									TLSConfig: &apiserver.TLSConfig{
										CABundle:   "/etc/srv/kubernetes/pki/konnectivity-server/ca.crt",
										ClientKey:  "/etc/srv/kubernetes/pki/konnectivity-server/client.key",
										ClientCert: "/etc/srv/kubernetes/pki/konnectivity-server/client.crt",
									},
								},
							},
						},
					},
					{
						Name: "controlplane",
						Connection: apiserver.Connection{
							ProxyProtocol: "HTTPConnect",
							Transport: &apiserver.Transport{
								TCP: &apiserver.TCPTransport{
									URL: "https://127.0.0.1:8132",
									TLSConfig: &apiserver.TLSConfig{
										CABundle:   "/etc/srv/kubernetes/pki/konnectivity-server-master/ca.crt",
										ClientKey:  "/etc/srv/kubernetes/pki/konnectivity-server-master/client.key",
										ClientCert: "/etc/srv/kubernetes/pki/konnectivity-server-master/client.crt",
									},
								},
							},
						},
					},
					{
						Name: "etcd",
						Connection: apiserver.Connection{
							ProxyProtocol: "Direct",
						},
					},
				},
			},
			expectedError: nil,
		},
		{
			name:       "v1beta1 using deprecated 'master' type",
			createFile: true,
			contents: `
apiVersion: apiserver.k8s.io/v1beta1
kind: EgressSelectorConfiguration
egressSelections:
- name: "cluster"
  connection:
    proxyProtocol: "HTTPConnect"
    transport:
      tcp:
        url: "https://127.0.0.1:8131"
        tlsConfig:
          caBundle: "/etc/srv/kubernetes/pki/konnectivity-server/ca.crt"
          clientKey: "/etc/srv/kubernetes/pki/konnectivity-server/client.key"
          clientCert: "/etc/srv/kubernetes/pki/konnectivity-server/client.crt"
- name: "master"
  connection:
    proxyProtocol: "HTTPConnect"
    transport:
      tcp:
        url: "https://127.0.0.1:8132"
        tlsConfig:
          caBundle: "/etc/srv/kubernetes/pki/konnectivity-server-master/ca.crt"
          clientKey: "/etc/srv/kubernetes/pki/konnectivity-server-master/client.key"
          clientCert: "/etc/srv/kubernetes/pki/konnectivity-server-master/client.crt"
- name: "etcd"
  connection:
    proxyProtocol: "Direct"
`,
			expectedResult: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "cluster",
						Connection: apiserver.Connection{
							ProxyProtocol: "HTTPConnect",
							Transport: &apiserver.Transport{
								TCP: &apiserver.TCPTransport{
									URL: "https://127.0.0.1:8131",

									TLSConfig: &apiserver.TLSConfig{
										CABundle:   "/etc/srv/kubernetes/pki/konnectivity-server/ca.crt",
										ClientKey:  "/etc/srv/kubernetes/pki/konnectivity-server/client.key",
										ClientCert: "/etc/srv/kubernetes/pki/konnectivity-server/client.crt",
									},
								},
							},
						},
					},
					{
						Name: "controlplane",
						Connection: apiserver.Connection{
							ProxyProtocol: "HTTPConnect",
							Transport: &apiserver.Transport{
								TCP: &apiserver.TCPTransport{
									URL: "https://127.0.0.1:8132",
									TLSConfig: &apiserver.TLSConfig{
										CABundle:   "/etc/srv/kubernetes/pki/konnectivity-server-master/ca.crt",
										ClientKey:  "/etc/srv/kubernetes/pki/konnectivity-server-master/client.key",
										ClientCert: "/etc/srv/kubernetes/pki/konnectivity-server-master/client.crt",
									},
								},
							},
						},
					},
					{
						Name: "etcd",
						Connection: apiserver.Connection{
							ProxyProtocol: "Direct",
						},
					},
				},
			},
			expectedError: nil,
		},
		{
			name:       "wrong_type",
			createFile: true,
			contents: `
apiVersion: apps/v1
kind: DaemonSet
metadata:
  labels:
    addonmanager.kubernetes.io/mode: Reconcile
    k8s-app: konnectivity-agent
  namespace: kube-system
  name: proxy-agent
spec:
  selector:
    matchLabels:
      k8s-app: konnectivity-agent
  updateStrategy:
    type: RollingUpdate
  template:
    metadata:
      labels:
        k8s-app: proxy-agent
    spec:
      priorityClassName: system-cluster-critical
      # Necessary to reboot node
      hostPID: true
      volumes:
        - name: pki
          hostPath:
            path: /etc/srv/kubernetes/pki/konnectivity-agent
      containers:
        - image: registry.k8s.io/proxy-agent:v0.0.3
          name: proxy-agent
          command: ["/proxy-agent"]
          args: ["--caCert=/etc/srv/kubernetes/pki/proxy-agent/ca.crt", "--agentCert=/etc/srv/kubernetes/pki/proxy-agent/client.crt", "--agentKey=/etc/srv/kubernetes/pki/proxy-agent/client.key", "--proxyServerHost=127.0.0.1", "--proxyServerPort=8132"]
          securityContext:
            capabilities:
              add: ["SYS_BOOT"]
          env:
            - name: wrong-type
              valueFrom:
                fieldRef:
                  fieldPath: metadata.name
            - name: kube-system
              valueFrom:
                fieldRef:
                  fieldPath: metadata.namespace
          resources:
            limits:
              cpu: 50m
              memory: 30Mi
          volumeMounts:
            - name: pki
              mountPath: /etc/srv/kubernetes/pki/konnectivity-agent
`,
			expectedResult: nil,
			expectedError:  strptr("invalid service configuration object \"DaemonSet\""),
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			proxyConfig := fmt.Sprintf("test-egress-selector-config-%s", tc.name)
			if tc.createFile {
				f, err := ioutil.TempFile("", proxyConfig)
				if err != nil {
					t.Fatal(err)
				}
				defer os.Remove(f.Name())
				if err := ioutil.WriteFile(f.Name(), []byte(tc.contents), os.FileMode(0755)); err != nil {
					t.Fatal(err)
				}
				proxyConfig = f.Name()
			}
			config, err := ReadEgressSelectorConfiguration(proxyConfig)
			if err == nil && tc.expectedError != nil {
				t.Errorf("calling ReadEgressSelectorConfiguration expected error: %s, did not get it", *tc.expectedError)
			}
			if err != nil && tc.expectedError == nil {
				t.Errorf("unexpected error calling ReadEgressSelectorConfiguration got: %#v", err)
			}
			if err != nil && tc.expectedError != nil && err.Error() != *tc.expectedError {
				t.Errorf("calling ReadEgressSelectorConfiguration expected error: %s, got %#v", *tc.expectedError, err)
			}
			if !reflect.DeepEqual(config, tc.expectedResult) {
				t.Errorf("problem with configuration returned from ReadEgressSelectorConfiguration expected: %#v, got: %#v", tc.expectedResult, config)
			}
		})
	}
}

func TestValidateEgressSelectorConfiguration(t *testing.T) {
	testcases := []struct {
		name        string
		expectError bool
		contents    *apiserver.EgressSelectorConfiguration
	}{
		{
			name:        "direct-valid",
			expectError: false,
			contents: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "controlplane",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolDirect,
						},
					},
				},
			},
		},
		{
			name:        "direct-invalid-transport",
			expectError: true,
			contents: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "controlplane",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolDirect,
							Transport:     &apiserver.Transport{},
						},
					},
				},
			},
		},
		{
			name:        "httpconnect-no-https",
			expectError: false,
			contents: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "cluster",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolHTTPConnect,
							Transport: &apiserver.Transport{
								TCP: &apiserver.TCPTransport{
									URL: "http://127.0.0.1:8131",
								},
							},
						},
					},
				},
			},
		},
		{
			name:        "httpconnect-https-no-cert-error",
			expectError: true,
			contents: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "cluster",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolHTTPConnect,
							Transport: &apiserver.Transport{
								TCP: &apiserver.TCPTransport{
									URL: "https://127.0.0.1:8131",
								},
							},
						},
					},
				},
			},
		},
		{
			name:        "httpconnect-tcp-uds-both-set",
			expectError: true,
			contents: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "cluster",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolHTTPConnect,
							Transport: &apiserver.Transport{
								TCP: &apiserver.TCPTransport{
									URL: "http://127.0.0.1:8131",
								},
								UDS: &apiserver.UDSTransport{
									UDSName: "/etc/srv/kubernetes/konnectivity/konnectivity-server.socket",
								},
							},
						},
					},
				},
			},
		},
		{
			name:        "httpconnect-uds",
			expectError: false,
			contents: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "cluster",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolHTTPConnect,
							Transport: &apiserver.Transport{
								UDS: &apiserver.UDSTransport{
									UDSName: "/etc/srv/kubernetes/konnectivity/konnectivity-server.socket",
								},
							},
						},
					},
				},
			},
		},
		{
			name:        "grpc-https-invalid",
			expectError: true,
			contents: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "cluster",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolGRPC,
							Transport: &apiserver.Transport{
								TCP: &apiserver.TCPTransport{
									URL: "http://127.0.0.1:8131",
									TLSConfig: &apiserver.TLSConfig{
										CABundle:   "",
										ClientKey:  "",
										ClientCert: "",
									},
								},
							},
						},
					},
				},
			},
		},
		{
			name:        "grpc-uds",
			expectError: false,
			contents: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "cluster",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolGRPC,
							Transport: &apiserver.Transport{
								UDS: &apiserver.UDSTransport{
									UDSName: "/etc/srv/kubernetes/konnectivity/konnectivity-server.socket",
								},
							},
						},
					},
				},
			},
		},
		{
			name:        "invalid egress selection name",
			expectError: true,
			contents: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "invalid",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolDirect,
						},
					},
				},
			},
		},
		{
			name:        "duplicate egress selections configured",
			expectError: true,
			contents: &apiserver.EgressSelectorConfiguration{
				TypeMeta: metav1.TypeMeta{
					Kind:       "",
					APIVersion: "",
				},
				EgressSelections: []apiserver.EgressSelection{
					{
						Name: "controlplane",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolDirect,
						},
					},
					{
						Name: "controlplane",
						Connection: apiserver.Connection{
							ProxyProtocol: apiserver.ProtocolDirect,
						},
					},
				},
			},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			errs := ValidateEgressSelectorConfiguration(tc.contents)
			if tc.expectError == false && len(errs) != 0 {
				t.Errorf("Calling ValidateEgressSelectorConfiguration expected no error, got %v", errs)
			} else if tc.expectError == true && len(errs) == 0 {
				t.Errorf("Calling ValidateEgressSelectorConfiguration expected error, got no error")
			}
		})
	}
}
