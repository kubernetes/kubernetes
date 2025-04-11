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

package dns

import (
	"bytes"
	"context"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/lithammer/dedent"

	apps "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/intstr"
	clientset "k8s.io/client-go/kubernetes"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"

	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
	kubeadmutil "k8s.io/kubernetes/cmd/kubeadm/app/util"
)

func TestCompileManifests(t *testing.T) {
	replicas := int32(coreDNSReplicas)
	var tests = []struct {
		name     string
		manifest string
		data     interface{}
	}{
		{
			name:     "CoreDNSDeployment manifest",
			manifest: CoreDNSDeployment,
			data: struct {
				DeploymentName, Image, ControlPlaneTaintKey string
				Replicas                                    *int32
			}{
				DeploymentName:       "foo",
				Image:                "foo",
				ControlPlaneTaintKey: "foo",
				Replicas:             &replicas,
			},
		},
		{
			name:     "CoreDNSConfigMap manifest",
			manifest: CoreDNSConfigMap,
			data: struct{ DNSDomain string }{
				DNSDomain: "foo",
			},
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			_, err := kubeadmutil.ParseTemplate(rt.manifest, rt.data)
			if err != nil {
				t.Errorf("unexpected ParseTemplate failure: %+v", err)
			}
		})
	}
}

func TestGetDNSIP(t *testing.T) {
	var tests = []struct {
		name, svcSubnet, expectedDNSIP string
		isDualStack                    bool
	}{
		{
			name:          "subnet mask 12",
			svcSubnet:     "10.96.0.0/12",
			expectedDNSIP: "10.96.0.10",
		},
		{
			name:          "subnet mask 26",
			svcSubnet:     "10.87.116.64/26",
			expectedDNSIP: "10.87.116.74",
		},
		{
			name:          "dual-stack ipv4 primary, subnet mask 26",
			svcSubnet:     "10.87.116.64/26,fd03::/112",
			expectedDNSIP: "10.87.116.74",
		},
		{
			name:          "dual-stack ipv6 primary, subnet mask 112",
			svcSubnet:     "fd03::/112,10.87.116.64/26",
			expectedDNSIP: "fd03::a",
		},
	}
	for _, rt := range tests {
		t.Run(rt.name, func(t *testing.T) {
			dnsIP, err := kubeadmconstants.GetDNSIP(rt.svcSubnet)
			if err != nil {
				t.Fatalf("couldn't get dnsIP : %v", err)
			}

			actualDNSIP := dnsIP.String()
			if actualDNSIP != rt.expectedDNSIP {
				t.Errorf(
					"failed GetDNSIP\n\texpected: %s\n\t  actual: %s",
					rt.expectedDNSIP,
					actualDNSIP,
				)
			}
		})
	}
}

func TestDeploymentsHaveSystemClusterCriticalPriorityClassName(t *testing.T) {
	replicas := int32(coreDNSReplicas)
	testCases := []struct {
		name     string
		manifest string
		data     interface{}
	}{
		{
			name:     "CoreDNSDeployment",
			manifest: CoreDNSDeployment,
			data: struct {
				DeploymentName, Image, ControlPlaneTaintKey, CoreDNSConfigMapName string
				Replicas                                                          *int32
			}{
				DeploymentName:       "foo",
				Image:                "foo",
				ControlPlaneTaintKey: "foo",
				CoreDNSConfigMapName: "foo",
				Replicas:             &replicas,
			},
		},
	}
	for _, testCase := range testCases {
		t.Run(testCase.name, func(t *testing.T) {
			deploymentBytes, _ := kubeadmutil.ParseTemplate(testCase.manifest, testCase.data)
			deployment := &apps.Deployment{}
			if err := runtime.DecodeInto(clientsetscheme.Codecs.UniversalDecoder(), deploymentBytes, deployment); err != nil {
				t.Errorf("unexpected error: %v", err)
			}
			if deployment.Spec.Template.Spec.PriorityClassName != "system-cluster-critical" {
				t.Errorf("expected to see system-cluster-critical priority class name. Got %q instead", deployment.Spec.Template.Spec.PriorityClassName)
			}
		})
	}
}

func TestCreateCoreDNSAddon(t *testing.T) {
	tests := []struct {
		name                 string
		initialCorefileData  string
		expectedCorefileData string
		coreDNSVersion       string
	}{
		{
			name:                "Empty Corefile",
			initialCorefileData: "",
			expectedCorefileData: `.:53 {
    errors
    health {
       lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure
       fallthrough in-addr.arpa ip6.arpa
       ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
       max_concurrent 1000
    }
    cache 30 {
       disable success cluster.local
       disable denial cluster.local
    }
    loop
    reload
    loadbalance
}
`,
			coreDNSVersion: "1.6.7",
		},
		{
			name: "Default Corefile",
			initialCorefileData: `.:53 {
        errors
        health {
            lameduck 5s
        }
        ready
        kubernetes cluster.local in-addr.arpa ip6.arpa {
            pods insecure
            fallthrough in-addr.arpa ip6.arpa
            ttl 30
        }
        prometheus :9153
        forward . /etc/resolv.conf
        cache 30
        loop
        reload
        loadbalance
    }
`,
			expectedCorefileData: `.:53 {
    errors
    health {
       lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure
       fallthrough in-addr.arpa ip6.arpa
       ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
       max_concurrent 1000
    }
    cache 30 {
       disable success cluster.local
       disable denial cluster.local
    }
    loop
    reload
    loadbalance
}
`,
			coreDNSVersion: "1.6.7",
		},
		{
			name: "Modified Corefile with only newdefaults needed",
			initialCorefileData: `.:53 {
        errors
        log
        health
        ready
        kubernetes cluster.local in-addr.arpa ip6.arpa {
            pods insecure
            fallthrough in-addr.arpa ip6.arpa
            ttl 30
        }
        prometheus :9153
        forward . /etc/resolv.conf
        cache 30
        loop
        reload
        loadbalance
    }
`,
			expectedCorefileData: `.:53 {
    errors
    log
    health {
        lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
        pods insecure
        fallthrough in-addr.arpa ip6.arpa
        ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
        max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
}
`,
			coreDNSVersion: "1.6.2",
		},
		{
			name: "Default Corefile with rearranged plugins",
			initialCorefileData: `.:53 {
        errors
        cache 30
        prometheus :9153
        forward . /etc/resolv.conf
        loop
        reload
        loadbalance
        kubernetes cluster.local in-addr.arpa ip6.arpa {
           pods insecure
           upstream
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        health
    }
`,
			expectedCorefileData: `.:53 {
    errors
    health {
       lameduck 5s
    }
    ready
    kubernetes cluster.local in-addr.arpa ip6.arpa {
       pods insecure
       fallthrough in-addr.arpa ip6.arpa
       ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
       max_concurrent 1000
    }
    cache 30 {
       disable success cluster.local
       disable denial cluster.local
    }
    loop
    reload
    loadbalance
}
`,
			coreDNSVersion: "1.3.1",
		},
		{
			name: "Remove Deprecated options",
			initialCorefileData: `.:53 {
        errors
        logs
        health
        kubernetes cluster.local in-addr.arpa ip6.arpa {
           pods insecure
           upstream
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        prometheus :9153
        forward . /etc/resolv.conf
        cache 30
        loop
        reload
        loadbalance
    }`,
			expectedCorefileData: `.:53 {
    errors
    logs
    health {
        lameduck 5s
    }
    kubernetes cluster.local in-addr.arpa ip6.arpa {
        pods insecure
        fallthrough in-addr.arpa ip6.arpa
        ttl 30
    }
    prometheus :9153
    forward . /etc/resolv.conf {
        max_concurrent 1000
    }
    cache 30
    loop
    reload
    loadbalance
    ready
}
`,
			coreDNSVersion: "1.3.1",
		},
		{
			name: "Update proxy plugin to forward plugin",
			initialCorefileData: `.:53 {
        errors
        health
        kubernetes cluster.local in-addr.arpa ip6.arpa {
           pods insecure
           upstream
           fallthrough in-addr.arpa ip6.arpa
        }
        prometheus :9153
        proxy . /etc/resolv.conf
        k8s_external example.com
        cache 30
        loop
        reload
        loadbalance
    }`,
			expectedCorefileData: `.:53 {
    errors
    health {
        lameduck 5s
    }
    kubernetes cluster.local in-addr.arpa ip6.arpa {
        pods insecure
        fallthrough in-addr.arpa ip6.arpa
    }
    prometheus :9153
    forward . /etc/resolv.conf {
        max_concurrent 1000
    }
    k8s_external example.com
    cache 30
    loop
    reload
    loadbalance
    ready
}
`,
			coreDNSVersion: "1.3.1",
		},
		{
			name: "Modified Corefile with no migration required",
			initialCorefileData: `consul {
        errors
        forward . 10.10.96.16:8600 10.10.96.17:8600 10.10.96.18:8600 {
            max_concurrent 1000
        }
        loadbalance
        cache 5
        reload
    }
    domain.int {
       errors
       forward . 10.10.0.140 10.10.0.240 10.10.51.40 {
           max_concurrent 1000
       }
       loadbalance
       cache 3600
       reload
    }
    .:53 {
      errors
      health {
          lameduck 5s
      }
      ready
      kubernetes cluster.local in-addr.arpa ip6.arpa {
          pods insecure
          fallthrough in-addr.arpa ip6.arpa
      }
      prometheus :9153
      forward . /etc/resolv.conf {
          prefer_udp
          max_concurrent 1000
      }
      cache 30
      loop
      reload
      loadbalance
    }
`,
			expectedCorefileData: `consul {
        errors
        forward . 10.10.96.16:8600 10.10.96.17:8600 10.10.96.18:8600 {
            max_concurrent 1000
        }
        loadbalance
        cache 5
        reload
    }
    domain.int {
       errors
       forward . 10.10.0.140 10.10.0.240 10.10.51.40 {
           max_concurrent 1000
       }
       loadbalance
       cache 3600
       reload
    }
    .:53 {
      errors
      health {
          lameduck 5s
      }
      ready
      kubernetes cluster.local in-addr.arpa ip6.arpa {
          pods insecure
          fallthrough in-addr.arpa ip6.arpa
      }
      prometheus :9153
      forward . /etc/resolv.conf {
          prefer_udp
          max_concurrent 1000
      }
      cache 30
      loop
      reload
      loadbalance
    }
`,
			coreDNSVersion: "1.6.7",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			client := createClientAndCoreDNSManifest(t, tc.initialCorefileData, tc.coreDNSVersion)

			configMapBytes, err := kubeadmutil.ParseTemplate(CoreDNSConfigMap, struct{ DNSDomain, UpstreamNameserver, StubDomain string }{
				DNSDomain:          "cluster.local",
				UpstreamNameserver: "/etc/resolv.conf",
				StubDomain:         "",
			})
			if err != nil {
				t.Errorf("unexpected ParseTemplate failure: %+v", err)
			}

			err = createCoreDNSAddon(nil, nil, configMapBytes, client)
			if err != nil {
				t.Fatalf("error creating the CoreDNS Addon: %v", err)
			}
			migratedConfigMap, _ := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Get(context.TODO(), kubeadmconstants.CoreDNSConfigMap, metav1.GetOptions{})
			if !strings.EqualFold(migratedConfigMap.Data["Corefile"], tc.expectedCorefileData) {
				t.Fatalf("expected to get %v, but got %v", tc.expectedCorefileData, migratedConfigMap.Data["Corefile"])
			}
		})
	}
}

func createClientAndCoreDNSManifest(t *testing.T, corefile, coreDNSVersion string) *clientsetfake.Clientset {
	client := clientsetfake.NewSimpleClientset()
	_, err := client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(context.TODO(), &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.CoreDNSConfigMap,
			Namespace: metav1.NamespaceSystem,
		},
		Data: map[string]string{
			"Corefile": corefile,
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating ConfigMap: %v", err)
	}
	_, err = client.AppsV1().Deployments(metav1.NamespaceSystem).Create(context.TODO(), &apps.Deployment{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Deployment",
			APIVersion: "apps/v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      kubeadmconstants.CoreDNSConfigMap,
			Namespace: metav1.NamespaceSystem,
			Labels: map[string]string{
				"k8s-app": "kube-dns",
			},
		},
		Spec: apps.DeploymentSpec{
			Template: v1.PodTemplateSpec{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Image: "test:" + coreDNSVersion,
						},
					},
				},
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating deployment: %v", err)
	}
	return client
}

func TestDeployedDNSReplicas(t *testing.T) {
	tests := []struct {
		name           string
		deploymentSize int
		want           int32
		wantErr        bool
	}{
		{
			name:           "one coredns addon deployment",
			deploymentSize: 1,
			want:           2,
			wantErr:        false,
		},
		{
			name:           "no coredns addon deployment",
			deploymentSize: 0,
			want:           5,
			wantErr:        false,
		},
		{
			name:           "multiple coredns addon deployments",
			deploymentSize: 3,
			want:           5,
			wantErr:        true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := newMockClientForTest(t, 2, tt.deploymentSize, "", "", "")
			got, err := deployedDNSReplicas(client, 5)
			if (err != nil) != tt.wantErr {
				t.Errorf("deployedDNSReplicas() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if *got != tt.want {
				t.Errorf("deployedDNSReplicas() = %v, want %v", *got, tt.want)
			}
		})
	}
}

func TestCoreDNSAddon(t *testing.T) {
	type args struct {
		cfg           *kubeadmapi.ClusterConfiguration
		client        clientset.Interface
		printManifest bool
	}
	tests := []struct {
		name    string
		args    args
		wantOut string
		wantErr bool
	}{
		{
			name: "cfg is empty",
			args: args{
				cfg:           &kubeadmapi.ClusterConfiguration{},
				client:        newMockClientForTest(t, 2, 1, "", "", ""),
				printManifest: false,
			},
			wantOut: "",
			wantErr: true,
		},
		{
			name: "cfg is valid and not print Manifest",
			args: args{
				cfg: &kubeadmapi.ClusterConfiguration{
					DNS: kubeadmapi.DNS{
						ImageMeta: kubeadmapi.ImageMeta{
							ImageRepository: "foo.bar.io",
						},
					},
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "10.0.0.0/16",
					},
				},
				client:        newMockClientForTest(t, 2, 1, "", "", ""),
				printManifest: false,
			},
			wantOut: "[addons] Applied essential addon: CoreDNS\n",
			wantErr: false,
		},
		{
			name: "cfg is valid and print Manifest",
			args: args{
				cfg: &kubeadmapi.ClusterConfiguration{
					DNS: kubeadmapi.DNS{
						ImageMeta: kubeadmapi.ImageMeta{
							ImageRepository: "foo.bar.io",
						},
					},
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "10.0.0.0/16",
					},
				},
				client:        newMockClientForTest(t, 2, 1, "", "", ""),
				printManifest: true,
			},
			wantOut: dedent.Dedent(fmt.Sprintf(`---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coredns
  namespace: kube-system
  labels:
    k8s-app: kube-dns
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
  selector:
    matchLabels:
      k8s-app: kube-dns
  template:
    metadata:
      labels:
        k8s-app: kube-dns
    spec:
      priorityClassName: system-cluster-critical
      serviceAccountName: coredns
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: k8s-app
                  operator: In
                  values: ["kube-dns"]
              topologyKey: kubernetes.io/hostname
      tolerations:
      - key: CriticalAddonsOnly
        operator: Exists
      - key: node-role.kubernetes.io/control-plane
        effect: NoSchedule
      nodeSelector:
        kubernetes.io/os: linux
      containers:
      - name: coredns
        image: foo.bar.io/coredns:%s
        imagePullPolicy: IfNotPresent
        resources:
          limits:
            memory: 170Mi
          requests:
            cpu: 100m
            memory: 70Mi
        args: [ "-conf", "/etc/coredns/Corefile" ]
        volumeMounts:
        - name: config-volume
          mountPath: /etc/coredns
          readOnly: true
        ports:
        - containerPort: 53
          name: dns
          protocol: UDP
        - containerPort: 53
          name: dns-tcp
          protocol: TCP
        - containerPort: 9153
          name: metrics
          protocol: TCP
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 60
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 5
        readinessProbe:
          httpGet:
            path: /ready
            port: 8181
            scheme: HTTP
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            add:
            - NET_BIND_SERVICE
            drop:
            - ALL
          readOnlyRootFilesystem: true
      dnsPolicy: Default
      volumes:
        - name: config-volume
          configMap:
            name: coredns
            items:
            - key: Corefile
              path: Corefile
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns
  namespace: kube-system
data:
  Corefile: |
    .:53 {
        errors
        health {
           lameduck 5s
        }
        ready
        kubernetes  in-addr.arpa ip6.arpa {
           pods insecure
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        prometheus :9153
        forward . /etc/resolv.conf {
           max_concurrent 1000
        }
        cache 30
        loop
        reload
        loadbalance
    }
---
apiVersion: v1
kind: Service
metadata:
  labels:
    k8s-app: kube-dns
    kubernetes.io/cluster-service: "true"
    kubernetes.io/name: "CoreDNS"
  name: kube-dns
  namespace: kube-system
  annotations:
    prometheus.io/port: "9153"
    prometheus.io/scrape: "true"
  # Without this resourceVersion value, an update of the Service between versions will yield:
  #   Service "kube-dns" is invalid: metadata.resourceVersion: Invalid value: "": must be specified for an update
  resourceVersion: "0"
spec:
  clusterIP: 10.0.0.10
  ports:
  - name: dns
    port: 53
    protocol: UDP
    targetPort: 53
  - name: dns-tcp
    port: 53
    protocol: TCP
    targetPort: 53
  - name: metrics
    port: 9153
    protocol: TCP
    targetPort: 9153
  selector:
    k8s-app: kube-dns
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: system:coredns
rules:
- apiGroups:
  - ""
  resources:
  - endpoints
  - services
  - pods
  - namespaces
  verbs:
  - list
  - watch
- apiGroups:
  - discovery.k8s.io
  resources:
  - endpointslices
  verbs:
  - list
  - watch
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: system:coredns
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: system:coredns
subjects:
- kind: ServiceAccount
  name: coredns
  namespace: kube-system
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: coredns
  namespace: kube-system
`, kubeadmconstants.CoreDNSVersion)),
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := &bytes.Buffer{}
			var replicas int32 = 3
			if err := coreDNSAddon(tt.args.cfg, tt.args.client, &replicas, "", out, tt.args.printManifest); (err != nil) != tt.wantErr {
				t.Errorf("coreDNSAddon() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if gotOut := out.String(); gotOut != tt.wantOut {
				t.Errorf("Actual output of coreDNSAddon() does not match expected.\nActual:  %v\nExpected: %v\n", gotOut, tt.wantOut)
			}
		})
	}
}

func TestEnsureDNSAddon(t *testing.T) {
	type args struct {
		cfg           *kubeadmapi.ClusterConfiguration
		client        clientset.Interface
		printManifest bool
	}
	tests := []struct {
		name    string
		args    args
		wantOut string
		wantErr bool
	}{
		{
			name: "not print Manifest",
			args: args{
				cfg: &kubeadmapi.ClusterConfiguration{
					DNS: kubeadmapi.DNS{
						ImageMeta: kubeadmapi.ImageMeta{
							ImageRepository: "foo.bar.io",
						},
					},
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "10.0.0.0/16",
					},
				},
				client:        newMockClientForTest(t, 0, 1, "", "", ""),
				printManifest: false,
			},
			wantOut: "[addons] Applied essential addon: CoreDNS\n",
		},
		{
			name: "get dns replicas failed",
			args: args{
				cfg: &kubeadmapi.ClusterConfiguration{
					DNS: kubeadmapi.DNS{
						ImageMeta: kubeadmapi.ImageMeta{
							ImageRepository: "foo.bar.io",
						},
					},
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "10.0.0.0/16",
					},
				},
				client:        newMockClientForTest(t, 0, 2, "", "", ""),
				printManifest: false,
			},
			wantErr: true,
			wantOut: "",
		},
		{
			name: "not print Manifest",
			args: args{
				cfg: &kubeadmapi.ClusterConfiguration{
					DNS: kubeadmapi.DNS{
						ImageMeta: kubeadmapi.ImageMeta{
							ImageRepository: "foo.bar.io",
						},
					},
					Networking: kubeadmapi.Networking{
						ServiceSubnet: "10.0.0.0/16",
					},
				},
				client:        newMockClientForTest(t, 0, 1, "", "", ""),
				printManifest: true,
			},
			wantOut: dedent.Dedent(fmt.Sprintf(`---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: coredns
  namespace: kube-system
  labels:
    k8s-app: kube-dns
spec:
  replicas: 2
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1
  selector:
    matchLabels:
      k8s-app: kube-dns
  template:
    metadata:
      labels:
        k8s-app: kube-dns
    spec:
      priorityClassName: system-cluster-critical
      serviceAccountName: coredns
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
          - weight: 100
            podAffinityTerm:
              labelSelector:
                matchExpressions:
                - key: k8s-app
                  operator: In
                  values: ["kube-dns"]
              topologyKey: kubernetes.io/hostname
      tolerations:
      - key: CriticalAddonsOnly
        operator: Exists
      - key: node-role.kubernetes.io/control-plane
        effect: NoSchedule
      nodeSelector:
        kubernetes.io/os: linux
      containers:
      - name: coredns
        image: foo.bar.io/coredns:%s
        imagePullPolicy: IfNotPresent
        resources:
          limits:
            memory: 170Mi
          requests:
            cpu: 100m
            memory: 70Mi
        args: [ "-conf", "/etc/coredns/Corefile" ]
        volumeMounts:
        - name: config-volume
          mountPath: /etc/coredns
          readOnly: true
        ports:
        - containerPort: 53
          name: dns
          protocol: UDP
        - containerPort: 53
          name: dns-tcp
          protocol: TCP
        - containerPort: 9153
          name: metrics
          protocol: TCP
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 60
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 5
        readinessProbe:
          httpGet:
            path: /ready
            port: 8181
            scheme: HTTP
        securityContext:
          allowPrivilegeEscalation: false
          capabilities:
            add:
            - NET_BIND_SERVICE
            drop:
            - ALL
          readOnlyRootFilesystem: true
      dnsPolicy: Default
      volumes:
        - name: config-volume
          configMap:
            name: coredns
            items:
            - key: Corefile
              path: Corefile
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: coredns
  namespace: kube-system
data:
  Corefile: |
    .:53 {
        errors
        health {
           lameduck 5s
        }
        ready
        kubernetes  in-addr.arpa ip6.arpa {
           pods insecure
           fallthrough in-addr.arpa ip6.arpa
           ttl 30
        }
        prometheus :9153
        forward . /etc/resolv.conf {
           max_concurrent 1000
        }
        cache 30
        loop
        reload
        loadbalance
    }
---
apiVersion: v1
kind: Service
metadata:
  labels:
    k8s-app: kube-dns
    kubernetes.io/cluster-service: "true"
    kubernetes.io/name: "CoreDNS"
  name: kube-dns
  namespace: kube-system
  annotations:
    prometheus.io/port: "9153"
    prometheus.io/scrape: "true"
  # Without this resourceVersion value, an update of the Service between versions will yield:
  #   Service "kube-dns" is invalid: metadata.resourceVersion: Invalid value: "": must be specified for an update
  resourceVersion: "0"
spec:
  clusterIP: 10.0.0.10
  ports:
  - name: dns
    port: 53
    protocol: UDP
    targetPort: 53
  - name: dns-tcp
    port: 53
    protocol: TCP
    targetPort: 53
  - name: metrics
    port: 9153
    protocol: TCP
    targetPort: 9153
  selector:
    k8s-app: kube-dns
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRole
metadata:
  name: system:coredns
rules:
- apiGroups:
  - ""
  resources:
  - endpoints
  - services
  - pods
  - namespaces
  verbs:
  - list
  - watch
- apiGroups:
  - discovery.k8s.io
  resources:
  - endpointslices
  verbs:
  - list
  - watch
---
apiVersion: rbac.authorization.k8s.io/v1
kind: ClusterRoleBinding
metadata:
  name: system:coredns
roleRef:
  apiGroup: rbac.authorization.k8s.io
  kind: ClusterRole
  name: system:coredns
subjects:
- kind: ServiceAccount
  name: coredns
  namespace: kube-system
---
apiVersion: v1
kind: ServiceAccount
metadata:
  name: coredns
  namespace: kube-system
`, kubeadmconstants.CoreDNSVersion)),
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			out := &bytes.Buffer{}
			if err := EnsureDNSAddon(tt.args.cfg, tt.args.client, "", out, tt.args.printManifest); (err != nil) != tt.wantErr {
				t.Errorf("EnsureDNSAddon() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if gotOut := out.String(); gotOut != tt.wantOut {
				t.Errorf("Actual output of EnsureDNSAddon() does not match expected.\nActual:  %v\nExpected: %v\n", gotOut, tt.wantOut)
			}
		})
	}
}

func TestCreateDNSService(t *testing.T) {
	coreDNSServiceBytes, _ := kubeadmutil.ParseTemplate(CoreDNSService, struct{ DNSIP string }{
		DNSIP: "10.233.0.3",
	})
	type args struct {
		dnsService   *v1.Service
		serviceBytes []byte
	}
	tests := []struct {
		name    string
		args    args
		wantErr bool
	}{
		{
			name: "dnsService and serviceBytes are nil",
			args: args{
				dnsService:   nil,
				serviceBytes: nil,
			},
			wantErr: true,
		},
		{
			name: "invalid dns",
			args: args{
				dnsService:   nil,
				serviceBytes: coreDNSServiceBytes,
			},
			wantErr: true,
		},
		{
			name: "serviceBytes is not valid",
			args: args{
				dnsService: &v1.Service{
					TypeMeta: metav1.TypeMeta{
						Kind:       "Service",
						APIVersion: "v1",
					},
					ObjectMeta: metav1.ObjectMeta{Name: "coredns",
						Labels: map[string]string{"k8s-app": "kube-dns",
							"kubernetes.io/name": "coredns"},
						Namespace: "kube-system",
					},
					Spec: v1.ServiceSpec{
						Ports: []v1.ServicePort{
							{
								Name:     "dns",
								Port:     53,
								Protocol: v1.ProtocolUDP,
								TargetPort: intstr.IntOrString{
									Type:   0,
									IntVal: 53,
								},
							},
							{
								Name:     "dns-tcp",
								Port:     53,
								Protocol: v1.ProtocolTCP,
								TargetPort: intstr.IntOrString{
									Type:   0,
									IntVal: 53,
								},
							},
						},
						Selector: map[string]string{
							"k8s-app": "kube-dns",
						},
					},
				},
				serviceBytes: []byte{
					'f', 'o', 'o',
				},
			},
			wantErr: true,
		},
		{
			name: "dnsService is valid and serviceBytes is nil",
			args: args{
				dnsService: &v1.Service{
					ObjectMeta: metav1.ObjectMeta{Name: "coredns",
						Labels: map[string]string{"k8s-app": "kube-dns",
							"kubernetes.io/name": "coredns"},
						Namespace: "kube-system",
					},
					Spec: v1.ServiceSpec{
						Ports: []v1.ServicePort{
							{
								Name:     "dns",
								Port:     53,
								Protocol: v1.ProtocolUDP,
								TargetPort: intstr.IntOrString{
									Type:   0,
									IntVal: 53,
								},
							},
							{
								Name:     "dns-tcp",
								Port:     53,
								Protocol: v1.ProtocolTCP,
								TargetPort: intstr.IntOrString{
									Type:   0,
									IntVal: 53,
								},
							},
						},
						Selector: map[string]string{
							"k8s-app": "kube-dns",
						},
					},
				},
				serviceBytes: nil,
			},
			wantErr: false,
		},
		{
			name: "dnsService and serviceBytes are not nil and valid",
			args: args{
				dnsService: &v1.Service{
					TypeMeta: metav1.TypeMeta{
						Kind:       "Service",
						APIVersion: "v1",
					},
					ObjectMeta: metav1.ObjectMeta{Name: "coredns",
						Labels: map[string]string{"k8s-app": "kube-dns",
							"kubernetes.io/name": "coredns"},
						Namespace: "kube-system",
					},
					Spec: v1.ServiceSpec{
						ClusterIP: "10.233.0.3",
						Ports: []v1.ServicePort{
							{
								Name:     "dns",
								Port:     53,
								Protocol: v1.ProtocolUDP,
								TargetPort: intstr.IntOrString{
									Type:   0,
									IntVal: 53,
								},
							},
						},
						Selector: map[string]string{
							"k8s-app": "kube-dns",
						},
					},
				},
				serviceBytes: coreDNSServiceBytes,
			},
			wantErr: false,
		},
		{
			name: "the namespace of dnsService is not kube-system",
			args: args{
				dnsService: &v1.Service{
					TypeMeta: metav1.TypeMeta{
						Kind:       "Service",
						APIVersion: "v1",
					},
					ObjectMeta: metav1.ObjectMeta{Name: "coredns",
						Labels: map[string]string{"k8s-app": "kube-dns",
							"kubernetes.io/name": "coredns"},
						Namespace: "kube-system-test",
					},
					Spec: v1.ServiceSpec{
						Ports: []v1.ServicePort{
							{
								Name:     "dns",
								Port:     53,
								Protocol: v1.ProtocolUDP,
								TargetPort: intstr.IntOrString{
									Type:   0,
									IntVal: 53,
								},
							},
							{
								Name:     "dns-tcp",
								Port:     53,
								Protocol: v1.ProtocolTCP,
								TargetPort: intstr.IntOrString{
									Type:   0,
									IntVal: 53,
								},
							},
						},
						Selector: map[string]string{
							"k8s-app": "kube-dns",
						},
					},
				},
				serviceBytes: nil,
			},
			wantErr: true,
		},
		{
			name: "the name of dnsService is not coredns",
			args: args{
				dnsService: &v1.Service{
					TypeMeta: metav1.TypeMeta{
						Kind:       "Service",
						APIVersion: "v1",
					},
					ObjectMeta: metav1.ObjectMeta{Name: "coredns-test",
						Labels: map[string]string{"k8s-app": "kube-dns",
							"kubernetes.io/name": "coredns"},
						Namespace: "kube-system",
					},
					Spec: v1.ServiceSpec{
						Ports: []v1.ServicePort{
							{
								Name:     "dns",
								Port:     53,
								Protocol: v1.ProtocolUDP,
								TargetPort: intstr.IntOrString{
									Type:   0,
									IntVal: 53,
								},
							},
							{
								Name:     "dns-tcp",
								Port:     53,
								Protocol: v1.ProtocolTCP,
								TargetPort: intstr.IntOrString{
									Type:   0,
									IntVal: 53,
								},
							},
						},
						Selector: map[string]string{
							"k8s-app": "kube-dns",
						},
					},
				},
				serviceBytes: nil,
			},
			wantErr: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := newMockClientForTest(t, 1, 1, "", "", "")
			if err := createDNSService(tt.args.dnsService, tt.args.serviceBytes, client); (err != nil) != tt.wantErr {
				t.Errorf("createDNSService() error = %v, wantErr %v", err, tt.wantErr)
			}
		})
	}
}

func TestDeployedDNSAddon(t *testing.T) {
	tests := []struct {
		name           string
		image          string
		wantVersion    string
		deploymentSize int
		wantErr        bool
	}{
		{
			name:           "default",
			image:          "registry.k8s.io/coredns/coredns:" + kubeadmconstants.CoreDNSVersion,
			deploymentSize: 1,
			wantVersion:    kubeadmconstants.CoreDNSVersion,
		},
		{
			name:           "no dns addon deployment",
			image:          "registry.k8s.io/coredns/coredns:" + kubeadmconstants.CoreDNSVersion,
			deploymentSize: 0,
			wantVersion:    "",
		},
		{
			name:           "multiple dns addon deployment",
			image:          "registry.k8s.io/coredns/coredns:" + kubeadmconstants.CoreDNSVersion,
			deploymentSize: 2,
			wantVersion:    "",
			wantErr:        true,
		},
		{
			name:           "with digest",
			image:          "registry.k8s.io/coredns/coredns:v1.12.1@sha256:a0ead06651cf580044aeb0a0feba63591858fb2e43ade8c9dea45a6a89ae7e5e",
			deploymentSize: 1,
			wantVersion:    kubeadmconstants.CoreDNSVersion,
		},
		{
			name:           "without registry",
			image:          "coredns/coredns:coredns-s390x",
			deploymentSize: 1,
			wantVersion:    "coredns-s390x",
		},
		{
			name:           "without registry and tag",
			image:          "coredns/coredns",
			deploymentSize: 1,
			wantVersion:    "",
		},
		{
			name:           "with explicit port",
			image:          "localhost:4711/coredns/coredns:v1.11.2-pre.1",
			deploymentSize: 1,
			wantVersion:    "v1.11.2-pre.1",
		},
		{
			name:           "with explicit port but without tag",
			image:          "localhost:4711/coredns/coredns@sha256:a0ead06651cf580044aeb0a0feba63591858fb2e43ade8c9dea45a6a89ae7e5e",
			deploymentSize: 1,
			wantVersion:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := newMockClientForTest(t, 1, tt.deploymentSize, tt.image, "", "")

			version, err := DeployedDNSAddon(client)
			if (err != nil) != tt.wantErr {
				t.Errorf("DeployedDNSAddon() error = %v, wantErr %v", err, tt.wantErr)
			}
			if version != tt.wantVersion {
				t.Errorf("DeployedDNSAddon() for image %q returned %q, want %q", tt.image, version, tt.wantVersion)
			}
		})
	}
}

func TestGetCoreDNSInfo(t *testing.T) {
	tests := []struct {
		name          string
		client        clientset.Interface
		wantConfigMap *v1.ConfigMap
		wantCorefile  string
		wantVersion   string
		wantErr       bool
	}{
		{
			name:          "no coredns configmap",
			client:        newMockClientForTest(t, 1, 1, "localhost:4711/coredns/coredns:v1.11.2-pre.1", "", ""),
			wantConfigMap: nil,
			wantCorefile:  "",
			wantVersion:   "",
			wantErr:       false,
		},
		{
			name:          "the key of coredns configmap data does not contain corefile",
			client:        newMockClientForTest(t, 1, 1, "localhost:4711/coredns/coredns:v1.11.2-pre.1", "coredns", "Corefilefake"),
			wantConfigMap: nil,
			wantCorefile:  "",
			wantVersion:   "",
			wantErr:       true,
		},
		{
			name:          "failed to obtain coredns version",
			client:        newMockClientForTest(t, 1, 2, "localhost:4711/coredns/coredns:v1.11.2-pre.1", "coredns", "Corefile"),
			wantConfigMap: nil,
			wantCorefile:  "",
			wantVersion:   "",
			wantErr:       true,
		},
		{
			name:   "coredns information can be obtained normally",
			client: newMockClientForTest(t, 1, 1, "localhost:4711/coredns/coredns:v1.11.2-pre.1", "coredns", "Corefile"),
			wantConfigMap: &v1.ConfigMap{
				TypeMeta: metav1.TypeMeta{
					Kind:       "ConfigMap",
					APIVersion: "v1",
				},
				ObjectMeta: metav1.ObjectMeta{
					Name: "coredns",
					Labels: map[string]string{
						"k8s-app":            "kube-dns",
						"kubernetes.io/name": "coredns",
					},
					Namespace: "kube-system",
				},
				Data: map[string]string{
					"Corefile": dedent.Dedent(`
    .:53 {
        errors
        health {
            lameduck 5s
        }
        ready
        kubernetes cluster.local in-addr.arpa ip6.arpa {
          pods insecure
          fallthrough in-addr.arpa ip6.arpa
        }
        prometheus :9153
        forward . /etc/resolv.conf {
          prefer_udp
          max_concurrent 1000
        }
        cache 30

        loop
        reload
        loadbalance
    }
	`),
				},
			},
			wantCorefile: dedent.Dedent(`
    .:53 {
        errors
        health {
            lameduck 5s
        }
        ready
        kubernetes cluster.local in-addr.arpa ip6.arpa {
          pods insecure
          fallthrough in-addr.arpa ip6.arpa
        }
        prometheus :9153
        forward . /etc/resolv.conf {
          prefer_udp
          max_concurrent 1000
        }
        cache 30

        loop
        reload
        loadbalance
    }
	`),
			wantVersion: "v1.11.2-pre.1",
			wantErr:     false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, got1, got2, err := GetCoreDNSInfo(tt.client)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetCoreDNSInfo() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.wantConfigMap) {
				t.Errorf("GetCoreDNSInfo() got = %v, want %v", got, tt.wantConfigMap)
			}
			if got1 != tt.wantCorefile {
				t.Errorf("GetCoreDNSInfo() got1 = %v, want %v", got1, tt.wantCorefile)
			}
			if got2 != tt.wantVersion {
				t.Errorf("GetCoreDNSInfo() got2 = %v, want %v", got2, tt.wantVersion)
			}
		})
	}
}

func TestIsCoreDNSConfigMapMigrationRequired(t *testing.T) {
	tests := []struct {
		name                           string
		corefile                       string
		currentInstalledCoreDNSVersion string
		want                           bool
		wantErr                        bool
	}{
		{
			name:                           "currentInstalledCoreDNSVersion is empty",
			corefile:                       "",
			currentInstalledCoreDNSVersion: "",
			want:                           false,
			wantErr:                        false,
		},
		{
			name:                           "currentInstalledCoreDNSVersion is consistent with the standard version",
			corefile:                       "",
			currentInstalledCoreDNSVersion: kubeadmconstants.CoreDNSVersion,
			want:                           false,
			wantErr:                        false,
		},
		{
			name:                           "Coredns Configmap needs to be migrated",
			corefile:                       "Corefile: fake",
			currentInstalledCoreDNSVersion: "v1.2.0",
			want:                           true,
			wantErr:                        false,
		},
		{
			name:                           "currentInstalledCoreDNSVersion is not supported",
			corefile:                       "",
			currentInstalledCoreDNSVersion: "v0.11.1",
			want:                           false,
			wantErr:                        true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := isCoreDNSConfigMapMigrationRequired(tt.corefile, tt.currentInstalledCoreDNSVersion)
			if (err != nil) != tt.wantErr {
				t.Errorf("isCoreDNSConfigMapMigrationRequired() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("isCoreDNSConfigMapMigrationRequired() = %v, want %v", got, tt.want)
			}
		})
	}
}

// replicas is replica of each DNS deployment
// deploymentSize is the number of deployments with `k8s-app=kube-dns` label.
func newMockClientForTest(t *testing.T, replicas int32, deploymentSize int, image string, configMap string, configData string) *clientsetfake.Clientset {
	if image == "" {
		image = "registry.k8s.io/coredns/coredns:" + kubeadmconstants.CoreDNSVersion
	}
	client := clientsetfake.NewSimpleClientset()
	for i := 0; i < deploymentSize; i++ {
		_, err := client.AppsV1().Deployments(metav1.NamespaceSystem).Create(context.TODO(), &apps.Deployment{
			TypeMeta: metav1.TypeMeta{
				Kind:       "Deployment",
				APIVersion: "apps/v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name:      fmt.Sprintf("coredns-%d", i),
				Namespace: metav1.NamespaceSystem,
				Labels: map[string]string{
					"k8s-app": "kube-dns",
				},
			},
			Spec: apps.DeploymentSpec{
				Replicas: &replicas,
				Template: v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						Containers: []v1.Container{{Image: image}},
					},
				},
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("error creating deployment: %v", err)
		}
	}
	_, err := client.CoreV1().Services(metav1.NamespaceSystem).Create(context.TODO(), &v1.Service{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Service",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{Name: "coredns",
			Labels: map[string]string{"k8s-app": "kube-dns",
				"kubernetes.io/name": "coredns"},
			Namespace: "kube-system",
		},
		Spec: v1.ServiceSpec{
			ClusterIP: "10.233.0.3",
			Ports: []v1.ServicePort{
				{
					Name:     "dns",
					Port:     53,
					Protocol: v1.ProtocolUDP,
					TargetPort: intstr.IntOrString{
						Type:   0,
						IntVal: 53,
					},
				},
			},
			Selector: map[string]string{
				"k8s-app": "kube-dns",
			},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("error creating service: %v", err)
	}

	if configMap != "" {
		if configMap == "" {
			configMap = "Corefile"
		}
		_, err = client.CoreV1().ConfigMaps(metav1.NamespaceSystem).Create(context.TODO(), &v1.ConfigMap{
			TypeMeta: metav1.TypeMeta{
				Kind:       "ConfigMap",
				APIVersion: "v1",
			},
			ObjectMeta: metav1.ObjectMeta{
				Name: configMap,
				Labels: map[string]string{
					"k8s-app":            "kube-dns",
					"kubernetes.io/name": "coredns",
				},
				Namespace: "kube-system",
			},
			Data: map[string]string{
				configData: dedent.Dedent(`
    .:53 {
        errors
        health {
            lameduck 5s
        }
        ready
        kubernetes cluster.local in-addr.arpa ip6.arpa {
          pods insecure
          fallthrough in-addr.arpa ip6.arpa
        }
        prometheus :9153
        forward . /etc/resolv.conf {
          prefer_udp
          max_concurrent 1000
        }
        cache 30

        loop
        reload
        loadbalance
    }
	`),
			},
		}, metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("error creating service: %v", err)
		}
	}
	return client
}
