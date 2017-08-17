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

package selfhosting

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"testing"

	"github.com/ghodss/yaml"
	kubeadmapi "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/cmd/kubeadm/app/cmd/features"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

const (
	apiProjectedSecret = `- name: k8s
    projected:
      sources:
      - secret:
          items:
          - key: tls.crt
            path: ca.crt
          - key: tls.key
            path: ca.key
          name: ca
      - secret:
          items:
          - key: tls.crt
            path: apiserver.crt
          - key: tls.key
            path: apiserver.key
          name: apiserver
      - secret:
          items:
          - key: tls.crt
            path: apiserver-kubelet-client.crt
          - key: tls.key
            path: apiserver-kubelet-client.key
          name: apiserver-kubelet-client
      - secret:
          items:
          - key: tls.crt
            path: sa.pub
          - key: tls.key
            path: sa.key
          name: sa
      - secret:
          items:
          - key: tls.crt
            path: front-proxy-ca.crt
          name: front-proxy-ca
      - secret:
          items:
          - key: tls.crt
            path: front-proxy-client.crt
          - key: tls.key
            path: front-proxy-client.key
          name: front-proxy-client`

	controllerManagerProjectedSecret = `- name: k8s
    projected:
      sources:
      - secret:
          name: controller-manager.conf
      - secret:
          items:
          - key: tls.crt
            path: ca.crt
          - key: tls.key
            path: ca.key
          name: ca
      - secret:
          items:
          - key: tls.key
            path: sa.key
          name: sa`

	schedulerProjectedSecret = `- name: k8s
    projected:
      sources:
      - secret:
          name: scheduler.conf`

	hostPathVol = `- hostPath:
      path: /etc/kubernetes
    name: k8s`

	testAPIServerPod = `
apiVersion: v1
kind: Pod
metadata:
  annotations:
    scheduler.alpha.kubernetes.io/critical-pod: ""
  creationTimestamp: null
  labels:
    component: kube-apiserver
    tier: control-plane
  name: kube-apiserver
  namespace: kube-system
spec:
  containers:
  - command:
    - kube-apiserver
    - --client-ca-file=/etc/kubernetes/pki/ca.crt
    - --tls-private-key-file=/etc/kubernetes/pki/apiserver.key
    - --allow-privileged=true
    - --service-cluster-ip-range=10.96.0.0/12
    - --service-account-key-file=/etc/kubernetes/pki/sa.pub
    - --tls-cert-file=/etc/kubernetes/pki/apiserver.crt
    - --kubelet-client-certificate=/etc/kubernetes/pki/apiserver-kubelet-client.crt
    - --secure-port=6443
    - --insecure-port=0
    - --admission-control=Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota
    - --requestheader-extra-headers-prefix=X-Remote-Extra-
    - --proxy-client-cert-file=/etc/kubernetes/pki/front-proxy-client.crt
    - --experimental-bootstrap-token-auth=true
    - --requestheader-group-headers=X-Remote-Group
    - --requestheader-allowed-names=front-proxy-client
    - --kubelet-client-key=/etc/kubernetes/pki/apiserver-kubelet-client.key
    - --requestheader-client-ca-file=/etc/kubernetes/pki/front-proxy-ca.crt
    - --proxy-client-key-file=/etc/kubernetes/pki/front-proxy-client.key
    - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
    - --requestheader-username-headers=X-Remote-User
    - --authorization-mode=Node,RBAC
    - --advertise-address=192.168.200.101
    - --etcd-servers=http://127.0.0.1:2379
    image: gcr.io/google_containers/kube-apiserver-amd64:v1.7.0
    livenessProbe:
      failureThreshold: 8
      httpGet:
        host: 127.0.0.1
        path: /healthz
        port: 6443
        scheme: HTTPS
      initialDelaySeconds: 15
      timeoutSeconds: 15
    name: kube-apiserver
    resources:
      requests:
        cpu: 250m
    volumeMounts:
    - mountPath: /etc/kubernetes
      name: k8s
      readOnly: true
    - mountPath: /etc/ssl/certs
      name: certs
    - mountPath: /etc/pki
      name: pki
  hostNetwork: true
  volumes:
  %s
  - hostPath:
      path: /etc/ssl/certs
    name: certs
  - hostPath:
      path: /etc/pki
    name: pki
status: {}
`

	testAPIServerDaemonSet = `metadata:
  creationTimestamp: null
  labels:
    k8s-app: self-hosted-kube-apiserver
  name: self-hosted-kube-apiserver
  namespace: kube-system
spec:
  template:
    metadata:
      creationTimestamp: null
      labels:
        k8s-app: self-hosted-kube-apiserver
    spec:
      containers:
      - command:
        - kube-apiserver
        - --client-ca-file=/etc/kubernetes/pki/ca.crt
        - --tls-private-key-file=/etc/kubernetes/pki/apiserver.key
        - --allow-privileged=true
        - --service-cluster-ip-range=10.96.0.0/12
        - --service-account-key-file=/etc/kubernetes/pki/sa.pub
        - --tls-cert-file=/etc/kubernetes/pki/apiserver.crt
        - --kubelet-client-certificate=/etc/kubernetes/pki/apiserver-kubelet-client.crt
        - --secure-port=6443
        - --insecure-port=0
        - --admission-control=Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota
        - --requestheader-extra-headers-prefix=X-Remote-Extra-
        - --proxy-client-cert-file=/etc/kubernetes/pki/front-proxy-client.crt
        - --experimental-bootstrap-token-auth=true
        - --requestheader-group-headers=X-Remote-Group
        - --requestheader-allowed-names=front-proxy-client
        - --kubelet-client-key=/etc/kubernetes/pki/apiserver-kubelet-client.key
        - --requestheader-client-ca-file=/etc/kubernetes/pki/front-proxy-ca.crt
        - --proxy-client-key-file=/etc/kubernetes/pki/front-proxy-client.key
        - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
        - --requestheader-username-headers=X-Remote-User
        - --authorization-mode=Node,RBAC
        - --advertise-address=192.168.200.101
        - --etcd-servers=http://127.0.0.1:2379
        image: gcr.io/google_containers/kube-apiserver-amd64:v1.7.0
        livenessProbe:
          failureThreshold: 8
          httpGet:
            host: 127.0.0.1
            path: /healthz
            port: 6443
            scheme: HTTPS
          initialDelaySeconds: 15
          timeoutSeconds: 15
        name: kube-apiserver
        resources:
          requests:
            cpu: 250m
        volumeMounts:
        - mountPath: /etc/kubernetes
          name: k8s
          readOnly: true
        - mountPath: /etc/ssl/certs
          name: certs
        - mountPath: /etc/pki
          name: pki
      dnsPolicy: ClusterFirstWithHostNet
      hostNetwork: true
      nodeSelector:
        node-role.kubernetes.io/master: ""
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
      volumes:
      %s
      - hostPath:
          path: /etc/ssl/certs
        name: certs
      - hostPath:
          path: /etc/pki
        name: pki
  updateStrategy: {}
status:
  currentNumberScheduled: 0
  desiredNumberScheduled: 0
  numberMisscheduled: 0
  numberReady: 0
`

	testControllerManagerPod = `
apiVersion: v1
kind: Pod
metadata:
  annotations:
    scheduler.alpha.kubernetes.io/critical-pod: ""
  creationTimestamp: null
  labels:
    component: kube-controller-manager
    tier: control-plane
  name: kube-controller-manager
  namespace: kube-system
spec:
  containers:
  - command:
    - kube-controller-manager
    - --service-account-private-key-file=/etc/kubernetes/pki/sa.key
    - --cluster-signing-cert-file=/etc/kubernetes/pki/ca.crt
    - --cluster-signing-key-file=/etc/kubernetes/pki/ca.key
    - --leader-elect=true
    - --kubeconfig=/etc/kubernetes/controller-manager.conf
    - --controllers=*,bootstrapsigner,tokencleaner
    - --root-ca-file=/etc/kubernetes/pki/ca.crt
    - --address=127.0.0.1
    - --use-service-account-credentials=true
    image: gcr.io/google_containers/kube-controller-manager-amd64:v1.7.0
    livenessProbe:
      failureThreshold: 8
      httpGet:
        host: 127.0.0.1
        path: /healthz
        port: 10252
        scheme: HTTP
      initialDelaySeconds: 15
      timeoutSeconds: 15
    name: kube-controller-manager
    resources:
      requests:
        cpu: 200m
    volumeMounts:
    - mountPath: /etc/kubernetes
      name: k8s
      readOnly: true
    - mountPath: /etc/ssl/certs
      name: certs
    - mountPath: /etc/pki
      name: pki
  hostNetwork: true
  volumes:
  %s
  - hostPath:
      path: /etc/ssl/certs
    name: certs
  - hostPath:
      path: /etc/pki
    name: pki
status: {}
`

	testControllerManagerDaemonSet = `metadata:
  creationTimestamp: null
  labels:
    k8s-app: self-hosted-kube-controller-manager
  name: self-hosted-kube-controller-manager
  namespace: kube-system
spec:
  template:
    metadata:
      creationTimestamp: null
      labels:
        k8s-app: self-hosted-kube-controller-manager
    spec:
      containers:
      - command:
        - kube-controller-manager
        - --service-account-private-key-file=/etc/kubernetes/pki/sa.key
        - --cluster-signing-cert-file=/etc/kubernetes/pki/ca.crt
        - --cluster-signing-key-file=/etc/kubernetes/pki/ca.key
        - --leader-elect=true
        - --kubeconfig=/etc/kubernetes/controller-manager.conf
        - --controllers=*,bootstrapsigner,tokencleaner
        - --root-ca-file=/etc/kubernetes/pki/ca.crt
        - --address=127.0.0.1
        - --use-service-account-credentials=true
        image: gcr.io/google_containers/kube-controller-manager-amd64:v1.7.0
        livenessProbe:
          failureThreshold: 8
          httpGet:
            host: 127.0.0.1
            path: /healthz
            port: 10252
            scheme: HTTP
          initialDelaySeconds: 15
          timeoutSeconds: 15
        name: kube-controller-manager
        resources:
          requests:
            cpu: 200m
        volumeMounts:
        - mountPath: /etc/kubernetes
          name: k8s
          readOnly: true
        - mountPath: /etc/ssl/certs
          name: certs
        - mountPath: /etc/pki
          name: pki
      dnsPolicy: ClusterFirstWithHostNet
      hostNetwork: true
      nodeSelector:
        node-role.kubernetes.io/master: ""
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
      volumes:
      %s
      - hostPath:
          path: /etc/ssl/certs
        name: certs
      - hostPath:
          path: /etc/pki
        name: pki
  updateStrategy: {}
status:
  currentNumberScheduled: 0
  desiredNumberScheduled: 0
  numberMisscheduled: 0
  numberReady: 0
`

	testSchedulerPod = `
apiVersion: v1
kind: Pod
metadata:
  annotations:
    scheduler.alpha.kubernetes.io/critical-pod: ""
  creationTimestamp: null
  labels:
    component: kube-scheduler
    tier: control-plane
  name: kube-scheduler
  namespace: kube-system
spec:
  containers:
  - command:
    - kube-scheduler
    - --address=127.0.0.1
    - --leader-elect=true
    - --kubeconfig=/etc/kubernetes/scheduler.conf
    image: gcr.io/google_containers/kube-scheduler-amd64:v1.7.0
    livenessProbe:
      failureThreshold: 8
      httpGet:
        host: 127.0.0.1
        path: /healthz
        port: 10251
        scheme: HTTP
      initialDelaySeconds: 15
      timeoutSeconds: 15
    name: kube-scheduler
    resources:
      requests:
        cpu: 100m
    volumeMounts:
    - mountPath: /etc/kubernetes
      name: k8s
      readOnly: true
  hostNetwork: true
  volumes:
  %s
status: {}
`

	testSchedulerDaemonSet = `metadata:
  creationTimestamp: null
  labels:
    k8s-app: self-hosted-kube-scheduler
  name: self-hosted-kube-scheduler
  namespace: kube-system
spec:
  template:
    metadata:
      creationTimestamp: null
      labels:
        k8s-app: self-hosted-kube-scheduler
    spec:
      containers:
      - command:
        - kube-scheduler
        - --address=127.0.0.1
        - --leader-elect=true
        - --kubeconfig=/etc/kubernetes/scheduler.conf
        image: gcr.io/google_containers/kube-scheduler-amd64:v1.7.0
        livenessProbe:
          failureThreshold: 8
          httpGet:
            host: 127.0.0.1
            path: /healthz
            port: 10251
            scheme: HTTP
          initialDelaySeconds: 15
          timeoutSeconds: 15
        name: kube-scheduler
        resources:
          requests:
            cpu: 100m
        volumeMounts:
        - mountPath: /etc/kubernetes
          name: k8s
          readOnly: true
      dnsPolicy: ClusterFirstWithHostNet
      hostNetwork: true
      nodeSelector:
        node-role.kubernetes.io/master: ""
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
      volumes:
      %s
  updateStrategy: {}
status:
  currentNumberScheduled: 0
  desiredNumberScheduled: 0
  numberMisscheduled: 0
  numberReady: 0
`
)

var (
	testAPIServerSecretsPod  = fmt.Sprintf(testAPIServerPod, apiProjectedSecret)
	testAPIServerSecretsDS   = fmt.Sprintf(testAPIServerDaemonSet, indentString(apiProjectedSecret, 4))
	testAPIServerHostPathPod = fmt.Sprintf(testAPIServerPod, hostPathVol)
	testAPIServerHostPathDS  = fmt.Sprintf(testAPIServerDaemonSet, indentString(hostPathVol, 4))

	testSchedulerSecretsPod  = fmt.Sprintf(testSchedulerPod, schedulerProjectedSecret)
	testSchedulerSecretsDS   = fmt.Sprintf(testSchedulerDaemonSet, indentString(schedulerProjectedSecret, 4))
	testSchedulerHostPathPod = fmt.Sprintf(testSchedulerPod, hostPathVol)
	testSchedulerHostPathDS  = fmt.Sprintf(testSchedulerDaemonSet, indentString(hostPathVol, 4))

	testControllerManagerSecretsPod  = fmt.Sprintf(testControllerManagerPod, controllerManagerProjectedSecret)
	testControllerManagerSecretsDS   = fmt.Sprintf(testControllerManagerDaemonSet, indentString(controllerManagerProjectedSecret, 4))
	testControllerManagerHostPathPod = fmt.Sprintf(testControllerManagerPod, hostPathVol)
	testControllerManagerHostPathDS  = fmt.Sprintf(testControllerManagerDaemonSet, indentString(hostPathVol, 4))
)

func TestBuildDaemonSet(t *testing.T) {
	var tests = []struct {
		component         string
		podBytes          []byte
		dsBytes           []byte
		selfHostedSecrets bool
	}{
		// vols as secrets
		{
			component:         kubeadmconstants.KubeAPIServer,
			podBytes:          []byte(testAPIServerSecretsPod),
			dsBytes:           []byte(testAPIServerSecretsDS),
			selfHostedSecrets: true,
		},
		{
			component:         kubeadmconstants.KubeControllerManager,
			podBytes:          []byte(testControllerManagerSecretsPod),
			dsBytes:           []byte(testControllerManagerSecretsDS),
			selfHostedSecrets: true,
		},
		{
			component:         kubeadmconstants.KubeScheduler,
			podBytes:          []byte(testSchedulerSecretsPod),
			dsBytes:           []byte(testSchedulerSecretsDS),
			selfHostedSecrets: true,
		},
		// hostPath vols
		{
			component:         kubeadmconstants.KubeAPIServer,
			podBytes:          []byte(testAPIServerHostPathPod),
			dsBytes:           []byte(testAPIServerHostPathDS),
			selfHostedSecrets: false,
		},
		{
			component:         kubeadmconstants.KubeControllerManager,
			podBytes:          []byte(testControllerManagerHostPathPod),
			dsBytes:           []byte(testControllerManagerHostPathDS),
			selfHostedSecrets: false,
		},
		{
			component:         kubeadmconstants.KubeScheduler,
			podBytes:          []byte(testSchedulerHostPathPod),
			dsBytes:           []byte(testSchedulerHostPathDS),
			selfHostedSecrets: false,
		},
	}

	for _, rt := range tests {
		tempFile, err := createTempFileWithContent(rt.podBytes)
		defer os.Remove(tempFile)

		podSpec, err := loadPodSpecFromFile(tempFile)
		if err != nil {
			t.Fatalf("couldn't load the specified Pod: %v", err)
		}

		cfg := &kubeadmapi.MasterConfiguration{
			FeatureFlags: map[string]bool{string(features.StoreCertsInSecrets): rt.selfHostedSecrets},
		}

		ds := buildDaemonSet(cfg, rt.component, podSpec)
		dsBytes, err := yaml.Marshal(ds)
		if err != nil {
			t.Fatalf("failed to marshal daemonset to YAML: %v", err)
		}

		if !bytes.Equal(dsBytes, rt.dsBytes) {
			t.Errorf("failed TestBuildDaemonSet for name=%s (secrets=%t):\nexpected:\n%s\nsaw:\n%s", rt.component, rt.selfHostedSecrets, rt.dsBytes, dsBytes)
		}
	}
}

func TestLoadPodSpecFromFile(t *testing.T) {
	tests := []struct {
		content     string
		expectError bool
	}{
		{
			// Good YAML
			content: `
apiVersion: v1
kind: Pod
metadata:
  name: testpod
spec:
  containers:
    - image: gcr.io/google_containers/busybox
`,
			expectError: false,
		},
		{
			// Good JSON
			content: `
{
  "apiVersion": "v1",
  "kind": "Pod",
  "metadata": {
    "name": "testpod"
  },
  "spec": {
    "containers": [
      {
        "image": "gcr.io/google_containers/busybox"
      }
    ]
  }
}`,
			expectError: false,
		},
		{
			// Bad PodSpec
			content: `
apiVersion: v1
kind: Pod
metadata:
  name: testpod
spec:
  - image: gcr.io/google_containers/busybox
`,
			expectError: true,
		},
	}

	for _, rt := range tests {
		tempFile, err := createTempFileWithContent([]byte(rt.content))
		defer os.Remove(tempFile)

		_, err = loadPodSpecFromFile(tempFile)
		if (err != nil) != rt.expectError {
			t.Errorf("failed TestLoadPodSpecFromFile:\nexpected error:\n%t\nsaw:\n%v", rt.expectError, err)
		}
	}
}

func createTempFileWithContent(content []byte) (string, error) {
	tempFile, err := ioutil.TempFile("", "")
	if err != nil {
		return "", fmt.Errorf("cannot create temporary file: %v", err)
	}
	if _, err = tempFile.Write([]byte(content)); err != nil {
		return "", fmt.Errorf("cannot save temporary file: %v", err)
	}
	if err = tempFile.Close(); err != nil {
		return "", fmt.Errorf("cannot close temporary file: %v", err)
	}
	return tempFile.Name(), nil
}

func indentString(input string, count int) string {
	output := ""
	lines := strings.Split(input, "\n")
	for i, line := range lines {
		if i > 0 {
			output += strings.Repeat(" ", count)
		}
		output += line
		if i < len(lines)-1 {
			output += "\n"
		}
	}
	return output
}
