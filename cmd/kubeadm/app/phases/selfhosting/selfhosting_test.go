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
	"testing"

	"github.com/ghodss/yaml"
	"k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

const (
	testAPIServerPod = `
apiVersion: v1
kind: Pod
metadata:
  annotations:
    scheduler.alpha.kubernetes.io/critical-pod: ""
  creationTimestamp: null
  name: kube-apiserver
  namespace: kube-system
spec:
  containers:
  - command:
    - kube-apiserver
    - --service-account-key-file=/etc/kubernetes/pki/sa.pub
    - --kubelet-client-key=/etc/kubernetes/pki/apiserver-kubelet-client.key
    - --secure-port=6443
    - --proxy-client-cert-file=/etc/kubernetes/pki/front-proxy-client.crt
    - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
    - --requestheader-group-headers=X-Remote-Group
    - --service-cluster-ip-range=10.96.0.0/12
    - --tls-cert-file=/etc/kubernetes/pki/apiserver.crt
    - --kubelet-client-certificate=/etc/kubernetes/pki/apiserver-kubelet-client.crt
    - --advertise-address=192.168.1.115
    - --requestheader-client-ca-file=/etc/kubernetes/pki/front-proxy-ca.crt
    - --insecure-port=0
    - --experimental-bootstrap-token-auth=true
    - --requestheader-username-headers=X-Remote-User
    - --requestheader-extra-headers-prefix=X-Remote-Extra-
    - --requestheader-allowed-names=front-proxy-client
    - --admission-control=Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota
    - --allow-privileged=true
    - --client-ca-file=/etc/kubernetes/pki/ca.crt
    - --tls-private-key-file=/etc/kubernetes/pki/apiserver.key
    - --proxy-client-key-file=/etc/kubernetes/pki/front-proxy-client.key
    - --authorization-mode=Node,RBAC
    - --etcd-servers=http://127.0.0.1:2379
    image: gcr.io/google_containers/kube-apiserver-amd64:v1.7.4
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
    - mountPath: /etc/kubernetes/pki
      name: k8s-certs
      readOnly: true
    - mountPath: /etc/ssl/certs
      name: ca-certs
      readOnly: true
    - mountPath: /etc/pki
      name: ca-certs-etc-pki
      readOnly: true
  hostNetwork: true
  volumes:
  - hostPath:
      path: /etc/kubernetes/pki
    name: k8s-certs
  - hostPath:
      path: /etc/ssl/certs
    name: ca-certs
  - hostPath:
      path: /etc/pki
    name: ca-certs-etc-pki
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
        - --service-account-key-file=/etc/kubernetes/pki/sa.pub
        - --kubelet-client-key=/etc/kubernetes/pki/apiserver-kubelet-client.key
        - --secure-port=6443
        - --proxy-client-cert-file=/etc/kubernetes/pki/front-proxy-client.crt
        - --kubelet-preferred-address-types=InternalIP,ExternalIP,Hostname
        - --requestheader-group-headers=X-Remote-Group
        - --service-cluster-ip-range=10.96.0.0/12
        - --tls-cert-file=/etc/kubernetes/pki/apiserver.crt
        - --kubelet-client-certificate=/etc/kubernetes/pki/apiserver-kubelet-client.crt
        - --advertise-address=192.168.1.115
        - --requestheader-client-ca-file=/etc/kubernetes/pki/front-proxy-ca.crt
        - --insecure-port=0
        - --experimental-bootstrap-token-auth=true
        - --requestheader-username-headers=X-Remote-User
        - --requestheader-extra-headers-prefix=X-Remote-Extra-
        - --requestheader-allowed-names=front-proxy-client
        - --admission-control=Initializers,NamespaceLifecycle,LimitRanger,ServiceAccount,PersistentVolumeLabel,DefaultStorageClass,DefaultTolerationSeconds,NodeRestriction,ResourceQuota
        - --allow-privileged=true
        - --client-ca-file=/etc/kubernetes/pki/ca.crt
        - --tls-private-key-file=/etc/kubernetes/pki/apiserver.key
        - --proxy-client-key-file=/etc/kubernetes/pki/front-proxy-client.key
        - --authorization-mode=Node,RBAC
        - --etcd-servers=http://127.0.0.1:2379
        image: gcr.io/google_containers/kube-apiserver-amd64:v1.7.4
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
        - mountPath: /etc/kubernetes/pki
          name: k8s-certs
          readOnly: true
        - mountPath: /etc/ssl/certs
          name: ca-certs
          readOnly: true
        - mountPath: /etc/pki
          name: ca-certs-etc-pki
          readOnly: true
      dnsPolicy: ClusterFirstWithHostNet
      hostNetwork: true
      nodeSelector:
        node-role.kubernetes.io/master: ""
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
      volumes:
      - hostPath:
          path: /etc/kubernetes/pki
        name: k8s-certs
      - hostPath:
          path: /etc/ssl/certs
        name: ca-certs
      - hostPath:
          path: /etc/pki
        name: ca-certs-etc-pki
  updateStrategy:
    type: RollingUpdate
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
  name: kube-controller-manager
  namespace: kube-system
spec:
  containers:
  - command:
    - kube-controller-manager
    - --leader-elect=true
    - --controllers=*,bootstrapsigner,tokencleaner
    - --kubeconfig=/etc/kubernetes/controller-manager.conf
    - --root-ca-file=/etc/kubernetes/pki/ca.crt
    - --service-account-private-key-file=/etc/kubernetes/pki/sa.key
    - --cluster-signing-cert-file=/etc/kubernetes/pki/ca.crt
    - --cluster-signing-key-file=/etc/kubernetes/pki/ca.key
    - --address=127.0.0.1
    - --use-service-account-credentials=true
    image: gcr.io/google_containers/kube-controller-manager-amd64:v1.7.4
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
    - mountPath: /etc/kubernetes/pki
      name: k8s-certs
      readOnly: true
    - mountPath: /etc/ssl/certs
      name: ca-certs
      readOnly: true
    - mountPath: /etc/kubernetes/controller-manager.conf
      name: kubeconfig
      readOnly: true
    - mountPath: /etc/pki
      name: ca-certs-etc-pki
      readOnly: true
  hostNetwork: true
  volumes:
  - hostPath:
      path: /etc/kubernetes/pki
    name: k8s-certs
  - hostPath:
      path: /etc/ssl/certs
    name: ca-certs
  - hostPath:
      path: /etc/kubernetes/controller-manager.conf
      type: FileOrCreate
    name: kubeconfig
  - hostPath:
      path: /etc/pki
    name: ca-certs-etc-pki
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
        - --leader-elect=true
        - --controllers=*,bootstrapsigner,tokencleaner
        - --kubeconfig=/etc/kubernetes/controller-manager.conf
        - --root-ca-file=/etc/kubernetes/pki/ca.crt
        - --service-account-private-key-file=/etc/kubernetes/pki/sa.key
        - --cluster-signing-cert-file=/etc/kubernetes/pki/ca.crt
        - --cluster-signing-key-file=/etc/kubernetes/pki/ca.key
        - --address=127.0.0.1
        - --use-service-account-credentials=true
        image: gcr.io/google_containers/kube-controller-manager-amd64:v1.7.4
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
        - mountPath: /etc/kubernetes/pki
          name: k8s-certs
          readOnly: true
        - mountPath: /etc/ssl/certs
          name: ca-certs
          readOnly: true
        - mountPath: /etc/kubernetes/controller-manager.conf
          name: kubeconfig
          readOnly: true
        - mountPath: /etc/pki
          name: ca-certs-etc-pki
          readOnly: true
      dnsPolicy: ClusterFirstWithHostNet
      hostNetwork: true
      nodeSelector:
        node-role.kubernetes.io/master: ""
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
      volumes:
      - hostPath:
          path: /etc/kubernetes/pki
        name: k8s-certs
      - hostPath:
          path: /etc/ssl/certs
        name: ca-certs
      - hostPath:
          path: /etc/kubernetes/controller-manager.conf
          type: FileOrCreate
        name: kubeconfig
      - hostPath:
          path: /etc/pki
        name: ca-certs-etc-pki
  updateStrategy:
    type: RollingUpdate
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
  name: kube-scheduler
  namespace: kube-system
spec:
  containers:
  - command:
    - kube-scheduler
    - --leader-elect=true
    - --kubeconfig=/etc/kubernetes/scheduler.conf
    - --address=127.0.0.1
    image: gcr.io/google_containers/kube-scheduler-amd64:v1.7.4
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
    - mountPath: /etc/kubernetes/scheduler.conf
      name: kubeconfig
      readOnly: true
  hostNetwork: true
  volumes:
  - hostPath:
      path: /etc/kubernetes/scheduler.conf
      type: FileOrCreate
    name: kubeconfig
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
        - --leader-elect=true
        - --kubeconfig=/etc/kubernetes/scheduler.conf
        - --address=127.0.0.1
        image: gcr.io/google_containers/kube-scheduler-amd64:v1.7.4
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
        - mountPath: /etc/kubernetes/scheduler.conf
          name: kubeconfig
          readOnly: true
      dnsPolicy: ClusterFirstWithHostNet
      hostNetwork: true
      nodeSelector:
        node-role.kubernetes.io/master: ""
      tolerations:
      - effect: NoSchedule
        key: node-role.kubernetes.io/master
      volumes:
      - hostPath:
          path: /etc/kubernetes/scheduler.conf
          type: FileOrCreate
        name: kubeconfig
  updateStrategy:
    type: RollingUpdate
status:
  currentNumberScheduled: 0
  desiredNumberScheduled: 0
  numberMisscheduled: 0
  numberReady: 0
`
)

func TestBuildDaemonSet(t *testing.T) {
	var tests = []struct {
		component string
		podBytes  []byte
		dsBytes   []byte
	}{
		{
			component: constants.KubeAPIServer,
			podBytes:  []byte(testAPIServerPod),
			dsBytes:   []byte(testAPIServerDaemonSet),
		},
		{
			component: constants.KubeControllerManager,
			podBytes:  []byte(testControllerManagerPod),
			dsBytes:   []byte(testControllerManagerDaemonSet),
		},
		{
			component: constants.KubeScheduler,
			podBytes:  []byte(testSchedulerPod),
			dsBytes:   []byte(testSchedulerDaemonSet),
		},
	}

	for _, rt := range tests {
		tempFile, err := createTempFileWithContent(rt.podBytes)
		defer os.Remove(tempFile)

		podSpec, err := loadPodSpecFromFile(tempFile)
		if err != nil {
			t.Fatalf("couldn't load the specified Pod")
		}

		ds := BuildDaemonSet(rt.component, podSpec, GetDefaultMutators())
		dsBytes, err := yaml.Marshal(ds)
		if err != nil {
			t.Fatalf("failed to marshal daemonset to YAML: %v", err)
		}

		if !bytes.Equal(dsBytes, rt.dsBytes) {
			t.Errorf("failed TestBuildDaemonSet:\nexpected:\n%s\nsaw:\n%s", rt.dsBytes, dsBytes)
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
