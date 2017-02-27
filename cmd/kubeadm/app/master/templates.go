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

package master

const (
	DummyDeployment = `
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  labels:
    app: dummy
  name: dummy
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: dummy
  template:
    metadata:
      labels:
        app: dummy
    spec:
      containers:
      - image: {{ .ImageRepository }}/pause-{{ .Arch }}:3.0
        name: dummy
      hostNetwork: true
`
	KubeDiscoveryDeployment = `
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  labels:
    k8s-app: kube-discovery
  name: kube-discovery
  namespace: kube-system
spec:
  replicas: 1
  selector:
    matchLabels:
      k8s-app: kube-discovery
  template:
    metadata:
      labels:
        k8s-app: kube-discovery
        # TODO: I guess we can remove all these cluster-service labels...
        kubernetes.io/cluster-service: "true"
    spec:
      containers:
      - name: kube-discovery
        image: {{ .ImageRepository }}/kube-discovery-{{ .Arch }}:1.0
        imagePullPolicy: IfNotPresent
        command:
        - /usr/local/bin/kube-discovery
        ports:
        - containerPort: 9898
          hostPort: 9898
          name: http
        volumeMounts:
        - mountPath: /tmp/secret
          name: clusterinfo
          readOnly: true
      hostNetwork: true
      # TODO: Why doesn't the Decoder recognize this new field and decode it properly? Right now it's ignored
      # tolerations:
      # - key: {{ .MasterTaintKey }}
      #  effect: NoSchedule
      securityContext:
          seLinuxOptions:
            type: spc_t
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: beta.kubernetes.io/arch
                operator: In
                values:
                - {{ .Arch }}
      volumes:
      - name: clusterinfo
        secret:
          defaultMode: 420
          secretName: clusterinfo
`
)
