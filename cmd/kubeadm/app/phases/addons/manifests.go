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

package addons

const (
	KubeProxyConfigMap = `
kind: ConfigMap
apiVersion: v1
metadata:
  name: kube-proxy
  namespace: kube-system
  labels:
    app: kube-proxy
data:
  kubeconfig.conf: |
    apiVersion: v1
    kind: Config
    clusters:
    - cluster:
        certificate-authority: /var/run/secrets/kubernetes.io/serviceaccount/ca.crt
        server: {{ .MasterEndpoint }}
      name: default
    contexts:
    - context:
        cluster: default
        namespace: default
        user: default
      name: default
    current-context: default
    users:
    - name: default
      user:
        tokenFile: /var/run/secrets/kubernetes.io/serviceaccount/token
`

	KubeProxyDaemonSet = `
apiVersion: extensions/v1beta1
kind: DaemonSet
metadata:
  labels:
    k8s-app: kube-proxy
  name: kube-proxy
  namespace: kube-system
spec:
  selector:
    matchLabels:
      k8s-app: kube-proxy
  template:
    metadata:
      labels:
        k8s-app: kube-proxy
      annotations:
        # TODO: Move this to the beta tolerations field below as soon as the Tolerations field exists in PodSpec
        scheduler.alpha.kubernetes.io/tolerations: '[{"key":"dedicated","value":"master","effect":"NoSchedule"}]'
    spec:
      containers:
      - name: kube-proxy
        image: {{ .Image }}
        imagePullPolicy: IfNotPresent
        # TODO: This is gonna work with hyperkube v1.6.0-alpha.2+: https://github.com/kubernetes/kubernetes/pull/41017
        command:
        - kube-proxy
        - --kubeconfig=/var/lib/kube-proxy/kubeconfig.conf
        {{ .ClusterCIDR }}
        securityContext:
          privileged: true
        volumeMounts:
        - mountPath: /var/lib/kube-proxy
          name: kube-proxy
      hostNetwork: true
      serviceAccountName: kube-proxy
      # Tolerate running on the master
      # tolerations:
      # - key: dedicated
      #   value: master
      #   effect: NoSchedule
      volumes:
      - name: kube-proxy
        configMap:
          name: kube-proxy
`

	KubeDNSVersion = "1.11.0"

	KubeDNSDeployment = `
apiVersion: extensions/v1beta1
kind: Deployment
metadata:
  labels:
    k8s-app: kube-dns
  name: kube-dns
  namespace: kube-system
spec:
  replicas: {{ .Replicas }}
  selector:
    matchLabels:
      k8s-app: kube-dns
  strategy:
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 1
    type: RollingUpdate
  template:
    metadata:
      labels:
        k8s-app: kube-dns
      annotations:
        # TODO: Move this to the beta tolerations field below as soon as the Tolerations field exists in PodSpec
        scheduler.alpha.kubernetes.io/tolerations: '[{"key":"dedicated","value":"master","effect":"NoSchedule"}]'
    spec:
      containers:
      - name: kubedns
        image: {{ .ImageRepository }}/k8s-dns-kube-dns-{{ .Arch }}:{{ .Version }}
        imagePullPolicy: IfNotPresent
        args:
        - --domain={{ .DNSDomain }}
        - --dns-port=10053
        - --config-map=kube-dns
        - --v=2
        env:
        - name: PROMETHEUS_PORT
          value: "10055"
        ports:
        - containerPort: 10053
          name: dns-local
          protocol: UDP
        - containerPort: 10053
          name: dns-tcp-local
          protocol: TCP
        - containerPort: 10055
          name: metrics
          protocol: TCP
        livenessProbe:
          failureThreshold: 5
          httpGet:
            path: /healthcheck/kubedns
            port: 10054
            scheme: HTTP
          initialDelaySeconds: 60
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 5
        readinessProbe:
          failureThreshold: 3
          httpGet:
            path: /readiness
            port: 8081
            scheme: HTTP
          initialDelaySeconds: 3
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 5
        resources:
          limits:
            memory: 170Mi
          requests:
            cpu: 100m
            memory: 70Mi
      - name: dnsmasq
        image: {{ .ImageRepository }}/k8s-dns-dnsmasq-{{ .Arch }}:{{ .Version }}
        imagePullPolicy: IfNotPresent
        args:
        - --cache-size=1000
        - --no-resolv
        - --server=127.0.0.1#10053
        - --log-facility=-
        ports:
        - containerPort: 53
          name: dns
          protocol: UDP
        - containerPort: 53
          name: dns-tcp
          protocol: TCP
        livenessProbe:
          failureThreshold: 5
          httpGet:
            path: /healthcheck/dnsmasq
            port: 10054
            scheme: HTTP
          initialDelaySeconds: 60
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 5
        resources:
          requests:
            cpu: 150m
            memory: 10Mi
      - name: sidecar
        image: {{ .ImageRepository }}/k8s-dns-sidecar-{{ .Arch }}:{{ .Version }}
        imagePullPolicy: IfNotPresent
        args:
        - --v=2
        - --logtostderr
        - --probe=kubedns,127.0.0.1:10053,kubernetes.default.svc.{{ .DNSDomain }},5,A
        - --probe=dnsmasq,127.0.0.1:53,kubernetes.default.svc.{{ .DNSDomain }},5,A
        ports:
        - containerPort: 10054
          name: metrics
          protocol: TCP
        livenessProbe:
          failureThreshold: 5
          httpGet:
            path: /metrics
            port: 10054
            scheme: HTTP
          initialDelaySeconds: 60
          periodSeconds: 10
          successThreshold: 1
          timeoutSeconds: 5
        resources:
          requests:
            cpu: 10m
            memory: 20Mi
      dnsPolicy: Default
      serviceAccountName: kube-dns
      # tolerations:
      # - key: dedicated
      #   value: master
      #   effect: NoSchedule
      # TODO: Remove this affinity field as soon as we are using manifest lists
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: beta.kubernetes.io/arch
                operator: In
                values:
                - {{ .Arch }}
`

	KubeDNSService = `
apiVersion: v1
kind: Service
metadata:
  labels:
    k8s-app: kube-dns
    kubernetes.io/cluster-service: "true"
    kubernetes.io/name: "KubeDNS"
  name: kube-dns
  namespace: kube-system
spec:
  clusterIP: {{ .DNSIP }}
  ports:
  - name: dns
    port: 53
    protocol: UDP
    targetPort: 53
  - name: dns-tcp
    port: 53
    protocol: TCP
    targetPort: 53
  selector:
    k8s-app: kube-dns
`
)
