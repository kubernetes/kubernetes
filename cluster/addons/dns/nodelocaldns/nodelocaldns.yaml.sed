# Copyright 2018 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Warning: This is a file generated from the base underscore template file: nodelocaldns.yaml.base

apiVersion: v1
kind: ServiceAccount
metadata:
  name: nodelocaldns
  namespace: kube-system
  labels:
    kubernetes.io/cluster-service: "true"
    addonmanager.kubernetes.io/mode: Reconcile
---

apiVersion: v1
kind: ConfigMap
metadata:
  name: nodelocaldns
  namespace: kube-system
  labels:
    addonmanager.kubernetes.io/mode: EnsureExists

data:
  Corefile: |
    $DNS_DOMAIN:53 {
        errors
        cache 30
        reload
        loop
        bind $LOCAL_DNS_IP
        forward . $DNS_SERVER_IP {
                force_tcp
        }
        prometheus :9253
        health $LOCAL_DNS_IP:8080
        }
    in-addr.arpa:53 {
        errors
        cache 30
        reload
        loop
        bind $LOCAL_DNS_IP
        forward . $DNS_SERVER_IP {
                force_tcp
        }
        prometheus :9253
        }
    ip6.arpa:53 {
        errors
        cache 30
        reload
        loop
        bind $LOCAL_DNS_IP
        forward . $DNS_SERVER_IP {
                force_tcp
        }
        prometheus :9253
        }
    .:53 {
        errors
        cache 30
        reload
        loop
        bind $LOCAL_DNS_IP
        forward . /etc/resolv.conf {
                force_tcp
        }
        prometheus :9253
        }
---
apiVersion: apps/v1
kind: DaemonSet
metadata:
  name: nodelocaldns
  namespace: kube-system
  labels:
    k8s-app: kube-dns
    kubernetes.io/cluster-service: "true"
    addonmanager.kubernetes.io/mode: Reconcile
spec:
  selector:
    matchLabels:
      k8s-app: nodelocaldns
  template:
    metadata:
       labels:
          k8s-app: nodelocaldns
    spec:
      priorityClassName: system-node-critical
      serviceAccountName: nodelocaldns
      hostNetwork: true
      dnsPolicy: Default  # Don't use cluster DNS.
      tolerations:
      - key: "CriticalAddonsOnly"
        operator: "Exists"
      containers:
      - name: node-cache
        image: k8s.gcr.io/k8s-dns-node-cache:1.15.0
        resources:
          limits:
            memory: 30Mi
          requests:
            cpu: 25m
            memory: 5Mi
        args: [ "-localip", "$LOCAL_DNS_IP", "-conf", "/etc/coredns/Corefile" ]
        securityContext:
                privileged: true
        ports:
        - containerPort: 53
          name: dns
          protocol: UDP
        - containerPort: 53
          name: dns-tcp
          protocol: TCP
        - containerPort: 9253
          name: metrics
          protocol: TCP
        livenessProbe:
          httpGet:
            host: $LOCAL_DNS_IP
            path: /health
            port: 8080
            scheme: HTTP
          initialDelaySeconds: 60
          timeoutSeconds: 5
          successThreshold: 1
          failureThreshold: 3
        volumeMounts:
        - name: config-volume
          mountPath: /etc/coredns
      volumes:
        - name: config-volume
          configMap:
            name: nodelocaldns
            items:
            - key: Corefile
              path: Corefile
      terminationGracePeriodSeconds: 30
