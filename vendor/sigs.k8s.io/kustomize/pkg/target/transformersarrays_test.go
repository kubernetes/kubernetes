// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package target_test

import (
	"testing"

	kusttest_test "sigs.k8s.io/kustomize/pkg/kusttest"
)

func makeStatefulSetKustomization(th *kusttest_test.KustTestHarness) {
	th.WriteK("/app", `
commonLabels:
  notIn: arrays
resources:
- statefulset.yaml
- statefulset-with-template.yaml
`)
	th.WriteF("/app/statefulset.yaml", `
kind: StatefulSet
apiVersion: apps/v1
metadata:
  name: test
  labels:
    notIn: arrays
spec:
  serviceName: test
  replicas: 1
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - name: nginx
        image: k8s.gcr.io/nginx-slim:0.8
        ports:
        - containerPort: 80
          name: web
`)
	th.WriteF("/app/statefulset-with-template.yaml", `
kind: StatefulSet
apiVersion: apps/v1
metadata:
  name: persisted-test
  labels:
    notIn: arrays
spec:
  serviceName: test
  replicas: 1
  selector:
    matchLabels:
      app: test
  template:
    metadata:
      labels:
        app: test
    spec:
      containers:
      - name: nginx
        image: k8s.gcr.io/nginx-slim:0.8
        ports:
        - containerPort: 80
          name: web
        volumeMounts:
        - name: www
          mountPath: /usr/share/nginx/html
        - name: data
          mountPath: /usr/share/nginx/data
  volumeClaimTemplates:
  - metadata:
      name: www
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: my-storage-class
      resources:
        requests:
          storage: 1Gi
  - metadata:
      name: data
    spec:
      accessModes: [ "ReadWriteOnce" ]
      storageClassName: my-storage-class
      resources:
        requests:
          storage: 100Gi
`)
}

func TestTransformersNoCreateArrays(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app")
	makeStatefulSetKustomization(th)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: apps/v1
kind: StatefulSet
metadata:
  labels:
    notIn: arrays
  name: test
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test
      notIn: arrays
  serviceName: test
  template:
    metadata:
      labels:
        app: test
        notIn: arrays
    spec:
      containers:
      - image: k8s.gcr.io/nginx-slim:0.8
        name: nginx
        ports:
        - containerPort: 80
          name: web
---
apiVersion: apps/v1
kind: StatefulSet
metadata:
  labels:
    notIn: arrays
  name: persisted-test
spec:
  replicas: 1
  selector:
    matchLabels:
      app: test
      notIn: arrays
  serviceName: test
  template:
    metadata:
      labels:
        app: test
        notIn: arrays
    spec:
      containers:
      - image: k8s.gcr.io/nginx-slim:0.8
        name: nginx
        ports:
        - containerPort: 80
          name: web
        volumeMounts:
        - mountPath: /usr/share/nginx/html
          name: www
        - mountPath: /usr/share/nginx/data
          name: data
  volumeClaimTemplates:
  - metadata:
      labels:
        notIn: arrays
      name: www
    spec:
      accessModes:
      - ReadWriteOnce
      resources:
        requests:
          storage: 1Gi
      storageClassName: my-storage-class
  - metadata:
      labels:
        notIn: arrays
      name: data
    spec:
      accessModes:
      - ReadWriteOnce
      resources:
        requests:
          storage: 100Gi
      storageClassName: my-storage-class
`)
}
