/*
Copyright 2018 The Kubernetes Authors.

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

package target_test

import (
	"strings"
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
)

func makeCommonFileForMultiplePatchTest(th *kusttest_test.KustTestHarness) {
	th.WriteK("/app/base", `
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namePrefix: team-foo-
commonLabels:
  app: mynginx
  org: example.com
  team: foo
commonAnnotations:
  note: This is a test annotation
resources:
  - deployment.yaml
  - service.yaml
configMapGenerator:
- name: configmap-in-base
  literals:
  - foo=bar
`)
	th.WriteF("/app/base/deployment.yaml", `
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx
        volumeMounts:
        - name: nginx-persistent-storage
          mountPath: /tmp/ps
      volumes:
      - name: nginx-persistent-storage
        emptyDir: {}
      - configMap:
          name: configmap-in-base
        name: configmap-in-base
`)
	th.WriteF("/app/base/service.yaml", `
apiVersion: v1
kind: Service
metadata:
  name: nginx
  labels:
    app: nginx
spec:
  ports:
    - port: 80
  selector:
    app: nginx
`)
	th.WriteK("/app/overlay/staging", `
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namePrefix: staging-
commonLabels:
  env: staging
patchesStrategicMerge:
  - deployment-patch1.yaml
  - deployment-patch2.yaml
resources:
  - ../../base
configMapGenerator:
- name: configmap-in-overlay
  literals:
  - hello=world
`)
}

func TestMultiplePatchesNoConflict(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/overlay/staging")
	makeCommonFileForMultiplePatchTest(th)
	th.WriteF("/app/overlay/staging/deployment-patch1.yaml", `
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: nginx
spec:
  template:
    spec:
      containers:
      - name: nginx
        image: nginx:latest
        env:
        - name: ENVKEY
          value: ENVVALUE
      volumes:
      - name: nginx-persistent-storage
        emptyDir: null
        gcePersistentDisk:
          pdName: nginx-persistent-storage
      - configMap:
          name: configmap-in-overlay
        name: configmap-in-overlay
`)
	th.WriteF("/app/overlay/staging/deployment-patch2.yaml", `
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: nginx
spec:
  template:
    spec:
      containers:
      - name: nginx
        env:
        - name: ANOTHERENV
          value: FOO
      - name: sidecar
        image: sidecar
      volumes:
      - name: nginx-persistent-storage
`)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: mynginx
    env: staging
    org: example.com
    team: foo
  name: staging-team-foo-nginx
spec:
  selector:
    matchLabels:
      app: mynginx
      env: staging
      org: example.com
      team: foo
  template:
    metadata:
      annotations:
        note: This is a test annotation
      labels:
        app: mynginx
        env: staging
        org: example.com
        team: foo
    spec:
      containers:
      - env:
        - name: ANOTHERENV
          value: FOO
        - name: ENVKEY
          value: ENVVALUE
        image: nginx:latest
        name: nginx
        volumeMounts:
        - mountPath: /tmp/ps
          name: nginx-persistent-storage
      - image: sidecar
        name: sidecar
      volumes:
      - gcePersistentDisk:
          pdName: nginx-persistent-storage
        name: nginx-persistent-storage
      - configMap:
          name: staging-configmap-in-overlay-k7cbc75tg8
        name: configmap-in-overlay
      - configMap:
          name: staging-team-foo-configmap-in-base-g7k6gt2889
        name: configmap-in-base
---
apiVersion: v1
kind: Service
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: mynginx
    env: staging
    org: example.com
    team: foo
  name: staging-team-foo-nginx
spec:
  ports:
  - port: 80
  selector:
    app: mynginx
    env: staging
    org: example.com
    team: foo
---
apiVersion: v1
data:
  foo: bar
kind: ConfigMap
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: mynginx
    env: staging
    org: example.com
    team: foo
  name: staging-team-foo-configmap-in-base-g7k6gt2889
---
apiVersion: v1
data:
  hello: world
kind: ConfigMap
metadata:
  labels:
    env: staging
  name: staging-configmap-in-overlay-k7cbc75tg8
`)
}

func TestMultiplePatchesWithConflict(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/overlay/staging")
	makeCommonFileForMultiplePatchTest(th)
	th.WriteF("/app/overlay/staging/deployment-patch1.yaml", `
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: nginx
spec:
  template:
    spec:
      containers:
      - name: nginx
        env:
        - name: ENABLE_FEATURE_FOO
          value: TRUE
      volumes:
      - name: nginx-persistent-storage
        emptyDir: null
        gcePersistentDisk:
          pdName: nginx-persistent-storage
      - configMap:
          name: configmap-in-overlay
        name: configmap-in-overlay
`)
	th.WriteF("/app/overlay/staging/deployment-patch2.yaml", `
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: nginx
spec:
  template:
    spec:
      containers:
      - name: nginx
        env:
        - name: ENABLE_FEATURE_FOO
          value: FALSE
`)
	_, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err == nil {
		t.Fatalf("expected conflict")
	}
	if !strings.Contains(
		err.Error(), "conflict between ") {
		t.Fatalf("Unexpected err: %v", err)
	}
}
