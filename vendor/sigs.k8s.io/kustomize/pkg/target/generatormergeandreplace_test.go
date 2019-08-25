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

func TestSimpleBase(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/base")
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
  - service.yaml
  - deployment.yaml
  - networkpolicy.yaml
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
	th.WriteF("/app/base/networkpolicy.yaml", `
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: nginx
spec:
  podSelector:
    matchExpressions:
      - {key: app, operator: In, values: [test]}
  ingress:
    - from:
        - podSelector:
            matchLabels:
              app: nginx
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
`)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
kind: Service
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: mynginx
    org: example.com
    team: foo
  name: team-foo-nginx
spec:
  ports:
  - port: 80
  selector:
    app: mynginx
    org: example.com
    team: foo
---
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: mynginx
    org: example.com
    team: foo
  name: team-foo-nginx
spec:
  selector:
    matchLabels:
      app: mynginx
      org: example.com
      team: foo
  template:
    metadata:
      annotations:
        note: This is a test annotation
      labels:
        app: mynginx
        org: example.com
        team: foo
    spec:
      containers:
      - image: nginx
        name: nginx
---
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: mynginx
    org: example.com
    team: foo
  name: team-foo-nginx
spec:
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: mynginx
          org: example.com
          team: foo
  podSelector:
    matchExpressions:
    - key: app
      operator: In
      values:
      - test
`)
}

func makeBaseWithGenerators(th *kusttest_test.KustTestHarness) {
	th.WriteK("/app", `
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
secretGenerator:
- name: secret-in-base
  literals:
  - username=admin
  - password=somepw
`)
	th.WriteF("/app/deployment.yaml", `
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
	th.WriteF("/app/service.yaml", `
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
}

func TestBaseWithGeneratorsAlone(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app")
	makeBaseWithGenerators(th)
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
    org: example.com
    team: foo
  name: team-foo-nginx
spec:
  selector:
    matchLabels:
      app: mynginx
      org: example.com
      team: foo
  template:
    metadata:
      annotations:
        note: This is a test annotation
      labels:
        app: mynginx
        org: example.com
        team: foo
    spec:
      containers:
      - image: nginx
        name: nginx
        volumeMounts:
        - mountPath: /tmp/ps
          name: nginx-persistent-storage
      volumes:
      - emptyDir: {}
        name: nginx-persistent-storage
      - configMap:
          name: team-foo-configmap-in-base-bbdmdh7m8t
        name: configmap-in-base
---
apiVersion: v1
kind: Service
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: mynginx
    org: example.com
    team: foo
  name: team-foo-nginx
spec:
  ports:
  - port: 80
  selector:
    app: mynginx
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
    org: example.com
    team: foo
  name: team-foo-configmap-in-base-bbdmdh7m8t
---
apiVersion: v1
data:
  password: c29tZXB3
  username: YWRtaW4=
kind: Secret
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: mynginx
    org: example.com
    team: foo
  name: team-foo-secret-in-base-tkm7hhtf8d
type: Opaque
`)
}

func TestMergeAndReplaceGenerators(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/overlay")
	makeBaseWithGenerators(th)
	th.WriteF("/overlay/deployment.yaml", `
apiVersion: apps/v1beta2
kind: Deployment
metadata:
  name: nginx
spec:
  template:
    spec:
      volumes:
      - name: nginx-persistent-storage
        emptyDir: null
        gcePersistentDisk:
          pdName: nginx-persistent-storage
      - configMap:
          name: configmap-in-overlay
        name: configmap-in-overlay
`)
	th.WriteK("/overlay", `
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namePrefix: staging-
commonLabels:
  env: staging
  team: override-foo
patchesStrategicMerge:
- deployment.yaml
resources:
- ../app
configMapGenerator:
- name: configmap-in-overlay
  literals:
  - hello=world
- name: configmap-in-base
  behavior: replace
  literals:
  - foo=override-bar
secretGenerator:
- name: secret-in-base
  behavior: merge
  literals:
  - proxy=haproxy
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
    team: override-foo
  name: staging-team-foo-nginx
spec:
  selector:
    matchLabels:
      app: mynginx
      env: staging
      org: example.com
      team: override-foo
  template:
    metadata:
      annotations:
        note: This is a test annotation
      labels:
        app: mynginx
        env: staging
        org: example.com
        team: override-foo
    spec:
      containers:
      - image: nginx
        name: nginx
        volumeMounts:
        - mountPath: /tmp/ps
          name: nginx-persistent-storage
      volumes:
      - gcePersistentDisk:
          pdName: nginx-persistent-storage
        name: nginx-persistent-storage
      - configMap:
          name: staging-configmap-in-overlay-k7cbc75tg8
        name: configmap-in-overlay
      - configMap:
          name: staging-team-foo-configmap-in-base-gh9d7t85gb
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
    team: override-foo
  name: staging-team-foo-nginx
spec:
  ports:
  - port: 80
  selector:
    app: mynginx
    env: staging
    org: example.com
    team: override-foo
---
apiVersion: v1
data:
  foo: override-bar
kind: ConfigMap
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: mynginx
    env: staging
    org: example.com
    team: override-foo
  name: staging-team-foo-configmap-in-base-gh9d7t85gb
---
apiVersion: v1
data:
  password: c29tZXB3
  proxy: aGFwcm94eQ==
  username: YWRtaW4=
kind: Secret
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: mynginx
    env: staging
    org: example.com
    team: override-foo
  name: staging-team-foo-secret-in-base-c8db7gk2m2
type: Opaque
---
apiVersion: v1
data:
  hello: world
kind: ConfigMap
metadata:
  labels:
    env: staging
    team: override-foo
  name: staging-configmap-in-overlay-k7cbc75tg8
`)
}

func TestGeneratingIntoNamespaces(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app")
	th.WriteK("/app", `
configMapGenerator:
- name: test
  namespace: bob
  literals:
  - key=value
- name: test
  namespace: kube-system
  literals:
  - key=value
`)
	_, err := th.MakeKustTarget().MakeCustomizedResMap()
	// Document #1155
	// This actually should be nil; it should work, and
	// have some expected output.
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "must merge or replace") {
		t.Fatalf("unexpected error %v", err)
	}
}
