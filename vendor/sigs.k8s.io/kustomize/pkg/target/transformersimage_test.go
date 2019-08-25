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

package target_test

import (
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
)

func makeTransfomersImageBase(th *kusttest_test.KustTestHarness) {
	th.WriteK("/app/base", `
resources:
- deploy1.yaml
- random.yaml
images:
- name: nginx
  newTag: v2
- name: my-nginx
  newTag: previous
- name: myprivaterepohostname:1234/my/image
  newTag: v1.0.1
- name: foobar
  digest: sha256:24a0c4b4
- name: alpine
  newName: myprivaterepohostname:1234/my/cool-alpine
- name: gcr.io:8080/my-project/my-cool-app
  newName: my-cool-app
- name: postgres
  newName: my-postgres
  newTag: v3
- name: docker
  newName: my-docker
  digest: sha256:25a0d4b4
`)
	th.WriteF("/app/base/deploy1.yaml", `
group: apps
apiVersion: v1
kind: Deployment
metadata:
  name: deploy1
spec:
  template:
    spec:
      initContainers:
      - name: nginx2
        image: my-nginx:1.8.0
      - name: init-alpine
        image: alpine:1.8.0
      containers:
      - name: ngnix
        image: nginx:1.7.9
      - name: repliaced-with-digest
        image: foobar:1
      - name: postgresdb
        image: postgres:1.8.0
`)
	th.WriteF("/app/base/random.yaml", `
kind: randomKind
metadata:
  name: random
spec:
  template:
    spec:
      containers:
      - name: ngnix1
        image: nginx
spec2:
  template:
    spec:
      containers:
      - name: nginx3
        image: nginx:v1
      - name: nginx4
        image: my-nginx:latest
spec3:
  template:
    spec:
      initContainers:
      - name: postgresdb
        image: postgres:alpine-9
      - name: init-docker
        image: docker:17-git
      - name: myImage
        image: myprivaterepohostname:1234/my/image:latest
      - name: myImage2
        image: myprivaterepohostname:1234/my/image
      - name: my-app
        image: my-app-image:v1
      - name: my-cool-app
        image: gcr.io:8080/my-project/my-cool-app:latest
`)
}

func TestTransfomersImageDefaultConfig(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/base")
	makeTransfomersImageBase(th)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
group: apps
kind: Deployment
metadata:
  name: deploy1
spec:
  template:
    spec:
      containers:
      - image: nginx:v2
        name: ngnix
      - image: foobar@sha256:24a0c4b4
        name: repliaced-with-digest
      - image: my-postgres:v3
        name: postgresdb
      initContainers:
      - image: my-nginx:previous
        name: nginx2
      - image: myprivaterepohostname:1234/my/cool-alpine:1.8.0
        name: init-alpine
---
kind: randomKind
metadata:
  name: random
spec:
  template:
    spec:
      containers:
      - image: nginx:v2
        name: ngnix1
spec2:
  template:
    spec:
      containers:
      - image: nginx:v2
        name: nginx3
      - image: my-nginx:previous
        name: nginx4
spec3:
  template:
    spec:
      initContainers:
      - image: my-postgres:v3
        name: postgresdb
      - image: my-docker@sha256:25a0d4b4
        name: init-docker
      - image: myprivaterepohostname:1234/my/image:v1.0.1
        name: myImage
      - image: myprivaterepohostname:1234/my/image:v1.0.1
        name: myImage2
      - image: my-app-image:v1
        name: my-app
      - image: my-cool-app:latest
        name: my-cool-app
`)
}

func makeTransfomersImageCustomBase(th *kusttest_test.KustTestHarness) {
	th.WriteK("/app/base", `
resources:
- custom.yaml
configurations:
- config/custom.yaml
images:
- name: nginx
  newTag: v2
- name: my-nginx
  newTag: previous
- name: myprivaterepohostname:1234/my/image
  newTag: v1.0.1
- name: foobar
  digest: sha256:24a0c4b4
- name: alpine
  newName: myprivaterepohostname:1234/my/cool-alpine
- name: gcr.io:8080/my-project/my-cool-app
  newName: my-cool-app
- name: postgres
  newName: my-postgres
  newTag: v3
- name: docker
  newName: my-docker
  digest: sha256:25a0d4b4
`)
	th.WriteF("/app/base/custom.yaml", `
kind: customKind
metadata:
  name: custom
spec:
  template:
    spec:
      myContainers:
      - name: ngnix1
        image: nginx
spec2:
  template:
    spec:
      myContainers:
      - name: nginx3
        image: nginx:v1
      - name: nginx4
        image: my-nginx:latest
spec3:
  template:
    spec:
      myInitContainers:
      - name: postgresdb
        image: postgres:alpine-9
      - name: init-docker
        image: docker:17-git
      - name: myImage
        image: myprivaterepohostname:1234/my/image:latest
      - name: myImage2
        image: myprivaterepohostname:1234/my/image
      - name: my-app
        image: my-app-image:v1
      - name: my-cool-app
        image: gcr.io:8080/my-project/my-cool-app:latest
`)
	th.WriteF("/app/base/config/custom.yaml", `
images:
- kind: Custom
  path: spec/template/spec/myContainers/image
- kind: Custom
  path: spec2/template/spec/myContainers/image
- kind: Custom
  path: spec3/template/spec/myInitContainers/image
`)
}
func TestTransfomersImageCustomConfig(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/base")
	makeTransfomersImageCustomBase(th)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
kind: customKind
metadata:
  name: custom
spec:
  template:
    spec:
      myContainers:
      - image: nginx
        name: ngnix1
spec2:
  template:
    spec:
      myContainers:
      - image: nginx:v1
        name: nginx3
      - image: my-nginx:latest
        name: nginx4
spec3:
  template:
    spec:
      myInitContainers:
      - image: postgres:alpine-9
        name: postgresdb
      - image: docker:17-git
        name: init-docker
      - image: myprivaterepohostname:1234/my/image:latest
        name: myImage
      - image: myprivaterepohostname:1234/my/image
        name: myImage2
      - image: my-app-image:v1
        name: my-app
      - image: gcr.io:8080/my-project/my-cool-app:latest
        name: my-cool-app
`)
}

func makeTransfomersImageKnativeBase(th *kusttest_test.KustTestHarness) {
	th.WriteK("/app/base", `
resources:
- knative.yaml
configurations:
- config/knative.yaml
images:
- name: solsa-echo
  newTag: foo
`)
	th.WriteF("/app/base/knative.yaml", `
apiVersion: serving.knative.dev/v1alpha1
kind: Service
metadata:
  name: echo
spec:
  runLatest:
    configuration:
      revisionTemplate:
        spec:
          container:
            image: solsa-echo
`)
	th.WriteF("/app/base/config/knative.yaml", `
images:
- path: spec/runLatest/configuration/revisionTemplate/spec/container/image
  apiVersion: serving.knative.dev/v1alpha1
  kind: Service
`)
}

func TestTransfomersImageKnativeConfig(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/base")
	makeTransfomersImageKnativeBase(th)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: serving.knative.dev/v1alpha1
kind: Service
metadata:
  name: echo
spec:
  runLatest:
    configuration:
      revisionTemplate:
        spec:
          container:
            image: solsa-echo:foo
`)
}
