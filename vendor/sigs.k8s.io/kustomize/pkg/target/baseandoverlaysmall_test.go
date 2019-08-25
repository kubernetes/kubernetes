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
	"sigs.k8s.io/kustomize/pkg/loader"
	"sigs.k8s.io/kustomize/pkg/plugins"
)

func TestOrderPreserved(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/prod")
	th.WriteK("/app/base", `
namePrefix: b-
resources:
- namespace.yaml
- role.yaml
- service.yaml
- deployment.yaml
`)
	th.WriteF("/app/base/service.yaml", `
apiVersion: v1
kind: Service
metadata:
  name: myService
`)
	th.WriteF("/app/base/namespace.yaml", `
apiVersion: v1
kind: Namespace
metadata:
  name: myNs
`)
	th.WriteF("/app/base/role.yaml", `
apiVersion: v1
kind: Role
metadata:
  name: myRole
`)
	th.WriteF("/app/base/deployment.yaml", `
apiVersion: v1
kind: Deployment
metadata:
  name: myDep
`)
	th.WriteK("/app/prod", `
namePrefix: p-
resources:
- ../base
- service.yaml
- namespace.yaml
`)
	th.WriteF("/app/prod/service.yaml", `
apiVersion: v1
kind: Service
metadata:
  name: myService2
`)
	th.WriteF("/app/prod/namespace.yaml", `
apiVersion: v1
kind: Namespace
metadata:
  name: myNs2
`)

	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
kind: Namespace
metadata:
  name: p-b-myNs
---
apiVersion: v1
kind: Role
metadata:
  name: p-b-myRole
---
apiVersion: v1
kind: Service
metadata:
  name: p-b-myService
---
apiVersion: v1
kind: Deployment
metadata:
  name: p-b-myDep
---
apiVersion: v1
kind: Service
metadata:
  name: p-myService2
---
apiVersion: v1
kind: Namespace
metadata:
  name: p-myNs2
`)
}

func TestBaseInResourceList(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/prod")
	th.WriteK("/app/prod", `
namePrefix: b-
resources:
- ../base
`)
	th.WriteK("/app/base", `
namePrefix: a-
resources:
- service.yaml
`)
	th.WriteF("/app/base/service.yaml", `
apiVersion: v1
kind: Service
metadata:
  name: myService
spec:
  selector:
    backend: bungie
`)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
kind: Service
metadata:
  name: b-a-myService
spec:
  selector:
    backend: bungie
`)
}

func writeSmallBase(th *kusttest_test.KustTestHarness) {
	th.WriteK("/app/base", `
namePrefix: a-
commonLabels:
  app: myApp
resources:
- deployment.yaml
- service.yaml
`)
	th.WriteF("/app/base/service.yaml", `
apiVersion: v1
kind: Service
metadata:
  name: myService
spec:
  selector:
    backend: bungie
  ports:
    - port: 7002
`)
	th.WriteF("/app/base/deployment.yaml", `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myDeployment
spec:
  template:
    metadata:
      labels:
        backend: awesome
    spec:
      containers:
      - name: whatever
        image: whatever
`)
}

func TestSmallBase(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/base")
	writeSmallBase(th)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: myApp
  name: a-myDeployment
spec:
  selector:
    matchLabels:
      app: myApp
  template:
    metadata:
      labels:
        app: myApp
        backend: awesome
    spec:
      containers:
      - image: whatever
        name: whatever
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app: myApp
  name: a-myService
spec:
  ports:
  - port: 7002
  selector:
    app: myApp
    backend: bungie
`)
}

func TestSmallOverlay(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/overlay")
	writeSmallBase(th)
	th.WriteK("/app/overlay", `
namePrefix: b-
commonLabels:
  env: prod
resources:
- ../base
patchesStrategicMerge:
- deployment/deployment.yaml
images:
- name: whatever
  newTag: 1.8.0
`)

	th.WriteF("/app/overlay/configmap/app.env", `
DB_USERNAME=admin
DB_PASSWORD=somepw
`)
	th.WriteF("/app/overlay/configmap/app-init.ini", `
FOO=bar
BAR=baz
`)
	th.WriteF("/app/overlay/deployment/deployment.yaml", `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myDeployment
spec:
  replicas: 1000
`)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: myApp
    env: prod
  name: b-a-myDeployment
spec:
  replicas: 1000
  selector:
    matchLabels:
      app: myApp
      env: prod
  template:
    metadata:
      labels:
        app: myApp
        backend: awesome
        env: prod
    spec:
      containers:
      - image: whatever:1.8.0
        name: whatever
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app: myApp
    env: prod
  name: b-a-myService
spec:
  ports:
  - port: 7002
  selector:
    app: myApp
    backend: bungie
    env: prod
`)
}

func TestSharedPatchDisAllowed(t *testing.T) {
	th := kusttest_test.NewKustTestHarnessFull(
		t, "/app/overlay",
		loader.RestrictionRootOnly, plugins.DefaultPluginConfig())
	writeSmallBase(th)
	th.WriteK("/app/overlay", `
commonLabels:
  env: prod
resources:
- ../base
patchesStrategicMerge:
- ../shared/deployment-patch.yaml
`)
	th.WriteF("/app/shared/deployment-patch.yaml", `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myDeployment
spec:
  replicas: 1000
`)
	_, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(
		err.Error(),
		"security; file '/app/shared/deployment-patch.yaml' is not in or below '/app/overlay'") {
		t.Fatalf("unexpected error: %s", err)
	}
}

func TestSharedPatchAllowed(t *testing.T) {
	th := kusttest_test.NewKustTestHarnessFull(
		t, "/app/overlay",
		loader.RestrictionNone, plugins.DefaultPluginConfig())
	writeSmallBase(th)
	th.WriteK("/app/overlay", `
commonLabels:
  env: prod
resources:
- ../base
patchesStrategicMerge:
- ../shared/deployment-patch.yaml
`)
	th.WriteF("/app/shared/deployment-patch.yaml", `
apiVersion: apps/v1
kind: Deployment
metadata:
  name: myDeployment
spec:
  replicas: 1000
`)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: myApp
    env: prod
  name: a-myDeployment
spec:
  replicas: 1000
  selector:
    matchLabels:
      app: myApp
      env: prod
  template:
    metadata:
      labels:
        app: myApp
        backend: awesome
        env: prod
    spec:
      containers:
      - image: whatever
        name: whatever
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app: myApp
    env: prod
  name: a-myService
spec:
  ports:
  - port: 7002
  selector:
    app: myApp
    backend: bungie
    env: prod
`)
}

func TestSmallOverlayJSONPatch(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/overlay")
	writeSmallBase(th)
	th.WriteK("/app/overlay", `
resources:
- ../base
patchesJson6902:
- target:
    version: v1
    kind: Service
    name: myService # BUG (https://github.com/kubernetes-sigs/kustomize/issues/972): this should be a-myService, because that is what the output for the base contains
  path: service-patch.yaml
`)

	th.WriteF("/app/overlay/service-patch.yaml", `
- op: add
  path: /spec/selector/backend
  value: beagle
`)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: apps/v1
kind: Deployment
metadata:
  labels:
    app: myApp
  name: a-myDeployment
spec:
  selector:
    matchLabels:
      app: myApp
  template:
    metadata:
      labels:
        app: myApp
        backend: awesome
    spec:
      containers:
      - image: whatever
        name: whatever
---
apiVersion: v1
kind: Service
metadata:
  labels:
    app: myApp
  name: a-myService
spec:
  ports:
  - port: 7002
  selector:
    app: myApp
    backend: beagle
`)
}
