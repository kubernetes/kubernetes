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
	"testing"

	"sigs.k8s.io/kustomize/pkg/kusttest"
)

func TestNamespacedGenerator(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app")
	th.WriteK("/app", `
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
configMapGenerator:
- name: the-non-default-namespace-map
  namespace: non-default
  literals:
  - altGreeting=Good Morning from non-default namespace!
  - enableRisky="false"
- name: the-map
  literals:
  - altGreeting=Good Morning from default namespace!
  - enableRisky="false"

secretGenerator:
- name: the-non-default-namespace-secret
  namespace: non-default
  literals:
  - password.txt=verySecret
- name: the-secret
  literals:
  - password.txt=anotherSecret
`)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
data:
  altGreeting: Good Morning from non-default namespace!
  enableRisky: "false"
kind: ConfigMap
metadata:
  name: the-non-default-namespace-map-b6h49k7mt8
  namespace: non-default
---
apiVersion: v1
data:
  altGreeting: Good Morning from default namespace!
  enableRisky: "false"
kind: ConfigMap
metadata:
  name: the-map-4959m5tm6c
---
apiVersion: v1
data:
  password.txt: dmVyeVNlY3JldA==
kind: Secret
metadata:
  name: the-non-default-namespace-secret-h8d9hkgtb9
  namespace: non-default
type: Opaque
---
apiVersion: v1
data:
  password.txt: YW5vdGhlclNlY3JldA==
kind: Secret
metadata:
  name: the-secret-fgb45h45bh
type: Opaque
`)
}
