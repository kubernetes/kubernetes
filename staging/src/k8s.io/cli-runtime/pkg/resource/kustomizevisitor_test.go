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

package resource

import (
	"github.com/davecgh/go-spew/spew"
	"sigs.k8s.io/kustomize/api/konfig"
	"sigs.k8s.io/kustomize/kyaml/filesys"
	"testing"
)

const (
	kustomizationContent1 = `
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namePrefix: foo-
nameSuffix: -bar
namespace: ns1
commonLabels:
  app: nginx
commonAnnotations:
  note: This is a test annotation
resources:
  - deployment.yaml
  - namespace.yaml
configMapGenerator:
- name: literalConfigMap
  literals:
  - DB_USERNAME=admin
  - DB_PASSWORD=somepw
secretGenerator:
- name: secret
  literals:
    - DB_USERNAME=admin
    - DB_PASSWORD=somepw
  type: Opaque
patchesJson6902:
- target:
    group: apps
    version: v1
    kind: Deployment
    name: dply1
  path: jsonpatch.json
`
	deploymentContent = `
apiVersion: apps/v1
metadata:
  name: dply1
kind: Deployment
`
	namespaceContent = `
apiVersion: v1
kind: Namespace
metadata:
  name: ns1
`
	jsonpatchContent = `[
    {"op": "add", "path": "/spec/replica", "value": "3"}
]`

	expectedContent = `apiVersion: v1
kind: Namespace
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: nginx
  name: ns1
---
apiVersion: v1
data:
  DB_PASSWORD: somepw
  DB_USERNAME: admin
kind: ConfigMap
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: nginx
  name: foo-literalConfigMap-bar-g5f6t456f5
  namespace: ns1
---
apiVersion: v1
data:
  DB_PASSWORD: c29tZXB3
  DB_USERNAME: YWRtaW4=
kind: Secret
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: nginx
  name: foo-secret-bar-82c2g5f8f6
  namespace: ns1
type: Opaque
---
apiVersion: apps/v1
kind: Deployment
metadata:
  annotations:
    note: This is a test annotation
  labels:
    app: nginx
  name: foo-dply1-bar
  namespace: ns1
spec:
  replica: "3"
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      annotations:
        note: This is a test annotation
      labels:
        app: nginx
`
)

func TestKustomizeVisitor(t *testing.T) {
	fSys := filesys.MakeFsInMemory()
	fSys.WriteFile(
		konfig.DefaultKustomizationFileName(),
		[]byte(kustomizationContent1))
	fSys.WriteFile("deployment.yaml", []byte(deploymentContent))
	fSys.WriteFile("namespace.yaml", []byte(namespaceContent))
	fSys.WriteFile("jsonpatch.json", []byte(jsonpatchContent))
	b := newDefaultBuilder()
	kv := KustomizeVisitor{
		mapper:  b.mapper,
		dirPath: ".",
		schema:  b.schema,
		fSys:    fSys,
	}
	tv := &testVisitor{}
	if err := kv.Visit(tv.Handle); err != nil {
		t.Fatal(err)
	}
	if len(tv.Infos) != 4 {
		t.Fatal(spew.Sdump(tv.Infos))
	}
	if string(kv.yml) != expectedContent {
		t.Fatalf("expected:\n%s\nbut got:\n%s", expectedContent, string(kv.yml))
	}
}
