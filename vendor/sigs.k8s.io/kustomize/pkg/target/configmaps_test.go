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

// Generate a Secret and a ConfigMap from the same data
// to compare the result.
func TestGeneratorBasics(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app")
	th.WriteK("/app", `
namePrefix: blah-
configMapGenerator:
- name: bob
  literals:
  - fruit=apple
  - vegetable=broccoli
  envs:
  - foo.env
  files:
  - passphrase=phrase.dat
  - forces.txt
- name: json
  literals:
  - 'v2=[{"path": "var/druid/segment-cache"}]'
  - >- 
    druid_segmentCache_locations=[{"path": 
    "var/druid/segment-cache", 
    "maxSize": 32000000000, 
    "freeSpacePercent": 1.0}]
secretGenerator:
- name: bob
  literals:
  - fruit=apple
  - vegetable=broccoli
  envs:
  - foo.env
  files:
  - passphrase=phrase.dat
  - forces.txt
`)
	th.WriteF("/app/foo.env", `
MOUNTAIN=everest
OCEAN=pacific
`)
	th.WriteF("/app/phrase.dat", `
Life is short.
But the years are long.
Not while the evil days come not.
`)
	th.WriteF("/app/forces.txt", `
gravitational
electromagnetic
strong nuclear
weak nuclear
`)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
data:
  MOUNTAIN: everest
  OCEAN: pacific
  forces.txt: |2

    gravitational
    electromagnetic
    strong nuclear
    weak nuclear
  fruit: apple
  passphrase: |2

    Life is short.
    But the years are long.
    Not while the evil days come not.
  vegetable: broccoli
kind: ConfigMap
metadata:
  name: blah-bob-k772g5db55
---
apiVersion: v1
data:
  druid_segmentCache_locations: '[{"path":  "var/druid/segment-cache",  "maxSize":
    32000000000,  "freeSpacePercent": 1.0}]'
  v2: '[{"path": "var/druid/segment-cache"}]'
kind: ConfigMap
metadata:
  name: blah-json-9gtcc2fgb4
---
apiVersion: v1
data:
  MOUNTAIN: ZXZlcmVzdA==
  OCEAN: cGFjaWZpYw==
  forces.txt: CmdyYXZpdGF0aW9uYWwKZWxlY3Ryb21hZ25ldGljCnN0cm9uZyBudWNsZWFyCndlYWsgbnVjbGVhcgo=
  fruit: YXBwbGU=
  passphrase: CkxpZmUgaXMgc2hvcnQuCkJ1dCB0aGUgeWVhcnMgYXJlIGxvbmcuCk5vdCB3aGlsZSB0aGUgZXZpbCBkYXlzIGNvbWUgbm90Lgo=
  vegetable: YnJvY2NvbGk=
kind: Secret
metadata:
  name: blah-bob-gmc2824f4b
type: Opaque
`)
}

// TODO: These should be errors instead.
func TestGeneratorRepeatsInKustomization(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app")
	th.WriteK("/app", `
namePrefix: blah-
configMapGenerator:
- name: bob
  behavior: create
  literals:
  - bean=pinto
  - star=wolf-rayet
  literals:
  - fruit=apple
  - vegetable=broccoli
  files:
  - forces.txt
  files:
  - nobles=nobility.txt
`)
	th.WriteF("/app/forces.txt", `
gravitational
electromagnetic
strong nuclear
weak nuclear
`)
	th.WriteF("/app/nobility.txt", `
helium
neon
argon
krypton
xenon
radon
`)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
data:
  fruit: apple
  nobles: |2

    helium
    neon
    argon
    krypton
    xenon
    radon
  vegetable: broccoli
kind: ConfigMap
metadata:
  name: blah-bob-gfkcbk5ckf
`)
}

func TestGeneratorOverlays(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/overlay")
	th.WriteK("/app/base1", `
namePrefix: p1-
configMapGenerator:
- name: com1
  behavior: create
  literals:
  - from=base
`)
	th.WriteK("/app/base2", `
namePrefix: p2-
configMapGenerator:
- name: com2
  behavior: create
  literals:
  - from=base
`)
	th.WriteK("/app/overlay/o1", `
resources:
- ../../base1
configMapGenerator:
- name: com1
  behavior: merge
  literals:
  - from=overlay
`)
	th.WriteK("/app/overlay/o2", `
resources:
- ../../base2
configMapGenerator:
- name: com2
  behavior: merge
  literals:
  - from=overlay
`)
	th.WriteK("/app/overlay", `
resources:
- o1
- o2
configMapGenerator:
- name: com1
  behavior: merge
  literals:
  - foo=bar
  - baz=qux
`)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
data:
  baz: qux
  foo: bar
  from: overlay
kind: ConfigMap
metadata:
  annotations: {}
  labels: {}
  name: p1-com1-dhbbm922gd
---
apiVersion: v1
data:
  from: overlay
kind: ConfigMap
metadata:
  annotations: {}
  labels: {}
  name: p2-com2-c4b8md75k9
`)
}
