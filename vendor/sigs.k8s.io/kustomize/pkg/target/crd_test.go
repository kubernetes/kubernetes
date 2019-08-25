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

func writeBaseWithCrd(th *kusttest_test.KustTestHarness) {
	th.WriteK("/app/base", `
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
crds:
- mycrd.json

resources:
- secret.yaml
- mykind.yaml
- bee.yaml

namePrefix: x-
`)
	th.WriteF("/app/base/bee.yaml", `
apiVersion: v1beta1
kind: Bee
metadata:
  name: bee
spec:
  action: fly
`)
	th.WriteF("/app/base/mykind.yaml", `
apiVersion: jingfang.example.com/v1beta1
kind: MyKind
metadata:
  name: mykind
spec:
  secretRef:
    name: crdsecret
  beeRef:
    name: bee
`)
	th.WriteF("/app/base/secret.yaml", `
apiVersion: v1
kind: Secret
metadata:
  name: crdsecret
data:
  PATH: yellowBrickRoad
`)
	th.WriteF("/app/base/mycrd.json", `
{
  "github.com/example/pkg/apis/jingfang/v1beta1.Bee": {
    "Schema": {
      "description": "Bee",
      "properties": {
        "apiVersion": {
          "description": "APIVersion defines the versioned schema of this representation of an object.",
          "type": "string"
        },
        "kind": {
          "description": "Kind is a string value representing the REST resource this object represents.",
          "type": "string"
        },
        "metadata": {
          "$ref": "k8s.io/apimachinery/pkg/apis/meta/v1.ObjectMeta"
        },
        "spec": {
          "$ref": "github.com/example/pkg/apis/jingfang/v1beta1.BeeSpec"
        },
        "status": {
          "$ref": "github.com/example/pkg/apis/jingfang/v1beta1.BeeStatus"
        }
      }
    },
    "Dependencies": [
      "github.com/example/pkg/apis/jingfang/v1beta1.BeeSpec",
      "github.com/example/pkg/apis/jingfang/v1beta1.BeeStatus",
      "k8s.io/apimachinery/pkg/apis/meta/v1.ObjectMeta"
    ]
  },
  "github.com/example/pkg/apis/jingfang/v1beta1.BeeList": {
    "Schema": {
      "required": [
        "items"
      ],
      "properties": {
        "apiVersion": {
          "description": "APIVersion defines the versioned schema of this representation of an object.",
          "type": "string"
        },
        "items": {
          "type": "array",
          "items": {
            "$ref": "github.com/example/pkg/apis/jingfang/v1beta1.Bee"
          }
        },
        "kind": {
          "description": "Kind is a string value representing the REST resource this object represents.",
          "type": "string"
        },
        "metadata": {
          "$ref": "k8s.io/apimachinery/pkg/apis/meta/v1.ListMeta"
        }
      }
    },
    "Dependencies": [
      "github.com/example/pkg/apis/jingfang/v1beta1.Bee",
      "k8s.io/apimachinery/pkg/apis/meta/v1.ListMeta"
    ]
  },
  "github.com/example/pkg/apis/jingfang/v1beta1.BeeObjectReference": {
    "Schema": {
      "properties": {
        "name": {
          "type": "string"
        }
      }
    },
    "Dependencies": []
  },
  "github.com/example/pkg/apis/jingfang/v1beta1.BeeSpec": {
    "Schema": {
      "description": "BeeSpec defines the desired state of Bee"
    },
    "Dependencies": []
  },
  "github.com/example/pkg/apis/jingfang/v1beta1.BeeStatus": {
    "Schema": {
      "description": "BeeStatus defines the observed state of Bee"
    },
    "Dependencies": []
  },
  "github.com/example/pkg/apis/jingfang/v1beta1.MyKind": {
    "Schema": {
      "description": "MyKind",
      "properties": {
        "apiVersion": {
          "description": "APIVersion defines the versioned schema of this representation of an object.",
          "type": "string"
        },
        "kind": {
          "description": "Kind is a string value representing the REST resource this object represents.",
          "type": "string"
        },
        "metadata": {
          "$ref": "k8s.io/apimachinery/pkg/apis/meta/v1.ObjectMeta"
        },
        "spec": {
          "$ref": "github.com/example/pkg/apis/jingfang/v1beta1.MyKindSpec"
        },
        "status": {
          "$ref": "github.com/example/pkg/apis/jingfang/v1beta1.MyKindStatus"
        }
      }
    },
    "Dependencies": [
      "github.com/example/pkg/apis/jingfang/v1beta1.MyKindSpec",
      "github.com/example/pkg/apis/jingfang/v1beta1.MyKindStatus",
      "k8s.io/apimachinery/pkg/apis/meta/v1.ObjectMeta"
    ]
  },
  "github.com/example/pkg/apis/jingfang/v1beta1.MyKindList": {
    "Schema": {
      "required": [
        "items"
      ],
      "properties": {
        "apiVersion": {
          "description": "APIVersion defines the versioned schema of this representation of an object.",
          "type": "string"
        },
        "items": {
          "type": "array",
          "items": {
            "$ref": "github.com/example/pkg/apis/jingfang/v1beta1.MyKind"
          }
        },
        "kind": {
          "description": "Kind is a string value representing the REST resource this object represents.",
          "type": "string"
        },
        "metadata": {
          "$ref": "k8s.io/apimachinery/pkg/apis/meta/v1.ListMeta"
        }
      }
    },
    "Dependencies": [
      "github.com/example/pkg/apis/jingfang/v1beta1.MyKind",
      "k8s.io/apimachinery/pkg/apis/meta/v1.ListMeta"
    ]
  },
  "github.com/example/pkg/apis/jingfang/v1beta1.MyKindSpec": {
    "Schema": {
      "description": "MyKindSpec defines the desired state of MyKind",
      "properties": {
        "beeRef": {
          "x-kubernetes-object-ref-api-version": "v1beta1",
          "x-kubernetes-object-ref-kind": "Bee",
          "$ref": "github.com/example/pkg/apis/jingfang/v1beta1.BeeObjectReference"
        },
        "secretRef": {
          "description": "If defined, use this secret for configuring the MYSQL_ROOT_PASSWORD",
          "x-kubernetes-object-ref-api-version": "v1",
          "x-kubernetes-object-ref-kind": "Secret",
          "$ref": "k8s.io/api/core/v1.LocalObjectReference"
        }
      }
    },
    "Dependencies": [
      "github.com/example/pkg/apis/jingfang/v1beta1.BeeObjectReference",
      "k8s.io/api/core/v1.LocalObjectReference"
    ]
  },
  "github.com/example/pkg/apis/jingfang/v1beta1.MyKindStatus": {
    "Schema": {
      "description": "MyKindStatus defines the observed state of MyKind"
    },
    "Dependencies": []
  }
}
`)
}

func TestCrdBase(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/base")
	writeBaseWithCrd(th)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: v1
data:
  PATH: yellowBrickRoad
kind: Secret
metadata:
  name: x-crdsecret
---
apiVersion: jingfang.example.com/v1beta1
kind: MyKind
metadata:
  name: x-mykind
spec:
  beeRef:
    name: x-bee
  secretRef:
    name: x-crdsecret
---
apiVersion: v1beta1
kind: Bee
metadata:
  name: x-bee
spec:
  action: fly
`)
}

func TestCrdWithOverlay(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/overlay")
	writeBaseWithCrd(th)
	th.WriteK("/app/overlay", `
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
namePrefix: prod-
resources:
- ../base
patchesStrategicMerge:
- bee.yaml
`)
	th.WriteF("/app/overlay/bee.yaml", `
apiVersion: v1beta1
kind: Bee
metadata:
  name: bee
spec:
  action: makehoney
`)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}

	th.AssertActualEqualsExpected(m, `
apiVersion: v1
data:
  PATH: yellowBrickRoad
kind: Secret
metadata:
  name: prod-x-crdsecret
---
apiVersion: jingfang.example.com/v1beta1
kind: MyKind
metadata:
  name: prod-x-mykind
spec:
  beeRef:
    name: prod-x-bee
  secretRef:
    name: prod-x-crdsecret
---
apiVersion: v1beta1
kind: Bee
metadata:
  name: prod-x-bee
spec:
  action: makehoney
`)
}

func TestCrdWithContainers(t *testing.T) {
	th := kusttest_test.NewKustTestHarness(t, "/app/crd/containers")
	th.WriteK("/app/crd/containers", `
apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization
resources:
  - crd.yaml
images:
  - name: test/test
    newName: registry.gitlab.com/test
    newTag: latest
`)
	th.WriteF("/app/crd/containers/crd.yaml", `
apiVersion: apiextensions.k8s.io/v1beta1
kind: CustomResourceDefinition
metadata:
  name: crontabs.stable.example.com
spec:
  group: stable.example.com
  scope: Namespaced
  names:
    plural: crontabs
    singular: crontab
    kind: CronTab
    shortNames:
    - ct
  validation:
    openAPIV3Schema:
      properties:
        spec:
          containers:
            description: Containers allows injecting additional containers
  `)
	m, err := th.MakeKustTarget().MakeCustomizedResMap()
	if err != nil {
		t.Fatalf("Err: %v", err)
	}
	th.AssertActualEqualsExpected(m, `
apiVersion: apiextensions.k8s.io/v1beta1
kind: CustomResourceDefinition
metadata:
  name: crontabs.stable.example.com
spec:
  group: stable.example.com
  names:
    kind: CronTab
    plural: crontabs
    shortNames:
    - ct
    singular: crontab
  scope: Namespaced
  validation:
    openAPIV3Schema:
      properties:
        spec:
          containers:
            description: Containers allows injecting additional containers
`)
}
