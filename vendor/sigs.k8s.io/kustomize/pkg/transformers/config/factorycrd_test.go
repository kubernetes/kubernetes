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

package config

import (
	"reflect"
	"testing"

	"sigs.k8s.io/kustomize/internal/loadertest"
	"sigs.k8s.io/kustomize/pkg/gvk"
	"sigs.k8s.io/kustomize/pkg/ifc"
)

// This defines two CRD's:  Bee and MyKind.
//
// Bee is boring, it's spec has no dependencies.
//
// MyKind, however, has a spec that contains
// a Bee and a (k8s native) Secret.
const (
	crdContent = `
{
	"github.com/example/pkg/apis/jingfang/v1beta1.Bee": {
		"Schema": {
			"description": "Bee",
			"properties": {
				"apiVersion": {
					"description": "APIVersion defines the versioned schema of this representation of an object. Servers should convert 
recognized schemas to the latest internal value, and may reject unrecognized values. 
More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#resources",
					"type": "string"
				},
				"kind": {
					"description": "Kind is a string value representing the REST resource this object represents. Servers may infer
this from the endpoint the client submits requests to. Cannot be updated. In CamelCase.
More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#types-kinds",
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
					"description": "APIVersion defines the versioned schema of this representation of an object.
Servers should convert recognized schemas to the latest internal value, and may reject unrecognized values.
More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#resources",
					"type": "string"
				},
				"kind": {
					"description": "Kind is a string value representing the REST resource this object represents.
Servers may infer this from the endpoint the client submits requests to. Cannot be updated.
In CamelCase. More info: https://git.k8s.io/community/contributors/devel/api-conventions.md#types-kinds",
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
	"github.com/example/pkg/apis/jingfang/v1beta1.MyKindSpec": {
		"Schema": {
			"description": "MyKindSpec defines the desired state of MyKind",
			"properties": {
				"beeRef": {
					"x-kubernetes-object-ref-api-version": "v1beta1",
					"x-kubernetes-object-ref-kind": "Bee",
					"$ref": "github.com/example/pkg/apis/jingfang/v1beta1.Bee"
				},
				"secretRef": {
					"description": "If defined, we use this secret for configuring the MYSQL_ROOT_PASSWORD 
If it is not set we generate a secret dynamically",
					"x-kubernetes-object-ref-api-version": "v1",
					"x-kubernetes-object-ref-kind": "Secret",
					"$ref": "k8s.io/api/core/v1.LocalObjectReference"
				}
			}
		},
		"Dependencies": [
			"github.com/example/pkg/apis/jingfang/v1beta1.Bee",
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
`
)

func makeLoader(t *testing.T) ifc.Loader {
	ldr := loadertest.NewFakeLoader("/testpath")
	err := ldr.AddFile("/testpath/crd.json", []byte(crdContent))
	if err != nil {
		t.Fatalf("Failed to setup fake ldr.")
	}
	return ldr
}

func TestLoadCRDs(t *testing.T) {
	nbrs := []NameBackReferences{
		{
			Gvk: gvk.Gvk{Kind: "Secret", Version: "v1"},
			FieldSpecs: []FieldSpec{
				{
					CreateIfNotPresent: false,
					Gvk:                gvk.Gvk{Kind: "MyKind"},
					Path:               "spec/secretRef/name",
				},
			},
		},
		{
			Gvk: gvk.Gvk{Kind: "Bee", Version: "v1beta1"},
			FieldSpecs: []FieldSpec{
				{
					CreateIfNotPresent: false,
					Gvk:                gvk.Gvk{Kind: "MyKind"},
					Path:               "spec/beeRef/name",
				},
			},
		},
	}

	expectedTc := &TransformerConfig{
		NameReference: nbrs,
	}

	actualTc, err := LoadConfigFromCRDs(makeLoader(t), []string{"crd.json"})
	if err != nil {
		t.Fatalf("unexpected error:%v", err)
	}
	if !reflect.DeepEqual(actualTc, expectedTc) {
		t.Fatalf("expected\n %v\n but got\n %v\n", expectedTc, actualTc)
	}
}
