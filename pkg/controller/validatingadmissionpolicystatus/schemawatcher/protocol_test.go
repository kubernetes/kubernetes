/*
Copyright 2024 The Kubernetes Authors.

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

package schemawatcher

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

var jsonGVDocs = []string{
	// doc0 contains only apis.example.com/v1.Foo
	`
{
  "info": {
    "title": "Kubernetes",
    "version": "unversioned"
  },
  "openapi": "3.0.0",
  "components": {
    "schemas": {
      "com.example.apis.v1.Foo": {
        "description": "Foo",
        "properties": {
          "name": {
            "default": "",
            "description": "The name.",
            "type": "string"
          }
        },
        "required": [
          "name"
        ],
        "type": "object",
        "x-kubernetes-group-version-kind": [
          {
            "group": "apis.example.com",
            "kind": "Foo",
            "version": "v1"
          }
        ]
      }
    }
  }
}
`,
	// doc1 contains apis.example.com/v1.Foo and apis.example.com/v1.Bar
	// Foo is unchanged.
	`
{
  "info": {
    "title": "Kubernetes",
    "version": "unversioned"
  },
  "openapi": "3.0.0",
  "components": {
    "schemas": {
      "com.example.apis.v1.Foo": {
        "description": "Foo",
        "properties": {
          "name": {
            "default": "",
            "description": "The name.",
            "type": "string"
          }
        },
        "required": [
          "name"
        ],
        "type": "object",
        "x-kubernetes-group-version-kind": [
          {
            "group": "apis.example.com",
            "kind": "Foo",
            "version": "v1"
          }
        ]
      },
      "com.example.apis.v1.Bar": {
        "description": "Bar",
        "properties": {
          "type": {
            "default": "",
            "description": "The type.",
            "type": "string"
          }
        },
        "required": [
          "type"
        ],
        "type": "object",
        "x-kubernetes-group-version-kind": [
          {
            "group": "apis.example.com",
            "kind": "Bar",
            "version": "v1"
          }
        ]
      }
    }
  }
}
`,

	// doc2 contains apis.example.com/v1.Foo and apis.example.com/v1.Bar
	// but v1.Foo changes.
	`
{
  "info": {
    "title": "Kubernetes",
    "version": "unversioned"
  },
  "openapi": "3.0.0",
  "components": {
    "schemas": {
      "com.example.apis.v1.Foo": {
        "description": "Foo",
        "properties": {
          "name": {
            "default": "",
            "description": "The name, but fancier",
            "type": "string"
          }
        },
        "required": [
          "name"
        ],
        "type": "object",
        "x-kubernetes-group-version-kind": [
          {
            "group": "apis.example.com",
            "kind": "Foo",
            "version": "v1"
          }
        ]
      },
      "com.example.apis.v1.Bar": {
        "description": "Bar",
        "properties": {
          "type": {
            "default": "",
            "description": "The type.",
            "type": "string"
          }
        },
        "required": [
          "type"
        ],
        "type": "object",
        "x-kubernetes-group-version-kind": [
          {
            "group": "apis.example.com",
            "kind": "Bar",
            "version": "v1"
          }
        ]
      }
    }
  }
}
`,
}

func TestSchemaHashing(t *testing.T) {
	r1 := parseOpenAPIv3Doc([]byte(jsonGVDocs[0]))
	if len(r1) != 1 {
		t.Fatalf("unexpected r1 length: expected %d but got %d", 1, len(r1))
	}
	r2 := parseOpenAPIv3Doc([]byte(jsonGVDocs[1]))
	if len(r2) != 2 {
		t.Fatalf("unexpected r2 length: expected %d but got %d", 2, len(r2))
	}
	r3 := parseOpenAPIv3Doc([]byte(jsonGVDocs[2]))
	if len(r3) != 2 {
		t.Fatalf("unexpected r3 length: expected %d but got %d", 2, len(r3))
	}

	hashOf := func(results []schemaWithHash, gvk schema.GroupVersionKind) [32]byte {
		for _, r := range results {
			if s, gvks := r.Schema, []schema.GroupVersionKind{}; s.Extensions.GetObject(gvkExtensionName, &gvks) != nil ||
				len(gvks) != 1 {
				t.Fatalf("invalid gvk: %v", gvks)
			} else if gvks[0] == gvk {
				return r.Hash
			}
		}
		t.Fatalf("gvk not found: %v", gvk)
		return [32]byte{}
	}

	gv := schema.GroupVersion{
		Group:   "apis.example.com",
		Version: "v1",
	}
	gvkFoo := gv.WithKind("Foo")
	gvkBar := gv.WithKind("Bar")
	// Foo is unchanged from doc1 -> doc2, but changed doc2 -> doc3
	// Bar appears doc1 -> doc2, unchanged doc2 -> doc3.
	if h1, h2 := hashOf(r1, gvkFoo), hashOf(r2, gvkFoo); h1 != h2 {
		t.Fatalf("unexpected unequalive hash, %x vs %x.", h1, h2)
	}
	if h1, h3 := hashOf(r1, gvkFoo), hashOf(r3, gvkFoo); h1 == h3 {
		t.Fatalf("unexpected unchanged hash, %x vs %x.", h1, h3)
	}
	if h1, h3 := hashOf(r2, gvkBar), hashOf(r3, gvkBar); h1 != h3 {
		t.Fatalf("unexpected unequalive hash, %x vs %x.", h1, h3)
	}
}
