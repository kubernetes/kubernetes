/*
Copyright 2025 The Kubernetes Authors.

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

package controllerrevision

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apps "k8s.io/kubernetes/pkg/apis/apps"
	_ "k8s.io/kubernetes/pkg/apis/apps/install"
	"k8s.io/kubernetes/pkg/registry/apps/controllerrevision"
	"k8s.io/kubernetes/test/declarative_validation/meta"
	"testing"
)

// TODO: remove this apiVersions variable once coverage tests are generated for this package.
var apiVersions = []string{"v1", "v1beta1", "v1beta2"}

// Helper function to create a baseline valid ControllerRevision with optional tweaks
func mkControllerRevision(tweaks ...func(*apps.ControllerRevision)) apps.ControllerRevision {
	obj := apps.ControllerRevision{
		ObjectMeta: metav1.ObjectMeta{
			Name: "valid-resource-name",
		},
		Data:     runtime.RawExtension{Raw: []byte(`{"kind":"Foo"}`)},
		Revision: 1,
	}
	for _, tweak := range tweaks {
		tweak(&obj)
	}
	return obj
}

func TestDeclarativeValidate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := controllerrevision.Strategy
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "apps",
				APIVersion:        apiVersion,
				Resource:          "controllerrevisions",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "create",
			})
			obj := mkControllerRevision(func(o *apps.ControllerRevision) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaTestCases(t, ctx, &obj, strategy)
		})
	}
}

func TestDeclarativeValidateUpdate(t *testing.T) {
	for _, apiVersion := range apiVersions {
		t.Run(apiVersion, func(t *testing.T) {
			strategy := controllerrevision.Strategy
			var namespace string
			if strategy.NamespaceScoped() {
				namespace = metav1.NamespaceDefault
			}
			ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
				APIGroup:          "apps",
				APIVersion:        apiVersion,
				Resource:          "controllerrevisions",
				Namespace:         namespace,
				IsResourceRequest: true,
				Verb:              "update",
			})
			obj := mkControllerRevision(func(o *apps.ControllerRevision) {
				o.Namespace = namespace
			})
			meta.RunObjectMetaUpdateTestCases(t, ctx, &obj, strategy)
		})
	}
}
