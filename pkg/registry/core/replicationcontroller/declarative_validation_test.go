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

package replicationcontroller

import (
	"fmt"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	podtest "k8s.io/kubernetes/pkg/api/pod/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

func TestDeclarativeValidateForDeclarative(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		input        api.ReplicationController
		expectedErrs field.ErrorList
	}{
		// baseline
		"empty resource": {
			input: mkValidReplicationController(),
		},
		// metadata.name
		"name: empty": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name = ""
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata.name"), ""),
			},
		},
		"name: label format": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name = "this-is-a-label"
			}),
		},
		"name: subdomain format": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name = "this.is.a.subdomain"
			}),
		},
		"name: invalid label format": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name = "-this-is-not-a-label"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"name: invalid subdomain format": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name = ".this.is.not.a.subdomain"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"name: label format with trailing dash": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name = "this-is-a-label-"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"name: subdomain format with trailing dash": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name = "this.is.a.subdomain-"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"name: long label format": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name = strings.Repeat("x", 253)
			}),
		},
		"name: long subdomain format": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name = strings.Repeat("x.", 126) + "x"
			}),
		},
		"name: too long label format": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name = strings.Repeat("x", 254)
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		"name: too long subdomain format": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name = strings.Repeat("x.", 126) + "xx"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, "").WithOrigin("format=k8s-long-name"),
			},
		},
		// metadata.generateName (note: it's is not really validated)
		"generateName: valid with empty name": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name = ""
				rc.GenerateName = "name"
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("metadata.name"), ""),
				field.InternalError(field.NewPath("metadata.name"), fmt.Errorf("")),
			},
		},

		"generateName: label format": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.GenerateName = "this-is-a-label"
			}),
		},
		"generateName: subdomain format": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.GenerateName = "this.is.a.subdomain"
			}),
		},
		"generateName: invalid format": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.GenerateName = "Obviously not a valid generateName!!"
			}),
		},
		"generateName: too long": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.GenerateName = strings.Repeat("x", 4096)
			}),
		},
		// spec.replicas
		"replicas: nil": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Replicas = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec.replicas"), ""),
			},
		},
		"replicas: 0": {
			input: mkValidReplicationController(setSpecReplicas(0)),
		},
		"replicas: positive": {
			input: mkValidReplicationController(setSpecReplicas(100)),
		},
		"replicas: negative": {
			input: mkValidReplicationController(setSpecReplicas(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.replicas"), nil, "").WithOrigin("minimum"),
			},
		},
		// spec.minReadySeconds
		"minReadySeconds: 0": {
			input: mkValidReplicationController(setSpecMinReadySeconds(0)),
		},
		"minReadySeconds: positive": {
			input: mkValidReplicationController(setSpecMinReadySeconds(100)),
		},
		"minReadySeconds: negative": {
			input: mkValidReplicationController(setSpecMinReadySeconds(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.minReadySeconds"), nil, "").WithOrigin("minimum"),
			},
		},
		// spec.selector
		"selector: empty": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Selector = map[string]string{}
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec.selector"), ""),
			},
		},
		"selector: valid": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Selector = map[string]string{"a": "b"}
			}),
		},
		// spec.template
		"template: nil": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Template = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec.template"), ""),
			},
		},
		"template : valid": {
			input: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Template = &api.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{"a": "b"},
					},
					Spec: podtest.MakePodSpec(),
				}
			}),
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			apitesting.VerifyValidationEquivalence(t, ctx, &tc.input, Strategy.Validate, tc.expectedErrs)
		})
	}
}

func TestValidateUpdateForDeclarative(t *testing.T) {
	ctx := genericapirequest.WithRequestInfo(genericapirequest.NewDefaultContext(), &genericapirequest.RequestInfo{
		APIGroup:   "",
		APIVersion: "v1",
	})
	testCases := map[string]struct {
		old          api.ReplicationController
		update       api.ReplicationController
		expectedErrs field.ErrorList
	}{
		// baseline
		"no change": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(),
		},
		// metadata.name
		"name: changed": {
			old: mkValidReplicationController(),
			update: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Name += "x"
			}),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("metadata.name"), nil, ""),
			},
		},
		// metadata.generateName
		"generateName: changed": {
			old: mkValidReplicationController(),
			update: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.GenerateName += "x"
			}),
			// This is, oddly, mutable.  We should probably ratchet this off
			// and make it immutable.
		},
		// spec.replicas
		"replicas: nil": {
			old: mkValidReplicationController(),
			update: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Replicas = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec.replicas"), ""),
			},
		},
		"replicas: 0": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecReplicas(0)),
		},
		"replicas: positive": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecReplicas(100)),
		},
		"replicas: negative": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecReplicas(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.replicas"), nil, "").WithOrigin("minimum"),
			},
		},
		// spec.minReadySeconds
		"minReadySeconds: 0": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecMinReadySeconds(0)),
		},
		"minReadySeconds: positive": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecMinReadySeconds(3)),
		},
		"minReadySeconds: negative": {
			old:    mkValidReplicationController(),
			update: mkValidReplicationController(setSpecMinReadySeconds(-1)),
			expectedErrs: field.ErrorList{
				field.Invalid(field.NewPath("spec.minReadySeconds"), nil, "").WithOrigin("minimum"),
			},
		},
		// spec.selector
		"selector: empty": {
			old: mkValidReplicationController(),
			update: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Selector = map[string]string{}
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec.selector"), ""),
			},
		},
		"selector: valid": {
			old: mkValidReplicationController(),
			update: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Selector = map[string]string{"a": "b"}
			}),
		},
		// spec.template
		"template: nil": {
			old: mkValidReplicationController(),
			update: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Template = nil
			}),
			expectedErrs: field.ErrorList{
				field.Required(field.NewPath("spec.template"), ""),
			},
		},
		"template: valid": {
			old: mkValidReplicationController(),
			update: mkValidReplicationController(func(rc *api.ReplicationController) {
				rc.Spec.Template = &api.PodTemplateSpec{
					ObjectMeta: metav1.ObjectMeta{
						Labels: map[string]string{"a": "b"},
					},
					Spec: podtest.MakePodSpec(),
				}
			}),
		},
	}
	for k, tc := range testCases {
		t.Run(k, func(t *testing.T) {
			tc.old.ObjectMeta.ResourceVersion = "1"
			tc.update.ObjectMeta.ResourceVersion = "1"
			apitesting.VerifyUpdateValidationEquivalence(t, ctx, &tc.update, &tc.old, Strategy.ValidateUpdate, tc.expectedErrs)
		})
	}
}

// mkValidReplicationController produces a ReplicationController which passes
// validation with no tweaks.
func mkValidReplicationController(tweaks ...func(rc *api.ReplicationController)) api.ReplicationController {
	rc := api.ReplicationController{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: api.ReplicationControllerSpec{
			Replicas: ptr.To[int32](1),
			Selector: map[string]string{"a": "b"},
			Template: &api.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: map[string]string{"a": "b"},
				},
				Spec: podtest.MakePodSpec(),
			},
		},
	}
	for _, tweak := range tweaks {
		tweak(&rc)
	}
	return rc
}

func setSpecReplicas(val int32) func(rc *api.ReplicationController) {
	return func(rc *api.ReplicationController) {
		rc.Spec.Replicas = ptr.To(val)
	}
}

func setSpecMinReadySeconds(val int32) func(rc *api.ReplicationController) {
	return func(rc *api.ReplicationController) {
		rc.Spec.MinReadySeconds = val
	}
}
