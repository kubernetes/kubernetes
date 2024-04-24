/*
Copyright 2015 The Kubernetes Authors.

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

package configmap

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"
)

func TestConfigMapStrategy(t *testing.T) {
	t.Parallel()
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("ConfigMap must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("ConfigMap should not allow create on update")
	}

	cfg := &api.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-config-data",
			Namespace: metav1.NamespaceDefault,
		},
		Data: map[string]string{
			"foo": "bar",
		},
	}

	Strategy.PrepareForCreate(ctx, cfg)
	if cfg.Generation != 1 {
		t.Errorf("expected generation to be 1, was %d", cfg.Generation)
	}

	errs := Strategy.Validate(ctx, cfg)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	newCfg := &api.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "valid-config-data-2",
			Namespace:       metav1.NamespaceDefault,
			ResourceVersion: "4",
		},
		Data: map[string]string{
			"invalidKey": "updatedValue",
		},
	}

	Strategy.PrepareForUpdate(ctx, newCfg, cfg)
	if expected, actual := cfg.Generation+1, newCfg.Generation; expected != actual {
		t.Errorf("expected generation to be %d, was %d", expected, actual)
	}

	errs = Strategy.ValidateUpdate(ctx, newCfg, cfg)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}

	newCfg = cfg.DeepCopy()
	newCfg.BinaryData = map[string][]byte{
		"foo": []byte("bar"),
	}

	Strategy.PrepareForUpdate(ctx, newCfg, cfg)
	if expected, actual := cfg.Generation+1, newCfg.Generation; expected != actual {
		t.Errorf("expected generation to be %d, was %d", expected, actual)
	}

	identicalDataCfg := cfg.DeepCopy()
	identicalDataCfg.Labels = map[string]string{"foo": "bar"}
	Strategy.PrepareForUpdate(ctx, cfg, identicalDataCfg)
	if expected, actual := cfg.Generation, identicalDataCfg.Generation; expected != actual {
		t.Errorf("expected generation to be %d, was %d", expected, actual)
	}

	enabledImmutability := cfg.DeepCopy()
	enabledImmutability.Immutable = ptr.To(true)
	Strategy.PrepareForUpdate(ctx, enabledImmutability, cfg)
	if expected, actual := cfg.Generation+1, enabledImmutability.Generation; expected != actual {
		t.Errorf("expected generation to be %d, was %d", expected, actual)
	}
}
