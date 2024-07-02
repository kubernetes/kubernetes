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

package secret

import (
	"context"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/ptr"

	// ensure types are installed
	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		"v1",
		"Secret",
		SelectableFields(&api.Secret{}),
		nil,
	)
}

func TestStrategy(t *testing.T) {
	t.Parallel()

	obj := &api.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
		},
	}
	Strategy.PrepareForCreate(context.Background(), obj)
	if obj.Generation != 1 {
		t.Errorf("expected generation to be 1, was %d", obj.Generation)
	}

	newSecret := obj.DeepCopy()
	newSecret.Labels = map[string]string{"foo": "bar"}

	Strategy.PrepareForUpdate(context.Background(), newSecret, obj)
	if expected, got := obj.Generation, newSecret.Generation; expected != got {
		t.Errorf("expected generation to be %d, was %d", expected, got)
	}

	newSecret.Data = map[string][]byte{"foo": []byte("bar")}

	Strategy.PrepareForUpdate(context.Background(), newSecret, obj)
	if expected, got := obj.Generation+1, newSecret.Generation; expected != got {
		t.Errorf("expected generation to be %d, was %d", expected, got)
	}

	immutableSecret := obj.DeepCopy()
	immutableSecret.Immutable = ptr.To(true)

	Strategy.PrepareForUpdate(context.Background(), immutableSecret, obj)
	if expected, got := obj.Generation+1, immutableSecret.Generation; expected != got {
		t.Errorf("expected generation to be %d, was %d", expected, got)
	}
}
