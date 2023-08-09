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

// Package webhook implements the authorizer.Authorizer interface using HTTP webhooks.
package webhook

import (
	"math/rand"
	"reflect"
	"testing"
	"time"

	fuzz "github.com/google/gofuzz"

	authorizationv1 "k8s.io/api/authorization/v1"
	authorizationv1beta1 "k8s.io/api/authorization/v1beta1"
	"k8s.io/apimachinery/pkg/util/diff"
)

func TestRoundTrip(t *testing.T) {
	f := fuzz.New()
	seed := time.Now().UnixNano()
	t.Logf("seed = %v", seed)
	f.RandSource(rand.New(rand.NewSource(seed)))

	for i := 0; i < 1000; i++ {
		original := &authorizationv1.SubjectAccessReview{}
		f.Fuzz(&original.Spec)
		f.Fuzz(&original.Status)
		converted := &authorizationv1beta1.SubjectAccessReview{
			Spec:   v1SpecToV1beta1Spec(&original.Spec),
			Status: v1StatusToV1beta1Status(original.Status),
		}
		roundtripped := &authorizationv1.SubjectAccessReview{
			Spec:   v1beta1SpecToV1Spec(converted.Spec),
			Status: v1beta1StatusToV1Status(&converted.Status),
		}
		if !reflect.DeepEqual(original, roundtripped) {
			t.Errorf("diff %s", diff.ObjectReflectDiff(original, roundtripped))
		}
	}
}

// v1StatusToV1beta1Status is only needed to verify round-trip fidelity
func v1StatusToV1beta1Status(in authorizationv1.SubjectAccessReviewStatus) authorizationv1beta1.SubjectAccessReviewStatus {
	return authorizationv1beta1.SubjectAccessReviewStatus{
		Allowed:         in.Allowed,
		Denied:          in.Denied,
		Reason:          in.Reason,
		EvaluationError: in.EvaluationError,
	}
}

// v1beta1SpecToV1Spec is only needed to verify round-trip fidelity
func v1beta1SpecToV1Spec(in authorizationv1beta1.SubjectAccessReviewSpec) authorizationv1.SubjectAccessReviewSpec {
	return authorizationv1.SubjectAccessReviewSpec{
		ResourceAttributes:    v1beta1ResourceAttributesToV1ResourceAttributes(in.ResourceAttributes),
		NonResourceAttributes: v1beta1NonResourceAttributesToV1NonResourceAttributes(in.NonResourceAttributes),
		User:                  in.User,
		Groups:                in.Groups,
		Extra:                 v1beta1ExtraToV1Extra(in.Extra),
		UID:                   in.UID,
	}
}

func v1beta1ResourceAttributesToV1ResourceAttributes(in *authorizationv1beta1.ResourceAttributes) *authorizationv1.ResourceAttributes {
	if in == nil {
		return nil
	}
	return &authorizationv1.ResourceAttributes{
		Namespace:   in.Namespace,
		Verb:        in.Verb,
		Group:       in.Group,
		Version:     in.Version,
		Resource:    in.Resource,
		Subresource: in.Subresource,
		Name:        in.Name,
	}
}

func v1beta1NonResourceAttributesToV1NonResourceAttributes(in *authorizationv1beta1.NonResourceAttributes) *authorizationv1.NonResourceAttributes {
	if in == nil {
		return nil
	}
	return &authorizationv1.NonResourceAttributes{
		Path: in.Path,
		Verb: in.Verb,
	}
}

func v1beta1ExtraToV1Extra(in map[string]authorizationv1beta1.ExtraValue) map[string]authorizationv1.ExtraValue {
	if in == nil {
		return nil
	}
	ret := make(map[string]authorizationv1.ExtraValue, len(in))
	for k, v := range in {
		ret[k] = authorizationv1.ExtraValue(v)
	}
	return ret
}
