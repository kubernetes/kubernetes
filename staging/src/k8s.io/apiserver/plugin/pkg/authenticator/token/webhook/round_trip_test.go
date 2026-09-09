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

	"github.com/google/go-cmp/cmp"
	"sigs.k8s.io/randfill"

	authenticationv1 "k8s.io/api/authentication/v1"
	authenticationv1beta1 "k8s.io/api/authentication/v1beta1"
)

func TestRoundTrip(t *testing.T) {
	f := randfill.New()
	seed := time.Now().UnixNano()
	t.Logf("seed = %v", seed)
	f.RandSource(rand.New(rand.NewSource(seed)))

	for i := 0; i < 1000; i++ {
		original := &authenticationv1.TokenReview{}
		f.Fill(&original.Spec)
		f.Fill(&original.Status)
		converted := &authenticationv1beta1.TokenReview{
			Spec:   v1SpecToV1beta1Spec(&original.Spec),
			Status: v1StatusToV1beta1Status(original.Status),
		}
		roundtripped := &authenticationv1.TokenReview{
			Spec:   v1beta1SpecToV1Spec(converted.Spec),
			Status: v1beta1StatusToV1Status(&converted.Status),
		}
		if !reflect.DeepEqual(original, roundtripped) {
			t.Errorf("diff %s", cmp.Diff(original, roundtripped))
		}
	}
}

func v1StatusToV1beta1Status(in authenticationv1.TokenReviewStatus) authenticationv1beta1.TokenReviewStatus {
	return authenticationv1beta1.TokenReviewStatus{
		Authenticated: in.Authenticated,
		User:          v1UserToV1beta1User(in.User),
		Audiences:     in.Audiences,
		Error:         in.Error,
	}
}

func v1UserToV1beta1User(u authenticationv1.UserInfo) authenticationv1beta1.UserInfo {
	var extra map[string]authenticationv1beta1.ExtraValue
	if u.Extra != nil {
		extra = make(map[string]authenticationv1beta1.ExtraValue, len(u.Extra))
		for k, v := range u.Extra {
			extra[k] = authenticationv1beta1.ExtraValue(v)
		}
	}
	return authenticationv1beta1.UserInfo{
		Username: u.Username,
		UID:      u.UID,
		Groups:   u.Groups,
		Extra:    extra,
	}
}

func v1beta1SpecToV1Spec(in authenticationv1beta1.TokenReviewSpec) authenticationv1.TokenReviewSpec {
	return authenticationv1.TokenReviewSpec{
		Token:     in.Token,
		Audiences: in.Audiences,
	}
}
