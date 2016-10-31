/*
Copyright 2014 The Kubernetes Authors.

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

package bootstrap

import (
	"testing"

	"github.com/davecgh/go-spew/spew"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/testing/core"
)

func newTokenSecret(tokenID, tokenSecret string) *v1.Secret {
	return &v1.Secret{
		ObjectMeta: v1.ObjectMeta{
			Namespace:       api.NamespaceSystem,
			Name:            "secretName",
			ResourceVersion: "1",
		},
		Type: v1.SecretTypeBootstrapToken,
		Data: map[string][]byte{
			v1.BootstrapTokenIdKey:     []byte(tokenID),
			v1.BootstrapTokenSecretKey: []byte(tokenSecret),
		},
	}
}

func addSecretExpiration(s *v1.Secret, expiration string) {
	s.Data[v1.BootstrapTokenExpirationKey] = []byte(expiration)
}

func addSecretSigningUsage(s *v1.Secret, value string) {
	s.Data[v1.BootstrapTokenUsageSigningKey] = []byte(value)
}

func verifyActions(t *testing.T, expected, actual []core.Action) {
	for i, a := range actual {
		if len(expected) < i+1 {
			t.Errorf("%d unexpected actions: %s", len(actual)-len(expected), spew.Sdump(actual[i:]))
			break
		}

		e := expected[i]
		if !api.Semantic.DeepEqual(e, a) {
			t.Errorf("Expected\n\t%s\ngot\n\t%s", spew.Sdump(e), spew.Sdump(a))
			continue
		}
	}

	if len(expected) > len(actual) {
		t.Errorf("%d additional expected actions", len(expected)-len(actual))
		for _, a := range expected[len(actual):] {
			t.Logf("    %s", spew.Sdump(a))
		}
	}
}
