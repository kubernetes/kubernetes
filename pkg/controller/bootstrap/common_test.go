/*
Copyright 2016 The Kubernetes Authors.

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

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/dump"
	core "k8s.io/client-go/testing"
	bootstrapapi "k8s.io/cluster-bootstrap/token/api"
	"k8s.io/kubernetes/pkg/apis/core/helper"
)

func newTokenSecret(tokenID, tokenSecret string) *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Namespace:       metav1.NamespaceSystem,
			Name:            bootstrapapi.BootstrapTokenSecretPrefix + tokenID,
			ResourceVersion: "1",
			UID:             types.UID("uid" + tokenID),
		},
		Type: bootstrapapi.SecretTypeBootstrapToken,
		Data: map[string][]byte{
			bootstrapapi.BootstrapTokenIDKey:     []byte(tokenID),
			bootstrapapi.BootstrapTokenSecretKey: []byte(tokenSecret),
		},
	}
}

func addSecretExpiration(s *v1.Secret, expiration string) {
	s.Data[bootstrapapi.BootstrapTokenExpirationKey] = []byte(expiration)
}

func addSecretSigningUsage(s *v1.Secret, value string) {
	s.Data[bootstrapapi.BootstrapTokenUsageSigningKey] = []byte(value)
}

func verifyActions(t *testing.T, expected, actual []core.Action) {
	for i, a := range actual {
		if len(expected) < i+1 {
			t.Errorf("%d unexpected actions: %s", len(actual)-len(expected), dump.Pretty(actual[i:]))
			break
		}

		e := expected[i]
		if !helper.Semantic.DeepEqual(e, a) {
			t.Errorf("Expected\n\t%s\ngot\n\t%s", dump.Pretty(e), dump.Pretty(a))
			continue
		}
	}

	if len(expected) > len(actual) {
		t.Errorf("%d additional expected actions", len(expected)-len(actual))
		for _, a := range expected[len(actual):] {
			t.Logf("    %s", dump.Pretty(a))
		}
	}
}
