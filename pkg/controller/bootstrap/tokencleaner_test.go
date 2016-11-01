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
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
)

func init() {
	spew.Config.DisableMethods = true
}

func newTokenCleaner() (*TokenCleaner, *fake.Clientset) {
	options := DefaultTokenCleanerOptions()
	cl := fake.NewSimpleClientset()
	return NewTokenCleaner(cl, options), cl
}

func newTokenSecret(tokenID, tokenSecret, expiration string) *api.Secret {
	return &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Namespace:       api.NamespaceSystem,
			Name:            "secretName",
			ResourceVersion: "1",
		},
		Type: api.SecretTypeBootstrapToken,
		Data: map[string][]byte{
			api.BootstrapTokenIdKey:         []byte(tokenID),
			api.BootstrapTokenSecretKey:     []byte(tokenSecret),
			api.BootstrapTokenExpirationKey: []byte(expiration),
		},
	}
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

func TestNoConfigMap(t *testing.T) {
	signer, cl := newBootstrapSigner()
	signer.signConfigMap()
	verifyActions(t, []core.Action{}, cl.Actions())
}

func TestSimpleSign(t *testing.T) {
	signer, cl := newBootstrapSigner()

	cm := newConfigMap("", "")
	signer.configMaps.Add(cm)

	secret := newTokenSecret("tokenID", "tokenSecret")
	signer.secrets.Add(secret)

	signer.signConfigMap()

	expected := []core.Action{
		core.NewUpdateAction(unversioned.GroupVersionResource{Resource: "configmaps"},
			api.NamespacePublic,
			newConfigMap("tokenID", "eyJhbGciOiJIUzI1NiIsImtpZCI6InRva2VuSUQifQ..QAvK9DAjF0hSyASEkH1MOTB5rJMmbWEY9j-z1NSYILE")),
	}

	verifyActions(t, expected, cl.Actions())
}

func TestNoSignNeeded(t *testing.T) {
	signer, cl := newBootstrapSigner()

	cm := newConfigMap("tokenID", "eyJhbGciOiJIUzI1NiIsImtpZCI6InRva2VuSUQifQ..QAvK9DAjF0hSyASEkH1MOTB5rJMmbWEY9j-z1NSYILE")
	signer.configMaps.Add(cm)

	secret := newTokenSecret("tokenID", "tokenSecret")
	signer.secrets.Add(secret)

	signer.signConfigMap()

	verifyActions(t, []core.Action{}, cl.Actions())
}

func TestUpdateSignature(t *testing.T) {
	signer, cl := newBootstrapSigner()

	cm := newConfigMap("tokenID", "old signature")
	signer.configMaps.Add(cm)

	secret := newTokenSecret("tokenID", "tokenSecret")
	signer.secrets.Add(secret)

	signer.signConfigMap()

	expected := []core.Action{
		core.NewUpdateAction(unversioned.GroupVersionResource{Resource: "configmaps"},
			api.NamespacePublic,
			newConfigMap("tokenID", "eyJhbGciOiJIUzI1NiIsImtpZCI6InRva2VuSUQifQ..QAvK9DAjF0hSyASEkH1MOTB5rJMmbWEY9j-z1NSYILE")),
	}

	verifyActions(t, expected, cl.Actions())
}

func TestRemoveSignature(t *testing.T) {
	signer, cl := newBootstrapSigner()

	cm := newConfigMap("tokenID", "old signature")
	signer.configMaps.Add(cm)

	signer.signConfigMap()

	expected := []core.Action{
		core.NewUpdateAction(unversioned.GroupVersionResource{Resource: "configmaps"},
			api.NamespacePublic,
			newConfigMap("", "")),
	}

	verifyActions(t, expected, cl.Actions())
}
