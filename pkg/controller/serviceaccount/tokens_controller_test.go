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

package serviceaccount

import (
	"reflect"
	"testing"
	"time"

	"gopkg.in/square/go-jose.v2/jwt"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/dump"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	core "k8s.io/client-go/testing"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/test/utils/ktesting"
)

type testGenerator struct {
	Token string
	Err   error
}

func (t *testGenerator) GenerateToken(sc *jwt.Claims, pc interface{}) (string, error) {
	return t.Token, t.Err
}

// emptySecretReferences is used by a service account without any secrets
func emptySecretReferences() []v1.ObjectReference {
	return []v1.ObjectReference{}
}

// missingSecretReferences is used by a service account that references secrets which do no exist
func missingSecretReferences() []v1.ObjectReference {
	return []v1.ObjectReference{{Name: "missing-secret-1"}}
}

// regularSecretReferences is used by a service account that references secrets which are not ServiceAccountTokens
func regularSecretReferences() []v1.ObjectReference {
	return []v1.ObjectReference{{Name: "regular-secret-1"}}
}

// tokenSecretReferences is used by a service account that references a ServiceAccountToken secret
func tokenSecretReferences() []v1.ObjectReference {
	return []v1.ObjectReference{{Name: "token-secret-1"}}
}

// serviceAccount returns a service account with the given secret refs
func serviceAccount(secretRefs []v1.ObjectReference) *v1.ServiceAccount {
	return &v1.ServiceAccount{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "default",
			UID:             "12345",
			Namespace:       "default",
			ResourceVersion: "1",
		},
		Secrets: secretRefs,
	}
}

// updatedServiceAccount returns a service account with the resource version modified
func updatedServiceAccount(secretRefs []v1.ObjectReference) *v1.ServiceAccount {
	sa := serviceAccount(secretRefs)
	sa.ResourceVersion = "2"
	return sa
}

// opaqueSecret returns a persisted non-ServiceAccountToken secret named "regular-secret-1"
func opaqueSecret() *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "regular-secret-1",
			Namespace:       "default",
			UID:             "23456",
			ResourceVersion: "1",
		},
		Type: "Opaque",
		Data: map[string][]byte{
			"mykey": []byte("mydata"),
		},
	}
}

// serviceAccountTokenSecret returns an existing ServiceAccountToken secret named "token-secret-1"
func serviceAccountTokenSecret() *v1.Secret {
	return &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "token-secret-1",
			Namespace:       "default",
			UID:             "23456",
			ResourceVersion: "1",
			Annotations: map[string]string{
				v1.ServiceAccountNameKey: "default",
				v1.ServiceAccountUIDKey:  "12345",
			},
		},
		Type: v1.SecretTypeServiceAccountToken,
		Data: map[string][]byte{
			"token":     []byte("ABC"),
			"ca.crt":    []byte("CA Data"),
			"namespace": []byte("default"),
		},
	}
}

// serviceAccountTokenSecretWithoutTokenData returns an existing ServiceAccountToken secret that lacks token data
func serviceAccountTokenSecretWithoutTokenData() *v1.Secret {
	secret := serviceAccountTokenSecret()
	delete(secret.Data, v1.ServiceAccountTokenKey)
	return secret
}

// serviceAccountTokenSecretWithoutCAData returns an existing ServiceAccountToken secret that lacks ca data
func serviceAccountTokenSecretWithoutCAData() *v1.Secret {
	secret := serviceAccountTokenSecret()
	delete(secret.Data, v1.ServiceAccountRootCAKey)
	return secret
}

// serviceAccountTokenSecretWithCAData returns an existing ServiceAccountToken secret with the specified ca data
func serviceAccountTokenSecretWithCAData(data []byte) *v1.Secret {
	secret := serviceAccountTokenSecret()
	secret.Data[v1.ServiceAccountRootCAKey] = data
	return secret
}

// serviceAccountTokenSecretWithoutNamespaceData returns an existing ServiceAccountToken secret that lacks namespace data
func serviceAccountTokenSecretWithoutNamespaceData() *v1.Secret {
	secret := serviceAccountTokenSecret()
	delete(secret.Data, v1.ServiceAccountNamespaceKey)
	return secret
}

// serviceAccountTokenSecretWithNamespaceData returns an existing ServiceAccountToken secret with the specified namespace data
func serviceAccountTokenSecretWithNamespaceData(data []byte) *v1.Secret {
	secret := serviceAccountTokenSecret()
	secret.Data[v1.ServiceAccountNamespaceKey] = data
	return secret
}

type reaction struct {
	verb     string
	resource string
	reactor  func(t *testing.T) core.ReactionFunc
}

func TestTokenCreation(t *testing.T) {
	testcases := map[string]struct {
		ClientObjects []runtime.Object

		IsAsync    bool
		MaxRetries int

		Reactors []reaction

		ExistingServiceAccount *v1.ServiceAccount
		ExistingSecrets        []*v1.Secret

		AddedServiceAccount   *v1.ServiceAccount
		UpdatedServiceAccount *v1.ServiceAccount
		DeletedServiceAccount *v1.ServiceAccount
		AddedSecret           *v1.Secret
		AddedSecretLocal      *v1.Secret
		UpdatedSecret         *v1.Secret
		DeletedSecret         *v1.Secret

		ExpectedActions []core.Action
	}{
		"new serviceaccount with no secrets": {
			ClientObjects: []runtime.Object{serviceAccount(emptySecretReferences())},

			AddedServiceAccount: serviceAccount(emptySecretReferences()),
			ExpectedActions:     []core.Action{},
		},
		"new serviceaccount with missing secrets": {
			ClientObjects: []runtime.Object{serviceAccount(missingSecretReferences())},

			AddedServiceAccount: serviceAccount(missingSecretReferences()),
			ExpectedActions:     []core.Action{},
		},
		"new serviceaccount with missing secrets and a local secret in the cache": {
			ClientObjects: []runtime.Object{serviceAccount(missingSecretReferences())},

			AddedServiceAccount: serviceAccount(tokenSecretReferences()),
			AddedSecretLocal:    serviceAccountTokenSecret(),
			ExpectedActions:     []core.Action{},
		},
		"new serviceaccount with non-token secrets": {
			ClientObjects: []runtime.Object{serviceAccount(regularSecretReferences()), opaqueSecret()},

			AddedServiceAccount: serviceAccount(regularSecretReferences()),
			ExpectedActions:     []core.Action{},
		},
		"new serviceaccount with token secrets": {
			ClientObjects:   []runtime.Object{serviceAccount(tokenSecretReferences()), serviceAccountTokenSecret()},
			ExistingSecrets: []*v1.Secret{serviceAccountTokenSecret()},

			AddedServiceAccount: serviceAccount(tokenSecretReferences()),
			ExpectedActions:     []core.Action{},
		},
		"updated serviceaccount with no secrets": {
			ClientObjects: []runtime.Object{serviceAccount(emptySecretReferences())},

			UpdatedServiceAccount: serviceAccount(emptySecretReferences()),
			ExpectedActions:       []core.Action{},
		},
		"updated serviceaccount with missing secrets": {
			ClientObjects: []runtime.Object{serviceAccount(missingSecretReferences())},

			UpdatedServiceAccount: serviceAccount(missingSecretReferences()),
			ExpectedActions:       []core.Action{},
		},
		"updated serviceaccount with non-token secrets": {
			ClientObjects: []runtime.Object{serviceAccount(regularSecretReferences()), opaqueSecret()},

			UpdatedServiceAccount: serviceAccount(regularSecretReferences()),
			ExpectedActions:       []core.Action{},
		},
		"updated serviceaccount with token secrets": {
			ExistingSecrets: []*v1.Secret{serviceAccountTokenSecret()},

			UpdatedServiceAccount: serviceAccount(tokenSecretReferences()),
			ExpectedActions:       []core.Action{},
		},
		"updated serviceaccount with no secrets with resource conflict": {
			ClientObjects: []runtime.Object{updatedServiceAccount(emptySecretReferences())},
			IsAsync:       true,
			MaxRetries:    1,

			UpdatedServiceAccount: serviceAccount(emptySecretReferences()),
			ExpectedActions:       []core.Action{},
		},

		"deleted serviceaccount with no secrets": {
			DeletedServiceAccount: serviceAccount(emptySecretReferences()),
			ExpectedActions:       []core.Action{},
		},
		"deleted serviceaccount with missing secrets": {
			DeletedServiceAccount: serviceAccount(missingSecretReferences()),
			ExpectedActions:       []core.Action{},
		},
		"deleted serviceaccount with non-token secrets": {
			ClientObjects: []runtime.Object{opaqueSecret()},

			DeletedServiceAccount: serviceAccount(regularSecretReferences()),
			ExpectedActions:       []core.Action{},
		},
		"deleted serviceaccount with token secrets": {
			ClientObjects:   []runtime.Object{serviceAccountTokenSecret()},
			ExistingSecrets: []*v1.Secret{serviceAccountTokenSecret()},

			DeletedServiceAccount: serviceAccount(tokenSecretReferences()),
			ExpectedActions: []core.Action{
				core.NewDeleteActionWithOptions(
					schema.GroupVersionResource{Version: "v1", Resource: "secrets"},
					metav1.NamespaceDefault, "token-secret-1",
					*metav1.NewPreconditionDeleteOptions("23456")),
			},
		},

		"added secret without serviceaccount": {
			ClientObjects: []runtime.Object{serviceAccountTokenSecret()},

			AddedSecret: serviceAccountTokenSecret(),
			ExpectedActions: []core.Action{
				core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "serviceaccounts"}, metav1.NamespaceDefault, "default"),
				core.NewDeleteActionWithOptions(
					schema.GroupVersionResource{Version: "v1", Resource: "secrets"},
					metav1.NamespaceDefault, "token-secret-1",
					*metav1.NewPreconditionDeleteOptions("23456")),
			},
		},
		"added secret with serviceaccount": {
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			AddedSecret:     serviceAccountTokenSecret(),
			ExpectedActions: []core.Action{},
		},
		"added token secret without token data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithoutTokenData()},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			AddedSecret: serviceAccountTokenSecretWithoutTokenData(),
			ExpectedActions: []core.Action{
				core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, "token-secret-1"),
				core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},
		"added token secret without ca data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithoutCAData()},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			AddedSecret: serviceAccountTokenSecretWithoutCAData(),
			ExpectedActions: []core.Action{
				core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, "token-secret-1"),
				core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},
		"added token secret with mismatched ca data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithCAData([]byte("mismatched"))},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			AddedSecret: serviceAccountTokenSecretWithCAData([]byte("mismatched")),
			ExpectedActions: []core.Action{
				core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, "token-secret-1"),
				core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},
		"added token secret without namespace data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithoutNamespaceData()},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			AddedSecret: serviceAccountTokenSecretWithoutNamespaceData(),
			ExpectedActions: []core.Action{
				core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, "token-secret-1"),
				core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},
		"added token secret with custom namespace data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithNamespaceData([]byte("custom"))},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			AddedSecret:     serviceAccountTokenSecretWithNamespaceData([]byte("custom")),
			ExpectedActions: []core.Action{
				// no update is performed... the custom namespace is preserved
			},
		},

		"updated secret without serviceaccount": {
			ClientObjects: []runtime.Object{serviceAccountTokenSecret()},

			UpdatedSecret: serviceAccountTokenSecret(),
			ExpectedActions: []core.Action{
				core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "serviceaccounts"}, metav1.NamespaceDefault, "default"),
				core.NewDeleteActionWithOptions(
					schema.GroupVersionResource{Version: "v1", Resource: "secrets"},
					metav1.NamespaceDefault, "token-secret-1",
					*metav1.NewPreconditionDeleteOptions("23456")),
			},
		},
		"updated secret with serviceaccount": {
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			UpdatedSecret:   serviceAccountTokenSecret(),
			ExpectedActions: []core.Action{},
		},
		"updated token secret without token data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithoutTokenData()},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			UpdatedSecret: serviceAccountTokenSecretWithoutTokenData(),
			ExpectedActions: []core.Action{
				core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, "token-secret-1"),
				core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},
		"updated token secret without ca data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithoutCAData()},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			UpdatedSecret: serviceAccountTokenSecretWithoutCAData(),
			ExpectedActions: []core.Action{
				core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, "token-secret-1"),
				core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},
		"updated token secret with mismatched ca data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithCAData([]byte("mismatched"))},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			UpdatedSecret: serviceAccountTokenSecretWithCAData([]byte("mismatched")),
			ExpectedActions: []core.Action{
				core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, "token-secret-1"),
				core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},
		"updated token secret without namespace data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithoutNamespaceData()},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			UpdatedSecret: serviceAccountTokenSecretWithoutNamespaceData(),
			ExpectedActions: []core.Action{
				core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, "token-secret-1"),
				core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "secrets"}, metav1.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},
		"updated token secret with custom namespace data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithNamespaceData([]byte("custom"))},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			UpdatedSecret:   serviceAccountTokenSecretWithNamespaceData([]byte("custom")),
			ExpectedActions: []core.Action{
				// no update is performed... the custom namespace is preserved
			},
		},

		"deleted secret without serviceaccount": {
			DeletedSecret:   serviceAccountTokenSecret(),
			ExpectedActions: []core.Action{},
		},
		"deleted secret with serviceaccount with reference": {
			ClientObjects:          []runtime.Object{serviceAccount(tokenSecretReferences())},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			DeletedSecret: serviceAccountTokenSecret(),
			ExpectedActions: []core.Action{
				core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "serviceaccounts"}, metav1.NamespaceDefault, "default"),
				core.NewUpdateAction(schema.GroupVersionResource{Version: "v1", Resource: "serviceaccounts"}, metav1.NamespaceDefault, serviceAccount(emptySecretReferences())),
			},
		},
		"deleted secret with serviceaccount without reference": {
			ExistingServiceAccount: serviceAccount(emptySecretReferences()),

			DeletedSecret: serviceAccountTokenSecret(),
			ExpectedActions: []core.Action{
				core.NewGetAction(schema.GroupVersionResource{Version: "v1", Resource: "serviceaccounts"}, metav1.NamespaceDefault, "default"),
			},
		},
	}

	for k, tc := range testcases {
		t.Run(k, func(t *testing.T) {
			_, ctx := ktesting.NewTestContext(t)

			// Re-seed to reset name generation
			utilrand.Seed(1)

			generator := &testGenerator{Token: "ABC"}

			client := fake.NewSimpleClientset(tc.ClientObjects...)
			for _, reactor := range tc.Reactors {
				client.Fake.PrependReactor(reactor.verb, reactor.resource, reactor.reactor(t))
			}
			informers := informers.NewSharedInformerFactory(client, controller.NoResyncPeriodFunc())
			secretInformer := informers.Core().V1().Secrets().Informer()
			secrets := secretInformer.GetStore()
			serviceAccounts := informers.Core().V1().ServiceAccounts().Informer().GetStore()
			controller, err := NewTokensController(informers.Core().V1().ServiceAccounts(), informers.Core().V1().Secrets(), client, TokensControllerOptions{TokenGenerator: generator, RootCA: []byte("CA Data"), MaxRetries: tc.MaxRetries})
			if err != nil {
				t.Fatalf("error creating Tokens controller: %v", err)
			}

			if tc.ExistingServiceAccount != nil {
				serviceAccounts.Add(tc.ExistingServiceAccount)
			}
			for _, s := range tc.ExistingSecrets {
				secrets.Add(s)
			}

			if tc.AddedServiceAccount != nil {
				serviceAccounts.Add(tc.AddedServiceAccount)
				controller.queueServiceAccountSync(tc.AddedServiceAccount)
			}
			if tc.UpdatedServiceAccount != nil {
				serviceAccounts.Add(tc.UpdatedServiceAccount)
				controller.queueServiceAccountUpdateSync(nil, tc.UpdatedServiceAccount)
			}
			if tc.DeletedServiceAccount != nil {
				serviceAccounts.Delete(tc.DeletedServiceAccount)
				controller.queueServiceAccountSync(tc.DeletedServiceAccount)
			}
			if tc.AddedSecret != nil {
				secrets.Add(tc.AddedSecret)
				controller.queueSecretSync(tc.AddedSecret)
			}
			if tc.AddedSecretLocal != nil {
				controller.updatedSecrets.Mutation(tc.AddedSecretLocal)
			}
			if tc.UpdatedSecret != nil {
				secrets.Add(tc.UpdatedSecret)
				controller.queueSecretUpdateSync(nil, tc.UpdatedSecret)
			}
			if tc.DeletedSecret != nil {
				secrets.Delete(tc.DeletedSecret)
				controller.queueSecretSync(tc.DeletedSecret)
			}

			// This is the longest we'll wait for async tests
			timeout := time.Now().Add(30 * time.Second)
			waitedForAdditionalActions := false

			for {
				if controller.syncServiceAccountQueue.Len() > 0 {
					controller.syncServiceAccount(ctx)
				}
				if controller.syncSecretQueue.Len() > 0 {
					controller.syncSecret(ctx)
				}

				// The queues still have things to work on
				if controller.syncServiceAccountQueue.Len() > 0 || controller.syncSecretQueue.Len() > 0 {
					continue
				}

				// If we expect this test to work asynchronously...
				if tc.IsAsync {
					// if we're still missing expected actions within our test timeout
					if len(client.Actions()) < len(tc.ExpectedActions) && time.Now().Before(timeout) {
						// wait for the expected actions (without hotlooping)
						time.Sleep(time.Millisecond)
						continue
					}

					// if we exactly match our expected actions, wait a bit to make sure no other additional actions show up
					if len(client.Actions()) == len(tc.ExpectedActions) && !waitedForAdditionalActions {
						time.Sleep(time.Second)
						waitedForAdditionalActions = true
						continue
					}
				}

				break
			}

			if controller.syncServiceAccountQueue.Len() > 0 {
				t.Errorf("%s: unexpected items in service account queue: %d", k, controller.syncServiceAccountQueue.Len())
			}
			if controller.syncSecretQueue.Len() > 0 {
				t.Errorf("%s: unexpected items in secret queue: %d", k, controller.syncSecretQueue.Len())
			}

			actions := client.Actions()
			for i, action := range actions {
				if len(tc.ExpectedActions) < i+1 {
					t.Errorf("%s: %d unexpected actions: %+v", k, len(actions)-len(tc.ExpectedActions), actions[i:])
					break
				}

				expectedAction := tc.ExpectedActions[i]
				if !reflect.DeepEqual(expectedAction, action) {
					t.Errorf("%s:\nExpected:\n%s\ngot:\n%s", k, dump.Pretty(expectedAction), dump.Pretty(action))
					continue
				}
			}

			if len(tc.ExpectedActions) > len(actions) {
				t.Errorf("%s: %d additional expected actions", k, len(tc.ExpectedActions)-len(actions))
				for _, a := range tc.ExpectedActions[len(actions):] {
					t.Logf("    %+v", a)
				}
			}
		})
	}
}
