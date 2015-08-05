/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/testclient"
	"k8s.io/kubernetes/pkg/runtime"
	utilrand "k8s.io/kubernetes/pkg/util/rand"
)

type testGenerator struct {
	GeneratedServiceAccounts []api.ServiceAccount
	GeneratedSecrets         []api.Secret
	Token                    string
	Err                      error
}

func (t *testGenerator) GenerateToken(serviceAccount api.ServiceAccount, secret api.Secret) (string, error) {
	t.GeneratedSecrets = append(t.GeneratedSecrets, secret)
	t.GeneratedServiceAccounts = append(t.GeneratedServiceAccounts, serviceAccount)
	return t.Token, t.Err
}

// emptySecretReferences is used by a service account without any secrets
func emptySecretReferences() []api.ObjectReference {
	return []api.ObjectReference{}
}

// missingSecretReferences is used by a service account that references secrets which do no exist
func missingSecretReferences() []api.ObjectReference {
	return []api.ObjectReference{{Name: "missing-secret-1"}}
}

// regularSecretReferences is used by a service account that references secrets which are not ServiceAccountTokens
func regularSecretReferences() []api.ObjectReference {
	return []api.ObjectReference{{Name: "regular-secret-1"}}
}

// tokenSecretReferences is used by a service account that references a ServiceAccountToken secret
func tokenSecretReferences() []api.ObjectReference {
	return []api.ObjectReference{{Name: "token-secret-1"}}
}

// addTokenSecretReference adds a reference to the ServiceAccountToken that will be created
func addTokenSecretReference(refs []api.ObjectReference) []api.ObjectReference {
	return append(refs, api.ObjectReference{Name: "default-token-fplln"})
}

// serviceAccount returns a service account with the given secret refs
func serviceAccount(secretRefs []api.ObjectReference) *api.ServiceAccount {
	return &api.ServiceAccount{
		ObjectMeta: api.ObjectMeta{
			Name:            "default",
			UID:             "12345",
			Namespace:       "default",
			ResourceVersion: "1",
		},
		Secrets: secretRefs,
	}
}

// opaqueSecret returns a persisted non-ServiceAccountToken secret named "regular-secret-1"
func opaqueSecret() *api.Secret {
	return &api.Secret{
		ObjectMeta: api.ObjectMeta{
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

// createdTokenSecret returns the ServiceAccountToken secret posted when creating a new token secret.
// Named "default-token-fplln", since that is the first generated name after rand.Seed(1)
func createdTokenSecret() *api.Secret {
	return &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name:      "default-token-fplln",
			Namespace: "default",
			Annotations: map[string]string{
				api.ServiceAccountNameKey: "default",
				api.ServiceAccountUIDKey:  "12345",
			},
		},
		Type: api.SecretTypeServiceAccountToken,
		Data: map[string][]byte{
			"token":  []byte("ABC"),
			"ca.crt": []byte("CA Data"),
		},
	}
}

// serviceAccountTokenSecret returns an existing ServiceAccountToken secret named "token-secret-1"
func serviceAccountTokenSecret() *api.Secret {
	return &api.Secret{
		ObjectMeta: api.ObjectMeta{
			Name:            "token-secret-1",
			Namespace:       "default",
			UID:             "23456",
			ResourceVersion: "1",
			Annotations: map[string]string{
				api.ServiceAccountNameKey: "default",
				api.ServiceAccountUIDKey:  "12345",
			},
		},
		Type: api.SecretTypeServiceAccountToken,
		Data: map[string][]byte{
			"token":  []byte("ABC"),
			"ca.crt": []byte("CA Data"),
		},
	}
}

// serviceAccountTokenSecretWithoutTokenData returns an existing ServiceAccountToken secret that lacks token data
func serviceAccountTokenSecretWithoutTokenData() *api.Secret {
	secret := serviceAccountTokenSecret()
	delete(secret.Data, api.ServiceAccountTokenKey)
	return secret
}

// serviceAccountTokenSecretWithoutCAData returns an existing ServiceAccountToken secret that lacks ca data
func serviceAccountTokenSecretWithoutCAData() *api.Secret {
	secret := serviceAccountTokenSecret()
	delete(secret.Data, api.ServiceAccountRootCAKey)
	return secret
}

// serviceAccountTokenSecretWithCAData returns an existing ServiceAccountToken secret with the specified ca data
func serviceAccountTokenSecretWithCAData(data []byte) *api.Secret {
	secret := serviceAccountTokenSecret()
	secret.Data[api.ServiceAccountRootCAKey] = data
	return secret
}

func TestTokenCreation(t *testing.T) {
	testcases := map[string]struct {
		ClientObjects []runtime.Object

		SecretsSyncPending         bool
		ServiceAccountsSyncPending bool

		ExistingServiceAccount *api.ServiceAccount
		ExistingSecrets        []*api.Secret

		AddedServiceAccount   *api.ServiceAccount
		UpdatedServiceAccount *api.ServiceAccount
		DeletedServiceAccount *api.ServiceAccount
		AddedSecret           *api.Secret
		UpdatedSecret         *api.Secret
		DeletedSecret         *api.Secret

		ExpectedActions []testclient.Action
	}{
		"new serviceaccount with no secrets": {
			ClientObjects: []runtime.Object{serviceAccount(emptySecretReferences()), createdTokenSecret()},

			AddedServiceAccount: serviceAccount(emptySecretReferences()),
			ExpectedActions: []testclient.Action{
				testclient.NewCreateAction("secrets", api.NamespaceDefault, createdTokenSecret()),
				testclient.NewGetAction("serviceaccounts", api.NamespaceDefault, "default"),
				testclient.NewUpdateAction("serviceaccounts", api.NamespaceDefault, serviceAccount(addTokenSecretReference(emptySecretReferences()))),
			},
		},
		"new serviceaccount with no secrets with unsynced secret store": {
			ClientObjects: []runtime.Object{serviceAccount(emptySecretReferences()), createdTokenSecret()},

			SecretsSyncPending: true,

			AddedServiceAccount: serviceAccount(emptySecretReferences()),
			ExpectedActions: []testclient.Action{
				testclient.NewCreateAction("secrets", api.NamespaceDefault, createdTokenSecret()),
				testclient.NewGetAction("serviceaccounts", api.NamespaceDefault, "default"),
				testclient.NewUpdateAction("serviceaccounts", api.NamespaceDefault, serviceAccount(addTokenSecretReference(emptySecretReferences()))),
			},
		},
		"new serviceaccount with missing secrets": {
			ClientObjects: []runtime.Object{serviceAccount(missingSecretReferences()), createdTokenSecret()},

			AddedServiceAccount: serviceAccount(missingSecretReferences()),
			ExpectedActions: []testclient.Action{
				testclient.NewCreateAction("secrets", api.NamespaceDefault, createdTokenSecret()),
				testclient.NewGetAction("serviceaccounts", api.NamespaceDefault, "default"),
				testclient.NewUpdateAction("serviceaccounts", api.NamespaceDefault, serviceAccount(addTokenSecretReference(missingSecretReferences()))),
			},
		},
		"new serviceaccount with missing secrets with unsynced secret store": {
			ClientObjects: []runtime.Object{serviceAccount(missingSecretReferences()), createdTokenSecret()},

			SecretsSyncPending: true,

			AddedServiceAccount: serviceAccount(missingSecretReferences()),
			ExpectedActions:     []testclient.Action{},
		},
		"new serviceaccount with non-token secrets": {
			ClientObjects: []runtime.Object{serviceAccount(regularSecretReferences()), createdTokenSecret(), opaqueSecret()},

			AddedServiceAccount: serviceAccount(regularSecretReferences()),
			ExpectedActions: []testclient.Action{
				testclient.NewCreateAction("secrets", api.NamespaceDefault, createdTokenSecret()),
				testclient.NewGetAction("serviceaccounts", api.NamespaceDefault, "default"),
				testclient.NewUpdateAction("serviceaccounts", api.NamespaceDefault, serviceAccount(addTokenSecretReference(regularSecretReferences()))),
			},
		},
		"new serviceaccount with token secrets": {
			ClientObjects:   []runtime.Object{serviceAccount(tokenSecretReferences()), serviceAccountTokenSecret()},
			ExistingSecrets: []*api.Secret{serviceAccountTokenSecret()},

			AddedServiceAccount: serviceAccount(tokenSecretReferences()),
			ExpectedActions:     []testclient.Action{},
		},

		"updated serviceaccount with no secrets": {
			ClientObjects: []runtime.Object{serviceAccount(emptySecretReferences()), createdTokenSecret()},

			UpdatedServiceAccount: serviceAccount(emptySecretReferences()),
			ExpectedActions: []testclient.Action{
				testclient.NewCreateAction("secrets", api.NamespaceDefault, createdTokenSecret()),
				testclient.NewGetAction("serviceaccounts", api.NamespaceDefault, "default"),
				testclient.NewUpdateAction("serviceaccounts", api.NamespaceDefault, serviceAccount(addTokenSecretReference(emptySecretReferences()))),
			},
		},
		"updated serviceaccount with no secrets with unsynced secret store": {
			ClientObjects: []runtime.Object{serviceAccount(emptySecretReferences()), createdTokenSecret()},

			SecretsSyncPending: true,

			UpdatedServiceAccount: serviceAccount(emptySecretReferences()),
			ExpectedActions: []testclient.Action{
				testclient.NewCreateAction("secrets", api.NamespaceDefault, createdTokenSecret()),
				testclient.NewGetAction("serviceaccounts", api.NamespaceDefault, "default"),
				testclient.NewUpdateAction("serviceaccounts", api.NamespaceDefault, serviceAccount(addTokenSecretReference(emptySecretReferences()))),
			},
		},
		"updated serviceaccount with missing secrets": {
			ClientObjects: []runtime.Object{serviceAccount(missingSecretReferences()), createdTokenSecret()},

			UpdatedServiceAccount: serviceAccount(missingSecretReferences()),
			ExpectedActions: []testclient.Action{
				testclient.NewCreateAction("secrets", api.NamespaceDefault, createdTokenSecret()),
				testclient.NewGetAction("serviceaccounts", api.NamespaceDefault, "default"),
				testclient.NewUpdateAction("serviceaccounts", api.NamespaceDefault, serviceAccount(addTokenSecretReference(missingSecretReferences()))),
			},
		},
		"updated serviceaccount with missing secrets with unsynced secret store": {
			ClientObjects: []runtime.Object{serviceAccount(missingSecretReferences()), createdTokenSecret()},

			SecretsSyncPending: true,

			UpdatedServiceAccount: serviceAccount(missingSecretReferences()),
			ExpectedActions:       []testclient.Action{},
		},
		"updated serviceaccount with non-token secrets": {
			ClientObjects: []runtime.Object{serviceAccount(regularSecretReferences()), createdTokenSecret(), opaqueSecret()},

			UpdatedServiceAccount: serviceAccount(regularSecretReferences()),
			ExpectedActions: []testclient.Action{
				testclient.NewCreateAction("secrets", api.NamespaceDefault, createdTokenSecret()),
				testclient.NewGetAction("serviceaccounts", api.NamespaceDefault, "default"),
				testclient.NewUpdateAction("serviceaccounts", api.NamespaceDefault, serviceAccount(addTokenSecretReference(regularSecretReferences()))),
			},
		},
		"updated serviceaccount with token secrets": {
			ExistingSecrets: []*api.Secret{serviceAccountTokenSecret()},

			UpdatedServiceAccount: serviceAccount(tokenSecretReferences()),
			ExpectedActions:       []testclient.Action{},
		},

		"deleted serviceaccount with no secrets": {
			DeletedServiceAccount: serviceAccount(emptySecretReferences()),
			ExpectedActions:       []testclient.Action{},
		},
		"deleted serviceaccount with missing secrets": {
			DeletedServiceAccount: serviceAccount(missingSecretReferences()),
			ExpectedActions:       []testclient.Action{},
		},
		"deleted serviceaccount with non-token secrets": {
			ClientObjects: []runtime.Object{opaqueSecret()},

			DeletedServiceAccount: serviceAccount(regularSecretReferences()),
			ExpectedActions:       []testclient.Action{},
		},
		"deleted serviceaccount with token secrets": {
			ClientObjects:   []runtime.Object{serviceAccountTokenSecret()},
			ExistingSecrets: []*api.Secret{serviceAccountTokenSecret()},

			DeletedServiceAccount: serviceAccount(tokenSecretReferences()),
			ExpectedActions: []testclient.Action{
				testclient.NewDeleteAction("secrets", api.NamespaceDefault, "token-secret-1"),
			},
		},

		"added secret without serviceaccount": {
			ClientObjects: []runtime.Object{serviceAccountTokenSecret()},

			AddedSecret: serviceAccountTokenSecret(),
			ExpectedActions: []testclient.Action{
				testclient.NewGetAction("serviceaccounts", api.NamespaceDefault, "default"),
				testclient.NewDeleteAction("secrets", api.NamespaceDefault, "token-secret-1"),
			},
		},
		"added secret with serviceaccount": {
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			AddedSecret:     serviceAccountTokenSecret(),
			ExpectedActions: []testclient.Action{},
		},
		"added token secret without token data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithoutTokenData()},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			AddedSecret: serviceAccountTokenSecretWithoutTokenData(),
			ExpectedActions: []testclient.Action{
				testclient.NewUpdateAction("secrets", api.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},
		"added token secret without ca data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithoutCAData()},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			AddedSecret: serviceAccountTokenSecretWithoutCAData(),
			ExpectedActions: []testclient.Action{
				testclient.NewUpdateAction("secrets", api.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},
		"added token secret with mismatched ca data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithCAData([]byte("mismatched"))},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			AddedSecret: serviceAccountTokenSecretWithCAData([]byte("mismatched")),
			ExpectedActions: []testclient.Action{
				testclient.NewUpdateAction("secrets", api.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},

		"updated secret without serviceaccount": {
			ClientObjects: []runtime.Object{serviceAccountTokenSecret()},

			UpdatedSecret: serviceAccountTokenSecret(),
			ExpectedActions: []testclient.Action{
				testclient.NewGetAction("serviceaccounts", api.NamespaceDefault, "default"),
				testclient.NewDeleteAction("secrets", api.NamespaceDefault, "token-secret-1"),
			},
		},
		"updated secret with serviceaccount": {
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			UpdatedSecret:   serviceAccountTokenSecret(),
			ExpectedActions: []testclient.Action{},
		},
		"updated token secret without token data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithoutTokenData()},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			UpdatedSecret: serviceAccountTokenSecretWithoutTokenData(),
			ExpectedActions: []testclient.Action{
				testclient.NewUpdateAction("secrets", api.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},
		"updated token secret without ca data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithoutCAData()},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			UpdatedSecret: serviceAccountTokenSecretWithoutCAData(),
			ExpectedActions: []testclient.Action{
				testclient.NewUpdateAction("secrets", api.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},
		"updated token secret with mismatched ca data": {
			ClientObjects:          []runtime.Object{serviceAccountTokenSecretWithCAData([]byte("mismatched"))},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			UpdatedSecret: serviceAccountTokenSecretWithCAData([]byte("mismatched")),
			ExpectedActions: []testclient.Action{
				testclient.NewUpdateAction("secrets", api.NamespaceDefault, serviceAccountTokenSecret()),
			},
		},

		"deleted secret without serviceaccount": {
			DeletedSecret:   serviceAccountTokenSecret(),
			ExpectedActions: []testclient.Action{},
		},
		"deleted secret with serviceaccount with reference": {
			ClientObjects:          []runtime.Object{serviceAccount(tokenSecretReferences())},
			ExistingServiceAccount: serviceAccount(tokenSecretReferences()),

			DeletedSecret: serviceAccountTokenSecret(),
			ExpectedActions: []testclient.Action{
				testclient.NewGetAction("serviceaccounts", api.NamespaceDefault, "default"),
				testclient.NewUpdateAction("serviceaccounts", api.NamespaceDefault, serviceAccount(emptySecretReferences())),
			},
		},
		"deleted secret with serviceaccount without reference": {
			ExistingServiceAccount: serviceAccount(emptySecretReferences()),

			DeletedSecret:   serviceAccountTokenSecret(),
			ExpectedActions: []testclient.Action{},
		},
	}

	for k, tc := range testcases {

		// Re-seed to reset name generation
		utilrand.Seed(1)

		generator := &testGenerator{Token: "ABC"}

		client := testclient.NewSimpleFake(tc.ClientObjects...)

		controller := NewTokensController(client, TokensControllerOptions{TokenGenerator: generator, RootCA: []byte("CA Data")})

		// Tell the token controller whether its stores have been synced
		controller.serviceAccountsSynced = func() bool { return !tc.ServiceAccountsSyncPending }
		controller.secretsSynced = func() bool { return !tc.SecretsSyncPending }

		if tc.ExistingServiceAccount != nil {
			controller.serviceAccounts.Add(tc.ExistingServiceAccount)
		}
		for _, s := range tc.ExistingSecrets {
			controller.secrets.Add(s)
		}

		if tc.AddedServiceAccount != nil {
			controller.serviceAccountAdded(tc.AddedServiceAccount)
		}
		if tc.UpdatedServiceAccount != nil {
			controller.serviceAccountUpdated(nil, tc.UpdatedServiceAccount)
		}
		if tc.DeletedServiceAccount != nil {
			controller.serviceAccountDeleted(tc.DeletedServiceAccount)
		}
		if tc.AddedSecret != nil {
			controller.secretAdded(tc.AddedSecret)
		}
		if tc.UpdatedSecret != nil {
			controller.secretUpdated(nil, tc.UpdatedSecret)
		}
		if tc.DeletedSecret != nil {
			controller.secretDeleted(tc.DeletedSecret)
		}

		actions := client.Actions()
		for i, action := range actions {
			if len(tc.ExpectedActions) < i+1 {
				t.Errorf("%s: %d unexpected actions: %+v", k, len(actions)-len(tc.ExpectedActions), actions[i:])
				break
			}

			expectedAction := tc.ExpectedActions[i]
			if !reflect.DeepEqual(expectedAction, action) {
				t.Errorf("%s: Expected\n\t%#v\ngot\n\t%#v", k, expectedAction, action)
				continue
			}
		}

		if len(tc.ExpectedActions) > len(actions) {
			t.Errorf("%s: %d additional expected actions:%+v", k, len(tc.ExpectedActions)-len(actions), tc.ExpectedActions[len(actions):])
		}
	}
}
