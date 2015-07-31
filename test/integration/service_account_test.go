// +build integration,!no-etcd

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

package integration

// This file tests authentication and (soon) authorization of HTTP requests to a master object.
// It does not use the client in pkg/client/... because authentication and authorization needs
// to work for any client of the HTTP interface.

import (
	"crypto/rand"
	"crypto/rsa"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authenticator/bearertoken"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/authorizer"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/auth/user"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/controller/serviceaccount"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/master"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools/etcdtest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util/wait"
	serviceaccountadmission "github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/admission/serviceaccount"
	"github.com/GoogleCloudPlatform/kubernetes/plugin/pkg/auth/authenticator/request/union"
)

const (
	rootUserName = "root"
	rootToken    = "root-user-token"

	readOnlyServiceAccountName  = "ro"
	readWriteServiceAccountName = "rw"
)

func init() {
	requireEtcd()
}

func TestServiceAccountAutoCreate(t *testing.T) {
	c, _, stopFunc := startServiceAccountTestServer(t)
	defer stopFunc()

	ns := "test-service-account-creation"

	// Create namespace
	_, err := c.Namespaces().Create(&api.Namespace{ObjectMeta: api.ObjectMeta{Name: ns}})
	if err != nil {
		t.Fatalf("could not create namespace: %v", err)
	}

	// Get service account
	defaultUser, err := getServiceAccount(c, ns, "default", true)
	if err != nil {
		t.Fatalf("Default serviceaccount not created: %v", err)
	}

	// Delete service account
	err = c.ServiceAccounts(ns).Delete(defaultUser.Name)
	if err != nil {
		t.Fatalf("Could not delete default serviceaccount: %v", err)
	}

	// Get recreated service account
	defaultUser2, err := getServiceAccount(c, ns, "default", true)
	if err != nil {
		t.Fatalf("Default serviceaccount not created: %v", err)
	}
	if defaultUser2.UID == defaultUser.UID {
		t.Fatalf("Expected different UID with recreated serviceaccount")
	}
}

func TestServiceAccountTokenAutoCreate(t *testing.T) {
	c, _, stopFunc := startServiceAccountTestServer(t)
	defer stopFunc()

	ns := "test-service-account-token-creation"
	name := "my-service-account"

	// Create namespace
	_, err := c.Namespaces().Create(&api.Namespace{ObjectMeta: api.ObjectMeta{Name: ns}})
	if err != nil {
		t.Fatalf("could not create namespace: %v", err)
	}

	// Create service account
	serviceAccount, err := c.ServiceAccounts(ns).Create(&api.ServiceAccount{ObjectMeta: api.ObjectMeta{Name: name}})
	if err != nil {
		t.Fatalf("Service Account not created: %v", err)
	}

	// Get token
	token1Name, token1, err := getReferencedServiceAccountToken(c, ns, name, true)
	if err != nil {
		t.Fatal(err)
	}

	// Delete token
	err = c.Secrets(ns).Delete(token1Name)
	if err != nil {
		t.Fatalf("Could not delete token: %v", err)
	}

	// Get recreated token
	token2Name, token2, err := getReferencedServiceAccountToken(c, ns, name, true)
	if err != nil {
		t.Fatal(err)
	}
	if token1Name == token2Name {
		t.Fatalf("Expected new auto-created token name")
	}
	if token1 == token2 {
		t.Fatalf("Expected new auto-created token value")
	}

	// Trigger creation of a new referenced token
	serviceAccount, err = c.ServiceAccounts(ns).Get(name)
	if err != nil {
		t.Fatal(err)
	}
	serviceAccount.Secrets = []api.ObjectReference{}
	_, err = c.ServiceAccounts(ns).Update(serviceAccount)
	if err != nil {
		t.Fatal(err)
	}

	// Get rotated token
	token3Name, token3, err := getReferencedServiceAccountToken(c, ns, name, true)
	if err != nil {
		t.Fatal(err)
	}
	if token3Name == token2Name {
		t.Fatalf("Expected new auto-created token name")
	}
	if token3 == token2 {
		t.Fatalf("Expected new auto-created token value")
	}

	// Delete service account
	err = c.ServiceAccounts(ns).Delete(name)
	if err != nil {
		t.Fatal(err)
	}

	// Wait for tokens to be deleted
	tokensToCleanup := util.NewStringSet(token1Name, token2Name, token3Name)
	err = wait.Poll(time.Second, 10*time.Second, func() (bool, error) {
		// Get all secrets in the namespace
		secrets, err := c.Secrets(ns).List(labels.Everything(), fields.Everything())
		// Retrieval errors should fail
		if err != nil {
			return false, err
		}
		for _, s := range secrets.Items {
			if tokensToCleanup.Has(s.Name) {
				// Still waiting for tokens to be cleaned up
				return false, nil
			}
		}
		// All clean
		return true, nil
	})
	if err != nil {
		t.Fatalf("Error waiting for tokens to be deleted: %v", err)
	}
}

func TestServiceAccountTokenAutoMount(t *testing.T) {
	c, _, stopFunc := startServiceAccountTestServer(t)
	defer stopFunc()

	ns := "auto-mount-ns"

	// Create "my" namespace
	_, err := c.Namespaces().Create(&api.Namespace{ObjectMeta: api.ObjectMeta{Name: ns}})
	if err != nil && !errors.IsAlreadyExists(err) {
		t.Fatalf("could not create namespace: %v", err)
	}

	// Get default token
	defaultTokenName, _, err := getReferencedServiceAccountToken(c, ns, serviceaccountadmission.DefaultServiceAccountName, true)
	if err != nil {
		t.Fatal(err)
	}

	// Pod to create
	protoPod := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "protopod"},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:  "container-1",
					Image: "container-1-image",
				},
				{
					Name:  "container-2",
					Image: "container-2-image",
					VolumeMounts: []api.VolumeMount{
						{Name: "empty-dir", MountPath: serviceaccountadmission.DefaultAPITokenMountPath},
					},
				},
			},
			Volumes: []api.Volume{
				{
					Name:         "empty-dir",
					VolumeSource: api.VolumeSource{EmptyDir: &api.EmptyDirVolumeSource{}},
				},
			},
		},
	}

	// Pod we expect to get created
	expectedServiceAccount := serviceaccountadmission.DefaultServiceAccountName
	expectedVolumes := append(protoPod.Spec.Volumes, api.Volume{
		Name: defaultTokenName,
		VolumeSource: api.VolumeSource{
			Secret: &api.SecretVolumeSource{
				SecretName: defaultTokenName,
			},
		},
	})
	expectedContainer1VolumeMounts := []api.VolumeMount{
		{Name: defaultTokenName, MountPath: serviceaccountadmission.DefaultAPITokenMountPath, ReadOnly: true},
	}
	expectedContainer2VolumeMounts := protoPod.Spec.Containers[1].VolumeMounts

	createdPod, err := c.Pods(ns).Create(&protoPod)
	if err != nil {
		t.Fatal(err)
	}
	if createdPod.Spec.ServiceAccountName != expectedServiceAccount {
		t.Fatalf("Expected %s, got %s", expectedServiceAccount, createdPod.Spec.ServiceAccountName)
	}
	if !api.Semantic.DeepEqual(&expectedVolumes, &createdPod.Spec.Volumes) {
		t.Fatalf("Expected\n\t%#v\n\tgot\n\t%#v", expectedVolumes, createdPod.Spec.Volumes)
	}
	if !api.Semantic.DeepEqual(&expectedContainer1VolumeMounts, &createdPod.Spec.Containers[0].VolumeMounts) {
		t.Fatalf("Expected\n\t%#v\n\tgot\n\t%#v", expectedContainer1VolumeMounts, createdPod.Spec.Containers[0].VolumeMounts)
	}
	if !api.Semantic.DeepEqual(&expectedContainer2VolumeMounts, &createdPod.Spec.Containers[1].VolumeMounts) {
		t.Fatalf("Expected\n\t%#v\n\tgot\n\t%#v", expectedContainer2VolumeMounts, createdPod.Spec.Containers[1].VolumeMounts)
	}
}

func TestServiceAccountTokenAuthentication(t *testing.T) {
	c, config, stopFunc := startServiceAccountTestServer(t)
	defer stopFunc()

	myns := "auth-ns"
	otherns := "other-ns"

	// Create "my" namespace
	_, err := c.Namespaces().Create(&api.Namespace{ObjectMeta: api.ObjectMeta{Name: myns}})
	if err != nil && !errors.IsAlreadyExists(err) {
		t.Fatalf("could not create namespace: %v", err)
	}

	// Create "other" namespace
	_, err = c.Namespaces().Create(&api.Namespace{ObjectMeta: api.ObjectMeta{Name: otherns}})
	if err != nil && !errors.IsAlreadyExists(err) {
		t.Fatalf("could not create namespace: %v", err)
	}

	// Create "ro" user in myns
	_, err = c.ServiceAccounts(myns).Create(&api.ServiceAccount{ObjectMeta: api.ObjectMeta{Name: readOnlyServiceAccountName}})
	if err != nil {
		t.Fatalf("Service Account not created: %v", err)
	}
	roTokenName, roToken, err := getReferencedServiceAccountToken(c, myns, readOnlyServiceAccountName, true)
	if err != nil {
		t.Fatal(err)
	}
	roClientConfig := config
	roClientConfig.BearerToken = roToken
	roClient := client.NewOrDie(&roClientConfig)
	doServiceAccountAPIRequests(t, roClient, myns, true, true, false)
	doServiceAccountAPIRequests(t, roClient, otherns, true, false, false)
	err = c.Secrets(myns).Delete(roTokenName)
	if err != nil {
		t.Fatalf("could not delete token: %v", err)
	}
	doServiceAccountAPIRequests(t, roClient, myns, false, false, false)

	// Create "rw" user in myns
	_, err = c.ServiceAccounts(myns).Create(&api.ServiceAccount{ObjectMeta: api.ObjectMeta{Name: readWriteServiceAccountName}})
	if err != nil {
		t.Fatalf("Service Account not created: %v", err)
	}
	_, rwToken, err := getReferencedServiceAccountToken(c, myns, readWriteServiceAccountName, true)
	if err != nil {
		t.Fatal(err)
	}
	rwClientConfig := config
	rwClientConfig.BearerToken = rwToken
	rwClient := client.NewOrDie(&rwClientConfig)
	doServiceAccountAPIRequests(t, rwClient, myns, true, true, true)
	doServiceAccountAPIRequests(t, rwClient, otherns, true, false, false)

	// Get default user and token which should have been automatically created
	_, defaultToken, err := getReferencedServiceAccountToken(c, myns, "default", true)
	if err != nil {
		t.Fatalf("could not get default user and token: %v", err)
	}
	defaultClientConfig := config
	defaultClientConfig.BearerToken = defaultToken
	defaultClient := client.NewOrDie(&defaultClientConfig)
	doServiceAccountAPIRequests(t, defaultClient, myns, true, false, false)
}

// startServiceAccountTestServer returns a started server
// It is the responsibility of the caller to ensure the returned stopFunc is called
func startServiceAccountTestServer(t *testing.T) (*client.Client, client.Config, func()) {

	deleteAllEtcdKeys()

	// Etcd
	etcdStorage, err := master.NewEtcdStorage(newEtcdClient(), latest.InterfacesFor, testapi.Version(), etcdtest.PathPrefix())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Listener
	var m *master.Master
	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		m.Handler.ServeHTTP(w, req)
	}))

	// Anonymous client config
	clientConfig := client.Config{Host: apiServer.URL, Version: testapi.Version()}
	// Root client
	rootClient := client.NewOrDie(&client.Config{Host: apiServer.URL, Version: testapi.Version(), BearerToken: rootToken})

	// Set up two authenticators:
	// 1. A token authenticator that maps the rootToken to the "root" user
	// 2. A ServiceAccountToken authenticator that validates ServiceAccount tokens
	rootTokenAuth := authenticator.TokenFunc(func(token string) (user.Info, bool, error) {
		if token == rootToken {
			return &user.DefaultInfo{rootUserName, "", []string{}}, true, nil
		}
		return nil, false, nil
	})
	serviceAccountKey, err := rsa.GenerateKey(rand.Reader, 2048)
	serviceAccountTokenGetter := serviceaccount.NewGetterFromClient(rootClient)
	serviceAccountTokenAuth := serviceaccount.JWTTokenAuthenticator([]*rsa.PublicKey{&serviceAccountKey.PublicKey}, true, serviceAccountTokenGetter)
	authenticator := union.New(
		bearertoken.New(rootTokenAuth),
		bearertoken.New(serviceAccountTokenAuth),
	)

	// Set up a stub authorizer:
	// 1. The "root" user is allowed to do anything
	// 2. ServiceAccounts named "ro" are allowed read-only operations in their namespace
	// 3. ServiceAccounts named "rw" are allowed any operation in their namespace
	authorizer := authorizer.AuthorizerFunc(func(attrs authorizer.Attributes) error {
		username := attrs.GetUserName()
		ns := attrs.GetNamespace()

		// If the user is "root"...
		if username == rootUserName {
			// allow them to do anything
			return nil
		}

		// If the user is a service account...
		if serviceAccountNamespace, serviceAccountName, err := serviceaccount.SplitUsername(username); err == nil {
			// Limit them to their own namespace
			if serviceAccountNamespace == ns {
				switch serviceAccountName {
				case readOnlyServiceAccountName:
					if attrs.IsReadOnly() {
						return nil
					}
				case readWriteServiceAccountName:
					return nil
				}
			}
		}

		return fmt.Errorf("User %s is denied (ns=%s, readonly=%v, resource=%s)", username, ns, attrs.IsReadOnly(), attrs.GetResource())
	})

	// Set up admission plugin to auto-assign serviceaccounts to pods
	serviceAccountAdmission := serviceaccountadmission.NewServiceAccount(rootClient)

	// Create a master and install handlers into mux.
	m = master.New(&master.Config{
		DatabaseStorage:   etcdStorage,
		KubeletClient:     client.FakeKubeletClient{},
		EnableLogsSupport: false,
		EnableUISupport:   false,
		EnableIndex:       true,
		APIPrefix:         "/api",
		Authenticator:     authenticator,
		Authorizer:        authorizer,
		AdmissionControl:  serviceAccountAdmission,
	})

	// Start the service account and service account token controllers
	tokenController := serviceaccount.NewTokensController(rootClient, serviceaccount.TokensControllerOptions{TokenGenerator: serviceaccount.JWTTokenGenerator(serviceAccountKey)})
	tokenController.Run()
	serviceAccountController := serviceaccount.NewServiceAccountsController(rootClient, serviceaccount.DefaultServiceAccountsControllerOptions())
	serviceAccountController.Run()
	// Start the admission plugin reflectors
	serviceAccountAdmission.Run()

	stop := func() {
		tokenController.Stop()
		serviceAccountController.Stop()
		serviceAccountAdmission.Stop()
		apiServer.Close()
	}

	return rootClient, clientConfig, stop
}

func getServiceAccount(c *client.Client, ns string, name string, shouldWait bool) (*api.ServiceAccount, error) {
	if !shouldWait {
		return c.ServiceAccounts(ns).Get(name)
	}

	var user *api.ServiceAccount
	var err error
	err = wait.Poll(time.Second, 10*time.Second, func() (bool, error) {
		user, err = c.ServiceAccounts(ns).Get(name)
		if errors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		return true, nil
	})
	return user, err
}

func getReferencedServiceAccountToken(c *client.Client, ns string, name string, shouldWait bool) (string, string, error) {
	tokenName := ""
	token := ""

	findToken := func() (bool, error) {
		user, err := c.ServiceAccounts(ns).Get(name)
		if errors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}

		for _, ref := range user.Secrets {
			secret, err := c.Secrets(ns).Get(ref.Name)
			if errors.IsNotFound(err) {
				continue
			}
			if err != nil {
				return false, err
			}
			if secret.Type != api.SecretTypeServiceAccountToken {
				continue
			}
			name := secret.Annotations[api.ServiceAccountNameKey]
			uid := secret.Annotations[api.ServiceAccountUIDKey]
			tokenData := secret.Data[api.ServiceAccountTokenKey]
			if name == user.Name && uid == string(user.UID) && len(tokenData) > 0 {
				tokenName = secret.Name
				token = string(tokenData)
				return true, nil
			}
		}

		return false, nil
	}

	if shouldWait {
		err := wait.Poll(time.Second, 10*time.Second, findToken)
		if err != nil {
			return "", "", err
		}
	} else {
		ok, err := findToken()
		if err != nil {
			return "", "", err
		}
		if !ok {
			return "", "", fmt.Errorf("No token found for %s/%s", ns, name)
		}
	}
	return tokenName, token, nil
}

type testOperation func() error

func doServiceAccountAPIRequests(t *testing.T, c *client.Client, ns string, authenticated bool, canRead bool, canWrite bool) {
	testSecret := &api.Secret{
		ObjectMeta: api.ObjectMeta{Name: "testSecret"},
		Data:       map[string][]byte{"test": []byte("data")},
	}

	readOps := []testOperation{
		func() error { _, err := c.Secrets(ns).List(labels.Everything(), fields.Everything()); return err },
		func() error { _, err := c.Pods(ns).List(labels.Everything(), fields.Everything()); return err },
	}
	writeOps := []testOperation{
		func() error { _, err := c.Secrets(ns).Create(testSecret); return err },
		func() error { return c.Secrets(ns).Delete(testSecret.Name) },
	}

	for _, op := range readOps {
		err := op()
		unauthorizedError := errors.IsUnauthorized(err)
		forbiddenError := errors.IsForbidden(err)

		switch {
		case !authenticated && !unauthorizedError:
			t.Fatalf("expected unauthorized error, got %v", err)
		case authenticated && unauthorizedError:
			t.Fatalf("unexpected unauthorized error: %v", err)
		case authenticated && canRead && forbiddenError:
			t.Fatalf("unexpected forbidden error: %v", err)
		case authenticated && !canRead && !forbiddenError:
			t.Fatalf("expected forbidden error, got: %v", err)
		}
	}

	for _, op := range writeOps {
		err := op()
		unauthorizedError := errors.IsUnauthorized(err)
		forbiddenError := errors.IsForbidden(err)

		switch {
		case !authenticated && !unauthorizedError:
			t.Fatalf("expected unauthorized error, got %v", err)
		case authenticated && unauthorizedError:
			t.Fatalf("unexpected unauthorized error: %v", err)
		case authenticated && canWrite && forbiddenError:
			t.Fatalf("unexpected forbidden error: %v", err)
		case authenticated && !canWrite && !forbiddenError:
			t.Fatalf("expected forbidden error, got: %v", err)
		}
	}
}
