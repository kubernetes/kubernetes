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

// This file tests authentication and (soon) authorization of HTTP requests to a master object.
// It does not use the client in pkg/client/... because authentication and authorization needs
// to work for any client of the HTTP interface.

import (
	"context"
	"crypto/rand"
	"crypto/rsa"
	"fmt"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/request/bearertoken"
	"k8s.io/apiserver/pkg/authentication/request/union"
	serviceaccountapiserver "k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/controller"
	serviceaccountcontroller "k8s.io/kubernetes/pkg/controller/serviceaccount"
	"k8s.io/kubernetes/pkg/serviceaccount"
	serviceaccountadmission "k8s.io/kubernetes/plugin/pkg/admission/serviceaccount"
	"k8s.io/kubernetes/test/integration/framework"
)

const (
	rootUserName = "root"
	rootToken    = "root-user-token"

	readOnlyServiceAccountName  = "ro"
	readWriteServiceAccountName = "rw"
)

func TestServiceAccountAutoCreate(t *testing.T) {
	c, _, stopFunc, err := startServiceAccountTestServer(t)
	defer stopFunc()
	if err != nil {
		t.Fatalf("failed to setup ServiceAccounts server: %v", err)
	}

	ns := "test-service-account-creation"

	// Create namespace
	_, err = c.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}})
	if err != nil {
		t.Fatalf("could not create namespace: %v", err)
	}

	// Get service account
	defaultUser, err := getServiceAccount(c, ns, "default", true)
	if err != nil {
		t.Fatalf("Default serviceaccount not created: %v", err)
	}

	// Delete service account
	err = c.CoreV1().ServiceAccounts(ns).Delete(defaultUser.Name, nil)
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
	c, _, stopFunc, err := startServiceAccountTestServer(t)
	defer stopFunc()
	if err != nil {
		t.Fatalf("failed to setup ServiceAccounts server: %v", err)
	}

	ns := "test-service-account-token-creation"
	name := "my-service-account"

	// Create namespace
	_, err = c.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}})
	if err != nil {
		t.Fatalf("could not create namespace: %v", err)
	}

	// Create service account
	_, err = c.CoreV1().ServiceAccounts(ns).Create(&v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: name}})
	if err != nil {
		t.Fatalf("Service Account not created: %v", err)
	}

	// Get token
	token1Name, token1, err := getReferencedServiceAccountToken(c, ns, name, true)
	if err != nil {
		t.Fatal(err)
	}

	// Delete token
	err = c.CoreV1().Secrets(ns).Delete(token1Name, nil)
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
	serviceAccount, err := c.CoreV1().ServiceAccounts(ns).Get(name, metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
	serviceAccount.Secrets = []v1.ObjectReference{}
	_, err = c.CoreV1().ServiceAccounts(ns).Update(serviceAccount)
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
	err = c.CoreV1().ServiceAccounts(ns).Delete(name, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Wait for tokens to be deleted
	tokensToCleanup := sets.NewString(token1Name, token2Name, token3Name)
	err = wait.Poll(time.Second, 10*time.Second, func() (bool, error) {
		// Get all secrets in the namespace
		secrets, err := c.CoreV1().Secrets(ns).List(metav1.ListOptions{})
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
	c, _, stopFunc, err := startServiceAccountTestServer(t)
	defer stopFunc()
	if err != nil {
		t.Fatalf("failed to setup ServiceAccounts server: %v", err)
	}

	ns := "auto-mount-ns"

	// Create "my" namespace
	_, err = c.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ns}})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatalf("could not create namespace: %v", err)
	}

	// Get default token
	defaultTokenName, _, err := getReferencedServiceAccountToken(c, ns, serviceaccountadmission.DefaultServiceAccountName, true)
	if err != nil {
		t.Fatal(err)
	}

	// Pod to create
	protoPod := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "protopod"},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  "container-1",
					Image: "container-1-image",
				},
				{
					Name:  "container-2",
					Image: "container-2-image",
					VolumeMounts: []v1.VolumeMount{
						{Name: "empty-dir", MountPath: serviceaccountadmission.DefaultAPITokenMountPath},
					},
				},
			},
			Volumes: []v1.Volume{
				{
					Name:         "empty-dir",
					VolumeSource: v1.VolumeSource{EmptyDir: &v1.EmptyDirVolumeSource{}},
				},
			},
		},
	}

	// Pod we expect to get created
	defaultMode := int32(0644)
	expectedServiceAccount := serviceaccountadmission.DefaultServiceAccountName
	expectedVolumes := append(protoPod.Spec.Volumes, v1.Volume{
		Name: defaultTokenName,
		VolumeSource: v1.VolumeSource{
			Secret: &v1.SecretVolumeSource{
				SecretName:  defaultTokenName,
				DefaultMode: &defaultMode,
			},
		},
	})
	expectedContainer1VolumeMounts := []v1.VolumeMount{
		{Name: defaultTokenName, MountPath: serviceaccountadmission.DefaultAPITokenMountPath, ReadOnly: true},
	}
	expectedContainer2VolumeMounts := protoPod.Spec.Containers[1].VolumeMounts

	createdPod, err := c.CoreV1().Pods(ns).Create(&protoPod)
	if err != nil {
		t.Fatal(err)
	}
	if createdPod.Spec.ServiceAccountName != expectedServiceAccount {
		t.Fatalf("Expected %s, got %s", expectedServiceAccount, createdPod.Spec.ServiceAccountName)
	}
	if !apiequality.Semantic.DeepEqual(&expectedVolumes, &createdPod.Spec.Volumes) {
		t.Fatalf("Expected\n\t%#v\n\tgot\n\t%#v", expectedVolumes, createdPod.Spec.Volumes)
	}
	if !apiequality.Semantic.DeepEqual(&expectedContainer1VolumeMounts, &createdPod.Spec.Containers[0].VolumeMounts) {
		t.Fatalf("Expected\n\t%#v\n\tgot\n\t%#v", expectedContainer1VolumeMounts, createdPod.Spec.Containers[0].VolumeMounts)
	}
	if !apiequality.Semantic.DeepEqual(&expectedContainer2VolumeMounts, &createdPod.Spec.Containers[1].VolumeMounts) {
		t.Fatalf("Expected\n\t%#v\n\tgot\n\t%#v", expectedContainer2VolumeMounts, createdPod.Spec.Containers[1].VolumeMounts)
	}
}

func TestServiceAccountTokenAuthentication(t *testing.T) {
	c, config, stopFunc, err := startServiceAccountTestServer(t)
	defer stopFunc()
	if err != nil {
		t.Fatalf("failed to setup ServiceAccounts server: %v", err)
	}

	myns := "auth-ns"
	otherns := "other-ns"

	// Create "my" namespace
	_, err = c.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: myns}})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatalf("could not create namespace: %v", err)
	}

	// Create "other" namespace
	_, err = c.CoreV1().Namespaces().Create(&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: otherns}})
	if err != nil && !apierrors.IsAlreadyExists(err) {
		t.Fatalf("could not create namespace: %v", err)
	}

	// Create "ro" user in myns
	_, err = c.CoreV1().ServiceAccounts(myns).Create(&v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: readOnlyServiceAccountName}})
	if err != nil {
		t.Fatalf("Service Account not created: %v", err)
	}
	roTokenName, roToken, err := getReferencedServiceAccountToken(c, myns, readOnlyServiceAccountName, true)
	if err != nil {
		t.Fatal(err)
	}
	roClientConfig := config
	roClientConfig.BearerToken = roToken
	roClient := clientset.NewForConfigOrDie(&roClientConfig)
	doServiceAccountAPIRequests(t, roClient, myns, true, true, false)
	doServiceAccountAPIRequests(t, roClient, otherns, true, false, false)
	err = c.CoreV1().Secrets(myns).Delete(roTokenName, nil)
	if err != nil {
		t.Fatalf("could not delete token: %v", err)
	}
	// wait for delete to be observed and reacted to via watch
	wait.PollImmediate(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		sa, err := c.CoreV1().ServiceAccounts(myns).Get(readOnlyServiceAccountName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		for _, secretRef := range sa.Secrets {
			if secretRef.Name == roTokenName {
				return false, nil
			}
		}
		return true, nil
	})
	doServiceAccountAPIRequests(t, roClient, myns, false, false, false)

	// Create "rw" user in myns
	_, err = c.CoreV1().ServiceAccounts(myns).Create(&v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: readWriteServiceAccountName}})
	if err != nil {
		t.Fatalf("Service Account not created: %v", err)
	}
	_, rwToken, err := getReferencedServiceAccountToken(c, myns, readWriteServiceAccountName, true)
	if err != nil {
		t.Fatal(err)
	}
	rwClientConfig := config
	rwClientConfig.BearerToken = rwToken
	rwClient := clientset.NewForConfigOrDie(&rwClientConfig)
	doServiceAccountAPIRequests(t, rwClient, myns, true, true, true)
	doServiceAccountAPIRequests(t, rwClient, otherns, true, false, false)

	// Get default user and token which should have been automatically created
	_, defaultToken, err := getReferencedServiceAccountToken(c, myns, "default", true)
	if err != nil {
		t.Fatalf("could not get default user and token: %v", err)
	}
	defaultClientConfig := config
	defaultClientConfig.BearerToken = defaultToken
	defaultClient := clientset.NewForConfigOrDie(&defaultClientConfig)
	doServiceAccountAPIRequests(t, defaultClient, myns, true, false, false)
}

// startServiceAccountTestServer returns a started server
// It is the responsibility of the caller to ensure the returned stopFunc is called
func startServiceAccountTestServer(t *testing.T) (*clientset.Clientset, restclient.Config, func(), error) {
	// Listener
	h := &framework.MasterHolder{Initialized: make(chan struct{})}
	apiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		<-h.Initialized
		h.M.GenericAPIServer.Handler.ServeHTTP(w, req)
	}))

	// Anonymous client config
	clientConfig := restclient.Config{Host: apiServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}}
	// Root client
	// TODO: remove rootClient after we refactor pkg/admission to use the clientset.
	rootClientset := clientset.NewForConfigOrDie(&restclient.Config{Host: apiServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}, BearerToken: rootToken})
	externalRootClientset := clientset.NewForConfigOrDie(&restclient.Config{Host: apiServer.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}, BearerToken: rootToken})

	externalInformers := informers.NewSharedInformerFactory(externalRootClientset, controller.NoResyncPeriodFunc())
	informers := informers.NewSharedInformerFactory(rootClientset, controller.NoResyncPeriodFunc())

	// Set up two authenticators:
	// 1. A token authenticator that maps the rootToken to the "root" user
	// 2. A ServiceAccountToken authenticator that validates ServiceAccount tokens
	rootTokenAuth := authenticator.TokenFunc(func(ctx context.Context, token string) (*authenticator.Response, bool, error) {
		if token == rootToken {
			return &authenticator.Response{User: &user.DefaultInfo{Name: rootUserName}}, true, nil
		}
		return nil, false, nil
	})
	serviceAccountKey, _ := rsa.GenerateKey(rand.Reader, 2048)
	serviceAccountTokenGetter := serviceaccountcontroller.NewGetterFromClient(
		rootClientset,
		externalInformers.Core().V1().Secrets().Lister(),
		externalInformers.Core().V1().ServiceAccounts().Lister(),
		externalInformers.Core().V1().Pods().Lister(),
	)
	serviceAccountTokenAuth := serviceaccount.JWTTokenAuthenticator(serviceaccount.LegacyIssuer, []interface{}{&serviceAccountKey.PublicKey}, nil, serviceaccount.NewLegacyValidator(true, serviceAccountTokenGetter))
	authenticator := union.New(
		bearertoken.New(rootTokenAuth),
		bearertoken.New(serviceAccountTokenAuth),
	)

	// Set up a stub authorizer:
	// 1. The "root" user is allowed to do anything
	// 2. ServiceAccounts named "ro" are allowed read-only operations in their namespace
	// 3. ServiceAccounts named "rw" are allowed any operation in their namespace
	authorizer := authorizer.AuthorizerFunc(func(attrs authorizer.Attributes) (authorizer.Decision, string, error) {
		username := ""
		if user := attrs.GetUser(); user != nil {
			username = user.GetName()
		}
		ns := attrs.GetNamespace()

		// If the user is "root"...
		if username == rootUserName {
			// allow them to do anything
			return authorizer.DecisionAllow, "", nil
		}

		// If the user is a service account...
		if serviceAccountNamespace, serviceAccountName, err := serviceaccountapiserver.SplitUsername(username); err == nil {
			// Limit them to their own namespace
			if serviceAccountNamespace == ns {
				switch serviceAccountName {
				case readOnlyServiceAccountName:
					if attrs.IsReadOnly() {
						return authorizer.DecisionAllow, "", nil
					}
				case readWriteServiceAccountName:
					return authorizer.DecisionAllow, "", nil
				}
			}
		}

		return authorizer.DecisionNoOpinion, fmt.Sprintf("User %s is denied (ns=%s, readonly=%v, resource=%s)", username, ns, attrs.IsReadOnly(), attrs.GetResource()), nil
	})

	// Set up admission plugin to auto-assign serviceaccounts to pods
	serviceAccountAdmission := serviceaccountadmission.NewServiceAccount()
	serviceAccountAdmission.SetExternalKubeClientSet(externalRootClientset)
	serviceAccountAdmission.SetExternalKubeInformerFactory(externalInformers)

	masterConfig := framework.NewMasterConfig()
	masterConfig.GenericConfig.EnableIndex = true
	masterConfig.GenericConfig.Authentication.Authenticator = authenticator
	masterConfig.GenericConfig.Authorization.Authorizer = authorizer
	masterConfig.GenericConfig.AdmissionControl = serviceAccountAdmission
	_, _, kubeAPIServerCloseFn := framework.RunAMasterUsingServer(masterConfig, apiServer, h)

	// Start the service account and service account token controllers
	stopCh := make(chan struct{})
	stop := func() {
		close(stopCh)
		kubeAPIServerCloseFn()
		apiServer.Close()
	}

	tokenGenerator, err := serviceaccount.JWTTokenGenerator(serviceaccount.LegacyIssuer, serviceAccountKey)
	if err != nil {
		return rootClientset, clientConfig, stop, err
	}
	tokenController, err := serviceaccountcontroller.NewTokensController(
		informers.Core().V1().ServiceAccounts(),
		informers.Core().V1().Secrets(),
		rootClientset,
		serviceaccountcontroller.TokensControllerOptions{TokenGenerator: tokenGenerator},
	)
	if err != nil {
		return rootClientset, clientConfig, stop, err
	}
	go tokenController.Run(1, stopCh)

	serviceAccountController, err := serviceaccountcontroller.NewServiceAccountsController(
		informers.Core().V1().ServiceAccounts(),
		informers.Core().V1().Namespaces(),
		rootClientset,
		serviceaccountcontroller.DefaultServiceAccountsControllerOptions(),
	)
	if err != nil {
		return rootClientset, clientConfig, stop, err
	}
	informers.Start(stopCh)
	externalInformers.Start(stopCh)
	go serviceAccountController.Run(5, stopCh)

	return rootClientset, clientConfig, stop, nil
}

func getServiceAccount(c *clientset.Clientset, ns string, name string, shouldWait bool) (*v1.ServiceAccount, error) {
	if !shouldWait {
		return c.CoreV1().ServiceAccounts(ns).Get(name, metav1.GetOptions{})
	}

	var user *v1.ServiceAccount
	var err error
	err = wait.Poll(time.Second, 10*time.Second, func() (bool, error) {
		user, err = c.CoreV1().ServiceAccounts(ns).Get(name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		return true, nil
	})
	return user, err
}

func getReferencedServiceAccountToken(c *clientset.Clientset, ns string, name string, shouldWait bool) (string, string, error) {
	tokenName := ""
	token := ""

	findToken := func() (bool, error) {
		user, err := c.CoreV1().ServiceAccounts(ns).Get(name, metav1.GetOptions{})
		if apierrors.IsNotFound(err) {
			return false, nil
		}
		if err != nil {
			return false, err
		}

		for _, ref := range user.Secrets {
			secret, err := c.CoreV1().Secrets(ns).Get(ref.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				continue
			}
			if err != nil {
				return false, err
			}
			if secret.Type != v1.SecretTypeServiceAccountToken {
				continue
			}
			name := secret.Annotations[v1.ServiceAccountNameKey]
			uid := secret.Annotations[v1.ServiceAccountUIDKey]
			tokenData := secret.Data[v1.ServiceAccountTokenKey]
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

func doServiceAccountAPIRequests(t *testing.T, c *clientset.Clientset, ns string, authenticated bool, canRead bool, canWrite bool) {
	testSecret := &v1.Secret{
		ObjectMeta: metav1.ObjectMeta{Name: "testSecret"},
		Data:       map[string][]byte{"test": []byte("data")},
	}

	readOps := []testOperation{
		func() error {
			_, err := c.CoreV1().Secrets(ns).List(metav1.ListOptions{})
			return err
		},
		func() error {
			_, err := c.CoreV1().Pods(ns).List(metav1.ListOptions{})
			return err
		},
	}
	writeOps := []testOperation{
		func() error { _, err := c.CoreV1().Secrets(ns).Create(testSecret); return err },
		func() error { return c.CoreV1().Secrets(ns).Delete(testSecret.Name, nil) },
	}

	for _, op := range readOps {
		err := op()
		unauthorizedError := apierrors.IsUnauthorized(err)
		forbiddenError := apierrors.IsForbidden(err)

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
		unauthorizedError := apierrors.IsUnauthorized(err)
		forbiddenError := apierrors.IsForbidden(err)

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
