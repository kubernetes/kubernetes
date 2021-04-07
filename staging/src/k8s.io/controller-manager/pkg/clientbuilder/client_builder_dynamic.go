/*
Copyright 2018 The Kubernetes Authors.

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

package clientbuilder

import (
	"context"
	"fmt"
	"net/http"
	"sync"
	"time"

	"golang.org/x/oauth2"
	v1authenticationapi "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/wait"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
	"k8s.io/client-go/discovery"
	clientset "k8s.io/client-go/kubernetes"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/transport"
	"k8s.io/klog/v2"
	utilpointer "k8s.io/utils/pointer"
)

var (
	// defaultExpirationSeconds defines the duration of a TokenRequest in seconds.
	defaultExpirationSeconds = int64(3600)
	// defaultLeewayPercent defines the percentage of expiration left before the client trigger a token rotation.
	// range[0, 100]
	defaultLeewayPercent = 20
)

type DynamicControllerClientBuilder struct {
	// ClientConfig is a skeleton config to clone and use as the basis for each controller client
	ClientConfig *restclient.Config

	// CoreClient is used to provision service accounts if needed and watch for their associated tokens
	// to construct a controller client
	CoreClient v1core.CoreV1Interface

	// Namespace is the namespace used to host the service accounts that will back the
	// controllers.  It must be highly privileged namespace which normal users cannot inspect.
	Namespace string

	// roundTripperFuncMap is a cache stores the corresponding roundtripper func for each
	// service account
	roundTripperFuncMap map[string]func(http.RoundTripper) http.RoundTripper

	// expirationSeconds defines the token expiration seconds
	expirationSeconds int64

	// leewayPercent defines the percentage of expiration left before the client trigger a token rotation.
	leewayPercent int

	mutex sync.Mutex

	clock clock.Clock
}

// NewDynamicClientBuilder returns client builder which uses TokenRequest feature and refresh service account token periodically
func NewDynamicClientBuilder(clientConfig *restclient.Config, coreClient v1core.CoreV1Interface, ns string) ControllerClientBuilder {
	builder := &DynamicControllerClientBuilder{
		ClientConfig:        clientConfig,
		CoreClient:          coreClient,
		Namespace:           ns,
		roundTripperFuncMap: map[string]func(http.RoundTripper) http.RoundTripper{},
		expirationSeconds:   defaultExpirationSeconds,
		leewayPercent:       defaultLeewayPercent,
		clock:               clock.RealClock{},
	}
	return builder
}

// this function only for test purpose, don't call it
func NewTestDynamicClientBuilder(clientConfig *restclient.Config, coreClient v1core.CoreV1Interface, ns string, expirationSeconds int64, leewayPercent int) ControllerClientBuilder {
	builder := &DynamicControllerClientBuilder{
		ClientConfig:        clientConfig,
		CoreClient:          coreClient,
		Namespace:           ns,
		roundTripperFuncMap: map[string]func(http.RoundTripper) http.RoundTripper{},
		expirationSeconds:   expirationSeconds,
		leewayPercent:       leewayPercent,
		clock:               clock.RealClock{},
	}
	return builder
}

func (t *DynamicControllerClientBuilder) Config(saName string) (*restclient.Config, error) {
	_, err := getOrCreateServiceAccount(t.CoreClient, t.Namespace, saName)
	if err != nil {
		return nil, err
	}

	configCopy := constructClient(t.Namespace, saName, t.ClientConfig)

	t.mutex.Lock()
	defer t.mutex.Unlock()

	rt, ok := t.roundTripperFuncMap[saName]
	if ok {
		configCopy.Wrap(rt)
	} else {
		cachedTokenSource := transport.NewCachedTokenSource(&tokenSourceImpl{
			namespace:          t.Namespace,
			serviceAccountName: saName,
			coreClient:         t.CoreClient,
			expirationSeconds:  t.expirationSeconds,
			leewayPercent:      t.leewayPercent,
		})
		configCopy.Wrap(transport.ResettableTokenSourceWrapTransport(cachedTokenSource))
		t.roundTripperFuncMap[saName] = configCopy.WrapTransport
	}

	return &configCopy, nil
}

func (t *DynamicControllerClientBuilder) ConfigOrDie(name string) *restclient.Config {
	clientConfig, err := t.Config(name)
	if err != nil {
		klog.Fatal(err)
	}
	return clientConfig
}

func (t *DynamicControllerClientBuilder) Client(name string) (clientset.Interface, error) {
	clientConfig, err := t.Config(name)
	if err != nil {
		return nil, err
	}
	return clientset.NewForConfig(clientConfig)
}

func (t *DynamicControllerClientBuilder) ClientOrDie(name string) clientset.Interface {
	client, err := t.Client(name)
	if err != nil {
		klog.Fatal(err)
	}
	return client
}

func (t *DynamicControllerClientBuilder) DiscoveryClient(name string) (discovery.DiscoveryInterface, error) {
	clientConfig, err := t.Config(name)
	if err != nil {
		return nil, err
	}
	// Discovery makes a lot of requests infrequently.  This allows the burst to succeed and refill to happen
	// in just a few seconds.
	clientConfig.Burst = 200
	clientConfig.QPS = 20
	return clientset.NewForConfig(clientConfig)
}

func (t *DynamicControllerClientBuilder) DiscoveryClientOrDie(name string) discovery.DiscoveryInterface {
	client, err := t.DiscoveryClient(name)
	if err != nil {
		klog.Fatal(err)
	}
	return client
}

type tokenSourceImpl struct {
	namespace          string
	serviceAccountName string
	coreClient         v1core.CoreV1Interface
	expirationSeconds  int64
	leewayPercent      int
}

func (ts *tokenSourceImpl) Token() (*oauth2.Token, error) {
	var retTokenRequest *v1authenticationapi.TokenRequest

	backoff := wait.Backoff{
		Duration: 500 * time.Millisecond,
		Factor:   2, // double the timeout for every failure
		Steps:    4,
	}
	if err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		if _, inErr := getOrCreateServiceAccount(ts.coreClient, ts.namespace, ts.serviceAccountName); inErr != nil {
			klog.Warningf("get or create service account failed: %v", inErr)
			return false, nil
		}

		tr, inErr := ts.coreClient.ServiceAccounts(ts.namespace).CreateToken(context.TODO(), ts.serviceAccountName, &v1authenticationapi.TokenRequest{
			Spec: v1authenticationapi.TokenRequestSpec{
				ExpirationSeconds: utilpointer.Int64Ptr(ts.expirationSeconds),
			},
		}, metav1.CreateOptions{})
		if inErr != nil {
			klog.Warningf("get token failed: %v", inErr)
			return false, nil
		}
		retTokenRequest = tr
		return true, nil
	}); err != nil {
		return nil, fmt.Errorf("failed to get token for %s/%s: %v", ts.namespace, ts.serviceAccountName, err)
	}

	if retTokenRequest.Spec.ExpirationSeconds == nil {
		return nil, fmt.Errorf("nil pointer of expiration in token request")
	}

	lifetime := retTokenRequest.Status.ExpirationTimestamp.Time.Sub(time.Now())
	if lifetime < time.Minute*10 {
		// possible clock skew issue, pin to minimum token lifetime
		lifetime = time.Minute * 10
	}

	leeway := time.Duration(int64(lifetime) * int64(ts.leewayPercent) / 100)
	expiry := time.Now().Add(lifetime).Add(-1 * leeway)

	return &oauth2.Token{
		AccessToken: retTokenRequest.Status.Token,
		TokenType:   "Bearer",
		Expiry:      expiry,
	}, nil
}

func constructClient(saNamespace, saName string, config *restclient.Config) restclient.Config {
	username := apiserverserviceaccount.MakeUsername(saNamespace, saName)
	// make a shallow copy
	// the caller already castrated the config during creation
	// this allows for potential extensions in the future
	// for example it preserve HTTP wrappers for custom behavior per request
	ret := *config
	restclient.AddUserAgent(&ret, username)
	return ret
}

func getOrCreateServiceAccount(coreClient v1core.CoreV1Interface, namespace, name string) (*v1.ServiceAccount, error) {
	sa, err := coreClient.ServiceAccounts(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err == nil {
		return sa, nil
	}
	if !apierrors.IsNotFound(err) {
		return nil, err
	}

	// Create the namespace if we can't verify it exists.
	// Tolerate errors, since we don't know whether this component has namespace creation permissions.
	if _, err := coreClient.Namespaces().Get(context.TODO(), namespace, metav1.GetOptions{}); apierrors.IsNotFound(err) {
		if _, err = coreClient.Namespaces().Create(context.TODO(), &v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: namespace}}, metav1.CreateOptions{}); err != nil && !apierrors.IsAlreadyExists(err) {
			klog.Warningf("create non-exist namespace %s failed:%v", namespace, err)
		}
	}

	// Create the service account
	sa, err = coreClient.ServiceAccounts(namespace).Create(context.TODO(), &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Namespace: namespace, Name: name}}, metav1.CreateOptions{})
	if apierrors.IsAlreadyExists(err) {
		// If we're racing to init and someone else already created it, re-fetch
		return coreClient.ServiceAccounts(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	}
	return sa, err
}
