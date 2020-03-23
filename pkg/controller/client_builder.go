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

package controller

import (
	"context"
	"fmt"
	"time"

	v1authenticationapi "k8s.io/api/authentication/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	apiserverserviceaccount "k8s.io/apiserver/pkg/authentication/serviceaccount"
	clientset "k8s.io/client-go/kubernetes"
	v1authentication "k8s.io/client-go/kubernetes/typed/authentication/v1"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/klog"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

// ControllerClientBuilder allows you to get clients and configs for controllers
// Please note a copy also exists in staging/src/k8s.io/cloud-provider/cloud.go
// TODO: Extract this into a separate controller utilities repo (issues/68947)
type ControllerClientBuilder interface {
	Config(name string) (*restclient.Config, error)
	ConfigOrDie(name string) *restclient.Config
	Client(name string) (clientset.Interface, error)
	ClientOrDie(name string) clientset.Interface
}

// SimpleControllerClientBuilder returns a fixed client with different user agents
type SimpleControllerClientBuilder struct {
	// ClientConfig is a skeleton config to clone and use as the basis for each controller client
	ClientConfig *restclient.Config
}

func (b SimpleControllerClientBuilder) Config(name string) (*restclient.Config, error) {
	clientConfig := *b.ClientConfig
	return restclient.AddUserAgent(&clientConfig, name), nil
}

func (b SimpleControllerClientBuilder) ConfigOrDie(name string) *restclient.Config {
	clientConfig, err := b.Config(name)
	if err != nil {
		klog.Fatal(err)
	}
	return clientConfig
}

func (b SimpleControllerClientBuilder) Client(name string) (clientset.Interface, error) {
	clientConfig, err := b.Config(name)
	if err != nil {
		return nil, err
	}
	return clientset.NewForConfig(clientConfig)
}

func (b SimpleControllerClientBuilder) ClientOrDie(name string) clientset.Interface {
	client, err := b.Client(name)
	if err != nil {
		klog.Fatal(err)
	}
	return client
}

// SAControllerClientBuilder is a ControllerClientBuilder that returns clients identifying as
// service accounts
type SAControllerClientBuilder struct {
	// ClientConfig is a skeleton config to clone and use as the basis for each controller client
	ClientConfig *restclient.Config

	// CoreClient is used to provision service accounts if needed and watch for their associated tokens
	// to construct a controller client
	CoreClient v1core.CoreV1Interface

	// AuthenticationClient is used to check API tokens to make sure they are valid before
	// building a controller client from them
	AuthenticationClient v1authentication.AuthenticationV1Interface

	// Namespace is the namespace used to host the service accounts that will back the
	// controllers.  It must be highly privileged namespace which normal users cannot inspect.
	Namespace string
}

// config returns a complete clientConfig for constructing clients.  This is separate in anticipation of composition
// which means that not all clientsets are known here
func (b SAControllerClientBuilder) Config(name string) (*restclient.Config, error) {
	sa, err := getOrCreateServiceAccount(b.CoreClient, b.Namespace, name)
	if err != nil {
		return nil, err
	}

	var clientConfig *restclient.Config
	fieldSelector := fields.SelectorFromSet(map[string]string{
		api.SecretTypeField: string(v1.SecretTypeServiceAccountToken),
	}).String()
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fieldSelector
			return b.CoreClient.Secrets(b.Namespace).List(context.TODO(), options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fieldSelector
			return b.CoreClient.Secrets(b.Namespace).Watch(context.TODO(), options)
		},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	_, err = watchtools.UntilWithSync(ctx, lw, &v1.Secret{}, nil,
		func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Deleted:
				return false, nil
			case watch.Error:
				return false, fmt.Errorf("error watching")

			case watch.Added, watch.Modified:
				secret, ok := event.Object.(*v1.Secret)
				if !ok {
					return false, fmt.Errorf("unexpected object type: %T", event.Object)
				}
				if !serviceaccount.IsServiceAccountToken(secret, sa) {
					return false, nil
				}
				if len(secret.Data[v1.ServiceAccountTokenKey]) == 0 {
					return false, nil
				}
				validConfig, valid, err := b.getAuthenticatedConfig(sa, string(secret.Data[v1.ServiceAccountTokenKey]))
				if err != nil {
					klog.Warningf("error validating API token for %s/%s in secret %s: %v", sa.Namespace, sa.Name, secret.Name, err)
					// continue watching for good tokens
					return false, nil
				}
				if !valid {
					klog.Warningf("secret %s contained an invalid API token for %s/%s", secret.Name, sa.Namespace, sa.Name)
					// try to delete the secret containing the invalid token
					if err := b.CoreClient.Secrets(secret.Namespace).Delete(context.TODO(), secret.Name, metav1.DeleteOptions{}); err != nil && !apierrors.IsNotFound(err) {
						klog.Warningf("error deleting secret %s containing invalid API token for %s/%s: %v", secret.Name, sa.Namespace, sa.Name, err)
					}
					// continue watching for good tokens
					return false, nil
				}
				clientConfig = validConfig
				return true, nil

			default:
				return false, fmt.Errorf("unexpected event type: %v", event.Type)
			}
		})
	if err != nil {
		return nil, fmt.Errorf("unable to get token for service account: %v", err)
	}

	return clientConfig, nil
}

func (b SAControllerClientBuilder) getAuthenticatedConfig(sa *v1.ServiceAccount, token string) (*restclient.Config, bool, error) {
	username := apiserverserviceaccount.MakeUsername(sa.Namespace, sa.Name)

	clientConfig := restclient.AnonymousClientConfig(b.ClientConfig)
	clientConfig.BearerToken = token
	restclient.AddUserAgent(clientConfig, username)

	// Try token review first
	tokenReview := &v1authenticationapi.TokenReview{Spec: v1authenticationapi.TokenReviewSpec{Token: token}}
	if tokenResult, err := b.AuthenticationClient.TokenReviews().Create(context.TODO(), tokenReview, metav1.CreateOptions{}); err == nil {
		if !tokenResult.Status.Authenticated {
			klog.Warningf("Token for %s/%s did not authenticate correctly", sa.Namespace, sa.Name)
			return nil, false, nil
		}
		if tokenResult.Status.User.Username != username {
			klog.Warningf("Token for %s/%s authenticated as unexpected username: %s", sa.Namespace, sa.Name, tokenResult.Status.User.Username)
			return nil, false, nil
		}
		klog.V(4).Infof("Verified credential for %s/%s", sa.Namespace, sa.Name)
		return clientConfig, true, nil
	}

	// If we couldn't run the token review, the API might be disabled or we might not have permission.
	// Try to make a request to /apis with the token. If we get a 401 we should consider the token invalid.
	clientConfigCopy := *clientConfig
	clientConfigCopy.NegotiatedSerializer = legacyscheme.Codecs
	client, err := restclient.UnversionedRESTClientFor(&clientConfigCopy)
	if err != nil {
		return nil, false, err
	}
	err = client.Get().AbsPath("/apis").Do(context.TODO()).Error()
	if apierrors.IsUnauthorized(err) {
		klog.Warningf("Token for %s/%s did not authenticate correctly: %v", sa.Namespace, sa.Name, err)
		return nil, false, nil
	}

	return clientConfig, true, nil
}

func (b SAControllerClientBuilder) ConfigOrDie(name string) *restclient.Config {
	clientConfig, err := b.Config(name)
	if err != nil {
		klog.Fatal(err)
	}
	return clientConfig
}

func (b SAControllerClientBuilder) Client(name string) (clientset.Interface, error) {
	clientConfig, err := b.Config(name)
	if err != nil {
		return nil, err
	}
	return clientset.NewForConfig(clientConfig)
}

func (b SAControllerClientBuilder) ClientOrDie(name string) clientset.Interface {
	client, err := b.Client(name)
	if err != nil {
		klog.Fatal(err)
	}
	return client
}
