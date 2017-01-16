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
	"fmt"
	"time"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	v1core "k8s.io/kubernetes/pkg/client/clientset_generated/clientset/typed/core/v1"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/serviceaccount"

	"github.com/golang/glog"
)

// ControllerClientBuilder allow syou to get clients and configs for controllers
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
		glog.Fatal(err)
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
		glog.Fatal(err)
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

	// Namespace is the namespace used to host the service accounts that will back the
	// controllers.  It must be highly privileged namespace which normal users cannot inspect.
	Namespace string
}

// config returns a complete clientConfig for constructing clients.  This is separate in anticipation of composition
// which means that not all clientsets are known here
func (b SAControllerClientBuilder) Config(name string) (*restclient.Config, error) {
	clientConfig := restclient.AnonymousClientConfig(b.ClientConfig)

	// we need the SA UID to find a matching SA token
	sa, err := b.CoreClient.ServiceAccounts(b.Namespace).Get(name, metav1.GetOptions{})
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	} else if apierrors.IsNotFound(err) {
		// check to see if the namespace exists.  If it isn't a NotFound, just try to create the SA.
		// It'll probably fail, but perhaps that will have a better message.
		if _, err := b.CoreClient.Namespaces().Get(b.Namespace, metav1.GetOptions{}); apierrors.IsNotFound(err) {
			_, err = b.CoreClient.Namespaces().Create(&v1.Namespace{ObjectMeta: v1.ObjectMeta{Name: b.Namespace}})
			if err != nil && !apierrors.IsAlreadyExists(err) {
				return nil, err
			}
		}

		sa, err = b.CoreClient.ServiceAccounts(b.Namespace).Create(
			&v1.ServiceAccount{ObjectMeta: v1.ObjectMeta{Namespace: b.Namespace, Name: name}})
		if err != nil {
			return nil, err
		}
	}

	lw := &cache.ListWatch{
		ListFunc: func(options v1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fields.SelectorFromSet(map[string]string{api.SecretTypeField: string(v1.SecretTypeServiceAccountToken)}).String()
			return b.CoreClient.Secrets(b.Namespace).List(options)
		},
		WatchFunc: func(options v1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fields.SelectorFromSet(map[string]string{api.SecretTypeField: string(v1.SecretTypeServiceAccountToken)}).String()
			return b.CoreClient.Secrets(b.Namespace).Watch(options)
		},
	}
	_, err = cache.ListWatchUntil(30*time.Second, lw,
		func(event watch.Event) (bool, error) {
			switch event.Type {
			case watch.Deleted:
				return false, nil
			case watch.Error:
				return false, fmt.Errorf("error watching")

			case watch.Added, watch.Modified:
				secret := event.Object.(*v1.Secret)
				if !serviceaccount.IsServiceAccountToken(secret, sa) ||
					len(secret.Data[v1.ServiceAccountTokenKey]) == 0 {
					return false, nil
				}
				// TODO maybe verify the token is valid
				clientConfig.BearerToken = string(secret.Data[v1.ServiceAccountTokenKey])
				restclient.AddUserAgent(clientConfig, serviceaccount.MakeUsername(b.Namespace, name))
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

func (b SAControllerClientBuilder) ConfigOrDie(name string) *restclient.Config {
	clientConfig, err := b.Config(name)
	if err != nil {
		glog.Fatal(err)
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
		glog.Fatal(err)
	}
	return client
}
