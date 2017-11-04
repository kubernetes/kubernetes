/*
Copyright 2017 The Kubernetes Authors.

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

package apiclient

import (
	"encoding/json"
	"fmt"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	kuberuntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/dynamic"
	clientset "k8s.io/client-go/kubernetes"
	clientsetscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	core "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/clientcmd"
)

// ClientBackedDryRunGetter implements the DryRunGetter interface for use in NewDryRunClient() and proxies all GET and LIST requests to the backing API server reachable via rest.Config
type ClientBackedDryRunGetter struct {
	client        clientset.Interface
	dynClientPool dynamic.ClientPool
}

// InitDryRunGetter should implement the DryRunGetter interface
var _ DryRunGetter = &ClientBackedDryRunGetter{}

// NewClientBackedDryRunGetter creates a new ClientBackedDryRunGetter instance based on the rest.Config object
func NewClientBackedDryRunGetter(config *rest.Config) (*ClientBackedDryRunGetter, error) {
	client, err := clientset.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	return &ClientBackedDryRunGetter{
		client:        client,
		dynClientPool: dynamic.NewDynamicClientPool(config),
	}, nil
}

// NewClientBackedDryRunGetterFromKubeconfig creates a new ClientBackedDryRunGetter instance from the given KubeConfig file
func NewClientBackedDryRunGetterFromKubeconfig(file string) (*ClientBackedDryRunGetter, error) {
	config, err := clientcmd.LoadFromFile(file)
	if err != nil {
		return nil, fmt.Errorf("failed to load kubeconfig: %v", err)
	}
	clientConfig, err := clientcmd.NewDefaultClientConfig(*config, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		return nil, fmt.Errorf("failed to create API client configuration from kubeconfig: %v", err)
	}
	return NewClientBackedDryRunGetter(clientConfig)
}

// HandleGetAction handles GET actions to the dryrun clientset this interface supports
func (clg *ClientBackedDryRunGetter) HandleGetAction(action core.GetAction) (bool, runtime.Object, error) {
	rc, err := clg.actionToResourceClient(action)
	if err != nil {
		return true, nil, err
	}

	unversionedObj, err := rc.Get(action.GetName(), metav1.GetOptions{})
	if err != nil {
		return true, nil, err
	}
	// If the unversioned object does not have .apiVersion; the inner object is probably nil
	if len(unversionedObj.GetAPIVersion()) == 0 {
		return true, nil, apierrors.NewNotFound(action.GetResource().GroupResource(), action.GetName())
	}
	newObj, err := decodeUnversionedIntoAPIObject(action, unversionedObj)
	if err != nil {
		fmt.Printf("error after decode: %v %v\n", unversionedObj, err)
		return true, nil, err
	}
	return true, newObj, err
}

// HandleListAction handles LIST actions to the dryrun clientset this interface supports
func (clg *ClientBackedDryRunGetter) HandleListAction(action core.ListAction) (bool, runtime.Object, error) {
	rc, err := clg.actionToResourceClient(action)
	if err != nil {
		return true, nil, err
	}

	listOpts := metav1.ListOptions{
		LabelSelector: action.GetListRestrictions().Labels.String(),
		FieldSelector: action.GetListRestrictions().Fields.String(),
	}

	unversionedList, err := rc.List(listOpts)
	if err != nil {
		return true, nil, err
	}
	// If the runtime.Object here is nil, we should return successfully with no result
	if unversionedList == nil {
		return true, unversionedList, nil
	}
	newObj, err := decodeUnversionedIntoAPIObject(action, unversionedList)
	if err != nil {
		fmt.Printf("error after decode: %v %v\n", unversionedList, err)
		return true, nil, err
	}
	return true, newObj, err
}

// Client gets the backing clientset.Interface
func (clg *ClientBackedDryRunGetter) Client() clientset.Interface {
	return clg.client
}

// actionToResourceClient returns the ResourceInterface for the given action
// First; the function gets the right API group interface from the resource type. The API group struct behind the interface
// returned may be cached in the dynamic client pool. Then, an APIResource object is constructed so that it can be passed to
// dynamic.Interface's Resource() function, which will give us the final ResourceInterface to query
func (clg *ClientBackedDryRunGetter) actionToResourceClient(action core.Action) (dynamic.ResourceInterface, error) {
	dynIface, err := clg.dynClientPool.ClientForGroupVersionResource(action.GetResource())
	if err != nil {
		return nil, err
	}

	apiResource := &metav1.APIResource{
		Name:       action.GetResource().Resource,
		Namespaced: action.GetNamespace() != "",
	}

	return dynIface.Resource(apiResource, action.GetNamespace()), nil
}

// decodeUnversionedIntoAPIObject converts the *unversioned.Unversioned object returned from the dynamic client
// to bytes; and then decodes it back _to an external api version (k8s.io/api vs k8s.io/kubernetes/pkg/api*)_ using the normal API machinery
func decodeUnversionedIntoAPIObject(action core.Action, unversionedObj runtime.Object) (runtime.Object, error) {
	objBytes, err := json.Marshal(unversionedObj)
	if err != nil {
		return nil, err
	}
	newObj, err := kuberuntime.Decode(clientsetscheme.Codecs.UniversalDecoder(action.GetResource().GroupVersion()), objBytes)
	if err != nil {
		return nil, err
	}
	return newObj, nil
}
