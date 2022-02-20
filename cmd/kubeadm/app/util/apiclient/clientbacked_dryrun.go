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
	"context"
	"encoding/json"
	"fmt"

	"github.com/pkg/errors"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
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
	dynamicClient dynamic.Interface
}

// InitDryRunGetter should implement the DryRunGetter interface
var _ DryRunGetter = &ClientBackedDryRunGetter{}

// NewClientBackedDryRunGetter creates a new ClientBackedDryRunGetter instance based on the rest.Config object
func NewClientBackedDryRunGetter(config *rest.Config) (*ClientBackedDryRunGetter, error) {
	client, err := clientset.NewForConfig(config)
	if err != nil {
		return nil, err
	}
	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		return nil, err
	}

	return &ClientBackedDryRunGetter{
		client:        client,
		dynamicClient: dynamicClient,
	}, nil
}

// NewClientBackedDryRunGetterFromKubeconfig creates a new ClientBackedDryRunGetter instance from the given KubeConfig file
func NewClientBackedDryRunGetterFromKubeconfig(file string) (*ClientBackedDryRunGetter, error) {
	config, err := clientcmd.LoadFromFile(file)
	if err != nil {
		return nil, errors.Wrap(err, "failed to load kubeconfig")
	}
	clientConfig, err := clientcmd.NewDefaultClientConfig(*config, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		return nil, errors.Wrap(err, "failed to create API client configuration from kubeconfig")
	}
	return NewClientBackedDryRunGetter(clientConfig)
}

// HandleGetAction handles GET actions to the dryrun clientset this interface supports
func (clg *ClientBackedDryRunGetter) HandleGetAction(action core.GetAction) (bool, runtime.Object, error) {
	unstructuredObj, err := clg.dynamicClient.Resource(action.GetResource()).Namespace(action.GetNamespace()).Get(context.TODO(), action.GetName(), metav1.GetOptions{})
	// Inform the user that the requested object wasn't found.
	printIfNotExists(err)
	if err != nil {
		return true, nil, err
	}
	newObj, err := decodeUnstructuredIntoAPIObject(action, unstructuredObj)
	if err != nil {
		fmt.Printf("error after decode: %v %v\n", unstructuredObj, err)
		return true, nil, err
	}
	return true, newObj, err
}

// HandleListAction handles LIST actions to the dryrun clientset this interface supports
func (clg *ClientBackedDryRunGetter) HandleListAction(action core.ListAction) (bool, runtime.Object, error) {
	listOpts := metav1.ListOptions{
		LabelSelector: action.GetListRestrictions().Labels.String(),
		FieldSelector: action.GetListRestrictions().Fields.String(),
	}

	unstructuredList, err := clg.dynamicClient.Resource(action.GetResource()).Namespace(action.GetNamespace()).List(context.TODO(), listOpts)
	if err != nil {
		return true, nil, err
	}
	newObj, err := decodeUnstructuredIntoAPIObject(action, unstructuredList)
	if err != nil {
		fmt.Printf("error after decode: %v %v\n", unstructuredList, err)
		return true, nil, err
	}
	return true, newObj, err
}

// Client gets the backing clientset.Interface
func (clg *ClientBackedDryRunGetter) Client() clientset.Interface {
	return clg.client
}

// decodeUnversionedIntoAPIObject converts the *unversioned.Unversioned object returned from the dynamic client
// to bytes; and then decodes it back _to an external api version (k8s.io/api)_ using the normal API machinery
func decodeUnstructuredIntoAPIObject(action core.Action, unstructuredObj runtime.Unstructured) (runtime.Object, error) {
	objBytes, err := json.Marshal(unstructuredObj)
	if err != nil {
		return nil, err
	}
	newObj, err := runtime.Decode(clientsetscheme.Codecs.UniversalDecoder(action.GetResource().GroupVersion()), objBytes)
	if err != nil {
		return nil, err
	}
	return newObj, nil
}

func printIfNotExists(err error) {
	if apierrors.IsNotFound(err) {
		fmt.Println("[dryrun] The GET request didn't yield any result, the API Server returned a NotFound error.")
	}
}
