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

package proxy

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"regexp"
	"strings"

	apiv1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/endpoints/request"
	apirest "k8s.io/apiserver/pkg/registry/rest"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/rest/fake"
	fedv1 "k8s.io/kubernetes/federation/apis/federation/v1beta1"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	kubeclient "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
)

func errorStatusResponse(codec runtime.Encoder, err error) *http.Response {
	status := err.(errors.APIStatus).Status()
	return &http.Response{
		StatusCode: int(status.Code),
		Header:     http.Header{"Content-Type": []string{runtime.ContentTypeJSON}},
		Body:       ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, &status)))),
	}
}

func fakeRestClientForClientset(testGroup testapi.TestGroup, kubeClientset kubeclient.Interface,
	listFunc func(kubeClientset kubeclient.Interface, namespace string, opts metav1.ListOptions) (runtime.Object, error),
	getFunc func(kubeClientset kubeclient.Interface, namespace string, name string, opts metav1.GetOptions) (runtime.Object, error),
) restclient.Interface {
	codec := testGroup.Codec()
	return &fake.RESTClient{
		APIRegistry:          api.Registry,
		NegotiatedSerializer: api.Codecs,
		Client: fake.CreateHTTPClient(func(req *http.Request) (*http.Response, error) {

			// [match, namespace, resource, name]
			re := regexp.MustCompile(`(?:/namespaces/([\w-.]+))?/([\w-.]+)(?:/([\w-.]+))?`)
			matches := re.FindStringSubmatch(req.URL.Path)
			namespace, _, name := matches[1], matches[2], matches[3]
			switch m := req.Method; {
			case m == http.MethodGet && name != "":
				opts := metav1.GetOptions{}
				api.ParameterCodec.DecodeParameters(req.URL.Query(), apiv1.SchemeGroupVersion, &opts)
				obj, err := getFunc(kubeClientset, namespace, name, opts)
				if err != nil {
					return errorStatusResponse(codec, err), nil
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{runtime.ContentTypeJSON}},
					Body:       ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, obj)))),
				}, nil
			case m == http.MethodGet:
				opts := metav1.ListOptions{}
				api.ParameterCodec.DecodeParameters(req.URL.Query(), apiv1.SchemeGroupVersion, &opts)
				obj, err := listFunc(kubeClientset, namespace, opts)
				if err != nil {
					return errorStatusResponse(codec, err), nil
				}
				return &http.Response{
					StatusCode: http.StatusOK,
					Header:     http.Header{"Content-Type": []string{runtime.ContentTypeJSON}},
					Body:       ioutil.NopCloser(bytes.NewReader([]byte(runtime.EncodeOrDie(codec, obj)))),
				}, nil
			default:
				return nil, fmt.Errorf("unexpected request: %#v\n%#v", req.URL, req)
			}
		}),
	}
}

func stripPort(hostport string) string {
	colon := strings.IndexByte(hostport, ':')
	if colon == -1 {
		return hostport
	}
	if i := strings.IndexByte(hostport, ']'); i != -1 {
		return strings.TrimPrefix(hostport[:i], "[")
	}
	return hostport[:colon]
}

// FakeRestClientFuncForClusters creates a rest client for the generated fake client where rest client wasn't implemented.
func FakeRestClientFuncForClusters(testGroup testapi.TestGroup, clusterClientsets map[string]kubeclient.Interface,
	listFunc func(kubeClientset kubeclient.Interface, namespace string, opts metav1.ListOptions) (runtime.Object, error),
	getFunc func(kubeClientset kubeclient.Interface, namespace string, name string, opts metav1.GetOptions) (runtime.Object, error),
) func(kubeClientset kubeclient.Interface) restclient.Interface {
	restClients := map[string]restclient.Interface{}
	for clusterName, clientset := range clusterClientsets {
		restClients[clusterName] = fakeRestClientForClientset(testGroup, clientset, listFunc, getFunc)
	}

	return func(kubeClientset kubeclient.Interface) restclient.Interface {
		restClient := kubeClientset.Discovery().RESTClient().(*restclient.RESTClient)
		return restClients[stripPort(restClient.Get().URL().Host)]
	}
}

// NewTestCluster creates a cluster object for testing.
func NewTestCluster(name string) *fedv1.Cluster {
	cluster := &fedv1.Cluster{
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: fedv1.ClusterSpec{
			ServerAddressByClientCIDRs: []fedv1.ServerAddressByClientCIDR{
				{
					ClientCIDR:    "0.0.0.0/0",
					ServerAddress: name + ":8080",
				},
			},
		},
		Status: fedv1.ClusterStatus{
			Conditions: []fedv1.ClusterCondition{
				{Type: fedv1.ClusterReady, Status: apiv1.ConditionTrue},
			},
		},
	}
	api.Scheme.Default(cluster)
	return cluster
}

// NewTestContex creates a request context for testing.
func NewTestContex() request.Context {
	return request.NewContext()
}

// NewTestContextWithNamespace creates a request context for testing and add namespace to it.
func NewTestContextWithNamespace(namespace string) request.Context {
	return request.WithNamespace(request.NewContext(), namespace)
}

// FakeStore create a store backed by the kube client for testing.
type FakeStore struct {
	NewFunc           func() runtime.Object
	NewListFunc       func() runtime.Object
	KubeClient        kubeclient.Interface
	QualifiedResource schema.GroupResource
	ListFunc          func(kubeClient kubeclient.Interface, namespace string, opts metav1.ListOptions) (runtime.Object, error)
	GetFunc           func(kubeClient kubeclient.Interface, namespace string, name string, opts metav1.GetOptions) (runtime.Object, error)
}

// List implements StandardStorage.List.
func (s *FakeStore) List(ctx request.Context, options *metainternalversion.ListOptions) (runtime.Object, error) {
	optsv1 := metav1.ListOptions{}
	api.Scheme.Convert(options, &optsv1, nil)
	return s.ListFunc(s.KubeClient, request.NamespaceValue(ctx), optsv1)
}

// Get implements StandardStorage.Get.
func (s *FakeStore) Get(ctx request.Context, name string, options *metav1.GetOptions) (runtime.Object, error) {
	return s.GetFunc(s.KubeClient, request.NamespaceValue(ctx), name, *options)
}

// New implements StandardStorage.New.
func (s *FakeStore) New() runtime.Object {
	return s.NewFunc()
}

// NewList implements StandardStorage.NewList.
func (s *FakeStore) NewList() runtime.Object {
	return s.NewListFunc()
}

// Create implements StandardStorage.Create.
func (s *FakeStore) Create(ctx request.Context, obj runtime.Object, includeUninitialized bool) (runtime.Object, error) {
	return nil, errors.NewMethodNotSupported(s.QualifiedResource, "create")
}

// Update implements StandardStorage.Update.
func (s *FakeStore) Update(ctx request.Context, name string, objInfo apirest.UpdatedObjectInfo) (runtime.Object, bool, error) {
	return nil, false, errors.NewMethodNotSupported(s.QualifiedResource, "update")
}

// Delete implements StandardStorage.Delete.
func (s *FakeStore) Delete(ctx request.Context, name string, options *metav1.DeleteOptions) (runtime.Object, bool, error) {
	return nil, false, errors.NewMethodNotSupported(s.QualifiedResource, "delete")
}

// DeleteCollection implements StandardStorage.DeleteCollection.
func (s *FakeStore) DeleteCollection(ctx request.Context, options *metav1.DeleteOptions, listOptions *metainternalversion.ListOptions) (runtime.Object, error) {
	return nil, errors.NewMethodNotSupported(s.QualifiedResource, "delete-collection")
}

// Watch implements StandardStorage.Watch.
func (s *FakeStore) Watch(ctx request.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return nil, errors.NewMethodNotSupported(s.QualifiedResource, "watch")
}
