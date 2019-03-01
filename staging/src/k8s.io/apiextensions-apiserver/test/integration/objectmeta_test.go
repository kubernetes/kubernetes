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

package integration

import (
	"path"
	"reflect"
	"strings"
	"testing"

	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/pkg/transport"

	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	serveroptions "k8s.io/apiextensions-apiserver/pkg/cmd/server/options"
	"k8s.io/apiextensions-apiserver/test/integration/fixtures"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/json"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/dynamic"
)

func TestPostInvalidObjectMeta(t *testing.T) {
	tearDown, apiExtensionClient, dynamicClient, err := fixtures.StartDefaultServerWithClients(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	noxuDefinition := fixtures.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	noxuResourceClient := newNamespacedCustomResourceClient("default", dynamicClient, noxuDefinition)

	obj := fixtures.NewNoxuInstance("default", "foo")
	unstructured.SetNestedField(obj.UnstructuredContent(), int64(42), "metadata", "unknown")
	unstructured.SetNestedField(obj.UnstructuredContent(), map[string]interface{}{"foo": int64(42), "bar": "abc"}, "metadata", "labels")
	_, err = instantiateCustomResource(t, obj, noxuResourceClient, noxuDefinition)
	if err == nil {
		t.Fatalf("unexpected non-error, expected invalid labels to be rejected: %v", err)
	}
	if status, ok := err.(errors.APIStatus); !ok {
		t.Fatalf("expected APIStatus error, but got: %#v", err)
	} else if !errors.IsBadRequest(err) {
		t.Fatalf("expected BadRequst error, but got: %v", errors.ReasonForError(err))
	} else if !strings.Contains(status.Status().Message, "cannot be handled") {
		t.Fatalf("expected 'cannot be handled' error message, got: %v", status.Status().Message)
	}

	unstructured.SetNestedField(obj.UnstructuredContent(), map[string]interface{}{"bar": "abc"}, "metadata", "labels")
	obj, err = instantiateCustomResource(t, obj, noxuResourceClient, noxuDefinition)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if unknown, found, err := unstructured.NestedInt64(obj.UnstructuredContent(), "metadata", "unknown"); err != nil {
		t.Errorf("unexpected error getting metadata.unknown: %v", err)
	} else if found {
		t.Errorf("unexpected metadata.unknown=%#v: expected this to be pruned", unknown)
	}
}

func TestInvalidObjectMetaInStorage(t *testing.T) {
	tearDown, config, options, err := fixtures.StartDefaultServer(t)
	if err != nil {
		t.Fatal(err)
	}
	defer tearDown()

	apiExtensionClient, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	dynamicClient, err := dynamic.NewForConfig(config)
	if err != nil {
		t.Fatal(err)
	}

	noxuDefinition := fixtures.NewNoxuCustomResourceDefinition(apiextensionsv1beta1.NamespaceScoped)
	noxuDefinition, err = fixtures.CreateNewCustomResourceDefinition(noxuDefinition, apiExtensionClient, dynamicClient)
	if err != nil {
		t.Fatal(err)
	}

	RESTOptionsGetter := serveroptions.NewCRDRESTOptionsGetter(*options.RecommendedOptions.Etcd)
	restOptions, err := RESTOptionsGetter.GetRESTOptions(schema.GroupResource{Group: noxuDefinition.Spec.Group, Resource: noxuDefinition.Spec.Names.Plural})
	if err != nil {
		t.Fatal(err)
	}
	tlsInfo := transport.TLSInfo{
		CertFile: restOptions.StorageConfig.Transport.CertFile,
		KeyFile:  restOptions.StorageConfig.Transport.KeyFile,
		CAFile:   restOptions.StorageConfig.Transport.CAFile,
	}
	tlsConfig, err := tlsInfo.ClientConfig()
	if err != nil {
		t.Fatal(err)
	}
	etcdConfig := clientv3.Config{
		Endpoints: restOptions.StorageConfig.Transport.ServerList,
		TLS:       tlsConfig,
	}
	etcdclient, err := clientv3.New(etcdConfig)
	if err != nil {
		t.Fatal(err)
	}

	t.Logf("Creating object with invalid labels manually in etcd")

	original := fixtures.NewNoxuInstance("default", "foo")
	unstructured.SetNestedField(original.UnstructuredContent(), int64(42), "metadata", "unknown")
	unstructured.SetNestedField(original.UnstructuredContent(), map[string]interface{}{"foo": int64(42), "bar": "abc"}, "metadata", "labels")

	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceDefault)
	key := path.Join("/", restOptions.StorageConfig.Prefix, noxuDefinition.Spec.Group, "noxus/default/foo")
	val, _ := json.Marshal(original.UnstructuredContent())
	if _, err := etcdclient.Put(ctx, key, string(val)); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	t.Logf("Checking that ObjectMeta is pruned from unknown fields")

	noxuResourceClient := newNamespacedCustomResourceClient("default", dynamicClient, noxuDefinition)
	obj, err := noxuResourceClient.Get("foo", metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if unknown, found, err := unstructured.NestedFieldNoCopy(obj.UnstructuredContent(), "metadata", "unknown"); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if found {
		t.Errorf("unexpected to find metadata.unknown=%#v", unknown)
	}

	t.Logf("Checking that ObjectMeta is pruned from invalid typed fields")

	if labels, found, err := unstructured.NestedStringMap(obj.UnstructuredContent(), "metadata", "labels"); err != nil {
		t.Errorf("unexpected error: %v", err)
	} else if found && !reflect.DeepEqual(labels, map[string]string{"bar": "abc"}) {
		t.Errorf("unexpected to find metadata.lables=%#v", labels)
	}
}
