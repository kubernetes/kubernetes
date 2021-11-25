/*
Copyright 2021 The Kubernetes Authors.

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

package fixtures

import (
	"context"
	"encoding/json"
	"path"

	clientv3 "go.etcd.io/etcd/client/v3"
	apiextensionsv1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	apiextensionsv1beta1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
	"k8s.io/apiextensions-apiserver/pkg/client/clientset/clientset"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/client-go/dynamic"
)

// CreateCRDUsingRemovedAPI creates a CRD directly using etcd.  This is must *ONLY* be used for checks of compatibility
// with removed data.  Do not use this just because you don't want to update your test to use v1.  Only use this
// when it actually matters.
func CreateCRDUsingRemovedAPI(etcdClient *clientv3.Client, etcdStoragePrefix string, betaCRD *apiextensionsv1beta1.CustomResourceDefinition, apiExtensionsClient clientset.Interface, dynamicClientSet dynamic.Interface) (*apiextensionsv1.CustomResourceDefinition, error) {
	crd, err := CreateCRDUsingRemovedAPIWatchUnsafe(etcdClient, etcdStoragePrefix, betaCRD, apiExtensionsClient)
	if err != nil {
		return nil, err
	}
	return waitForCRDReady(crd, apiExtensionsClient, dynamicClientSet)
}

// CreateCRDUsingRemovedAPIWatchUnsafe creates a CRD directly using etcd.  This is must *ONLY* be used for checks of compatibility
// with removed data.  Do not use this just because you don't want to update your test to use v1.  Only use this
// when it actually matters.
func CreateCRDUsingRemovedAPIWatchUnsafe(etcdClient *clientv3.Client, etcdStoragePrefix string, betaCRD *apiextensionsv1beta1.CustomResourceDefinition, apiExtensionsClient clientset.Interface) (*apiextensionsv1.CustomResourceDefinition, error) {
	// attempt defaulting, best effort
	apiextensionsv1beta1.SetDefaults_CustomResourceDefinition(betaCRD)
	betaCRD.Kind = "CustomResourceDefinition"
	betaCRD.APIVersion = apiextensionsv1beta1.SchemeGroupVersion.Group + "/" + apiextensionsv1beta1.SchemeGroupVersion.Version

	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), metav1.NamespaceNone)
	key := path.Join("/", etcdStoragePrefix, "apiextensions.k8s.io", "customresourcedefinitions", betaCRD.Name)
	val, _ := json.Marshal(betaCRD)
	if _, err := etcdClient.Put(ctx, key, string(val)); err != nil {
		return nil, err
	}

	return apiExtensionsClient.ApiextensionsV1().CustomResourceDefinitions().Get(context.TODO(), betaCRD.Name, metav1.GetOptions{})
}
