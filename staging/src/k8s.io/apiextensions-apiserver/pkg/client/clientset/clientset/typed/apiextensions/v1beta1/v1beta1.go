/*
Copyright The Kubernetes Authors.

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

package v1beta1

import (
	"net/http"

	v1beta1 "k8s.io/apiextensions/pkg/client/clientset/clientset/typed/apiextensions/v1beta1"
	"k8s.io/client-go/rest"
)

//go:fix inline
type ApiextensionsV1beta1Client = v1beta1.ApiextensionsV1beta1Client

//go:fix inline
type ApiextensionsV1beta1Interface = v1beta1.ApiextensionsV1beta1Interface

//go:fix inline
type CustomResourceDefinitionExpansion = v1beta1.CustomResourceDefinitionExpansion

//go:fix inline
type CustomResourceDefinitionInterface = v1beta1.CustomResourceDefinitionInterface

//go:fix inline
type CustomResourceDefinitionsGetter = v1beta1.CustomResourceDefinitionsGetter

//go:fix inline
func New(c rest.Interface) *v1beta1.ApiextensionsV1beta1Client {
	return v1beta1.New(c)
}

//go:fix inline
func NewForConfig(c *rest.Config) (*v1beta1.ApiextensionsV1beta1Client, error) {
	return v1beta1.NewForConfig(c)
}

//go:fix inline
func NewForConfigAndClient(c *rest.Config, h *http.Client) (*v1beta1.ApiextensionsV1beta1Client, error) {
	return v1beta1.NewForConfigAndClient(c, h)
}

//go:fix inline
func NewForConfigOrDie(c *rest.Config) *v1beta1.ApiextensionsV1beta1Client {
	return v1beta1.NewForConfigOrDie(c)
}
