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

package clientset

import (
	"net/http"

	clientset "k8s.io/apiextensions/pkg/client/clientset/clientset"
	"k8s.io/client-go/rest"
)

//go:fix inline
type Clientset = clientset.Clientset

//go:fix inline
type Interface = clientset.Interface

//go:fix inline
func New(c rest.Interface) *clientset.Clientset {
	return clientset.New(c)
}

//go:fix inline
func NewForConfig(c *rest.Config) (*clientset.Clientset, error) {
	return clientset.NewForConfig(c)
}

//go:fix inline
func NewForConfigAndClient(c *rest.Config, httpClient *http.Client) (*clientset.Clientset, error) {
	return clientset.NewForConfigAndClient(c, httpClient)
}

//go:fix inline
func NewForConfigOrDie(c *rest.Config) *clientset.Clientset {
	return clientset.NewForConfigOrDie(c)
}
