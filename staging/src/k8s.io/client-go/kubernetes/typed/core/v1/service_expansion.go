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

package v1

import (
	"k8s.io/apimachinery/pkg/util/net"
	"k8s.io/client-go/rest"
)

// The ServiceExpansion interface allows manually adding extra methods to the ServiceInterface.
type ServiceExpansion interface {
	ProxyGet(scheme, name, port, path string, params map[string]string) rest.ResponseWrapper
}

// ProxyGet returns a response of the service by calling it through the proxy.
func (c *services) ProxyGet(scheme, name, port, path string, params map[string]string) rest.ResponseWrapper {
	request := c.client.Get().
		Prefix("proxy").
		Namespace(c.ns).
		Resource("services").
		Name(net.JoinSchemeNamePort(scheme, name, port)).
		Suffix(path)
	for k, v := range params {
		request = request.Param(k, v)
	}
	return request
}
