/*
Copyright 2015 The Kubernetes Authors.

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

package server

import (
	"net/http"
	"strings"

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apiserver/pkg/authentication/authenticator"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/klog/v2"
)

// KubeletAuth implements AuthInterface
type KubeletAuth struct {
	// authenticator identifies the user for requests to the Kubelet API
	authenticator.Request
	// authorizerAttributeGetter builds authorization.Attributes for a request to the Kubelet API
	authorizer.RequestAttributesGetter
	// authorizer determines whether a given authorization.Attributes is allowed
	authorizer.Authorizer
}

// NewKubeletAuth returns a kubelet.AuthInterface composed of the given authenticator, attribute getter, and authorizer
func NewKubeletAuth(authenticator authenticator.Request, authorizerAttributeGetter authorizer.RequestAttributesGetter, authorizer authorizer.Authorizer) AuthInterface {
	return &KubeletAuth{authenticator, authorizerAttributeGetter, authorizer}
}

// NewNodeAuthorizerAttributesGetter creates a new authorizer.RequestAttributesGetter for the node.
func NewNodeAuthorizerAttributesGetter(nodeName types.NodeName) authorizer.RequestAttributesGetter {
	return nodeAuthorizerAttributesGetter{nodeName: nodeName}
}

type nodeAuthorizerAttributesGetter struct {
	nodeName types.NodeName
}

func isSubpath(subpath, path string) bool {
	path = strings.TrimSuffix(path, "/")
	return subpath == path || (strings.HasPrefix(subpath, path) && subpath[len(path)] == '/')
}

// GetRequestAttributes populates authorizer attributes for the requests to the kubelet API.
// Default attributes are: {apiVersion=v1,verb=<http verb from request>,resource=nodes,name=<node name>,subresource=proxy}
// More specific verb/resource is set for the following request patterns:
//
//	/stats/*   => verb=<api verb from request>, resource=nodes, name=<node name>, subresource=stats
//	/metrics/* => verb=<api verb from request>, resource=nodes, name=<node name>, subresource=metrics
//	/logs/*    => verb=<api verb from request>, resource=nodes, name=<node name>, subresource=log
func (n nodeAuthorizerAttributesGetter) GetRequestAttributes(u user.Info, r *http.Request) authorizer.Attributes {

	apiVerb := ""
	switch r.Method {
	case "POST":
		apiVerb = "create"
	case "GET":
		apiVerb = "get"
	case "PUT":
		apiVerb = "update"
	case "PATCH":
		apiVerb = "patch"
	case "DELETE":
		apiVerb = "delete"
	}

	requestPath := r.URL.Path

	// Default attributes mirror the API attributes that would allow this access to the kubelet API
	attrs := authorizer.AttributesRecord{
		User:            u,
		Verb:            apiVerb,
		Namespace:       "",
		APIGroup:        "",
		APIVersion:      "v1",
		Resource:        "nodes",
		Subresource:     "proxy",
		Name:            string(n.nodeName),
		ResourceRequest: true,
		Path:            requestPath,
	}

	// Override subresource for specific paths
	// This allows subdividing access to the kubelet API
	switch {
	case isSubpath(requestPath, statsPath):
		attrs.Subresource = "stats"
	case isSubpath(requestPath, metricsPath):
		attrs.Subresource = "metrics"
	case isSubpath(requestPath, logsPath):
		// "log" to match other log subresources (pods/log, etc)
		attrs.Subresource = "log"
	}

	klog.V(5).InfoS("Node request attributes", "user", attrs.GetUser().GetName(), "verb", attrs.GetVerb(), "resource", attrs.GetResource(), "subresource", attrs.GetSubresource())

	return attrs
}
