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
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/features"
)

// KubeletAuth implements AuthInterface
type KubeletAuth struct {
	// authenticator identifies the user for requests to the Kubelet API
	authenticator.Request
	// KubeletRequestAttributesGetter builds authorization.Attributes for a request to the Kubelet API
	NodeRequestAttributesGetter
	// authorizer determines whether a given authorization.Attributes is allowed
	authorizer.Authorizer
}

// NewKubeletAuth returns a kubelet.AuthInterface composed of the given authenticator, attribute getter, and authorizer
func NewKubeletAuth(authenticator authenticator.Request, authorizerAttributeGetter NodeRequestAttributesGetter, authorizer authorizer.Authorizer) AuthInterface {
	return &KubeletAuth{authenticator, authorizerAttributeGetter, authorizer}
}

// NewNodeAuthorizerAttributesGetter creates a new authorizer.RequestAttributesGetter for the node.
func NewNodeAuthorizerAttributesGetter(nodeName types.NodeName) NodeRequestAttributesGetter {
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
func (n nodeAuthorizerAttributesGetter) GetRequestAttributes(u user.Info, r *http.Request) []authorizer.Attributes {

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

	fineGrained := utilfeature.DefaultFeatureGate.Enabled(features.KubeletFineGrainedAuthz)
	var attrs []authorizer.Attributes
	switch {
	case isSubpath(requestPath, statsPath):
		attrs = append(attrs, attributes(u, apiVerb, "stats", string(n.nodeName), requestPath))
	case isSubpath(requestPath, metricsPath):
		attrs = append(attrs, attributes(u, apiVerb, "metrics", string(n.nodeName), requestPath))
	case isSubpath(requestPath, logsPath):
		// "log" to match other log subresources (pods/log, etc)
		attrs = append(attrs, attributes(u, apiVerb, "log", string(n.nodeName), requestPath))
	case isSubpath(requestPath, checkpointPath):
		attrs = append(attrs, attributes(u, apiVerb, "checkpoint", string(n.nodeName), requestPath))
	case isSubpath(requestPath, podsPath):
		if fineGrained {
			attrs = append(attrs, attributes(u, apiVerb, "pods", string(n.nodeName), requestPath))
		}
		attrs = append(attrs, attributes(u, apiVerb, "proxy", string(n.nodeName), requestPath))
	case isSubpath(requestPath, runningPodsPath):
		if fineGrained {
			attrs = append(attrs, attributes(u, apiVerb, "pods", string(n.nodeName), requestPath))
		}
		attrs = append(attrs, attributes(u, apiVerb, "proxy", string(n.nodeName), requestPath))
	case isSubpath(requestPath, healthzPath):
		if fineGrained {
			attrs = append(attrs, attributes(u, apiVerb, "healthz", string(n.nodeName), requestPath))
		}
		attrs = append(attrs, attributes(u, apiVerb, "proxy", string(n.nodeName), requestPath))
	case isSubpath(requestPath, configzPath):
		if fineGrained {
			attrs = append(attrs, attributes(u, apiVerb, "configz", string(n.nodeName), requestPath))
		}
		attrs = append(attrs, attributes(u, apiVerb, "proxy", string(n.nodeName), requestPath))
	default:
		attrs = append(attrs, attributes(u, apiVerb, "proxy", string(n.nodeName), requestPath))
	}

	for _, attr := range attrs {
		klog.V(5).InfoS("Node request attributes", "user", attr.GetUser().GetName(), "verb", attr.GetVerb(), "resource", attr.GetResource(), "subresource", attr.GetSubresource())
	}

	return attrs
}

func attributes(u user.Info, apiVerb, subresource, nodeName, requestPath string) authorizer.AttributesRecord {
	return authorizer.AttributesRecord{
		User:            u,
		Verb:            apiVerb,
		Namespace:       "",
		APIGroup:        "",
		APIVersion:      "v1",
		Resource:        "nodes",
		Subresource:     subresource,
		Name:            nodeName,
		ResourceRequest: true,
		Path:            requestPath,
	}
}
