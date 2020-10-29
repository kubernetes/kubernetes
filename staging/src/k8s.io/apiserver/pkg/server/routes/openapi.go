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

package routes

import (
	"strings"

	restful "github.com/emicklei/go-restful/v3"
	"k8s.io/klog/v2"

	"k8s.io/apiserver/pkg/server/mux"
	builder2 "k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/builder3"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/common/restfuladapter"
	"k8s.io/kube-openapi/pkg/handler"
	"k8s.io/kube-openapi/pkg/handler3"
	"k8s.io/kube-openapi/pkg/validation/spec"
)

// OpenAPI installs spec endpoints for each web service.
type OpenAPI struct {
	Config   *common.Config
	V3Config *common.OpenAPIV3Config
}

// Install adds the SwaggerUI webservice to the given mux.
func (oa OpenAPI) InstallV2(c *restful.Container, mux *mux.PathRecorderMux) (*handler.OpenAPIService, *spec.Swagger) {
	// we shadow ClustResourceQuotas, RoleBindingRestrictions, and SecurityContextContstraints
	// with a CRD. This loop removes all CRQ,RBR, SCC paths
	// from the OpenAPI spec such that they don't conflict with the CRD
	// apiextensions-apiserver spec during merging.
	oa.Config.IgnorePrefixes = append(oa.Config.IgnorePrefixes,
		"/apis/quota.openshift.io/v1/clusterresourcequotas",
		"/apis/security.openshift.io/v1/securitycontextconstraints",
		"/apis/authorization.openshift.io/v1/rolebindingrestrictions",
		"/apis/authorization.openshift.io/v1/namespaces/{namespace}/rolebindingrestrictions",
		"/apis/authorization.openshift.io/v1/watch/namespaces/{namespace}/rolebindingrestrictions",
		"/apis/authorization.openshift.io/v1/watch/rolebindingrestrictions")

	spec, err := builder2.BuildOpenAPISpecFromRoutes(restfuladapter.AdaptWebServices(c.RegisteredWebServices()), oa.Config)
	if err != nil {
		klog.Fatalf("Failed to build open api spec for root: %v", err)
	}

	// we shadow ClustResourceQuotas, RoleBindingRestrictions, and SecurityContextContstraints
	// with a CRD. This loop removes all CRQ,RBR, SCC paths
	// from the OpenAPI spec such that they don't conflict with the CRD
	// apiextensions-apiserver spec during merging.
	for pth := range spec.Paths.Paths {
		if strings.HasPrefix(pth, "/apis/quota.openshift.io/v1/clusterresourcequotas") ||
			strings.Contains(pth, "rolebindingrestrictions") ||
			strings.HasPrefix(pth, "/apis/security.openshift.io/v1/securitycontextconstraints") {
			delete(spec.Paths.Paths, pth)
		}
	}

	spec.Definitions = handler.PruneDefaults(spec.Definitions)
	openAPIVersionedService := handler.NewOpenAPIService(spec)
	openAPIVersionedService.RegisterOpenAPIVersionedService("/openapi/v2", mux)

	return openAPIVersionedService, spec
}

// InstallV3 adds the static group/versions defined in the RegisteredWebServices to the OpenAPI v3 spec
func (oa OpenAPI) InstallV3(c *restful.Container, mux *mux.PathRecorderMux) *handler3.OpenAPIService {
	openAPIVersionedService := handler3.NewOpenAPIService()
	err := openAPIVersionedService.RegisterOpenAPIV3VersionedService("/openapi/v3", mux)
	if err != nil {
		klog.Fatalf("Failed to register versioned open api spec for root: %v", err)
	}

	grouped := make(map[string][]*restful.WebService)

	for _, t := range c.RegisteredWebServices() {
		// Strip the "/" prefix from the name
		gvName := t.RootPath()[1:]
		grouped[gvName] = []*restful.WebService{t}
	}

	for gv, ws := range grouped {
		spec, err := builder3.BuildOpenAPISpecFromRoutes(restfuladapter.AdaptWebServices(ws), oa.V3Config)
		if err != nil {
			klog.Errorf("Failed to build OpenAPI v3 for group %s, %q", gv, err)

		}
		openAPIVersionedService.UpdateGroupVersion(gv, spec)
	}
	return openAPIVersionedService
}
