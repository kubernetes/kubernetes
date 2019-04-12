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

	restful "github.com/emicklei/go-restful"
	"github.com/go-openapi/spec"
	"k8s.io/klog/v2"

	"k8s.io/apiserver/pkg/server/mux"
	"k8s.io/kube-openapi/pkg/builder"
	"k8s.io/kube-openapi/pkg/common"
	"k8s.io/kube-openapi/pkg/handler"
)

// OpenAPI installs spec endpoints for each web service.
type OpenAPI struct {
	Config *common.Config
}

// Install adds the SwaggerUI webservice to the given mux.
func (oa OpenAPI) Install(c *restful.Container, mux *mux.PathRecorderMux) (*handler.OpenAPIService, *spec.Swagger) {
	spec, err := builder.BuildOpenAPISpec(c.RegisteredWebServices(), oa.Config)
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

	openAPIVersionedService, err := handler.NewOpenAPIService(spec)
	if err != nil {
		klog.Fatalf("Failed to create OpenAPIService: %v", err)
	}

	err = openAPIVersionedService.RegisterOpenAPIVersionedService("/openapi/v2", mux)
	if err != nil {
		klog.Fatalf("Failed to register versioned open api spec for root: %v", err)
	}

	return openAPIVersionedService, spec
}
