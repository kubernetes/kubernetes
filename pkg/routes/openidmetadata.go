/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"net/http"

	restful "github.com/emicklei/go-restful/v3"

	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/serviceaccount"
)

// This code is in package routes because many controllers import
// pkg/serviceaccount, but are not allowed by import-boss to depend on
// go-restful. All logic that deals with keys is kept in pkg/serviceaccount,
// and only the rendered JSON is passed into this server.

const (
	// cacheControl is the value of the Cache-Control header. Overrides the
	// global `private, no-cache` setting.
	headerCacheControl = "Cache-Control"

	cacheControlTemplate = "public, max-age=%d"

	// mimeJWKS is the content type of the keyset response
	mimeJWKS = "application/jwk-set+json"
)

// OpenIDMetadataServer is an HTTP server for metadata of the KSA token issuer.
type OpenIDMetadataServer struct {
	provider serviceaccount.OpenIDMetadataProvider
}

// NewOpenIDMetadataServer creates a new OpenIDMetadataServer.
// The issuer is the OIDC issuer; keys are the keys that may be used to sign
// KSA tokens.
func NewOpenIDMetadataServer(provider serviceaccount.OpenIDMetadataProvider) *OpenIDMetadataServer {
	return &OpenIDMetadataServer{provider: provider}
}

// Install adds this server to the request router c.
func (s *OpenIDMetadataServer) Install(c *restful.Container) {
	// Configuration WebService
	// Container.Add "will detect duplicate root paths and exit in that case",
	// so we need a root for /.well-known/openid-configuration to avoid conflicts.
	cfg := new(restful.WebService).
		Produces(restful.MIME_JSON)

	cfg.Path(serviceaccount.OpenIDConfigPath).Route(
		cfg.GET("").
			To(fromStandard(s.serveConfiguration)).
			Doc("get service account issuer OpenID configuration, also known as the 'OIDC discovery doc'").
			Operation("getServiceAccountIssuerOpenIDConfiguration").
			// Just include the OK, doesn't look like we include Internal Error in our openapi-spec.
			Returns(http.StatusOK, "OK", ""))
	c.Add(cfg)

	// JWKS WebService
	jwks := new(restful.WebService).
		Produces(mimeJWKS)

	jwks.Path(serviceaccount.JWKSPath).Route(
		jwks.GET("").
			To(fromStandard(s.serveKeys)).
			Doc("get service account issuer OpenID JSON Web Key Set (contains public token verification keys)").
			Operation("getServiceAccountIssuerOpenIDKeyset").
			// Just include the OK, doesn't look like we include Internal Error in our openapi-spec.
			Returns(http.StatusOK, "OK", ""))
	c.Add(jwks)
}

// fromStandard provides compatibility between the standard (net/http) handler signature and the restful signature.
func fromStandard(h http.HandlerFunc) restful.RouteFunction {
	return func(req *restful.Request, resp *restful.Response) {
		h(resp, req.Request)
	}
}

func (s *OpenIDMetadataServer) serveConfiguration(w http.ResponseWriter, req *http.Request) {
	configJSON, maxAge := s.provider.GetConfigJSON()
	w.Header().Set(restful.HEADER_ContentType, restful.MIME_JSON)
	w.Header().Set(headerCacheControl, fmt.Sprintf(cacheControlTemplate, maxAge))
	if _, err := w.Write(configJSON); err != nil {
		klog.Errorf("failed to write service account issuer metadata response: %v", err)
		return
	}
}

func (s *OpenIDMetadataServer) serveKeys(w http.ResponseWriter, req *http.Request) {
	keysetJSON, maxAge := s.provider.GetKeysetJSON()
	// Per RFC7517 : https://tools.ietf.org/html/rfc7517#section-8.5.1
	w.Header().Set(restful.HEADER_ContentType, mimeJWKS)
	w.Header().Set(headerCacheControl, fmt.Sprintf(cacheControlTemplate, maxAge))
	if _, err := w.Write(keysetJSON); err != nil {
		klog.Errorf("failed to write service account issuer JWKS response: %v", err)
		return
	}
}
