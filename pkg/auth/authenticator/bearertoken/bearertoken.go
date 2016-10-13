/*
Copyright 2014 The Kubernetes Authors.

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

package bearertoken

import (
	"net/http"
	"strings"

	"github.com/go-openapi/spec"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
)

type Authenticator struct {
	auth authenticator.Token
	// name is used in OpenAPI SecurityDefinition to distinguish different bearer authenticator
	name string
}

func New(auth authenticator.Token, name string) *Authenticator {
	return &Authenticator{auth, strings.Title(name)}
}

func (a *Authenticator) AuthenticateRequest(req *http.Request) (user.Info, bool, error) {
	auth := strings.TrimSpace(req.Header.Get("Authorization"))
	if auth == "" {
		return nil, false, nil
	}
	parts := strings.Split(auth, " ")
	if len(parts) < 2 || strings.ToLower(parts[0]) != "bearer" {
		return nil, false, nil
	}

	token := parts[1]
	return a.auth.AuthenticateToken(token)
}

// GetOpenAPISecurityDefinition returns a Bearer authentication SecurityDefinition.
func (a *Authenticator) GetOpenAPISecurityDefinition() (spec.SecurityDefinitions, error) {
	ret := spec.SecurityDefinitions{}
	ret[a.name+"Bearer"] = &spec.SecurityScheme{
		SecuritySchemeProps: spec.SecuritySchemeProps{
			Type:        "apiKey",
			Name:        "authorization",
			In:          "header",
			Description: "Bearer " + a.name + " authentication",
		},
	}
	return ret, nil
}
