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

package server

import (
	"net"
	"net/http"
	"time"

	"k8s.io/apiserver/pkg/server/serving"
	restclient "k8s.io/client-go/rest"
)

// The secure serving and request authentication and authorization configuration
// lives in k8s.io/apiserver/pkg/server/serving so that components which only
// serve HTTPS do not link the generic API server. The declarations below keep
// existing importers of this package compiling.

// SecureServingInfo is an alias for serving.SecureServingInfo.
type SecureServingInfo = serving.SecureServingInfo

// AuthenticationInfo is an alias for serving.AuthenticationInfo.
type AuthenticationInfo = serving.AuthenticationInfo

// AuthorizationInfo is an alias for serving.AuthorizationInfo.
type AuthorizationInfo = serving.AuthorizationInfo

// LoopbackClientServerNameOverride is passed to the apiserver from the loopback client in order to
// select the loopback certificate via SNI if TLS is used.
const LoopbackClientServerNameOverride = serving.LoopbackClientServerNameOverride

// RunServer spawns a go-routine continuously serving until the stopCh is
// closed. See serving.RunServer.
func RunServer(server *http.Server, ln net.Listener, shutDownTimeout time.Duration, stopCh <-chan struct{}) (<-chan struct{}, <-chan struct{}, error) {
	return serving.RunServer(server, ln, shutDownTimeout, stopCh)
}

// LoopbackHostPort returns the host and port loopback REST clients should use
// to contact the server.
func LoopbackHostPort(bindAddress string) (string, string, error) {
	return serving.LoopbackHostPort(bindAddress)
}

// AuthorizeClientBearerToken wraps the authenticator in loopback authentication logic
// if the loopback client config is specified AND it has a bearer token.
func AuthorizeClientBearerToken(loopback *restclient.Config, authn *AuthenticationInfo, authz *AuthorizationInfo) {
	serving.AuthorizeClientBearerToken(loopback, authn, authz)
}
