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

package handlers

import (
	"fmt"
	"net/http"
	"strconv"
	"strings"

	"github.com/golang/glog"
	"github.com/prometheus/client_golang/prometheus"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/auth/authenticator"
	"k8s.io/kubernetes/pkg/auth/user"
)

var (
	authenticatedUserCounter = prometheus.NewCounterVec(
		prometheus.CounterOpts{
			Name: "authenticated_user_requests",
			Help: "Counter of authenticated requests broken out by username.",
		},
		[]string{"username"},
	)
)

func init() {
	prometheus.MustRegister(authenticatedUserCounter)
}

// NewRequestAuthenticator creates an http handler that tries to authenticate the given request as a user, and then
// stores any such user found onto the provided context for the request. If authentication fails or returns an error
// the failed handler is used. On success, handler is invoked to serve the request.
func NewRequestAuthenticator(mapper api.RequestContextMapper, auth authenticator.Request, failed http.Handler, handler http.Handler) (http.Handler, error) {
	return api.NewRequestContextFilter(
		mapper,
		http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
			user, ok, err := auth.AuthenticateRequest(req)
			if err != nil || !ok {
				if err != nil {
					glog.Errorf("Unable to authenticate the request due to an error: %v", err)
				}
				failed.ServeHTTP(w, req)
				return
			}

			if ctx, ok := mapper.Get(req); ok {
				mapper.Update(req, api.WithUser(ctx, user))
			}

			authenticatedUserCounter.WithLabelValues(compressUsername(user.GetName())).Inc()
			setAuthenticationInfo(w, user)

			handler.ServeHTTP(w, req)
		}),
	)
}

// set the "Authentication-Info" header for a response.
func setAuthenticationInfo(w http.ResponseWriter, userInfo user.Info) {
	// Requirements for quoting these values is defined in RFC 7235 (which links to RFC 7230).
	// We use the strconv package because it encompasses the requirements, including escaping
	// inner quotes, spaces, and non-printable characters.
	//
	// QuoteToASCII because non-ASCII HTTP headers aren't guarenteed to work with many clients.
	//
	// https://tools.ietf.org/html/rfc7235#section-2.1
	// https://tools.ietf.org/html/rfc7230#section-3.2.6
	username := strconv.QuoteToASCII(userInfo.GetName())
	uid := strconv.QuoteToASCII(userInfo.GetUID())

	w.Header().Set("Authentication-Info", fmt.Sprintf("username=%s, uid=%s", username, uid))
}

func Unauthorized(supportsBasicAuth bool) http.HandlerFunc {
	if supportsBasicAuth {
		return unauthorizedBasicAuth
	}
	return unauthorized
}

// unauthorizedBasicAuth serves an unauthorized message to clients.
func unauthorizedBasicAuth(w http.ResponseWriter, req *http.Request) {
	w.Header().Set("WWW-Authenticate", `Basic realm="kubernetes-master"`)
	http.Error(w, "Unauthorized", http.StatusUnauthorized)
}

// unauthorized serves an unauthorized message to clients.
func unauthorized(w http.ResponseWriter, req *http.Request) {
	http.Error(w, "Unauthorized", http.StatusUnauthorized)
}

// compressUsername maps all possible usernames onto a small set of categories
// of usernames. This is done both to limit the cardinality of the
// authorized_user_requests metric, and to avoid pushing actual usernames in the
// metric.
func compressUsername(username string) string {
	switch {
	// Known internal identities.
	case username == "admin" ||
		username == "client" ||
		username == "kube_proxy" ||
		username == "kubelet" ||
		username == "system:serviceaccount:kube-system:default":
		return username
	// Probably an email address.
	case strings.Contains(username, "@"):
		return "email_id"
	// Anything else (custom service accounts, custom external identities, etc.)
	default:
		return "other"
	}
}
