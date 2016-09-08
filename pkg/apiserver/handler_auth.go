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

package apiserver

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

const (
	authInfoHeader              = "Authentication-Info"
	authInfoImpersonationHeader = "Impersonation-" + authInfoHeader
)

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
			setAuthInfo(w, user, authInfoHeader)

			if ctx, ok := mapper.Get(req); ok {
				mapper.Update(req, api.WithUser(ctx, user))
			}

			authenticatedUserCounter.WithLabelValues(compressUsername(user.GetName())).Inc()

			handler.ServeHTTP(w, req)
		}),
	)
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

// setAuthInfo sets the "Authentication-Info" header for the underlying response.
func setAuthInfo(w http.ResponseWriter, info user.Info, header string) {
	// We use the strconv package because it encompasses the requirements, including escaping
	// inner quotes, spaces, and non-printable characters.
	//
	// QuoteToASCII differs from the normal "%q" formatter by escaping unicode characters to valid
	// ASCII. Use this because non-ASCII HTTP headers aren't guarenteed to work with many HTTP clients.
	//
	// QuoteToASCII example: https://play.golang.org/p/B1jyPZGXlN
	//
	// Relevant RFCs:
	//   https://tools.ietf.org/html/rfc7615 ("Authentication-Info" header)
	//   https://tools.ietf.org/html/rfc7235#section-2.1 (format of auth params)
	//   https://tools.ietf.org/html/rfc7230#section-3.2.6 ("quoted-string" format)
	username := strconv.QuoteToASCII(info.GetName())
	uid := strconv.QuoteToASCII(info.GetUID())

	w.Header().Set(header, fmt.Sprintf("username=%s, uid=%s", username, uid))
}
