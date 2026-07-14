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

package verify

import "fmt"

// defaultServiceAudiencePort is the port the server-side derivation assumes when
// an admission webhook's Service reference omits one.
const defaultServiceAudiencePort = 443

// AudienceForService returns the token audience an admission webhook backed by a
// Service must present. It mirrors the server-side derivation in kube-apiserver
// (validateWebhookAudience): the host is the Service's cluster DNS name, the
// port is always explicit (defaulting to 443 when the Service reference omits
// one), and the reference path, if any, is appended verbatim.
//
// A URL-backed webhook does not use this helper: its audience is its configured
// URL verbatim.
//
// It lives in this leaf package (not oidc) so both the OIDC authenticator and
// the net/http adapter can share one source of truth without either importing
// the other's dependencies.
func AudienceForService(name, namespace string, port int32, path string) string {
	if port == 0 {
		port = defaultServiceAudiencePort
	}
	return fmt.Sprintf("https://%s.%s.svc:%d%s", name, namespace, port, path)
}
