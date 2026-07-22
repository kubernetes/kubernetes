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

import (
	"fmt"
	"strings"
)

// AudienceForService returns the token audience an admission webhook backed by a
// Service must present: https://<name>.<namespace>.svc:<port><path>, with the
// port defaulting to 443. It mirrors kube-apiserver's validateWebhookAudience.
//
// The path is normalized to always begin with a single "/", so that a missing
// or leading-slash-less path cannot silently change the audience the verifier
// compares against. Because this string identifies the verification target,
// callers must not depend on an un-normalized form.
//
// A URL-backed webhook does not use this helper; its audience is its configured
// URL verbatim.
func AudienceForService(name, namespace string, port int32, path string) string {
	if port == 0 {
		// 443 is the port the server-side derivation assumes when an admission
		// webhook's Service reference omits one.
		port = 443
	}
	if !strings.HasPrefix(path, "/") {
		path = "/" + path
	}
	return fmt.Sprintf("https://%s.%s.svc:%d%s", name, namespace, port, path)
}
