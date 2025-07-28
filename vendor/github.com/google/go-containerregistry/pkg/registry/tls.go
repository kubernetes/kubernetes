// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package registry

import (
	"net/http/httptest"

	ggcrtest "github.com/google/go-containerregistry/internal/httptest"
)

// TLS returns an httptest server, with an http client that has been configured to
// send all requests to the returned server. The TLS certs are generated for the given domain
// which should correspond to the domain the image is stored in.
// If you need a transport, Client().Transport is correctly configured.
func TLS(domain string) (*httptest.Server, error) {
	return ggcrtest.NewTLSServer(domain, New())
}
