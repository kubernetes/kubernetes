/*
 *
 * Copyright 2024 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package bootstrap

import (
	"encoding/json"

	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/google"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/internal/xds/bootstrap/tlscreds"
)

func init() {
	RegisterCredentials(&insecureCredsBuilder{})
	RegisterCredentials(&googleDefaultCredsBuilder{})
	RegisterCredentials(&tlsCredsBuilder{})
}

// insecureCredsBuilder implements the `Credentials` interface defined in
// package `xds/bootstrap` and encapsulates an insecure credential.
type insecureCredsBuilder struct{}

func (i *insecureCredsBuilder) Build(json.RawMessage) (credentials.Bundle, func(), error) {
	return insecure.NewBundle(), func() {}, nil
}

func (i *insecureCredsBuilder) Name() string {
	return "insecure"
}

// tlsCredsBuilder implements the `Credentials` interface defined in
// package `xds/bootstrap` and encapsulates a TLS credential.
type tlsCredsBuilder struct{}

func (t *tlsCredsBuilder) Build(config json.RawMessage) (credentials.Bundle, func(), error) {
	return tlscreds.NewBundle(config)
}

func (t *tlsCredsBuilder) Name() string {
	return "tls"
}

// googleDefaultCredsBuilder implements the `Credentials` interface defined in
// package `xds/bootstrap` and encapsulates a Google Default credential.
type googleDefaultCredsBuilder struct{}

func (d *googleDefaultCredsBuilder) Build(json.RawMessage) (credentials.Bundle, func(), error) {
	return google.NewDefaultCredentials(), func() {}, nil
}

func (d *googleDefaultCredsBuilder) Name() string {
	return "google_default"
}
