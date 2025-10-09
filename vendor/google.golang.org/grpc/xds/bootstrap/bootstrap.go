/*
 *
 * Copyright 2022 gRPC authors.
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

// Package bootstrap provides the functionality to register possible options
// for aspects of the xDS client through the bootstrap file.
//
// # Experimental
//
// Notice: This package is EXPERIMENTAL and may be changed or removed
// in a later release.
package bootstrap

import (
	"encoding/json"

	"google.golang.org/grpc/credentials"
)

// registry is a map from credential type name to Credential builder.
var registry = make(map[string]Credentials)

// Credentials interface encapsulates a credentials.Bundle builder
// that can be used for communicating with the xDS Management server.
type Credentials interface {
	// Build returns a credential bundle associated with this credential, and
	// a function to cleans up additional resources associated with this bundle
	// when it is no longer needed.
	Build(config json.RawMessage) (credentials.Bundle, func(), error)
	// Name returns the credential name associated with this credential.
	Name() string
}

// RegisterCredentials registers Credentials used for connecting to the xds
// management server.
//
// NOTE: this function must only be called during initialization time (i.e. in
// an init() function), and is not thread-safe. If multiple credentials are
// registered with the same name, the one registered last will take effect.
func RegisterCredentials(c Credentials) {
	registry[c.Name()] = c
}

// GetCredentials returns the credentials associated with a given name.
// If no credentials are registered with the name, nil will be returned.
func GetCredentials(name string) Credentials {
	if c, ok := registry[name]; ok {
		return c
	}

	return nil
}
