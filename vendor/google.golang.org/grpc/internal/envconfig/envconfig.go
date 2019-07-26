/*
 *
 * Copyright 2018 gRPC authors.
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

// Package envconfig contains grpc settings configured by environment variables.
package envconfig

import (
	"os"
	"strings"
)

const (
	prefix              = "GRPC_GO_"
	retryStr            = prefix + "RETRY"
	requireHandshakeStr = prefix + "REQUIRE_HANDSHAKE"
)

// RequireHandshakeSetting describes the settings for handshaking.
type RequireHandshakeSetting int

const (
	// RequireHandshakeOn indicates to wait for handshake before considering a
	// connection ready/successful.
	RequireHandshakeOn RequireHandshakeSetting = iota
	// RequireHandshakeOff indicates to not wait for handshake before
	// considering a connection ready/successful.
	RequireHandshakeOff
)

var (
	// Retry is set if retry is explicitly enabled via "GRPC_GO_RETRY=on".
	Retry = strings.EqualFold(os.Getenv(retryStr), "on")
	// RequireHandshake is set based upon the GRPC_GO_REQUIRE_HANDSHAKE
	// environment variable.
	//
	// Will be removed after the 1.18 release.
	RequireHandshake = RequireHandshakeOn
)

func init() {
	switch strings.ToLower(os.Getenv(requireHandshakeStr)) {
	case "on":
		fallthrough
	default:
		RequireHandshake = RequireHandshakeOn
	case "off":
		RequireHandshake = RequireHandshakeOff
	}
}
