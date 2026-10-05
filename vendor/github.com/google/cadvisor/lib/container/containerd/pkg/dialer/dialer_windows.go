// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//go:build windows

package dialer

import (
	"fmt"
	"net"
	"time"
)

// isNoent is a no-op on non-unix platforms; the containerd handler that uses
// this dialer is linux-only, so this exists solely so the package compiles.
func isNoent(_ error) bool {
	return false
}

// dialer is unsupported on this platform. cadvisor's containerd handler runs
// only on linux; on other platforms this returns an error rather than dialing.
func dialer(_ string, _ time.Duration) (net.Conn, error) {
	return nil, fmt.Errorf("containerd dialer is not supported on this platform")
}

// DialAddress returns the address unchanged. The unix variant prepends the
// unix:// scheme; on this platform the address is never actually dialed.
func DialAddress(address string) string {
	return address
}
