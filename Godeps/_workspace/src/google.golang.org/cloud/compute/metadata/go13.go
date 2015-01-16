// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build go1.3

package metadata

import (
	"net"
	"time"
)

// This is a workaround for https://github.com/golang/oauth2/issues/70, where
// net.Dialer.KeepAlive is unavailable on Go 1.2 (which App Engine as of
// Jan 2015 still runs).
//
// TODO(bradfitz,jbd,adg): remove this once App Engine supports Go
// 1.3+.
func init() {
	go13Dialer = func() *net.Dialer {
		return &net.Dialer{
			Timeout:   750 * time.Millisecond,
			KeepAlive: 30 * time.Second,
		}
	}
}
