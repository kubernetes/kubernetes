//go:build !go1.13
// +build !go1.13

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

package autorest

import "github.com/Azure/go-autorest/autorest/adal"

// IsTokenRefreshError returns true if the specified error implements the TokenRefreshError
// interface.  If err is a DetailedError it will walk the chain of Original errors.
func IsTokenRefreshError(err error) bool {
	if _, ok := err.(adal.TokenRefreshError); ok {
		return true
	}
	if de, ok := err.(DetailedError); ok {
		return IsTokenRefreshError(de.Original)
	}
	return false
}
