// Copyright 2016 The etcd Authors
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

package client

import (
	"errors"
	"regexp"
)

var (
	roleNotFoundRegExp *regexp.Regexp
	userNotFoundRegExp *regexp.Regexp
)

func init() {
	roleNotFoundRegExp = regexp.MustCompile("auth: Role .* does not exist.")
	userNotFoundRegExp = regexp.MustCompile("auth: User .* does not exist.")
}

// IsKeyNotFound returns true if the error code is ErrorCodeKeyNotFound.
func IsKeyNotFound(err error) bool {
	var cErr Error
	return errors.As(err, &cErr) && cErr.Code == ErrorCodeKeyNotFound
}

// IsRoleNotFound returns true if the error means role not found of v2 API.
func IsRoleNotFound(err error) bool {
	var ae authError
	return errors.As(err, &ae) && roleNotFoundRegExp.MatchString(ae.Message)
}

// IsUserNotFound returns true if the error means user not found of v2 API.
func IsUserNotFound(err error) bool {
	var ae authError
	return errors.As(err, &ae) && userNotFoundRegExp.MatchString(ae.Message)
}
