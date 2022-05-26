// Copyright (c) 2020-2022 Denis Tingaikin
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at:
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package goheader

type Issue interface {
	Location() Location
	Message() string
}

type issue struct {
	msg      string
	location Location
}

func (i *issue) Location() Location {
	return i.location
}

func (i *issue) Message() string {
	return i.msg
}

func NewIssueWithLocation(msg string, location Location) Issue {
	return &issue{
		msg:      msg,
		location: location,
	}
}

func NewIssue(msg string) Issue {
	return &issue{
		msg: msg,
	}
}
