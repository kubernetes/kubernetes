package log
// Copyright 2013, CoreOS, Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// author: David Fisher <ddf1991@gmail.com>
// based on previous package by: Cong Ding <dinggnu@gmail.com>

type Priority int

const (
	PriEmerg Priority = iota
	PriAlert
	PriCrit
	PriErr
	PriWarning
	PriNotice
	PriInfo
	PriDebug
)

func (priority Priority) String() string {
	switch priority {
	case PriEmerg:
		return "EMERGENCY"
	case PriAlert:
		return "ALERT"
	case PriCrit:
		return "CRITICAL"
	case PriErr:
		return "ERROR"
	case PriWarning:
		return "WARNING"
	case PriNotice:
		return "NOTICE"
	case PriInfo:
		return "INFO"
	case PriDebug:
		return "DEBUG"

	default:
		return "UNKNOWN"
	}
}
