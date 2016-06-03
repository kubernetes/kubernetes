// Copyright 2014 Gary Burd
//
// Licensed under the Apache License, Version 2.0 (the "License"): you may
// not use this file except in compliance with the License. You may obtain
// a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations
// under the License.

package internal

import (
	"strings"
)

const (
	WatchState = 1 << iota
	MultiState
	SubscribeState
	MonitorState
)

type CommandInfo struct {
	Set, Clear int
}

var commandInfos = map[string]CommandInfo{
	"WATCH":      {Set: WatchState},
	"UNWATCH":    {Clear: WatchState},
	"MULTI":      {Set: MultiState},
	"EXEC":       {Clear: WatchState | MultiState},
	"DISCARD":    {Clear: WatchState | MultiState},
	"PSUBSCRIBE": {Set: SubscribeState},
	"SUBSCRIBE":  {Set: SubscribeState},
	"MONITOR":    {Set: MonitorState},
}

func init() {
	for n, ci := range commandInfos {
		commandInfos[strings.ToLower(n)] = ci
	}
}

func LookupCommandInfo(commandName string) CommandInfo {
	if ci, ok := commandInfos[commandName]; ok {
		return ci
	}
	return commandInfos[strings.ToUpper(commandName)]
}
