// Copyright 2015 CoreOS, Inc.
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

// +build windows

package etcdmain

// TODO(barakmich): So because file locking on Windows is untested, the
// temporary fix is to default to unlimited snapshots and WAL files, with manual
// removal. Perhaps not the most elegant solution, but it's at least safe and
// we'd totally love a PR to fix the story around locking.
const (
	defaultMaxSnapshots = 0
	defaultMaxWALs      = 0
)
