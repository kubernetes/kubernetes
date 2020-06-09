// Copyright 2015 The etcd Authors
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

/*
Package integration implements tests built upon embedded etcd, and focus on
etcd correctness.

Features/goals of the integration tests:
1. test the whole code base except command-line parsing.
2. check internal data, including raft, store and etc.
3. based on goroutines, which is faster than process.
4. mainly tests user behavior and user-facing API.
*/
package integration
