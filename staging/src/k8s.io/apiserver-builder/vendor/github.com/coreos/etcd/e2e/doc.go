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

/*
Package e2e implements tests built upon etcd binaries, and focus on
end-to-end testing.

Features/goals of the end-to-end tests:
1. test command-line parsing and arguments.
2. test user-facing command-line API.
3. launch full processes and check for expected outputs.
*/
package e2e
