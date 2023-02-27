// Copyright 2013 Google Inc.  All rights reserved.
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

// Package pretty pretty-prints Go structures.
//
// This package uses reflection to examine a Go value and can
// print out in a nice, aligned fashion.  It supports three
// modes (normal, compact, and extended) for advanced use.
//
// See the Reflect and Print examples for what the output looks like.
package pretty

// TODO:
//   - Catch cycles
