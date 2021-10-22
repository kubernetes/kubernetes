// +build tools

// Copyright 2019 Google LLC
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

// This package exists to cause `go mod` and `go get` to believe these tools
// are dependencies, even though they are not runtime dependencies of any
// package (these are tools used by our CI builds). This means they will appear
// in our `go.mod` file, but will not be a part of the build. Also, since the
// build target is something non-existent, these should not be included in any
// binaries.

package genproto

import (
	_ "github.com/golang/protobuf/protoc-gen-go"
	_ "golang.org/x/lint/golint"
	_ "golang.org/x/tools/cmd/goimports"
)
