// Copyright 2015 The appc Authors
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

// Package lastditch provides fallback redefinitions of parts of
// schemas provided by schema package.
//
// Almost no validation of schemas is done (besides checking if data
// really is `JSON`-encoded and kind is either `ImageManifest` or
// `PodManifest`. This is to get as much data as possible from an
// invalid manifest. The main aim of the package is to be used for the
// better error reporting. The another aim might be to force some
// operation (like removing a pod), which would otherwise fail because
// of an invalid manifest.
//
// To avoid validation during deserialization, types provided by this
// package use plain strings.
package lastditch
