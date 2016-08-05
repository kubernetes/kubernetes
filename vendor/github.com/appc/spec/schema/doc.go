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

// Package schema provides definitions for the JSON schema of the different
// manifests in the App Container Specification. The manifests are canonically
// represented in their respective structs:
//   - `ImageManifest`
//   - `PodManifest`
//
// Validation is performed through serialization: if a blob of JSON data will
// unmarshal to one of the *Manifests, it is considered a valid implementation
// of the standard. Similarly, if a constructed *Manifest struct marshals
// successfully to JSON, it must be valid.
package schema
