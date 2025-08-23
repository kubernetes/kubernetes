// Copyright 2018 Google LLC All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package name defines structured types for representing image references.
//
// What's in a name? For image references, not nearly enough!
//
// Image references look a lot like URLs, but they differ in that they don't
// contain the scheme (http or https), they can end with a :tag or a @digest
// (the latter being validated), and they perform defaulting for missing
// components.
//
// Since image references don't contain the scheme, we do our best to infer
// if we use http or https from the given hostname. We allow http fallback for
// any host that looks like localhost (localhost, 127.0.0.1, ::1), ends in
// ".local", or is in the "private" address space per RFC 1918. For everything
// else, we assume https only. To override this heuristic, use the Insecure
// option.
//
// Image references with a digest signal to us that we should verify the content
// of the image matches the digest. E.g. when pulling a Digest reference, we'll
// calculate the sha256 of the manifest returned by the registry and error out
// if it doesn't match what we asked for.
//
// For defaulting, we interpret "ubuntu" as
// "index.docker.io/library/ubuntu:latest" because we add the missing repo
// "library", the missing registry "index.docker.io", and the missing tag
// "latest". To disable this defaulting, use the StrictValidation option. This
// is useful e.g. to only allow image references that explicitly set a tag or
// digest, so that you don't accidentally pull "latest".
package name
