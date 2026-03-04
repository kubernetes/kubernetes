/*
 *
 * Copyright 2025 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package internal contains code internal to the encoding package.
package internal

// RegisterCompressorForTesting registers a compressor in the global compressor
// registry. It returns a cleanup function that should be called at the end
// of the test to unregister the compressor.
//
// This prevents compressors registered in one test from appearing in the
// encoding headers of subsequent tests.
var RegisterCompressorForTesting any // func RegisterCompressor(c Compressor) func()
