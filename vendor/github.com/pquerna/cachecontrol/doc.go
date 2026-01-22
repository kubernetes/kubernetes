/**
 *  Copyright 2015 Paul Querna
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 */

// Package cachecontrol implements the logic for HTTP Caching
//
// Deciding if an HTTP Response can be cached is often harder
// and more bug prone than an actual cache storage backend.
// cachecontrol provides a simple interface to determine if
// request and response pairs are cachable as defined under
// RFC 7234 http://tools.ietf.org/html/rfc7234
package cachecontrol
