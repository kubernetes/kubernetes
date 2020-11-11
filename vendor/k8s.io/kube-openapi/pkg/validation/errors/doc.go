// Copyright 2015 go-swagger maintainers
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

/*

Package errors provides an Error interface and several concrete types
implementing this interface to manage API errors and JSON-schema validation
errors.

A middleware handler ServeError() is provided to serve the errors types
it defines.

It is used throughout the various go-openapi toolkit libraries
(https://github.com/go-openapi).

*/
package errors
