/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package resourcedefaults contains an example plug-in that
// cluster operators can use to prevent unlimited CPU and
// Memory usage for a container.  It intercepts all pod
// create and update requests and applies a default
// Memory and CPU quantity if none is supplied.
// This plug-in can be enhanced in the future to make the default value
// configurable via the admission control configuration file.
package resourcedefaults
