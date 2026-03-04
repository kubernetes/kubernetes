/*
Copyright 2014 The Kubernetes Authors.

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

package common

import "log"

// Runner is a client or server to run.
type Runner interface {
	// NewOptions returns a new empty options structure to be populated
	// by from the JSON -options argument.
	NewOptions() interface{}
	// Run the client or server, taking in options. This execute the
	// test code.
	Run(logger *log.Logger, options interface{}) error
}
