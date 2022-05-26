// Copyright 2022 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package promhttp

// Option are used to configure a middleware or round tripper..
type Option func(*option)

type option struct {
	extraMethods []string
}

// WithExtraMethods adds additional HTTP methods to the list of allowed methods.
// See https://developer.mozilla.org/en-US/docs/Web/HTTP/Methods for the default list.
//
// See the example for ExampleInstrumentHandlerWithExtraMethods for example usage.
func WithExtraMethods(methods ...string) Option {
	return func(o *option) {
		o.extraMethods = methods
	}
}
