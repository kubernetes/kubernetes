/*
Copyright 2019 The Kubernetes Authors.

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

package legacyflag

// TODO(mtaufen): wait until https://github.com/kubernetes/kubernetes/pull/76354
// is finalized and port those decisions to here.

// MapOptions contains options that control how the values are parsed
type MapOptions struct {
	// DisableCommaSeparatedPairs disables parsing multiple comma-separated
	// key-value pairs from a single invocation. Instead, the entire string
	// after the = separator will be parsed as the value. This can be convenient
	// if values contain commas.
	DisableCommaSeparatedPairs bool

	// KeyValueSep is the separator between a key and its corresponding value.
	// Default: equals sign (=).
	KeyValueSep string

	// PairSep is the separator between key-value pairs.
	// Default: comma (,).
	PairSep string
}

// Default applies defaults to uninitialized values in MapOptions.
func (o *MapOptions) Default() {
	if o.KeyValueSep == "" {
		o.KeyValueSep = "="
	}
	if o.PairSep == "" {
		o.PairSep = ","
	}
}
