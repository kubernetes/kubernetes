/*
Copyright 2025 The Kubernetes Authors.

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

package encodings

import "io"

func init() {
	Register(gzipEncoding{})
}

func Register(encoding Interface) {
	name := encoding.EncoderName()
	registry[name] = encoding
}

func Get(name string) (Interface, bool) {
	algorithm, ok := registry[name]
	return algorithm, ok
}

type Interface interface {
	NewWriter(w io.Writer) io.WriteCloser
	EncoderName() string
}

var registry = make(map[string]Interface)
