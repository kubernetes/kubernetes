/*
Copyright 2023 The Kubernetes Authors.

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

package coloredwriter

import (
	"io"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/kubectl/pkg/coloredwriter/prettyjson"
)

type ColoredJsonWriter struct {
	Delegate io.Writer
}

func (c ColoredJsonWriter) Write(p []byte) (int, error) {
	n := len(p)

	var v interface{}

	// TODO:
	// - check if is 1 line json

	if err := json.Unmarshal(p, &v); err != nil {
		return c.Delegate.Write(p)
	}

	s, err := prettyjson.Marshal(v)
	if err != nil {
		return c.Delegate.Write(p)
	}

	_, err = c.Delegate.Write(s)
	if err != nil {
		return c.Delegate.Write(p)
	}

	return n, nil
}
