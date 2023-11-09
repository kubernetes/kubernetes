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
	"bytes"
	"io"
	"k8s.io/kubectl/pkg/coloredwriter/yh"
)

type ColoredYamlWriter struct {
	Delegate io.Writer
}

func (c ColoredYamlWriter) Write(p []byte) (int, error) {
	n := len(p)

	h, err := yh.Highlight(bytes.NewReader(p))
	if err != nil {
		return c.Delegate.Write(p)
	}

	if _, err := c.Delegate.Write([]byte(h)); err != nil {
		return c.Delegate.Write(p)
	}

	return n, nil
}
