/*
Copyright 2017 The Kubernetes Authors.

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

package image

import "runtime"

const (
	AMD64   string = "amd64"
	ARM     string = "arm"
	ARM64   string = "arm64"
	PPC64LE string = "ppc64le"
	S390X   string = "s390x"
)

type MultiArchImage struct {
	AMD64   string
	ARM     string
	ARM64   string
	PPC64LE string
	S390X   string
}

func (m *MultiArchImage) getArchImage() string {
	switch arch := runtime.GOARCH; arch {
	case AMD64:
		return m.AMD64
	case ARM:
		return m.ARM
	case ARM64:
		return m.ARM64
	case PPC64LE:
		return m.PPC64LE
	case S390X:
		return m.S390X
	}
	return ""
}
