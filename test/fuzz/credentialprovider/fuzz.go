//go:build gofuzz
// +build gofuzz

/*
Copyright 2022 The Kubernetes Authors.

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

package credentialprovider

import (
	fuzz "github.com/AdaLogics/go-fuzz-headers"
	"k8s.io/kubernetes/pkg/credentialprovider"
)

// FuzzUrlsMatch implements a fuzzer
// that targets credentialprovider.URLsMatchStr
func FuzzUrlsMatch(data []byte) int {
	f := fuzz.NewConsumer(data)
	glob, err := f.GetString()
	if err != nil {
		return 0
	}
	target, err := f.GetString()
	if err != nil {
		return 0
	}
	_, _ = credentialprovider.URLsMatchStr(glob, target)
	return 1
}
