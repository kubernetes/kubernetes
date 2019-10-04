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

package dynamiccertificates

import (
	"bytes"
	"strings"
)

type unionCAContent []CAContentProvider

// NewUnionCAContentProvider returns a CAContentProvider that is a union of other CAContentProviders
func NewUnionCAContentProvider(caContentProviders ...CAContentProvider) CAContentProvider {
	return unionCAContent(caContentProviders)
}

// Name is just an identifier
func (c unionCAContent) Name() string {
	names := []string{}
	for _, curr := range c {
		names = append(names, curr.Name())
	}
	return strings.Join(names, ",")
}

// CurrentCABundleContent provides ca bundle byte content
func (c unionCAContent) CurrentCABundleContent() []byte {
	caBundles := [][]byte{}
	for _, curr := range c {
		caBundles = append(caBundles, curr.CurrentCABundleContent())
	}

	return bytes.Join(caBundles, []byte("\n"))
}
