/*
Copyright 2015 The Kubernetes Authors.

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

package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestWrap(t *testing.T) {
	tt := `before ->{{.Long | wrap "**"}}<- after`
	data := struct {
		Long string
	}{
		`Hodor, hodor; hodor hodor;
hodor hodor... Hodor hodor hodor? Hodor. Hodor hodor hodor hodor...
Hodor hodor hodor; hodor hodor hodor! Hodor, hodor. Hodor. Hodor,
HODOR hodor, hodor hodor; hodor hodor; hodor HODOR hodor, hodor hodor?
Hodor. Hodor hodor - hodor hodor. Hodor hodor HODOR! Hodor hodor - hodor...
Hodor hodor HODOR hodor, hodor hodor hodor! Hodor, hodor... Hodor hodor
hodor hodor hodor hodor! Hodor, hodor; hodor hodor. Hodor.`,
	}
	output, _ := ExecuteTemplateToString(tt, data)
	t.Logf("%q", output)

	assert.Equal(t, `before ->**Hodor, hodor; hodor hodor; hodor hodor... Hodor hodor hodor? Hodor. Hodor
**hodor hodor hodor... Hodor hodor hodor; hodor hodor hodor! Hodor, hodor.
**Hodor. Hodor, HODOR hodor, hodor hodor; hodor hodor; hodor HODOR hodor, hodor
**hodor? Hodor. Hodor hodor - hodor hodor. Hodor hodor HODOR! Hodor hodor -
**hodor... Hodor hodor HODOR hodor, hodor hodor hodor! Hodor, hodor... Hodor
**hodor hodor hodor hodor hodor! Hodor, hodor; hodor hodor. Hodor.
<- after`, output)
}

func TestTrim(t *testing.T) {
	tt := `before ->{{.Messy | trim }}<- after`
	data := struct {
		Messy string
	}{
		"\t  stuff\n \r ",
	}
	output, _ := ExecuteTemplateToString(tt, data)
	t.Logf("%q", output)

	assert.Equal(t, `before ->stuff<- after`, output)
}
