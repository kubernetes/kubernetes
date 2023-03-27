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

package ktesting

import (
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestFormat(t *testing.T) {
	obj := config{
		TypeMeta: metav1.TypeMeta{
			Kind:       "config",
			APIVersion: "v1",
		},
		RealField: 42,
	}

	assert.Equal(t, "&TypeMeta{Kind:config,APIVersion:v1,}", obj.String(), "config.String()")
	assert.Equal(t, `RealField: 42
apiVersion: v1
kind: config
`, Format(obj).String(), "Format(config).String()")
	// fmt.Sprintf would call String if it was available.
	assert.Equal(t, "&{{config v1} %!s(int=42)}", fmt.Sprintf("%s", Format(obj).MarshalLog()), "fmt.Sprintf")
}

type config struct {
	metav1.TypeMeta // implements fmt.Stringer

	RealField int
}
