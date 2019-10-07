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

package runtime

import (
	"errors"
	"fmt"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestStrictDecodingErrorWrapping(t *testing.T) {
	innerErrMsg := "some other error"
	strictErr := NewStrictDecodingError(fmt.Errorf(innerErrMsg), "foo")
	assert.NotNil(t, strictErr)

	innerErr := errors.Unwrap(strictErr)
	assert.Error(t, innerErr)
	assert.Contains(t, innerErr.Error(), innerErrMsg)

	wrappingErr := fmt.Errorf("wrapping: %w", strictErr)
	assert.True(t, IsStrictDecodingError(errors.Unwrap(wrappingErr)))
}
