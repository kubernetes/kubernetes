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
