package parser

import (
	"testing"
)

func TestPosError_impl(t *testing.T) {
	var _ error = new(PosError)
}
