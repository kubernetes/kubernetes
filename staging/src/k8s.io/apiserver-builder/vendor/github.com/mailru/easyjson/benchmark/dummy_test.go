package benchmark

import (
	"testing"
)

type DummyWriter struct{}

func (w DummyWriter) Write(data []byte) (int, error) { return len(data), nil }

func TestToSuppressNoTestsWarning(t *testing.T) {}
