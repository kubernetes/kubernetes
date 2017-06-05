package grpc

import "testing"

func TestBackoffConfigDefaults(t *testing.T) {
	b := BackoffConfig{}
	setDefaults(&b)
	if b != DefaultBackoffConfig {
		t.Fatalf("expected BackoffConfig to pickup default parameters: %v != %v", b, DefaultBackoffConfig)
	}
}
