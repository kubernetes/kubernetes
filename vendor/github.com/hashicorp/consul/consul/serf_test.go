package consul

import (
	"testing"
)

func TestUserEventNames(t *testing.T) {
	out := userEventName("foo")
	if out != "consul:event:foo" {
		t.Fatalf("bad: %v", out)
	}
	if !isUserEvent(out) {
		t.Fatalf("bad")
	}
	if isUserEvent("foo") {
		t.Fatalf("bad")
	}
	if raw := rawUserEventName(out); raw != "foo" {
		t.Fatalf("bad: %v", raw)
	}
}
