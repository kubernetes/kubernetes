package request

import (
	"testing"
)

func TestCopy(t *testing.T) {
	handlers := Handlers{}
	op := &Operation{}
	op.HTTPMethod = "Foo"
	req := &Request{}
	req.Operation = op
	req.Handlers = handlers

	r := req.copy()

	if r == req {
		t.Fatal("expect request pointer copy to be different")
	}
	if r.Operation == req.Operation {
		t.Errorf("expect request operation pointer to be different")
	}

	if e, a := req.Operation.HTTPMethod, r.Operation.HTTPMethod; e != a {
		t.Errorf("expect %q http method, got %q", e, a)
	}
}
