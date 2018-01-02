package libcontainer

import (
	"fmt"
	"io/ioutil"
	"testing"
)

func TestErrorDetail(t *testing.T) {
	err := newGenericError(fmt.Errorf("test error"), SystemError)
	if derr := err.Detail(ioutil.Discard); derr != nil {
		t.Fatal(derr)
	}
}

func TestErrorWithCode(t *testing.T) {
	err := newGenericError(fmt.Errorf("test error"), SystemError)
	if code := err.Code(); code != SystemError {
		t.Fatalf("expected err code %q but %q", SystemError, code)
	}
}

func TestErrorWithError(t *testing.T) {
	cc := []struct {
		errmsg string
		cause  string
	}{
		{
			errmsg: "test error",
		},
		{
			errmsg: "test error",
			cause:  "test",
		},
	}

	for _, v := range cc {
		err := newSystemErrorWithCause(fmt.Errorf(v.errmsg), v.cause)

		msg := err.Error()
		if v.cause == "" && msg != v.errmsg {
			t.Fatalf("expected err(%q) equal errmsg(%q)", msg, v.errmsg)
		}
		if v.cause != "" && msg == v.errmsg {
			t.Fatalf("unexpected err(%q) equal errmsg(%q)", msg, v.errmsg)
		}

	}
}
