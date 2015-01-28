package channel

import (
	"testing"

	"google.golang.org/appengine/internal"
)

func TestRemapError(t *testing.T) {
	err := &internal.APIError{
		Service: "xmpp",
	}
	err = remapError(err).(*internal.APIError)
	if err.Service != "channel" {
		t.Errorf("err.Service = %q, want %q", err.Service, "channel")
	}
}
