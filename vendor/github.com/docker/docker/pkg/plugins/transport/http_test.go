package transport

import (
	"io"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHTTPTransport(t *testing.T) {
	var r io.Reader
	roundTripper := &http.Transport{}
	newTransport := NewHTTPTransport(roundTripper, "http", "0.0.0.0")
	request, err := newTransport.NewRequest("", r)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, "POST", request.Method)
}
