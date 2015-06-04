package aws

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestHandlerList(t *testing.T) {
	s := ""
	r := &Request{}
	l := HandlerList{}
	l.PushBack(func(r *Request) {
		s += "a"
		r.Data = s
	})
	l.Run(r)
	assert.Equal(t, "a", s)
	assert.Equal(t, "a", r.Data)
}

func TestMultipleHandlers(t *testing.T) {
	r := &Request{}
	l := HandlerList{}
	l.PushBack(func(r *Request) { r.Data = nil })
	l.PushFront(func(r *Request) { r.Data = Boolean(true) })
	l.Run(r)
	if r.Data != nil {
		t.Error("Expected handler to execute")
	}
}
