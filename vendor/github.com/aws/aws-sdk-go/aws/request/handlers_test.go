package request_test

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/request"
)

func TestHandlerList(t *testing.T) {
	s := ""
	r := &request.Request{}
	l := request.HandlerList{}
	l.PushBack(func(r *request.Request) {
		s += "a"
		r.Data = s
	})
	l.Run(r)
	assert.Equal(t, "a", s)
	assert.Equal(t, "a", r.Data)
}

func TestMultipleHandlers(t *testing.T) {
	r := &request.Request{}
	l := request.HandlerList{}
	l.PushBack(func(r *request.Request) { r.Data = nil })
	l.PushFront(func(r *request.Request) { r.Data = aws.Bool(true) })
	l.Run(r)
	if r.Data != nil {
		t.Error("Expected handler to execute")
	}
}

func TestNamedHandlers(t *testing.T) {
	l := request.HandlerList{}
	named := request.NamedHandler{Name: "Name", Fn: func(r *request.Request) {}}
	named2 := request.NamedHandler{Name: "NotName", Fn: func(r *request.Request) {}}
	l.PushBackNamed(named)
	l.PushBackNamed(named)
	l.PushBackNamed(named2)
	l.PushBack(func(r *request.Request) {})
	assert.Equal(t, 4, l.Len())
	l.Remove(named)
	assert.Equal(t, 2, l.Len())
}
