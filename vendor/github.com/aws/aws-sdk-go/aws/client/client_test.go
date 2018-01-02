package client

import (
	"testing"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/client/metadata"
	"github.com/aws/aws-sdk-go/aws/request"
)

func pushBackTestHandler(name string, list *request.HandlerList) *bool {
	called := false
	(*list).PushBackNamed(request.NamedHandler{
		Name: name,
		Fn: func(r *request.Request) {
			called = true
		},
	})

	return &called
}

func pushFrontTestHandler(name string, list *request.HandlerList) *bool {
	called := false
	(*list).PushFrontNamed(request.NamedHandler{
		Name: name,
		Fn: func(r *request.Request) {
			called = true
		},
	})

	return &called
}

func TestNewClient_CopyHandlers(t *testing.T) {
	handlers := request.Handlers{}
	firstCalled := pushBackTestHandler("first", &handlers.Send)
	secondCalled := pushBackTestHandler("second", &handlers.Send)

	var clientHandlerCalled *bool
	c := New(aws.Config{}, metadata.ClientInfo{}, handlers,
		func(c *Client) {
			clientHandlerCalled = pushFrontTestHandler("client handler", &c.Handlers.Send)
		},
	)

	if e, a := 2, handlers.Send.Len(); e != a {
		t.Errorf("expect %d original handlers, got %d", e, a)
	}
	if e, a := 3, c.Handlers.Send.Len(); e != a {
		t.Errorf("expect %d client handlers, got %d", e, a)
	}

	handlers.Send.Run(nil)
	if !*firstCalled {
		t.Errorf("expect first handler to of been called")
	}
	*firstCalled = false
	if !*secondCalled {
		t.Errorf("expect second handler to of been called")
	}
	*secondCalled = false
	if *clientHandlerCalled {
		t.Errorf("expect client handler to not of been called, but was")
	}

	c.Handlers.Send.Run(nil)
	if !*firstCalled {
		t.Errorf("expect client's first handler to of been called")
	}
	if !*secondCalled {
		t.Errorf("expect client's second handler to of been called")
	}
	if !*clientHandlerCalled {
		t.Errorf("expect client's client handler to of been called")
	}

}
