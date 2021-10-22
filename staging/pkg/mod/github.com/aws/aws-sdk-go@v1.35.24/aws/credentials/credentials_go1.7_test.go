// +build go1.7

package credentials

import (
	"context"
	"testing"
)

func TestCredentialsGetWithContext(t *testing.T) {
	stub := &stubProviderConcurrent{
		stubProvider: stubProvider{
			creds: Value{
				AccessKeyID:     "AKIDEXAMPLE",
				SecretAccessKey: "KEYEXAMPLE",
			},
		},
		done: make(chan struct{}),
	}

	c := NewCredentials(stub)

	ctx, cancel1 := context.WithCancel(context.Background())
	ctx1 := &ContextWaiter{Context: ctx, waiting: make(chan struct{}, 1)}
	ctx2 := &ContextWaiter{Context: context.Background(), waiting: make(chan struct{}, 1)}

	var err1, err2 error
	var creds1, creds2 Value

	done1 := make(chan struct{})
	go func() {
		creds1, err1 = c.GetWithContext(ctx1)
		close(done1)
	}()
	<-ctx1.waiting
	<-ctx1.waiting

	done2 := make(chan struct{})
	go func() {
		creds2, err2 = c.GetWithContext(ctx2)
		close(done2)
	}()
	<-ctx2.waiting

	cancel1()
	<-done1

	close(stub.done)
	<-done2

	if err1 == nil {
		t.Errorf("expect first to have error")
	}
	if creds1.HasKeys() {
		t.Errorf("expect first not to have keys, %v", creds1)
	}

	if err2 != nil {
		t.Errorf("expect second not to have error, %v", err2)
	}
	if !creds2.HasKeys() {
		t.Errorf("expect second to have keys")
	}
}

type ContextWaiter struct {
	context.Context
	waiting chan struct{}
}

func (c *ContextWaiter) Done() <-chan struct{} {
	go func() {
		c.waiting <- struct{}{}
	}()

	return c.Context.Done()
}
