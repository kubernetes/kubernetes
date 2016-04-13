package waiter_test

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/client"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
	"github.com/aws/aws-sdk-go/private/waiter"
)

type mockClient struct {
	*client.Client
}
type MockInput struct{}
type MockOutput struct {
	States []*MockState
}
type MockState struct {
	State *string
}

func (c *mockClient) MockRequest(input *MockInput) (*request.Request, *MockOutput) {
	op := &request.Operation{
		Name:       "Mock",
		HTTPMethod: "POST",
		HTTPPath:   "/",
	}

	if input == nil {
		input = &MockInput{}
	}

	output := &MockOutput{}
	req := c.NewRequest(op, input, output)
	req.Data = output
	return req, output
}

func TestWaiterPathAll(t *testing.T) {
	svc := &mockClient{Client: awstesting.NewClient(&aws.Config{
		Region: aws.String("mock-region"),
	})}
	svc.Handlers.Send.Clear() // mock sending
	svc.Handlers.Unmarshal.Clear()
	svc.Handlers.UnmarshalMeta.Clear()
	svc.Handlers.ValidateResponse.Clear()

	reqNum := 0
	resps := []*MockOutput{
		{ // Request 1
			States: []*MockState{
				{State: aws.String("pending")},
				{State: aws.String("pending")},
			},
		},
		{ // Request 2
			States: []*MockState{
				{State: aws.String("running")},
				{State: aws.String("pending")},
			},
		},
		{ // Request 3
			States: []*MockState{
				{State: aws.String("running")},
				{State: aws.String("running")},
			},
		},
	}

	numBuiltReq := 0
	svc.Handlers.Build.PushBack(func(r *request.Request) {
		numBuiltReq++
	})
	svc.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		if reqNum >= len(resps) {
			assert.Fail(t, "too many polling requests made")
			return
		}
		r.Data = resps[reqNum]
		reqNum++
	})

	waiterCfg := waiter.Config{
		Operation:   "Mock",
		Delay:       0,
		MaxAttempts: 10,
		Acceptors: []waiter.WaitAcceptor{
			{
				State:    "success",
				Matcher:  "pathAll",
				Argument: "States[].State",
				Expected: "running",
			},
		},
	}
	w := waiter.Waiter{
		Client: svc,
		Input:  &MockInput{},
		Config: waiterCfg,
	}

	err := w.Wait()
	assert.NoError(t, err)
	assert.Equal(t, 3, numBuiltReq)
	assert.Equal(t, 3, reqNum)
}

func TestWaiterPath(t *testing.T) {
	svc := &mockClient{Client: awstesting.NewClient(&aws.Config{
		Region: aws.String("mock-region"),
	})}
	svc.Handlers.Send.Clear() // mock sending
	svc.Handlers.Unmarshal.Clear()
	svc.Handlers.UnmarshalMeta.Clear()
	svc.Handlers.ValidateResponse.Clear()

	reqNum := 0
	resps := []*MockOutput{
		{ // Request 1
			States: []*MockState{
				{State: aws.String("pending")},
				{State: aws.String("pending")},
			},
		},
		{ // Request 2
			States: []*MockState{
				{State: aws.String("running")},
				{State: aws.String("pending")},
			},
		},
		{ // Request 3
			States: []*MockState{
				{State: aws.String("running")},
				{State: aws.String("running")},
			},
		},
	}

	numBuiltReq := 0
	svc.Handlers.Build.PushBack(func(r *request.Request) {
		numBuiltReq++
	})
	svc.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		if reqNum >= len(resps) {
			assert.Fail(t, "too many polling requests made")
			return
		}
		r.Data = resps[reqNum]
		reqNum++
	})

	waiterCfg := waiter.Config{
		Operation:   "Mock",
		Delay:       0,
		MaxAttempts: 10,
		Acceptors: []waiter.WaitAcceptor{
			{
				State:    "success",
				Matcher:  "path",
				Argument: "States[].State",
				Expected: "running",
			},
		},
	}
	w := waiter.Waiter{
		Client: svc,
		Input:  &MockInput{},
		Config: waiterCfg,
	}

	err := w.Wait()
	assert.NoError(t, err)
	assert.Equal(t, 3, numBuiltReq)
	assert.Equal(t, 3, reqNum)
}

func TestWaiterFailure(t *testing.T) {
	svc := &mockClient{Client: awstesting.NewClient(&aws.Config{
		Region: aws.String("mock-region"),
	})}
	svc.Handlers.Send.Clear() // mock sending
	svc.Handlers.Unmarshal.Clear()
	svc.Handlers.UnmarshalMeta.Clear()
	svc.Handlers.ValidateResponse.Clear()

	reqNum := 0
	resps := []*MockOutput{
		{ // Request 1
			States: []*MockState{
				{State: aws.String("pending")},
				{State: aws.String("pending")},
			},
		},
		{ // Request 2
			States: []*MockState{
				{State: aws.String("running")},
				{State: aws.String("pending")},
			},
		},
		{ // Request 3
			States: []*MockState{
				{State: aws.String("running")},
				{State: aws.String("stopping")},
			},
		},
	}

	numBuiltReq := 0
	svc.Handlers.Build.PushBack(func(r *request.Request) {
		numBuiltReq++
	})
	svc.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		if reqNum >= len(resps) {
			assert.Fail(t, "too many polling requests made")
			return
		}
		r.Data = resps[reqNum]
		reqNum++
	})

	waiterCfg := waiter.Config{
		Operation:   "Mock",
		Delay:       0,
		MaxAttempts: 10,
		Acceptors: []waiter.WaitAcceptor{
			{
				State:    "success",
				Matcher:  "pathAll",
				Argument: "States[].State",
				Expected: "running",
			},
			{
				State:    "failure",
				Matcher:  "pathAny",
				Argument: "States[].State",
				Expected: "stopping",
			},
		},
	}
	w := waiter.Waiter{
		Client: svc,
		Input:  &MockInput{},
		Config: waiterCfg,
	}

	err := w.Wait().(awserr.Error)
	assert.Error(t, err)
	assert.Equal(t, "ResourceNotReady", err.Code())
	assert.Equal(t, "failed waiting for successful resource state", err.Message())
	assert.Equal(t, 3, numBuiltReq)
	assert.Equal(t, 3, reqNum)
}

func TestWaiterError(t *testing.T) {
	svc := &mockClient{Client: awstesting.NewClient(&aws.Config{
		Region: aws.String("mock-region"),
	})}
	svc.Handlers.Send.Clear() // mock sending
	svc.Handlers.Unmarshal.Clear()
	svc.Handlers.UnmarshalMeta.Clear()
	svc.Handlers.ValidateResponse.Clear()

	reqNum := 0
	resps := []*MockOutput{
		{ // Request 1
			States: []*MockState{
				{State: aws.String("pending")},
				{State: aws.String("pending")},
			},
		},
		{ // Request 2, error case
		},
		{ // Request 3
			States: []*MockState{
				{State: aws.String("running")},
				{State: aws.String("running")},
			},
		},
	}

	numBuiltReq := 0
	svc.Handlers.Build.PushBack(func(r *request.Request) {
		numBuiltReq++
	})
	svc.Handlers.Send.PushBack(func(r *request.Request) {
		if reqNum == 1 {
			r.Error = awserr.New("MockException", "mock exception message", nil)
			r.HTTPResponse = &http.Response{
				StatusCode: 400,
				Status:     http.StatusText(400),
				Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
			}
			reqNum++
		}
	})
	svc.Handlers.Unmarshal.PushBack(func(r *request.Request) {
		if reqNum >= len(resps) {
			assert.Fail(t, "too many polling requests made")
			return
		}
		r.Data = resps[reqNum]
		reqNum++
	})

	waiterCfg := waiter.Config{
		Operation:   "Mock",
		Delay:       0,
		MaxAttempts: 10,
		Acceptors: []waiter.WaitAcceptor{
			{
				State:    "success",
				Matcher:  "pathAll",
				Argument: "States[].State",
				Expected: "running",
			},
			{
				State:    "retry",
				Matcher:  "error",
				Argument: "",
				Expected: "MockException",
			},
		},
	}
	w := waiter.Waiter{
		Client: svc,
		Input:  &MockInput{},
		Config: waiterCfg,
	}

	err := w.Wait()
	assert.NoError(t, err)
	assert.Equal(t, 3, numBuiltReq)
	assert.Equal(t, 3, reqNum)
}

func TestWaiterStatus(t *testing.T) {
	svc := &mockClient{Client: awstesting.NewClient(&aws.Config{
		Region: aws.String("mock-region"),
	})}
	svc.Handlers.Send.Clear() // mock sending
	svc.Handlers.Unmarshal.Clear()
	svc.Handlers.UnmarshalMeta.Clear()
	svc.Handlers.ValidateResponse.Clear()

	reqNum := 0
	svc.Handlers.Build.PushBack(func(r *request.Request) {
		reqNum++
	})
	svc.Handlers.Send.PushBack(func(r *request.Request) {
		code := 200
		if reqNum == 3 {
			code = 404
		}
		r.HTTPResponse = &http.Response{
			StatusCode: code,
			Status:     http.StatusText(code),
			Body:       ioutil.NopCloser(bytes.NewReader([]byte{})),
		}
	})

	waiterCfg := waiter.Config{
		Operation:   "Mock",
		Delay:       0,
		MaxAttempts: 10,
		Acceptors: []waiter.WaitAcceptor{
			{
				State:    "success",
				Matcher:  "status",
				Argument: "",
				Expected: 404,
			},
		},
	}
	w := waiter.Waiter{
		Client: svc,
		Input:  &MockInput{},
		Config: waiterCfg,
	}

	err := w.Wait()
	assert.NoError(t, err)
	assert.Equal(t, 3, reqNum)
}
