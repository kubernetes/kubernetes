package corehandlers_test

import (
	"fmt"
	"net/http"
	"os"
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/corehandlers"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/awstesting"
)

func TestValidateEndpointHandler(t *testing.T) {
	os.Clearenv()

	svc := awstesting.NewClient(aws.NewConfig().WithRegion("us-west-2"))
	svc.Handlers.Clear()
	svc.Handlers.Validate.PushBackNamed(corehandlers.ValidateEndpointHandler)

	req := svc.NewRequest(&request.Operation{Name: "Operation"}, nil, nil)
	err := req.Build()

	assert.NoError(t, err)
}

func TestValidateEndpointHandlerErrorRegion(t *testing.T) {
	os.Clearenv()

	svc := awstesting.NewClient()
	svc.Handlers.Clear()
	svc.Handlers.Validate.PushBackNamed(corehandlers.ValidateEndpointHandler)

	req := svc.NewRequest(&request.Operation{Name: "Operation"}, nil, nil)
	err := req.Build()

	assert.Error(t, err)
	assert.Equal(t, aws.ErrMissingRegion, err)
}

type mockCredsProvider struct {
	expired        bool
	retrieveCalled bool
}

func (m *mockCredsProvider) Retrieve() (credentials.Value, error) {
	m.retrieveCalled = true
	return credentials.Value{}, nil
}

func (m *mockCredsProvider) IsExpired() bool {
	return m.expired
}

func TestAfterRetryRefreshCreds(t *testing.T) {
	os.Clearenv()
	credProvider := &mockCredsProvider{}

	svc := awstesting.NewClient(&aws.Config{
		Credentials: credentials.NewCredentials(credProvider),
		MaxRetries:  aws.Int(1),
	})

	svc.Handlers.Clear()
	svc.Handlers.ValidateResponse.PushBack(func(r *request.Request) {
		r.Error = awserr.New("UnknownError", "", nil)
		r.HTTPResponse = &http.Response{StatusCode: 400}
	})
	svc.Handlers.UnmarshalError.PushBack(func(r *request.Request) {
		r.Error = awserr.New("ExpiredTokenException", "", nil)
	})
	svc.Handlers.AfterRetry.PushBackNamed(corehandlers.AfterRetryHandler)

	assert.True(t, svc.Config.Credentials.IsExpired(), "Expect to start out expired")
	assert.False(t, credProvider.retrieveCalled)

	req := svc.NewRequest(&request.Operation{Name: "Operation"}, nil, nil)
	req.Send()

	assert.True(t, svc.Config.Credentials.IsExpired())
	assert.False(t, credProvider.retrieveCalled)

	_, err := svc.Config.Credentials.Get()
	assert.NoError(t, err)
	assert.True(t, credProvider.retrieveCalled)
}

type testSendHandlerTransport struct{}

func (t *testSendHandlerTransport) RoundTrip(r *http.Request) (*http.Response, error) {
	return nil, fmt.Errorf("mock error")
}

func TestSendHandlerError(t *testing.T) {
	svc := awstesting.NewClient(&aws.Config{
		HTTPClient: &http.Client{
			Transport: &testSendHandlerTransport{},
		},
	})
	svc.Handlers.Clear()
	svc.Handlers.Send.PushBackNamed(corehandlers.SendHandler)
	r := svc.NewRequest(&request.Operation{Name: "Operation"}, nil, nil)

	r.Send()

	assert.Error(t, r.Error)
	assert.NotNil(t, r.HTTPResponse)
}
