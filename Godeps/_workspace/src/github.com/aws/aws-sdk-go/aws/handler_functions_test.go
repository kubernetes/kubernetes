package aws

import (
	"net/http"
	"os"
	"testing"

	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/internal/apierr"
	"github.com/stretchr/testify/assert"
)

func TestValidateEndpointHandler(t *testing.T) {
	os.Clearenv()
	svc := NewService(&Config{Region: "us-west-2"})
	svc.Handlers.Clear()
	svc.Handlers.Validate.PushBack(ValidateEndpointHandler)

	req := NewRequest(svc, &Operation{Name: "Operation"}, nil, nil)
	err := req.Build()

	assert.NoError(t, err)
}

func TestValidateEndpointHandlerErrorRegion(t *testing.T) {
	os.Clearenv()
	svc := NewService(nil)
	svc.Handlers.Clear()
	svc.Handlers.Validate.PushBack(ValidateEndpointHandler)

	req := NewRequest(svc, &Operation{Name: "Operation"}, nil, nil)
	err := req.Build()

	assert.Error(t, err)
	assert.Equal(t, ErrMissingRegion, err)
}

type mockCredsProvider struct {
	expired        bool
	retreiveCalled bool
}

func (m *mockCredsProvider) Retrieve() (credentials.Value, error) {
	m.retreiveCalled = true
	return credentials.Value{}, nil
}

func (m *mockCredsProvider) IsExpired() bool {
	return m.expired
}

func TestAfterRetryRefreshCreds(t *testing.T) {
	os.Clearenv()
	credProvider := &mockCredsProvider{}
	svc := NewService(&Config{Credentials: credentials.NewCredentials(credProvider), MaxRetries: 1})

	svc.Handlers.Clear()
	svc.Handlers.ValidateResponse.PushBack(func(r *Request) {
		r.Error = apierr.New("UnknownError", "", nil)
		r.HTTPResponse = &http.Response{StatusCode: 400}
	})
	svc.Handlers.UnmarshalError.PushBack(func(r *Request) {
		r.Error = apierr.New("ExpiredTokenException", "", nil)
	})
	svc.Handlers.AfterRetry.PushBack(func(r *Request) {
		AfterRetryHandler(r)
	})

	assert.True(t, svc.Config.Credentials.IsExpired(), "Expect to start out expired")
	assert.False(t, credProvider.retreiveCalled)

	req := NewRequest(svc, &Operation{Name: "Operation"}, nil, nil)
	req.Send()

	assert.True(t, svc.Config.Credentials.IsExpired())
	assert.False(t, credProvider.retreiveCalled)

	_, err := svc.Config.Credentials.Get()
	assert.NoError(t, err)
	assert.True(t, credProvider.retreiveCalled)
}
