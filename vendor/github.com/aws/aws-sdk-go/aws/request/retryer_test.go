package request

import (
	"errors"
	"fmt"
	"testing"

	"github.com/aws/aws-sdk-go/aws/awserr"
)

func TestRequestThrottling(t *testing.T) {
	req := Request{}

	req.Error = awserr.New("Throttling", "", nil)
	if e, a := true, req.IsErrorThrottle(); e != a {
		t.Errorf("expect %t to be throttled, was %t", e, a)
	}
}

type mockTempError bool

func (e mockTempError) Error() string {
	return fmt.Sprintf("mock temporary error: %t", e.Temporary())
}
func (e mockTempError) Temporary() bool {
	return bool(e)
}

func TestIsErrorRetryable(t *testing.T) {
	cases := []struct {
		Err    error
		IsTemp bool
	}{
		{
			Err:    awserr.New(ErrCodeSerialization, "temporary error", mockTempError(true)),
			IsTemp: true,
		},
		{
			Err:    awserr.New(ErrCodeSerialization, "temporary error", mockTempError(false)),
			IsTemp: false,
		},
		{
			Err:    awserr.New(ErrCodeSerialization, "some error", errors.New("blah")),
			IsTemp: false,
		},
		{
			Err:    awserr.New("SomeError", "some error", nil),
			IsTemp: false,
		},
		{
			Err:    awserr.New("RequestError", "some error", nil),
			IsTemp: true,
		},
	}

	for i, c := range cases {
		retryable := IsErrorRetryable(c.Err)
		if e, a := c.IsTemp, retryable; e != a {
			t.Errorf("%d, expect %t temporary error, got %t", i, e, a)
		}
	}
}
