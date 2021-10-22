package request

import (
	"bytes"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"testing"
	"time"

	"github.com/aws/aws-sdk-go/aws/awserr"
)

type testReader struct {
	duration time.Duration
	count    int
}

func (r *testReader) Read(b []byte) (int, error) {
	if r.count > 0 {
		r.count--
		return len(b), nil
	}
	time.Sleep(r.duration)
	return 0, io.EOF
}

func (r *testReader) Close() error {
	return nil
}

func TestTimeoutReadCloser(t *testing.T) {
	reader := timeoutReadCloser{
		reader: &testReader{
			duration: time.Second,
			count:    5,
		},
		duration: time.Millisecond,
	}
	b := make([]byte, 100)
	_, err := reader.Read(b)
	if err != nil {
		t.Log(err)
	}
}

func TestTimeoutReadCloserSameDuration(t *testing.T) {
	reader := timeoutReadCloser{
		reader: &testReader{
			duration: time.Millisecond,
			count:    5,
		},
		duration: time.Millisecond,
	}
	b := make([]byte, 100)
	_, err := reader.Read(b)
	if err != nil {
		t.Log(err)
	}
}

func TestWithResponseReadTimeout(t *testing.T) {
	r := Request{
		HTTPRequest: &http.Request{
			URL: &url.URL{},
		},
		HTTPResponse: &http.Response{
			Body: ioutil.NopCloser(bytes.NewReader(nil)),
		},
	}
	r.ApplyOptions(WithResponseReadTimeout(time.Second))
	err := r.Send()
	if err != nil {
		t.Error(err)
	}
	v, ok := r.HTTPResponse.Body.(*timeoutReadCloser)
	if !ok {
		t.Error("Expected the body to be a timeoutReadCloser")
	}
	if v.duration != time.Second {
		t.Errorf("Expected %v, but receive %v\n", time.Second, v.duration)
	}
}

func TestAdaptToResponseTimeout(t *testing.T) {
	testCases := []struct {
		childErr         error
		r                Request
		expectedRootCode string
	}{
		{
			childErr: awserr.New(ErrCodeResponseTimeout, "timeout!", nil),
			r: Request{
				Error: awserr.New("ErrTest", "FooBar", awserr.New(ErrCodeResponseTimeout, "timeout!", nil)),
			},
			expectedRootCode: ErrCodeResponseTimeout,
		},
		{
			childErr: awserr.New(ErrCodeResponseTimeout+"1", "timeout!", nil),
			r: Request{
				Error: awserr.New("ErrTest", "FooBar", awserr.New(ErrCodeResponseTimeout+"1", "timeout!", nil)),
			},
			expectedRootCode: "ErrTest",
		},
		{
			r: Request{
				Error: awserr.New("ErrTest", "FooBar", nil),
			},
			expectedRootCode: "ErrTest",
		},
	}

	for i, c := range testCases {
		adaptToResponseTimeoutError(&c.r)
		if aerr, ok := c.r.Error.(awserr.Error); !ok {
			t.Errorf("Case %d: Expected 'awserr', but received %v", i+1, c.r.Error)
		} else if aerr.Code() != c.expectedRootCode {
			t.Errorf("Case %d: Expected %q, but received %s", i+1, c.expectedRootCode, aerr.Code())
		}
	}
}
