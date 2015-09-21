package gophercloud

import (
	"github.com/racker/perigee"
	"testing"
)

// This reauth-handler does nothing, and returns no error.
func doNothing(_ AccessProvider) error {
	return nil
}

func TestOtherErrorsPropegate(t *testing.T) {
	calls := 0
	c := TestContext().WithReauthHandler(doNothing)

	err := c.WithReauth(nil, func() error {
		calls++
		return &perigee.UnexpectedResponseCodeError{
			Expected: []int{204},
			Actual:   404,
		}
	})

	if err == nil {
		t.Error("Expected MyError to be returned; got nil instead.")
		return
	}
	if _, ok := err.(*perigee.UnexpectedResponseCodeError); !ok {
		t.Error("Expected UnexpectedResponseCodeError; got %#v", err)
		return
	}
	if calls != 1 {
		t.Errorf("Expected the body to be invoked once; found %d calls instead", calls)
		return
	}
}

func Test401ErrorCausesBodyInvokation2ndTime(t *testing.T) {
	calls := 0
	c := TestContext().WithReauthHandler(doNothing)

	err := c.WithReauth(nil, func() error {
		calls++
		return &perigee.UnexpectedResponseCodeError{
			Expected: []int{204},
			Actual:   401,
		}
	})

	if err == nil {
		t.Error("Expected MyError to be returned; got nil instead.")
		return
	}
	if calls != 2 {
		t.Errorf("Expected the body to be invoked once; found %d calls instead", calls)
		return
	}
}

func TestReauthAttemptShouldHappen(t *testing.T) {
	calls := 0
	c := TestContext().WithReauthHandler(func(_ AccessProvider) error {
		calls++
		return nil
	})
	c.WithReauth(nil, func() error {
		return &perigee.UnexpectedResponseCodeError{
			Expected: []int{204},
			Actual:   401,
		}
	})

	if calls != 1 {
		t.Errorf("Expected Reauthenticator to be called once; found %d instead", calls)
		return
	}
}

type MyError struct{}

func (*MyError) Error() string {
	return "MyError instance"
}

func TestReauthErrorShouldPropegate(t *testing.T) {
	c := TestContext().WithReauthHandler(func(_ AccessProvider) error {
		return &MyError{}
	})

	err := c.WithReauth(nil, func() error {
		return &perigee.UnexpectedResponseCodeError{
			Expected: []int{204},
			Actual:   401,
		}
	})

	if _, ok := err.(*MyError); !ok {
		t.Errorf("Expected a MyError; got %#v", err)
		return
	}
}

type MyAccess struct{}

func (my *MyAccess) FirstEndpointUrlByCriteria(ApiCriteria) string {
	return ""
}
func (my *MyAccess) AuthToken() string {
	return ""
}
func (my *MyAccess) Revoke(string) error {
	return nil
}
func (my *MyAccess) Reauthenticate() error {
	return nil
}

func TestReauthHandlerUsesSameAccessProvider(t *testing.T) {
	fakeAccess := &MyAccess{}
	c := TestContext().WithReauthHandler(func(acc AccessProvider) error {
		if acc != fakeAccess {
			t.Errorf("Expected acc = fakeAccess")
		}
		return nil
	})
	c.WithReauth(fakeAccess, func() error {
		return &perigee.UnexpectedResponseCodeError{
			Expected: []int{204},
			Actual:   401,
		}
	})
}
