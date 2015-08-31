package oidc

import (
	"errors"
	"net/http"
	"reflect"
	"testing"

	phttp "github.com/coreos/go-oidc/http"
	"github.com/coreos/go-oidc/jose"
)

type staticTokenRefresher struct {
	verify  func(jose.JWT) error
	refresh func() (jose.JWT, error)
}

func (s *staticTokenRefresher) Verify(jwt jose.JWT) error {
	return s.verify(jwt)
}

func (s *staticTokenRefresher) Refresh() (jose.JWT, error) {
	return s.refresh()
}

func TestAuthenticatedTransportVerifiedJWT(t *testing.T) {
	tests := []struct {
		refresher TokenRefresher
		startJWT  jose.JWT
		wantJWT   jose.JWT
		wantError error
	}{
		// verification succeeds, so refresh is not called
		{
			refresher: &staticTokenRefresher{
				verify:  func(jose.JWT) error { return nil },
				refresh: func() (jose.JWT, error) { return jose.JWT{RawPayload: "2"}, nil },
			},
			startJWT: jose.JWT{RawPayload: "1"},
			wantJWT:  jose.JWT{RawPayload: "1"},
		},

		// verification fails, refresh succeeds so cached JWT changes
		{
			refresher: &staticTokenRefresher{
				verify:  func(jose.JWT) error { return errors.New("fail!") },
				refresh: func() (jose.JWT, error) { return jose.JWT{RawPayload: "2"}, nil },
			},
			startJWT: jose.JWT{RawPayload: "1"},
			wantJWT:  jose.JWT{RawPayload: "2"},
		},

		// verification succeeds, so failing refresh isn't attempted
		{
			refresher: &staticTokenRefresher{
				verify:  func(jose.JWT) error { return nil },
				refresh: func() (jose.JWT, error) { return jose.JWT{}, errors.New("fail!") },
			},
			startJWT: jose.JWT{RawPayload: "1"},
			wantJWT:  jose.JWT{RawPayload: "1"},
		},

		// verification fails, but refresh fails, too
		{
			refresher: &staticTokenRefresher{
				verify:  func(jose.JWT) error { return errors.New("fail!") },
				refresh: func() (jose.JWT, error) { return jose.JWT{}, errors.New("fail!") },
			},
			startJWT:  jose.JWT{RawPayload: "1"},
			wantJWT:   jose.JWT{},
			wantError: errors.New("unable to acquire valid JWT: fail!"),
		},
	}

	for i, tt := range tests {
		at := &AuthenticatedTransport{
			TokenRefresher: tt.refresher,
			jwt:            tt.startJWT,
		}

		gotJWT, err := at.verifiedJWT()
		if !reflect.DeepEqual(tt.wantError, err) {
			t.Errorf("#%d: unexpected error: want=%#v got=%#v", i, tt.wantError, err)
		}
		if !reflect.DeepEqual(tt.wantJWT, gotJWT) {
			t.Errorf("#%d: incorrect JWT returned from verifiedJWT: want=%#v got=%#v", i, tt.wantJWT, gotJWT)
		}
	}
}

func TestAuthenticatedTransportJWTCaching(t *testing.T) {
	at := &AuthenticatedTransport{
		TokenRefresher: &staticTokenRefresher{
			verify:  func(jose.JWT) error { return errors.New("fail!") },
			refresh: func() (jose.JWT, error) { return jose.JWT{RawPayload: "2"}, nil },
		},
		jwt: jose.JWT{RawPayload: "1"},
	}

	wantJWT := jose.JWT{RawPayload: "2"}
	gotJWT, err := at.verifiedJWT()
	if err != nil {
		t.Fatalf("got non-nil error: %#v", err)
	}
	if !reflect.DeepEqual(wantJWT, gotJWT) {
		t.Fatalf("incorrect JWT returned from verifiedJWT: want=%#v got=%#v", wantJWT, gotJWT)
	}

	at.TokenRefresher = &staticTokenRefresher{
		verify:  func(jose.JWT) error { return nil },
		refresh: func() (jose.JWT, error) { return jose.JWT{RawPayload: "3"}, nil },
	}

	// the previous JWT should still be cached on the AuthenticatedTransport since
	// it is still valid, even though there's a new token ready to refresh
	gotJWT, err = at.verifiedJWT()
	if err != nil {
		t.Fatalf("got non-nil error: %#v", err)
	}
	if !reflect.DeepEqual(wantJWT, gotJWT) {
		t.Fatalf("incorrect JWT returned from verifiedJWT: want=%#v got=%#v", wantJWT, gotJWT)
	}
}

func TestAuthenticatedTransportRoundTrip(t *testing.T) {
	rr := &phttp.RequestRecorder{Response: &http.Response{StatusCode: http.StatusOK}}
	at := &AuthenticatedTransport{
		TokenRefresher: &staticTokenRefresher{
			verify: func(jose.JWT) error { return nil },
		},
		RoundTripper: rr,
		jwt:          jose.JWT{RawPayload: "1"},
	}

	req := http.Request{}
	_, err := at.RoundTrip(&req)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	if !reflect.DeepEqual(req, http.Request{}) {
		t.Errorf("http.Request object was modified")
	}

	want := []string{"Bearer .1."}
	got := rr.Request.Header["Authorization"]
	if !reflect.DeepEqual(want, got) {
		t.Errorf("incorrect Authorization header: want=%#v got=%#v", want, got)
	}
}

func TestAuthenticatedTransportRoundTripRefreshFail(t *testing.T) {
	rr := &phttp.RequestRecorder{Response: &http.Response{StatusCode: http.StatusOK}}
	at := &AuthenticatedTransport{
		TokenRefresher: &staticTokenRefresher{
			verify:  func(jose.JWT) error { return errors.New("fail!") },
			refresh: func() (jose.JWT, error) { return jose.JWT{}, errors.New("fail!") },
		},
		RoundTripper: rr,
		jwt:          jose.JWT{RawPayload: "1"},
	}

	_, err := at.RoundTrip(&http.Request{})
	if err == nil {
		t.Errorf("expected non-nil error")
	}
}
