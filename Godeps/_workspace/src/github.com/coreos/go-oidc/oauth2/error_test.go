package oauth2

import (
	"fmt"
	"reflect"
	"testing"
)

func TestUnmarshalError(t *testing.T) {
	tests := []struct {
		b []byte
		e *Error
		o bool
	}{
		{
			b: []byte("{ \"error\": \"invalid_client\", \"state\": \"foo\" }"),
			e: &Error{Type: ErrorInvalidClient, State: "foo"},
			o: true,
		},
		{
			b: []byte("{ \"error\": \"invalid_grant\", \"state\": \"bar\" }"),
			e: &Error{Type: ErrorInvalidGrant, State: "bar"},
			o: true,
		},
		{
			b: []byte("{ \"error\": \"invalid_request\", \"state\": \"\" }"),
			e: &Error{Type: ErrorInvalidRequest, State: ""},
			o: true,
		},
		{
			b: []byte("{ \"error\": \"server_error\", \"state\": \"elroy\" }"),
			e: &Error{Type: ErrorServerError, State: "elroy"},
			o: true,
		},
		{
			b: []byte("{ \"error\": \"unsupported_grant_type\", \"state\": \"\" }"),
			e: &Error{Type: ErrorUnsupportedGrantType, State: ""},
			o: true,
		},
		{
			b: []byte("{ \"error\": \"unsupported_response_type\", \"state\": \"\" }"),
			e: &Error{Type: ErrorUnsupportedResponseType, State: ""},
			o: true,
		},
		// Should fail json unmarshal
		{
			b: nil,
			e: nil,
			o: false,
		},
		{
			b: []byte("random string"),
			e: nil,
			o: false,
		},
	}

	for i, tt := range tests {
		err := unmarshalError(tt.b)
		oerr, ok := err.(*Error)

		if ok != tt.o {
			t.Errorf("%v != %v, %v", ok, tt.o, oerr)
			t.Errorf("case %d: want=%+v, got=%+v", i, tt.e, oerr)
		}

		if ok && !reflect.DeepEqual(tt.e, oerr) {
			t.Errorf("case %d: want=%+v, got=%+v", i, tt.e, oerr)
		}

		if !ok && tt.e != nil {
			want := fmt.Sprintf("unrecognized error: %s", string(tt.b))
			got := tt.e.Error()
			if want != got {
				t.Errorf("case %d: want=%+v, got=%+v", i, want, got)
			}
		}
	}
}
