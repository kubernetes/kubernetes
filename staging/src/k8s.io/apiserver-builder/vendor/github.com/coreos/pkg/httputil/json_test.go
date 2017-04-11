package httputil

import (
	"net/http/httptest"
	"testing"
)

func TestWriteJSONResponse(t *testing.T) {
	for i, test := range []struct {
		code         int
		resp         interface{}
		expectedJSON string
		expectErr    bool
	}{
		{
			200,
			struct {
				A string
				B string
			}{A: "foo", B: "bar"},
			`{"A":"foo","B":"bar"}`,
			false,
		},
		{
			500,
			// Something that json.Marshal cannot serialize.
			make(chan int),
			"",
			true,
		},
	} {
		w := httptest.NewRecorder()
		err := WriteJSONResponse(w, test.code, test.resp)

		if w.Code != test.code {
			t.Errorf("case %d: w.code == %v, want %v", i, w.Code, test.code)
		}

		if (err != nil) != test.expectErr {
			t.Errorf("case %d: (err != nil) == %v, want %v. err: %v", i, err != nil, test.expectErr, err)
		}

		if string(w.Body.Bytes()) != test.expectedJSON {
			t.Errorf("case %d: w.Body.Bytes()) == %q, want %q", i,
				string(w.Body.Bytes()), test.expectedJSON)
		}

		if !test.expectErr {
			contentType := w.Header()["Content-Type"][0]
			if contentType != JSONContentType {
				t.Errorf("case %d: contentType == %v, want %v", i, contentType, JSONContentType)
			}
		}
	}

}
