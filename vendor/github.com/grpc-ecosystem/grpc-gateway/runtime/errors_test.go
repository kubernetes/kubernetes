package runtime_test

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/grpc-ecosystem/grpc-gateway/runtime"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

func TestDefaultHTTPError(t *testing.T) {
	ctx := context.Background()

	for _, spec := range []struct {
		err    error
		status int
		msg    string
	}{
		{
			err:    fmt.Errorf("example error"),
			status: http.StatusInternalServerError,
			msg:    "example error",
		},
		{
			err:    grpc.Errorf(codes.NotFound, "no such resource"),
			status: http.StatusNotFound,
			msg:    "no such resource",
		},
	} {
		w := httptest.NewRecorder()
		req, _ := http.NewRequest("", "", nil) // Pass in an empty request to match the signature
		runtime.DefaultHTTPError(ctx, &runtime.JSONBuiltin{}, w, req, spec.err)

		if got, want := w.Header().Get("Content-Type"), "application/json"; got != want {
			t.Errorf(`w.Header().Get("Content-Type") = %q; want %q; on spec.err=%v`, got, want, spec.err)
		}
		if got, want := w.Code, spec.status; got != want {
			t.Errorf("w.Code = %d; want %d", got, want)
		}

		body := make(map[string]interface{})
		if err := json.Unmarshal(w.Body.Bytes(), &body); err != nil {
			t.Errorf("json.Unmarshal(%q, &body) failed with %v; want success", w.Body.Bytes(), err)
			continue
		}
		if got, want := body["error"].(string), spec.msg; !strings.Contains(got, want) {
			t.Errorf(`body["error"] = %q; want %q; on spec.err=%v`, got, want, spec.err)
		}
	}
}
