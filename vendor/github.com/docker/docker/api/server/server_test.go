package server

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/docker/docker/api"
	"github.com/docker/docker/api/server/httputils"
	"github.com/docker/docker/api/server/middleware"

	"golang.org/x/net/context"
)

func TestMiddlewares(t *testing.T) {
	cfg := &Config{
		Version: "0.1omega2",
	}
	srv := &Server{
		cfg: cfg,
	}

	srv.UseMiddleware(middleware.NewVersionMiddleware("0.1omega2", api.DefaultVersion, api.MinVersion))

	req, _ := http.NewRequest("GET", "/containers/json", nil)
	resp := httptest.NewRecorder()
	ctx := context.Background()

	localHandler := func(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
		if httputils.VersionFromContext(ctx) == "" {
			t.Fatal("Expected version, got empty string")
		}

		if sv := w.Header().Get("Server"); !strings.Contains(sv, "Docker/0.1omega2") {
			t.Fatalf("Expected server version in the header `Docker/0.1omega2`, got %s", sv)
		}

		return nil
	}

	handlerFunc := srv.handlerWithGlobalMiddlewares(localHandler)
	if err := handlerFunc(ctx, resp, req, map[string]string{}); err != nil {
		t.Fatal(err)
	}
}
