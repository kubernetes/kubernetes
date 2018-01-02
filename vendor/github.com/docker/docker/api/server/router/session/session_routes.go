package session

import (
	"net/http"

	apierrors "github.com/docker/docker/api/errors"
	"golang.org/x/net/context"
)

func (sr *sessionRouter) startSession(ctx context.Context, w http.ResponseWriter, r *http.Request, vars map[string]string) error {
	err := sr.backend.HandleHTTPRequest(ctx, w, r)
	if err != nil {
		return apierrors.NewBadRequestError(err)
	}
	return nil
}
