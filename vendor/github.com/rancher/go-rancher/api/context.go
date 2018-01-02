package api

import (
	"net/http"

	"github.com/gorilla/context"
	"github.com/rancher/go-rancher/client"
)

type key int

const (
	contextKey key = 0
)

type ApiContext struct {
	r                 *http.Request
	schemas           *client.Schemas
	UrlBuilder        UrlBuilder
	apiResponseWriter ApiResponseWriter
	responseWriter    http.ResponseWriter
}

func GetApiContext(r *http.Request) *ApiContext {
	if rv := context.Get(r, contextKey); rv != nil {
		return rv.(*ApiContext)
	}
	return nil
}

func CreateApiContext(rw http.ResponseWriter, r *http.Request, schemas *client.Schemas) error {
	urlBuilder, err := NewUrlBuilder(r, schemas)
	if err != nil {
		return err
	}

	apiContext := &ApiContext{
		r:                 r,
		schemas:           schemas,
		UrlBuilder:        urlBuilder,
		apiResponseWriter: parseResponseType(r),
		responseWriter:    rw,
	}

	context.Set(r, contextKey, apiContext)
	return nil
}
