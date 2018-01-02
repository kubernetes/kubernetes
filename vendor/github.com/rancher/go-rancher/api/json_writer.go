package api

import (
	"encoding/json"
	"net/http"
)

type JsonWriter struct {
	r *http.Request
}

func (j *JsonWriter) Write(obj interface{}, rw http.ResponseWriter) error {
	apiContext := GetApiContext(j.r)
	rw.Header().Set("X-API-Schemas", apiContext.UrlBuilder.Collection("schema"))

	if rw.Header().Get("Content-Type") == "" {
		rw.Header().Set("Content-Type", "application/json")
	}

	enc := json.NewEncoder(rw)
	return enc.Encode(obj)
}
