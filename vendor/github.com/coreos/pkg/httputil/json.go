package httputil

import (
	"encoding/json"
	"net/http"
)

const (
	JSONContentType = "application/json"
)

func WriteJSONResponse(w http.ResponseWriter, code int, resp interface{}) error {
	enc, err := json.Marshal(resp)
	if err != nil {
		w.WriteHeader(http.StatusInternalServerError)
		return err
	}

	w.Header().Set("Content-Type", JSONContentType)
	w.WriteHeader(code)

	_, err = w.Write(enc)
	if err != nil {
		return err
	}
	return nil
}
