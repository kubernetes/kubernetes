package osin

import (
	"encoding/json"
	"net/http"
)

// OutputJSON encodes the Response to JSON and writes to the http.ResponseWriter
func OutputJSON(rs *Response, w http.ResponseWriter, r *http.Request) error {
	// Add headers
	for i, k := range rs.Headers {
		for _, v := range k {
			w.Header().Add(i, v)
		}
	}

	if rs.Type == REDIRECT {
		// Output redirect with parameters
		u, err := rs.GetRedirectUrl()
		if err != nil {
			return err
		}
		w.Header().Add("Location", u)
		w.WriteHeader(302)
	} else {
		// Output json
		w.Header().Add("Content-Type", "application/json")
		w.WriteHeader(rs.StatusCode)

		encoder := json.NewEncoder(w)
		err := encoder.Encode(rs.Output)
		if err != nil {
			return err
		}
	}
	return nil
}
