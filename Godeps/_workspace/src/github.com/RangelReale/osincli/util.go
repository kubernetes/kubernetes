package osincli

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
)

// Parse basic authentication header
type BasicAuth struct {
	Username string
	Password string
}

// Download and parse OAuth2 JSON access request
func downloadData(method string, u *url.URL, auth *BasicAuth, transport http.RoundTripper, output ResponseData) error {
	var postdata url.Values
	var body io.Reader
	var contenttype string

	if method == "POST" {
		// Convert query parameters to post data
		postdata = u.Query()
		u.RawQuery = ""
		body = strings.NewReader(postdata.Encode())
		contenttype = "application/x-www-form-urlencoded"
	}

	// Create a new request
	preq, err := http.NewRequest(method, u.String(), body)
	if err != nil {
		return err
	}

	if auth != nil {
		preq.SetBasicAuth(auth.Username, auth.Password)
	}

	// Set content type for post request
	if contenttype != "" {
		preq.Header.Set("Content-Type", contenttype)
	}

	// Explicitly set accept header to JSON
	preq.Header.Set("Accept", "application/json")

	// do request
	client := &http.Client{}
	if transport != nil {
		client.Transport = transport
	}
	presp, err := client.Do(preq)
	if err != nil {
		return err
	}

	// must close body
	defer presp.Body.Close()

	if presp.StatusCode != 200 {
		return errors.New(fmt.Sprintf("Invalid status code (%d): %s", presp.StatusCode, presp.Status))
	}

	// decode JSON and detect OAuth error
	jdec := json.NewDecoder(presp.Body)
	if err = jdec.Decode(&output); err == nil {
		if em, eok := output["error"]; eok {
			return NewError(fmt.Sprintf("%v", em), fmt.Sprintf("%v", output["error_description"]),
				fmt.Sprintf("%v", output["error_uri"]), fmt.Sprintf("%v", output["state"]))
		}
	}
	return err
}
