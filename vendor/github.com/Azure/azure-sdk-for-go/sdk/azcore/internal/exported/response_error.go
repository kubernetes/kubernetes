//go:build go1.18
// +build go1.18

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

package exported

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"regexp"
)

// NewResponseError creates a new *ResponseError from the provided HTTP response.
// Exported as runtime.NewResponseError().
func NewResponseError(resp *http.Response) error {
	respErr := &ResponseError{
		StatusCode:  resp.StatusCode,
		RawResponse: resp,
	}

	// prefer the error code in the response header
	if ec := resp.Header.Get("x-ms-error-code"); ec != "" {
		respErr.ErrorCode = ec
		return respErr
	}

	// if we didn't get x-ms-error-code, check in the response body
	body, err := Payload(resp)
	if err != nil {
		return err
	}

	if len(body) > 0 {
		if code := extractErrorCodeJSON(body); code != "" {
			respErr.ErrorCode = code
		} else if code := extractErrorCodeXML(body); code != "" {
			respErr.ErrorCode = code
		}
	}

	return respErr
}

func extractErrorCodeJSON(body []byte) string {
	var rawObj map[string]interface{}
	if err := json.Unmarshal(body, &rawObj); err != nil {
		// not a JSON object
		return ""
	}

	// check if this is a wrapped error, i.e. { "error": { ... } }
	// if so then unwrap it
	if wrapped, ok := rawObj["error"]; ok {
		unwrapped, ok := wrapped.(map[string]interface{})
		if !ok {
			return ""
		}
		rawObj = unwrapped
	} else if wrapped, ok := rawObj["odata.error"]; ok {
		// check if this a wrapped odata error, i.e. { "odata.error": { ... } }
		unwrapped, ok := wrapped.(map[string]any)
		if !ok {
			return ""
		}
		rawObj = unwrapped
	}

	// now check for the error code
	code, ok := rawObj["code"]
	if !ok {
		return ""
	}
	codeStr, ok := code.(string)
	if !ok {
		return ""
	}
	return codeStr
}

func extractErrorCodeXML(body []byte) string {
	// regular expression is much easier than dealing with the XML parser
	rx := regexp.MustCompile(`<(?:\w+:)?[c|C]ode>\s*(\w+)\s*<\/(?:\w+:)?[c|C]ode>`)
	res := rx.FindStringSubmatch(string(body))
	if len(res) != 2 {
		return ""
	}
	// first submatch is the entire thing, second one is the captured error code
	return res[1]
}

// ResponseError is returned when a request is made to a service and
// the service returns a non-success HTTP status code.
// Use errors.As() to access this type in the error chain.
// Exported as azcore.ResponseError.
type ResponseError struct {
	// ErrorCode is the error code returned by the resource provider if available.
	ErrorCode string

	// StatusCode is the HTTP status code as defined in https://pkg.go.dev/net/http#pkg-constants.
	StatusCode int

	// RawResponse is the underlying HTTP response.
	RawResponse *http.Response
}

// Error implements the error interface for type ResponseError.
// Note that the message contents are not contractual and can change over time.
func (e *ResponseError) Error() string {
	// write the request method and URL with response status code
	msg := &bytes.Buffer{}
	fmt.Fprintf(msg, "%s %s://%s%s\n", e.RawResponse.Request.Method, e.RawResponse.Request.URL.Scheme, e.RawResponse.Request.URL.Host, e.RawResponse.Request.URL.Path)
	fmt.Fprintln(msg, "--------------------------------------------------------------------------------")
	fmt.Fprintf(msg, "RESPONSE %d: %s\n", e.RawResponse.StatusCode, e.RawResponse.Status)
	if e.ErrorCode != "" {
		fmt.Fprintf(msg, "ERROR CODE: %s\n", e.ErrorCode)
	} else {
		fmt.Fprintln(msg, "ERROR CODE UNAVAILABLE")
	}
	fmt.Fprintln(msg, "--------------------------------------------------------------------------------")
	body, err := Payload(e.RawResponse)
	if err != nil {
		// this really shouldn't fail at this point as the response
		// body is already cached (it was read in NewResponseError)
		fmt.Fprintf(msg, "Error reading response body: %v", err)
	} else if len(body) > 0 {
		if err := json.Indent(msg, body, "", "  "); err != nil {
			// failed to pretty-print so just dump it verbatim
			fmt.Fprint(msg, string(body))
		}
		// the standard library doesn't have a pretty-printer for XML
		fmt.Fprintln(msg)
	} else {
		fmt.Fprintln(msg, "Response contained no body")
	}
	fmt.Fprintln(msg, "--------------------------------------------------------------------------------")

	return msg.String()
}
