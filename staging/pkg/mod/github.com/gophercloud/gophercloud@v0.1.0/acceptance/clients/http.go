package clients

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"net/http"
	"sort"
	"strings"
)

// List of headers that need to be redacted
var REDACT_HEADERS = []string{"x-auth-token", "x-auth-key", "x-service-token",
	"x-storage-token", "x-account-meta-temp-url-key", "x-account-meta-temp-url-key-2",
	"x-container-meta-temp-url-key", "x-container-meta-temp-url-key-2", "set-cookie",
	"x-subject-token"}

// LogRoundTripper satisfies the http.RoundTripper interface and is used to
// customize the default http client RoundTripper to allow logging.
type LogRoundTripper struct {
	Rt http.RoundTripper
}

// RoundTrip performs a round-trip HTTP request and logs relevant information
// about it.
func (lrt *LogRoundTripper) RoundTrip(request *http.Request) (*http.Response, error) {
	defer func() {
		if request.Body != nil {
			request.Body.Close()
		}
	}()

	var err error

	log.Printf("[DEBUG] OpenStack Request URL: %s %s", request.Method, request.URL)
	log.Printf("[DEBUG] OpenStack request Headers:\n%s", formatHeaders(request.Header))

	if request.Body != nil {
		request.Body, err = lrt.logRequest(request.Body, request.Header.Get("Content-Type"))
		if err != nil {
			return nil, err
		}
	}

	response, err := lrt.Rt.RoundTrip(request)
	if response == nil {
		return nil, err
	}

	log.Printf("[DEBUG] OpenStack Response Code: %d", response.StatusCode)
	log.Printf("[DEBUG] OpenStack Response Headers:\n%s", formatHeaders(response.Header))

	response.Body, err = lrt.logResponse(response.Body, response.Header.Get("Content-Type"))

	return response, err
}

// logRequest will log the HTTP Request details.
// If the body is JSON, it will attempt to be pretty-formatted.
func (lrt *LogRoundTripper) logRequest(original io.ReadCloser, contentType string) (io.ReadCloser, error) {
	defer original.Close()

	var bs bytes.Buffer
	_, err := io.Copy(&bs, original)
	if err != nil {
		return nil, err
	}

	// Handle request contentType
	if strings.HasPrefix(contentType, "application/json") {
		debugInfo := lrt.formatJSON(bs.Bytes())
		log.Printf("[DEBUG] OpenStack Request Body: %s", debugInfo)
	}

	return ioutil.NopCloser(strings.NewReader(bs.String())), nil
}

// logResponse will log the HTTP Response details.
// If the body is JSON, it will attempt to be pretty-formatted.
func (lrt *LogRoundTripper) logResponse(original io.ReadCloser, contentType string) (io.ReadCloser, error) {
	if strings.HasPrefix(contentType, "application/json") {
		var bs bytes.Buffer
		defer original.Close()
		_, err := io.Copy(&bs, original)
		if err != nil {
			return nil, err
		}
		debugInfo := lrt.formatJSON(bs.Bytes())
		if debugInfo != "" {
			log.Printf("[DEBUG] OpenStack Response Body: %s", debugInfo)
		}
		return ioutil.NopCloser(strings.NewReader(bs.String())), nil
	}

	log.Printf("[DEBUG] Not logging because OpenStack response body isn't JSON")
	return original, nil
}

// formatJSON will try to pretty-format a JSON body.
// It will also mask known fields which contain sensitive information.
func (lrt *LogRoundTripper) formatJSON(raw []byte) string {
	var rawData interface{}

	err := json.Unmarshal(raw, &rawData)
	if err != nil {
		log.Printf("[DEBUG] Unable to parse OpenStack JSON: %s", err)
		return string(raw)
	}

	data, ok := rawData.(map[string]interface{})
	if !ok {
		pretty, err := json.MarshalIndent(rawData, "", "  ")
		if err != nil {
			log.Printf("[DEBUG] Unable to re-marshal OpenStack JSON: %s", err)
			return string(raw)
		}

		return string(pretty)
	}

	// Mask known password fields
	if v, ok := data["auth"].(map[string]interface{}); ok {
		if v, ok := v["identity"].(map[string]interface{}); ok {
			if v, ok := v["password"].(map[string]interface{}); ok {
				if v, ok := v["user"].(map[string]interface{}); ok {
					v["password"] = "***"
				}
			}
			if v, ok := v["application_credential"].(map[string]interface{}); ok {
				v["secret"] = "***"
			}
			if v, ok := v["token"].(map[string]interface{}); ok {
				v["id"] = "***"
			}
		}
	}

	// Ignore the catalog
	if v, ok := data["token"].(map[string]interface{}); ok {
		if _, ok := v["catalog"]; ok {
			return ""
		}
	}

	pretty, err := json.MarshalIndent(data, "", "  ")
	if err != nil {
		log.Printf("[DEBUG] Unable to re-marshal OpenStack JSON: %s", err)
		return string(raw)
	}

	return string(pretty)
}

// redactHeaders processes a headers object, returning a redacted list
func redactHeaders(headers http.Header) (processedHeaders []string) {
	for name, header := range headers {
		var sensitive bool

		for _, redact_header := range REDACT_HEADERS {
			if strings.ToLower(name) == strings.ToLower(redact_header) {
				sensitive = true
			}
		}

		for _, v := range header {
			if sensitive {
				processedHeaders = append(processedHeaders, fmt.Sprintf("%v: %v", name, "***"))
			} else {
				processedHeaders = append(processedHeaders, fmt.Sprintf("%v: %v", name, v))
			}
		}
	}
	return
}

// formatHeaders processes a headers object plus a deliminator, returning a string
func formatHeaders(headers http.Header) string {
	redactedHeaders := redactHeaders(headers)
	sort.Strings(redactedHeaders)

	return strings.Join(redactedHeaders, "\n")
}
