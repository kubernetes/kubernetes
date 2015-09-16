package pagination

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/url"
	"strings"

	"github.com/rackspace/gophercloud"
)

// PageResult stores the HTTP response that returned the current page of results.
type PageResult struct {
	gophercloud.Result
	url.URL
}

// PageResultFrom parses an HTTP response as JSON and returns a PageResult containing the
// results, interpreting it as JSON if the content type indicates.
func PageResultFrom(resp *http.Response) (PageResult, error) {
	var parsedBody interface{}

	defer resp.Body.Close()
	rawBody, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return PageResult{}, err
	}

	if strings.HasPrefix(resp.Header.Get("Content-Type"), "application/json") {
		err = json.Unmarshal(rawBody, &parsedBody)
		if err != nil {
			return PageResult{}, err
		}
	} else {
		parsedBody = rawBody
	}

	return PageResultFromParsed(resp, parsedBody), err
}

// PageResultFromParsed constructs a PageResult from an HTTP response that has already had its
// body parsed as JSON (and closed).
func PageResultFromParsed(resp *http.Response, body interface{}) PageResult {
	return PageResult{
		Result: gophercloud.Result{
			Body:   body,
			Header: resp.Header,
		},
		URL: *resp.Request.URL,
	}
}

// Request performs an HTTP request and extracts the http.Response from the result.
func Request(client *gophercloud.ServiceClient, headers map[string]string, url string) (*http.Response, error) {
	return client.Request("GET", url, gophercloud.RequestOpts{
		MoreHeaders: headers,
		OkCodes:     []int{200, 204},
	})
}
