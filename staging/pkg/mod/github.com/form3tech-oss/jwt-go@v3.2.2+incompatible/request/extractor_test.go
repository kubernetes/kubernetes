package request

import (
	"fmt"
	"net/http"
	"net/url"
	"testing"
)

var extractorTestTokenA = "A"
var extractorTestTokenB = "B"

var extractorTestData = []struct {
	name      string
	extractor Extractor
	headers   map[string]string
	query     url.Values
	token     string
	err       error
}{
	{
		name:      "simple header",
		extractor: HeaderExtractor{"Foo"},
		headers:   map[string]string{"Foo": extractorTestTokenA},
		query:     nil,
		token:     extractorTestTokenA,
		err:       nil,
	},
	{
		name:      "simple argument",
		extractor: ArgumentExtractor{"token"},
		headers:   map[string]string{},
		query:     url.Values{"token": {extractorTestTokenA}},
		token:     extractorTestTokenA,
		err:       nil,
	},
	{
		name: "multiple extractors",
		extractor: MultiExtractor{
			HeaderExtractor{"Foo"},
			ArgumentExtractor{"token"},
		},
		headers: map[string]string{"Foo": extractorTestTokenA},
		query:   url.Values{"token": {extractorTestTokenB}},
		token:   extractorTestTokenA,
		err:     nil,
	},
	{
		name:      "simple miss",
		extractor: HeaderExtractor{"This-Header-Is-Not-Set"},
		headers:   map[string]string{"Foo": extractorTestTokenA},
		query:     nil,
		token:     "",
		err:       ErrNoTokenInRequest,
	},
	{
		name:      "filter",
		extractor: AuthorizationHeaderExtractor,
		headers:   map[string]string{"Authorization": "Bearer " + extractorTestTokenA},
		query:     nil,
		token:     extractorTestTokenA,
		err:       nil,
	},
}

func TestExtractor(t *testing.T) {
	// Bearer token request
	for _, data := range extractorTestData {
		// Make request from test struct
		r := makeExampleRequest("GET", "/", data.headers, data.query)

		// Test extractor
		token, err := data.extractor.ExtractToken(r)
		if token != data.token {
			t.Errorf("[%v] Expected token '%v'.  Got '%v'", data.name, data.token, token)
			continue
		}
		if err != data.err {
			t.Errorf("[%v] Expected error '%v'.  Got '%v'", data.name, data.err, err)
			continue
		}
	}
}

func makeExampleRequest(method, path string, headers map[string]string, urlArgs url.Values) *http.Request {
	r, _ := http.NewRequest(method, fmt.Sprintf("%v?%v", path, urlArgs.Encode()), nil)
	for k, v := range headers {
		r.Header.Set(k, v)
	}
	return r
}
