// Package webfonts provides access to the Google Fonts Developer API.
//
// See https://developers.google.com/fonts/docs/developer_api
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/webfonts/v1"
//   ...
//   webfontsService, err := webfonts.New(oauthHttpClient)
package webfonts

import (
	"bytes"
	"code.google.com/p/google-api-go-client/googleapi"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
)

// Always reference these packages, just in case the auto-generated code
// below doesn't.
var _ = bytes.NewBuffer
var _ = strconv.Itoa
var _ = fmt.Sprintf
var _ = json.NewDecoder
var _ = io.Copy
var _ = url.Parse
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace

const apiId = "webfonts:v1"
const apiName = "webfonts"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/webfonts/v1/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Webfonts = NewWebfontsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Webfonts *WebfontsService
}

func NewWebfontsService(s *Service) *WebfontsService {
	rs := &WebfontsService{s: s}
	return rs
}

type WebfontsService struct {
	s *Service
}

type Webfont struct {
	// Category: The category of the font.
	Category string `json:"category,omitempty"`

	// Family: The name of the font.
	Family string `json:"family,omitempty"`

	// Files: The font files (with all supported scripts) for each one of
	// the available variants, as a key : value map.
	Files map[string]string `json:"files,omitempty"`

	// Kind: This kind represents a webfont object in the webfonts service.
	Kind string `json:"kind,omitempty"`

	// LastModified: The date (format "yyyy-MM-dd") the font was modified
	// for the last time.
	LastModified string `json:"lastModified,omitempty"`

	// Subsets: The scripts supported by the font.
	Subsets []string `json:"subsets,omitempty"`

	// Variants: The available variants for the font.
	Variants []string `json:"variants,omitempty"`

	// Version: The font version.
	Version string `json:"version,omitempty"`
}

type WebfontList struct {
	// Items: The list of fonts currently served by the Google Fonts API.
	Items []*Webfont `json:"items,omitempty"`

	// Kind: This kind represents a list of webfont objects in the webfonts
	// service.
	Kind string `json:"kind,omitempty"`
}

// method id "webfonts.webfonts.list":

type WebfontsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieves the list of fonts currently served by the Google
// Fonts Developer API
func (r *WebfontsService) List() *WebfontsListCall {
	c := &WebfontsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// Sort sets the optional parameter "sort": Enables sorting of the list
func (c *WebfontsListCall) Sort(sort string) *WebfontsListCall {
	c.opt_["sort"] = sort
	return c
}

func (c *WebfontsListCall) Do() (*WebfontList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["sort"]; ok {
		params.Set("sort", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "webfonts")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(WebfontList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of fonts currently served by the Google Fonts Developer API",
	//   "httpMethod": "GET",
	//   "id": "webfonts.webfonts.list",
	//   "parameters": {
	//     "sort": {
	//       "description": "Enables sorting of the list",
	//       "enum": [
	//         "alpha",
	//         "date",
	//         "popularity",
	//         "style",
	//         "trending"
	//       ],
	//       "enumDescriptions": [
	//         "Sort alphabetically",
	//         "Sort by date added",
	//         "Sort by popularity",
	//         "Sort by number of styles",
	//         "Sort by trending"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "webfonts",
	//   "response": {
	//     "$ref": "WebfontList"
	//   }
	// }

}
