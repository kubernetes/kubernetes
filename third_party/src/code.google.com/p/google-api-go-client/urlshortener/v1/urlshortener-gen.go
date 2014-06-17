// Package urlshortener provides access to the URL Shortener API.
//
// See http://code.google.com/apis/urlshortener/v1/getting_started.html
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/urlshortener/v1"
//   ...
//   urlshortenerService, err := urlshortener.New(oauthHttpClient)
package urlshortener

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

const apiId = "urlshortener:v1"
const apiName = "urlshortener"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/urlshortener/v1/"

// OAuth2 scopes used by this API.
const (
	// Manage your goo.gl short URLs
	UrlshortenerScope = "https://www.googleapis.com/auth/urlshortener"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Url = NewUrlService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Url *UrlService
}

func NewUrlService(s *Service) *UrlService {
	rs := &UrlService{s: s}
	return rs
}

type UrlService struct {
	s *Service
}

type AnalyticsSnapshot struct {
	// Browsers: Top browsers, e.g. "Chrome"; sorted by (descending) click
	// counts. Only present if this data is available.
	Browsers []*StringCount `json:"browsers,omitempty"`

	// Countries: Top countries (expressed as country codes), e.g. "US" or
	// "DE"; sorted by (descending) click counts. Only present if this data
	// is available.
	Countries []*StringCount `json:"countries,omitempty"`

	// LongUrlClicks: Number of clicks on all goo.gl short URLs pointing to
	// this long URL.
	LongUrlClicks int64 `json:"longUrlClicks,omitempty,string"`

	// Platforms: Top platforms or OSes, e.g. "Windows"; sorted by
	// (descending) click counts. Only present if this data is available.
	Platforms []*StringCount `json:"platforms,omitempty"`

	// Referrers: Top referring hosts, e.g. "www.google.com"; sorted by
	// (descending) click counts. Only present if this data is available.
	Referrers []*StringCount `json:"referrers,omitempty"`

	// ShortUrlClicks: Number of clicks on this short URL.
	ShortUrlClicks int64 `json:"shortUrlClicks,omitempty,string"`
}

type AnalyticsSummary struct {
	// AllTime: Click analytics over all time.
	AllTime *AnalyticsSnapshot `json:"allTime,omitempty"`

	// Day: Click analytics over the last day.
	Day *AnalyticsSnapshot `json:"day,omitempty"`

	// Month: Click analytics over the last month.
	Month *AnalyticsSnapshot `json:"month,omitempty"`

	// TwoHours: Click analytics over the last two hours.
	TwoHours *AnalyticsSnapshot `json:"twoHours,omitempty"`

	// Week: Click analytics over the last week.
	Week *AnalyticsSnapshot `json:"week,omitempty"`
}

type StringCount struct {
	// Count: Number of clicks for this top entry, e.g. for this particular
	// country or browser.
	Count int64 `json:"count,omitempty,string"`

	// Id: Label assigned to this top entry, e.g. "US" or "Chrome".
	Id string `json:"id,omitempty"`
}

type Url struct {
	// Analytics: A summary of the click analytics for the short and long
	// URL. Might not be present if not requested or currently unavailable.
	Analytics *AnalyticsSummary `json:"analytics,omitempty"`

	// Created: Time the short URL was created; ISO 8601 representation
	// using the yyyy-MM-dd'T'HH:mm:ss.SSSZZ format, e.g.
	// "2010-10-14T19:01:24.944+00:00".
	Created string `json:"created,omitempty"`

	// Id: Short URL, e.g. "http://goo.gl/l6MS".
	Id string `json:"id,omitempty"`

	// Kind: The fixed string "urlshortener#url".
	Kind string `json:"kind,omitempty"`

	// LongUrl: Long URL, e.g. "http://www.google.com/". Might not be
	// present if the status is "REMOVED".
	LongUrl string `json:"longUrl,omitempty"`

	// Status: Status of the target URL. Possible values: "OK", "MALWARE",
	// "PHISHING", or "REMOVED". A URL might be marked "REMOVED" if it was
	// flagged as spam, for example.
	Status string `json:"status,omitempty"`
}

type UrlHistory struct {
	// Items: A list of URL resources.
	Items []*Url `json:"items,omitempty"`

	// ItemsPerPage: Number of items returned with each full "page" of
	// results. Note that the last page could have fewer items than the
	// "itemsPerPage" value.
	ItemsPerPage int64 `json:"itemsPerPage,omitempty"`

	// Kind: The fixed string "urlshortener#urlHistory".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: A token to provide to get the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// TotalItems: Total number of short URLs associated with this user (may
	// be approximate).
	TotalItems int64 `json:"totalItems,omitempty"`
}

// method id "urlshortener.url.get":

type UrlGetCall struct {
	s        *Service
	shortUrl string
	opt_     map[string]interface{}
}

// Get: Expands a short URL or gets creation time and analytics.
func (r *UrlService) Get(shortUrl string) *UrlGetCall {
	c := &UrlGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.shortUrl = shortUrl
	return c
}

// Projection sets the optional parameter "projection": Additional
// information to return.
func (c *UrlGetCall) Projection(projection string) *UrlGetCall {
	c.opt_["projection"] = projection
	return c
}

func (c *UrlGetCall) Do() (*Url, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("shortUrl", fmt.Sprintf("%v", c.shortUrl))
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "url")
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
	ret := new(Url)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Expands a short URL or gets creation time and analytics.",
	//   "httpMethod": "GET",
	//   "id": "urlshortener.url.get",
	//   "parameterOrder": [
	//     "shortUrl"
	//   ],
	//   "parameters": {
	//     "projection": {
	//       "description": "Additional information to return.",
	//       "enum": [
	//         "ANALYTICS_CLICKS",
	//         "ANALYTICS_TOP_STRINGS",
	//         "FULL"
	//       ],
	//       "enumDescriptions": [
	//         "Returns only click counts.",
	//         "Returns only top string counts.",
	//         "Returns the creation timestamp and all available analytics."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "shortUrl": {
	//       "description": "The short URL, including the protocol.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "url",
	//   "response": {
	//     "$ref": "Url"
	//   }
	// }

}

// method id "urlshortener.url.insert":

type UrlInsertCall struct {
	s    *Service
	url  *Url
	opt_ map[string]interface{}
}

// Insert: Creates a new short URL.
func (r *UrlService) Insert(url *Url) *UrlInsertCall {
	c := &UrlInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.url = url
	return c
}

func (c *UrlInsertCall) Do() (*Url, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.url)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "url")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Url)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new short URL.",
	//   "httpMethod": "POST",
	//   "id": "urlshortener.url.insert",
	//   "path": "url",
	//   "request": {
	//     "$ref": "Url"
	//   },
	//   "response": {
	//     "$ref": "Url"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/urlshortener"
	//   ]
	// }

}

// method id "urlshortener.url.list":

type UrlListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieves a list of URLs shortened by a user.
func (r *UrlService) List() *UrlListCall {
	c := &UrlListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// Projection sets the optional parameter "projection": Additional
// information to return.
func (c *UrlListCall) Projection(projection string) *UrlListCall {
	c.opt_["projection"] = projection
	return c
}

// StartToken sets the optional parameter "start-token": Token for
// requesting successive pages of results.
func (c *UrlListCall) StartToken(startToken string) *UrlListCall {
	c.opt_["start-token"] = startToken
	return c
}

func (c *UrlListCall) Do() (*UrlHistory, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["projection"]; ok {
		params.Set("projection", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-token"]; ok {
		params.Set("start-token", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "url/history")
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
	ret := new(UrlHistory)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of URLs shortened by a user.",
	//   "httpMethod": "GET",
	//   "id": "urlshortener.url.list",
	//   "parameters": {
	//     "projection": {
	//       "description": "Additional information to return.",
	//       "enum": [
	//         "ANALYTICS_CLICKS",
	//         "FULL"
	//       ],
	//       "enumDescriptions": [
	//         "Returns short URL click counts.",
	//         "Returns short URL click counts."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "start-token": {
	//       "description": "Token for requesting successive pages of results.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "url/history",
	//   "response": {
	//     "$ref": "UrlHistory"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/urlshortener"
	//   ]
	// }

}
