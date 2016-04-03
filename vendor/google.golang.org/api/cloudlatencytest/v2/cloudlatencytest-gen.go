// Package cloudlatencytest provides access to the Google Cloud Network Performance Monitoring API.
//
// Usage example:
//
//   import "google.golang.org/api/cloudlatencytest/v2"
//   ...
//   cloudlatencytestService, err := cloudlatencytest.New(oauthHttpClient)
package cloudlatencytest // import "google.golang.org/api/cloudlatencytest/v2"

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	context "golang.org/x/net/context"
	ctxhttp "golang.org/x/net/context/ctxhttp"
	gensupport "google.golang.org/api/gensupport"
	googleapi "google.golang.org/api/googleapi"
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
var _ = gensupport.MarshalJSON
var _ = googleapi.Version
var _ = errors.New
var _ = strings.Replace
var _ = context.Canceled
var _ = ctxhttp.Do

const apiId = "cloudlatencytest:v2"
const apiName = "cloudlatencytest"
const apiVersion = "v2"
const basePath = "https://cloudlatencytest-pa.googleapis.com/v2/statscollection/"

// OAuth2 scopes used by this API.
const (
	// View monitoring data for all of your Google Cloud and API projects
	MonitoringReadonlyScope = "https://www.googleapis.com/auth/monitoring.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Statscollection = NewStatscollectionService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Statscollection *StatscollectionService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewStatscollectionService(s *Service) *StatscollectionService {
	rs := &StatscollectionService{s: s}
	return rs
}

type StatscollectionService struct {
	s *Service
}

type AggregatedStats struct {
	Stats []*Stats `json:"stats,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Stats") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AggregatedStats) MarshalJSON() ([]byte, error) {
	type noMethod AggregatedStats
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type AggregatedStatsReply struct {
	TestValue string `json:"testValue,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "TestValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AggregatedStatsReply) MarshalJSON() ([]byte, error) {
	type noMethod AggregatedStatsReply
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type DoubleValue struct {
	Label string `json:"label,omitempty"`

	Value float64 `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Label") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *DoubleValue) MarshalJSON() ([]byte, error) {
	type noMethod DoubleValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type IntValue struct {
	Label string `json:"label,omitempty"`

	Value int64 `json:"value,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Label") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *IntValue) MarshalJSON() ([]byte, error) {
	type noMethod IntValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type Stats struct {
	DoubleValues []*DoubleValue `json:"doubleValues,omitempty"`

	IntValues []*IntValue `json:"intValues,omitempty"`

	StringValues []*StringValue `json:"stringValues,omitempty"`

	Time float64 `json:"time,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DoubleValues") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Stats) MarshalJSON() ([]byte, error) {
	type noMethod Stats
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type StatsReply struct {
	TestValue string `json:"testValue,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "TestValue") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *StatsReply) MarshalJSON() ([]byte, error) {
	type noMethod StatsReply
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type StringValue struct {
	Label string `json:"label,omitempty"`

	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Label") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *StringValue) MarshalJSON() ([]byte, error) {
	type noMethod StringValue
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "cloudlatencytest.statscollection.updateaggregatedstats":

type StatscollectionUpdateaggregatedstatsCall struct {
	s               *Service
	aggregatedstats *AggregatedStats
	urlParams_      gensupport.URLParams
	ctx_            context.Context
}

// Updateaggregatedstats: RPC to update the new TCP stats.
func (r *StatscollectionService) Updateaggregatedstats(aggregatedstats *AggregatedStats) *StatscollectionUpdateaggregatedstatsCall {
	c := &StatscollectionUpdateaggregatedstatsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.aggregatedstats = aggregatedstats
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StatscollectionUpdateaggregatedstatsCall) QuotaUser(quotaUser string) *StatscollectionUpdateaggregatedstatsCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StatscollectionUpdateaggregatedstatsCall) UserIP(userIP string) *StatscollectionUpdateaggregatedstatsCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StatscollectionUpdateaggregatedstatsCall) Fields(s ...googleapi.Field) *StatscollectionUpdateaggregatedstatsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StatscollectionUpdateaggregatedstatsCall) Context(ctx context.Context) *StatscollectionUpdateaggregatedstatsCall {
	c.ctx_ = ctx
	return c
}

func (c *StatscollectionUpdateaggregatedstatsCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.aggregatedstats)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "updateaggregatedstats")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "cloudlatencytest.statscollection.updateaggregatedstats" call.
// Exactly one of *AggregatedStatsReply or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *AggregatedStatsReply.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *StatscollectionUpdateaggregatedstatsCall) Do() (*AggregatedStatsReply, error) {
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &AggregatedStatsReply{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "RPC to update the new TCP stats.",
	//   "httpMethod": "POST",
	//   "id": "cloudlatencytest.statscollection.updateaggregatedstats",
	//   "path": "updateaggregatedstats",
	//   "request": {
	//     "$ref": "AggregatedStats"
	//   },
	//   "response": {
	//     "$ref": "AggregatedStatsReply"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/monitoring.readonly"
	//   ]
	// }

}

// method id "cloudlatencytest.statscollection.updatestats":

type StatscollectionUpdatestatsCall struct {
	s          *Service
	stats      *Stats
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Updatestats: RPC to update the new TCP stats.
func (r *StatscollectionService) Updatestats(stats *Stats) *StatscollectionUpdatestatsCall {
	c := &StatscollectionUpdatestatsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.stats = stats
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *StatscollectionUpdatestatsCall) QuotaUser(quotaUser string) *StatscollectionUpdatestatsCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *StatscollectionUpdatestatsCall) UserIP(userIP string) *StatscollectionUpdatestatsCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *StatscollectionUpdatestatsCall) Fields(s ...googleapi.Field) *StatscollectionUpdatestatsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *StatscollectionUpdatestatsCall) Context(ctx context.Context) *StatscollectionUpdatestatsCall {
	c.ctx_ = ctx
	return c
}

func (c *StatscollectionUpdatestatsCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.stats)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "updatestats")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "cloudlatencytest.statscollection.updatestats" call.
// Exactly one of *StatsReply or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *StatsReply.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *StatscollectionUpdatestatsCall) Do() (*StatsReply, error) {
	res, err := c.doRequest("json")
	if res != nil && res.StatusCode == http.StatusNotModified {
		if res.Body != nil {
			res.Body.Close()
		}
		return nil, &googleapi.Error{
			Code:   res.StatusCode,
			Header: res.Header,
		}
	}
	if err != nil {
		return nil, err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := &StatsReply{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	if err := json.NewDecoder(res.Body).Decode(&ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "RPC to update the new TCP stats.",
	//   "httpMethod": "POST",
	//   "id": "cloudlatencytest.statscollection.updatestats",
	//   "path": "updatestats",
	//   "request": {
	//     "$ref": "Stats"
	//   },
	//   "response": {
	//     "$ref": "StatsReply"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/monitoring.readonly"
	//   ]
	// }

}
