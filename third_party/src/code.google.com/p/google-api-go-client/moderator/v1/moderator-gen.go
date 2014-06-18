// Package moderator provides access to the Moderator API.
//
// See http://code.google.com/apis/moderator/v1/using_rest.html
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/moderator/v1"
//   ...
//   moderatorService, err := moderator.New(oauthHttpClient)
package moderator

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

var _ = bytes.NewBuffer
var _ = strconv.Itoa
var _ = fmt.Sprintf
var _ = json.NewDecoder
var _ = io.Copy
var _ = url.Parse
var _ = googleapi.Version
var _ = errors.New

const apiId = "moderator:v1"
const apiName = "moderator"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/moderator/v1/"

// OAuth2 scopes used by this API.
const (
	// Manage your activity in Google Moderator
	ModeratorScope = "https://www.googleapis.com/auth/moderator"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client}
	s.Featured = &FeaturedService{s: s}
	s.Global = &GlobalService{s: s}
	s.My = &MyService{s: s}
	s.Myrecent = &MyrecentService{s: s}
	s.Profiles = &ProfilesService{s: s}
	s.Responses = &ResponsesService{s: s}
	s.Series = &SeriesService{s: s}
	s.Submissions = &SubmissionsService{s: s}
	s.Tags = &TagsService{s: s}
	s.Topics = &TopicsService{s: s}
	s.Votes = &VotesService{s: s}
	return s, nil
}

type Service struct {
	client *http.Client

	Featured *FeaturedService

	Global *GlobalService

	My *MyService

	Myrecent *MyrecentService

	Profiles *ProfilesService

	Responses *ResponsesService

	Series *SeriesService

	Submissions *SubmissionsService

	Tags *TagsService

	Topics *TopicsService

	Votes *VotesService
}

type FeaturedService struct {
	s *Service
}

type GlobalService struct {
	s *Service
}

type MyService struct {
	s *Service
}

type MyrecentService struct {
	s *Service
}

type ProfilesService struct {
	s *Service
}

type ResponsesService struct {
	s *Service
}

type SeriesService struct {
	s *Service
}

type SubmissionsService struct {
	s *Service
}

type TagsService struct {
	s *Service
}

type TopicsService struct {
	s *Service
}

type VotesService struct {
	s *Service
}

type ModeratorTopicsResourcePartial struct {
	Id *ModeratorTopicsResourcePartialId `json:"id,omitempty"`
}

type ModeratorTopicsResourcePartialId struct {
	SeriesId int64 `json:"seriesId,omitempty,string"`

	TopicId int64 `json:"topicId,omitempty,string"`
}

type ModeratorVotesResourcePartial struct {
	Flag string `json:"flag,omitempty"`

	Vote string `json:"vote,omitempty"`
}

type Profile struct {
	Attribution *ProfileAttribution `json:"attribution,omitempty"`

	Id *ProfileId `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`
}

type ProfileAttribution struct {
	AvatarUrl string `json:"avatarUrl,omitempty"`

	DisplayName string `json:"displayName,omitempty"`

	Geo *ProfileAttributionGeo `json:"geo,omitempty"`

	Location string `json:"location,omitempty"`
}

type ProfileAttributionGeo struct {
	Latitude float64 `json:"latitude,omitempty"`

	Location string `json:"location,omitempty"`

	Longitude float64 `json:"longitude,omitempty"`
}

type ProfileId struct {
	User string `json:"user,omitempty"`
}

type Series struct {
	AnonymousSubmissionAllowed bool `json:"anonymousSubmissionAllowed,omitempty"`

	Counters *SeriesCounters `json:"counters,omitempty"`

	Description string `json:"description,omitempty"`

	Id *SeriesId `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`

	Name string `json:"name,omitempty"`

	NumTopics int64 `json:"numTopics,omitempty"`

	Rules *SeriesRules `json:"rules,omitempty"`

	UnauthSubmissionAllowed bool `json:"unauthSubmissionAllowed,omitempty"`

	UnauthVotingAllowed bool `json:"unauthVotingAllowed,omitempty"`

	VideoSubmissionAllowed bool `json:"videoSubmissionAllowed,omitempty"`
}

type SeriesCounters struct {
	AnonymousSubmissions int64 `json:"anonymousSubmissions,omitempty"`

	MinusVotes int64 `json:"minusVotes,omitempty"`

	NoneVotes int64 `json:"noneVotes,omitempty"`

	PlusVotes int64 `json:"plusVotes,omitempty"`

	Submissions int64 `json:"submissions,omitempty"`

	Users int64 `json:"users,omitempty"`

	VideoSubmissions int64 `json:"videoSubmissions,omitempty"`
}

type SeriesId struct {
	SeriesId int64 `json:"seriesId,omitempty,string"`
}

type SeriesRules struct {
	Submissions *SeriesRulesSubmissions `json:"submissions,omitempty"`

	Votes *SeriesRulesVotes `json:"votes,omitempty"`
}

type SeriesRulesSubmissions struct {
	Close uint64 `json:"close,omitempty,string"`

	Open uint64 `json:"open,omitempty,string"`
}

type SeriesRulesVotes struct {
	Close uint64 `json:"close,omitempty,string"`

	Open uint64 `json:"open,omitempty,string"`
}

type SeriesList struct {
	Items []*Series `json:"items,omitempty"`

	Kind string `json:"kind,omitempty"`
}

type Submission struct {
	AttachmentUrl string `json:"attachmentUrl,omitempty"`

	Attribution *SubmissionAttribution `json:"attribution,omitempty"`

	Author string `json:"author,omitempty"`

	Counters *SubmissionCounters `json:"counters,omitempty"`

	Created uint64 `json:"created,omitempty,string"`

	Geo *SubmissionGeo `json:"geo,omitempty"`

	Id *SubmissionId `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`

	ParentSubmissionId *SubmissionParentSubmissionId `json:"parentSubmissionId,omitempty"`

	Text string `json:"text,omitempty"`

	Topics []*ModeratorTopicsResourcePartial `json:"topics,omitempty"`

	Translations []*SubmissionTranslations `json:"translations,omitempty"`

	Vote *ModeratorVotesResourcePartial `json:"vote,omitempty"`
}

type SubmissionAttribution struct {
	AvatarUrl string `json:"avatarUrl,omitempty"`

	DisplayName string `json:"displayName,omitempty"`

	Location string `json:"location,omitempty"`
}

type SubmissionCounters struct {
	MinusVotes int64 `json:"minusVotes,omitempty"`

	NoneVotes int64 `json:"noneVotes,omitempty"`

	PlusVotes int64 `json:"plusVotes,omitempty"`
}

type SubmissionGeo struct {
	Latitude float64 `json:"latitude,omitempty"`

	Location string `json:"location,omitempty"`

	Longitude float64 `json:"longitude,omitempty"`
}

type SubmissionId struct {
	SeriesId int64 `json:"seriesId,omitempty,string"`

	SubmissionId int64 `json:"submissionId,omitempty,string"`
}

type SubmissionParentSubmissionId struct {
	SeriesId int64 `json:"seriesId,omitempty,string"`

	SubmissionId int64 `json:"submissionId,omitempty,string"`
}

type SubmissionTranslations struct {
	Lang string `json:"lang,omitempty"`

	Text string `json:"text,omitempty"`
}

type SubmissionList struct {
	Items []*Submission `json:"items,omitempty"`

	Kind string `json:"kind,omitempty"`
}

type Tag struct {
	Id *TagId `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`

	Text string `json:"text,omitempty"`
}

type TagId struct {
	SeriesId int64 `json:"seriesId,omitempty,string"`

	SubmissionId int64 `json:"submissionId,omitempty,string"`

	TagId string `json:"tagId,omitempty"`
}

type TagList struct {
	Items []*Tag `json:"items,omitempty"`

	Kind string `json:"kind,omitempty"`
}

type Topic struct {
	Counters *TopicCounters `json:"counters,omitempty"`

	Description string `json:"description,omitempty"`

	FeaturedSubmission *Submission `json:"featuredSubmission,omitempty"`

	Id *TopicId `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`

	Name string `json:"name,omitempty"`

	Presenter string `json:"presenter,omitempty"`

	Rules *TopicRules `json:"rules,omitempty"`
}

type TopicCounters struct {
	MinusVotes int64 `json:"minusVotes,omitempty"`

	NoneVotes int64 `json:"noneVotes,omitempty"`

	PlusVotes int64 `json:"plusVotes,omitempty"`

	Submissions int64 `json:"submissions,omitempty"`

	Users int64 `json:"users,omitempty"`

	VideoSubmissions int64 `json:"videoSubmissions,omitempty"`
}

type TopicId struct {
	SeriesId int64 `json:"seriesId,omitempty,string"`

	TopicId int64 `json:"topicId,omitempty,string"`
}

type TopicRules struct {
	Submissions *TopicRulesSubmissions `json:"submissions,omitempty"`

	Votes *TopicRulesVotes `json:"votes,omitempty"`
}

type TopicRulesSubmissions struct {
	Close uint64 `json:"close,omitempty,string"`

	Open uint64 `json:"open,omitempty,string"`
}

type TopicRulesVotes struct {
	Close uint64 `json:"close,omitempty,string"`

	Open uint64 `json:"open,omitempty,string"`
}

type TopicList struct {
	Items []*Topic `json:"items,omitempty"`

	Kind string `json:"kind,omitempty"`
}

type Vote struct {
	Flag string `json:"flag,omitempty"`

	Id *VoteId `json:"id,omitempty"`

	Kind string `json:"kind,omitempty"`

	Vote string `json:"vote,omitempty"`
}

type VoteId struct {
	SeriesId int64 `json:"seriesId,omitempty,string"`

	SubmissionId int64 `json:"submissionId,omitempty,string"`
}

type VoteList struct {
	Items []*Vote `json:"items,omitempty"`

	Kind string `json:"kind,omitempty"`
}

// method id "moderator.profiles.get":

type ProfilesGetCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// Get: Returns the profile information for the authenticated user.
func (r *ProfilesService) Get() *ProfilesGetCall {
	c := &ProfilesGetCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *ProfilesGetCall) Do() (*Profile, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "profiles/@me")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Profile)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the profile information for the authenticated user.",
	//   "httpMethod": "GET",
	//   "id": "moderator.profiles.get",
	//   "path": "profiles/@me",
	//   "response": {
	//     "$ref": "Profile"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.profiles.patch":

type ProfilesPatchCall struct {
	s       *Service
	profile *Profile
	opt_    map[string]interface{}
}

// Patch: Updates the profile information for the authenticated user.
// This method supports patch semantics.
func (r *ProfilesService) Patch(profile *Profile) *ProfilesPatchCall {
	c := &ProfilesPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.profile = profile
	return c
}

func (c *ProfilesPatchCall) Do() (*Profile, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.profile)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "profiles/@me")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Profile)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the profile information for the authenticated user. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "moderator.profiles.patch",
	//   "path": "profiles/@me",
	//   "request": {
	//     "$ref": "Profile"
	//   },
	//   "response": {
	//     "$ref": "Profile"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.profiles.update":

type ProfilesUpdateCall struct {
	s       *Service
	profile *Profile
	opt_    map[string]interface{}
}

// Update: Updates the profile information for the authenticated user.
func (r *ProfilesService) Update(profile *Profile) *ProfilesUpdateCall {
	c := &ProfilesUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.profile = profile
	return c
}

func (c *ProfilesUpdateCall) Do() (*Profile, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.profile)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "profiles/@me")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Profile)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the profile information for the authenticated user.",
	//   "httpMethod": "PUT",
	//   "id": "moderator.profiles.update",
	//   "path": "profiles/@me",
	//   "request": {
	//     "$ref": "Profile"
	//   },
	//   "response": {
	//     "$ref": "Profile"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.responses.insert":

type ResponsesInsertCall struct {
	s                  *Service
	seriesId           int64
	topicId            int64
	parentSubmissionId int64
	submission         *Submission
	opt_               map[string]interface{}
}

// Insert: Inserts a response for the specified submission in the
// specified topic within the specified series.
func (r *ResponsesService) Insert(seriesId int64, topicId int64, parentSubmissionId int64, submission *Submission) *ResponsesInsertCall {
	c := &ResponsesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.topicId = topicId
	c.parentSubmissionId = parentSubmissionId
	c.submission = submission
	return c
}

// Anonymous sets the optional parameter "anonymous": Set to true to
// mark the new submission as anonymous.
func (c *ResponsesInsertCall) Anonymous(anonymous bool) *ResponsesInsertCall {
	c.opt_["anonymous"] = anonymous
	return c
}

// UnauthToken sets the optional parameter "unauthToken": User
// identifier for unauthenticated usage mode
func (c *ResponsesInsertCall) UnauthToken(unauthToken string) *ResponsesInsertCall {
	c.opt_["unauthToken"] = unauthToken
	return c
}

func (c *ResponsesInsertCall) Do() (*Submission, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.submission)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["anonymous"]; ok {
		params.Set("anonymous", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["unauthToken"]; ok {
		params.Set("unauthToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/topics/{topicId}/submissions/{parentSubmissionId}/responses")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{topicId}", strconv.FormatInt(c.topicId, 10), 1)
	urls = strings.Replace(urls, "{parentSubmissionId}", strconv.FormatInt(c.parentSubmissionId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Submission)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a response for the specified submission in the specified topic within the specified series.",
	//   "httpMethod": "POST",
	//   "id": "moderator.responses.insert",
	//   "parameterOrder": [
	//     "seriesId",
	//     "topicId",
	//     "parentSubmissionId"
	//   ],
	//   "parameters": {
	//     "anonymous": {
	//       "description": "Set to true to mark the new submission as anonymous.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "parentSubmissionId": {
	//       "description": "The decimal ID of the parent Submission within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "topicId": {
	//       "description": "The decimal ID of the Topic within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "unauthToken": {
	//       "description": "User identifier for unauthenticated usage mode",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "series/{seriesId}/topics/{topicId}/submissions/{parentSubmissionId}/responses",
	//   "request": {
	//     "$ref": "Submission"
	//   },
	//   "response": {
	//     "$ref": "Submission"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.responses.list":

type ResponsesListCall struct {
	s            *Service
	seriesId     int64
	submissionId int64
	opt_         map[string]interface{}
}

// List: Lists or searches the responses for the specified submission
// within the specified series and returns the search results.
func (r *ResponsesService) List(seriesId int64, submissionId int64) *ResponsesListCall {
	c := &ResponsesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.submissionId = submissionId
	return c
}

// Author sets the optional parameter "author": Restricts the results to
// submissions by a specific author.
func (c *ResponsesListCall) Author(author string) *ResponsesListCall {
	c.opt_["author"] = author
	return c
}

// HasAttachedVideo sets the optional parameter "hasAttachedVideo":
// Specifies whether to restrict to submissions that have videos
// attached.
func (c *ResponsesListCall) HasAttachedVideo(hasAttachedVideo bool) *ResponsesListCall {
	c.opt_["hasAttachedVideo"] = hasAttachedVideo
	return c
}

// MaxResults sets the optional parameter "max-results": Maximum number
// of results to return.
func (c *ResponsesListCall) MaxResults(maxResults int64) *ResponsesListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// Q sets the optional parameter "q": Search query.
func (c *ResponsesListCall) Q(q string) *ResponsesListCall {
	c.opt_["q"] = q
	return c
}

// Sort sets the optional parameter "sort": Sort order.
func (c *ResponsesListCall) Sort(sort string) *ResponsesListCall {
	c.opt_["sort"] = sort
	return c
}

// StartIndex sets the optional parameter "start-index": Index of the
// first result to be retrieved.
func (c *ResponsesListCall) StartIndex(startIndex int64) *ResponsesListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *ResponsesListCall) Do() (*SubmissionList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["author"]; ok {
		params.Set("author", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["hasAttachedVideo"]; ok {
		params.Set("hasAttachedVideo", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["q"]; ok {
		params.Set("q", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sort"]; ok {
		params.Set("sort", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/submissions/{submissionId}/responses")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{submissionId}", strconv.FormatInt(c.submissionId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(SubmissionList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists or searches the responses for the specified submission within the specified series and returns the search results.",
	//   "httpMethod": "GET",
	//   "id": "moderator.responses.list",
	//   "parameterOrder": [
	//     "seriesId",
	//     "submissionId"
	//   ],
	//   "parameters": {
	//     "author": {
	//       "description": "Restricts the results to submissions by a specific author.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "hasAttachedVideo": {
	//       "description": "Specifies whether to restrict to submissions that have videos attached.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "max-results": {
	//       "description": "Maximum number of results to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "q": {
	//       "description": "Search query.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "sort": {
	//       "description": "Sort order.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "start-index": {
	//       "description": "Index of the first result to be retrieved.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "submissionId": {
	//       "description": "The decimal ID of the Submission within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series/{seriesId}/submissions/{submissionId}/responses",
	//   "response": {
	//     "$ref": "SubmissionList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.series.get":

type SeriesGetCall struct {
	s        *Service
	seriesId int64
	opt_     map[string]interface{}
}

// Get: Returns the specified series.
func (r *SeriesService) Get(seriesId int64) *SeriesGetCall {
	c := &SeriesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	return c
}

func (c *SeriesGetCall) Do() (*Series, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Series)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified series.",
	//   "httpMethod": "GET",
	//   "id": "moderator.series.get",
	//   "parameterOrder": [
	//     "seriesId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series/{seriesId}",
	//   "response": {
	//     "$ref": "Series"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.series.insert":

type SeriesInsertCall struct {
	s      *Service
	series *Series
	opt_   map[string]interface{}
}

// Insert: Inserts a new series.
func (r *SeriesService) Insert(series *Series) *SeriesInsertCall {
	c := &SeriesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.series = series
	return c
}

func (c *SeriesInsertCall) Do() (*Series, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.series)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Series)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a new series.",
	//   "httpMethod": "POST",
	//   "id": "moderator.series.insert",
	//   "path": "series",
	//   "request": {
	//     "$ref": "Series"
	//   },
	//   "response": {
	//     "$ref": "Series"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.series.list":

type SeriesListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Searches the series and returns the search results.
func (r *SeriesService) List() *SeriesListCall {
	c := &SeriesListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// MaxResults sets the optional parameter "max-results": Maximum number
// of results to return.
func (c *SeriesListCall) MaxResults(maxResults int64) *SeriesListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// Q sets the optional parameter "q": Search query.
func (c *SeriesListCall) Q(q string) *SeriesListCall {
	c.opt_["q"] = q
	return c
}

// StartIndex sets the optional parameter "start-index": Index of the
// first result to be retrieved.
func (c *SeriesListCall) StartIndex(startIndex int64) *SeriesListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *SeriesListCall) Do() (*SeriesList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["q"]; ok {
		params.Set("q", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(SeriesList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Searches the series and returns the search results.",
	//   "httpMethod": "GET",
	//   "id": "moderator.series.list",
	//   "parameters": {
	//     "max-results": {
	//       "description": "Maximum number of results to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "q": {
	//       "description": "Search query.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "start-index": {
	//       "description": "Index of the first result to be retrieved.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series",
	//   "response": {
	//     "$ref": "SeriesList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.series.patch":

type SeriesPatchCall struct {
	s        *Service
	seriesId int64
	series   *Series
	opt_     map[string]interface{}
}

// Patch: Updates the specified series. This method supports patch
// semantics.
func (r *SeriesService) Patch(seriesId int64, series *Series) *SeriesPatchCall {
	c := &SeriesPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.series = series
	return c
}

func (c *SeriesPatchCall) Do() (*Series, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.series)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Series)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the specified series. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "moderator.series.patch",
	//   "parameterOrder": [
	//     "seriesId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series/{seriesId}",
	//   "request": {
	//     "$ref": "Series"
	//   },
	//   "response": {
	//     "$ref": "Series"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.series.update":

type SeriesUpdateCall struct {
	s        *Service
	seriesId int64
	series   *Series
	opt_     map[string]interface{}
}

// Update: Updates the specified series.
func (r *SeriesService) Update(seriesId int64, series *Series) *SeriesUpdateCall {
	c := &SeriesUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.series = series
	return c
}

func (c *SeriesUpdateCall) Do() (*Series, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.series)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Series)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the specified series.",
	//   "httpMethod": "PUT",
	//   "id": "moderator.series.update",
	//   "parameterOrder": [
	//     "seriesId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series/{seriesId}",
	//   "request": {
	//     "$ref": "Series"
	//   },
	//   "response": {
	//     "$ref": "Series"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.submissions.get":

type SubmissionsGetCall struct {
	s            *Service
	seriesId     int64
	submissionId int64
	opt_         map[string]interface{}
}

// Get: Returns the specified submission within the specified series.
func (r *SubmissionsService) Get(seriesId int64, submissionId int64) *SubmissionsGetCall {
	c := &SubmissionsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.submissionId = submissionId
	return c
}

// IncludeVotes sets the optional parameter "includeVotes": Specifies
// whether to include the current user's vote
func (c *SubmissionsGetCall) IncludeVotes(includeVotes bool) *SubmissionsGetCall {
	c.opt_["includeVotes"] = includeVotes
	return c
}

// Lang sets the optional parameter "lang": The language code for the
// language the client prefers resuls in.
func (c *SubmissionsGetCall) Lang(lang string) *SubmissionsGetCall {
	c.opt_["lang"] = lang
	return c
}

func (c *SubmissionsGetCall) Do() (*Submission, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["includeVotes"]; ok {
		params.Set("includeVotes", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["lang"]; ok {
		params.Set("lang", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/submissions/{submissionId}")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{submissionId}", strconv.FormatInt(c.submissionId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Submission)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified submission within the specified series.",
	//   "httpMethod": "GET",
	//   "id": "moderator.submissions.get",
	//   "parameterOrder": [
	//     "seriesId",
	//     "submissionId"
	//   ],
	//   "parameters": {
	//     "includeVotes": {
	//       "description": "Specifies whether to include the current user's vote",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "lang": {
	//       "description": "The language code for the language the client prefers resuls in.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "submissionId": {
	//       "description": "The decimal ID of the Submission within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series/{seriesId}/submissions/{submissionId}",
	//   "response": {
	//     "$ref": "Submission"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.submissions.insert":

type SubmissionsInsertCall struct {
	s          *Service
	seriesId   int64
	topicId    int64
	submission *Submission
	opt_       map[string]interface{}
}

// Insert: Inserts a new submission in the specified topic within the
// specified series.
func (r *SubmissionsService) Insert(seriesId int64, topicId int64, submission *Submission) *SubmissionsInsertCall {
	c := &SubmissionsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.topicId = topicId
	c.submission = submission
	return c
}

// Anonymous sets the optional parameter "anonymous": Set to true to
// mark the new submission as anonymous.
func (c *SubmissionsInsertCall) Anonymous(anonymous bool) *SubmissionsInsertCall {
	c.opt_["anonymous"] = anonymous
	return c
}

// UnauthToken sets the optional parameter "unauthToken": User
// identifier for unauthenticated usage mode
func (c *SubmissionsInsertCall) UnauthToken(unauthToken string) *SubmissionsInsertCall {
	c.opt_["unauthToken"] = unauthToken
	return c
}

func (c *SubmissionsInsertCall) Do() (*Submission, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.submission)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["anonymous"]; ok {
		params.Set("anonymous", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["unauthToken"]; ok {
		params.Set("unauthToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/topics/{topicId}/submissions")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{topicId}", strconv.FormatInt(c.topicId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Submission)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a new submission in the specified topic within the specified series.",
	//   "httpMethod": "POST",
	//   "id": "moderator.submissions.insert",
	//   "parameterOrder": [
	//     "seriesId",
	//     "topicId"
	//   ],
	//   "parameters": {
	//     "anonymous": {
	//       "description": "Set to true to mark the new submission as anonymous.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "topicId": {
	//       "description": "The decimal ID of the Topic within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "unauthToken": {
	//       "description": "User identifier for unauthenticated usage mode",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "series/{seriesId}/topics/{topicId}/submissions",
	//   "request": {
	//     "$ref": "Submission"
	//   },
	//   "response": {
	//     "$ref": "Submission"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.tags.delete":

type TagsDeleteCall struct {
	s            *Service
	seriesId     int64
	submissionId int64
	tagId        string
	opt_         map[string]interface{}
}

// Delete: Deletes the specified tag from the specified submission
// within the specified series.
func (r *TagsService) Delete(seriesId int64, submissionId int64, tagId string) *TagsDeleteCall {
	c := &TagsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.submissionId = submissionId
	c.tagId = tagId
	return c
}

func (c *TagsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/submissions/{submissionId}/tags/{tagId}")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{submissionId}", strconv.FormatInt(c.submissionId, 10), 1)
	urls = strings.Replace(urls, "{tagId}", cleanPathString(c.tagId), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes the specified tag from the specified submission within the specified series.",
	//   "httpMethod": "DELETE",
	//   "id": "moderator.tags.delete",
	//   "parameterOrder": [
	//     "seriesId",
	//     "submissionId",
	//     "tagId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "submissionId": {
	//       "description": "The decimal ID of the Submission within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "tagId": {
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "series/{seriesId}/submissions/{submissionId}/tags/{tagId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.tags.insert":

type TagsInsertCall struct {
	s            *Service
	seriesId     int64
	submissionId int64
	tag          *Tag
	opt_         map[string]interface{}
}

// Insert: Inserts a new tag for the specified submission within the
// specified series.
func (r *TagsService) Insert(seriesId int64, submissionId int64, tag *Tag) *TagsInsertCall {
	c := &TagsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.submissionId = submissionId
	c.tag = tag
	return c
}

func (c *TagsInsertCall) Do() (*Tag, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.tag)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/submissions/{submissionId}/tags")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{submissionId}", strconv.FormatInt(c.submissionId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Tag)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a new tag for the specified submission within the specified series.",
	//   "httpMethod": "POST",
	//   "id": "moderator.tags.insert",
	//   "parameterOrder": [
	//     "seriesId",
	//     "submissionId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "submissionId": {
	//       "description": "The decimal ID of the Submission within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series/{seriesId}/submissions/{submissionId}/tags",
	//   "request": {
	//     "$ref": "Tag"
	//   },
	//   "response": {
	//     "$ref": "Tag"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.tags.list":

type TagsListCall struct {
	s            *Service
	seriesId     int64
	submissionId int64
	opt_         map[string]interface{}
}

// List: Lists all tags for the specified submission within the
// specified series.
func (r *TagsService) List(seriesId int64, submissionId int64) *TagsListCall {
	c := &TagsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.submissionId = submissionId
	return c
}

func (c *TagsListCall) Do() (*TagList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/submissions/{submissionId}/tags")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{submissionId}", strconv.FormatInt(c.submissionId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(TagList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists all tags for the specified submission within the specified series.",
	//   "httpMethod": "GET",
	//   "id": "moderator.tags.list",
	//   "parameterOrder": [
	//     "seriesId",
	//     "submissionId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "submissionId": {
	//       "description": "The decimal ID of the Submission within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series/{seriesId}/submissions/{submissionId}/tags",
	//   "response": {
	//     "$ref": "TagList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.topics.get":

type TopicsGetCall struct {
	s        *Service
	seriesId int64
	topicId  int64
	opt_     map[string]interface{}
}

// Get: Returns the specified topic from the specified series.
func (r *TopicsService) Get(seriesId int64, topicId int64) *TopicsGetCall {
	c := &TopicsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.topicId = topicId
	return c
}

func (c *TopicsGetCall) Do() (*Topic, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/topics/{topicId}")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{topicId}", strconv.FormatInt(c.topicId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Topic)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the specified topic from the specified series.",
	//   "httpMethod": "GET",
	//   "id": "moderator.topics.get",
	//   "parameterOrder": [
	//     "seriesId",
	//     "topicId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "topicId": {
	//       "description": "The decimal ID of the Topic within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series/{seriesId}/topics/{topicId}",
	//   "response": {
	//     "$ref": "Topic"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.topics.insert":

type TopicsInsertCall struct {
	s        *Service
	seriesId int64
	topic    *Topic
	opt_     map[string]interface{}
}

// Insert: Inserts a new topic into the specified series.
func (r *TopicsService) Insert(seriesId int64, topic *Topic) *TopicsInsertCall {
	c := &TopicsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.topic = topic
	return c
}

func (c *TopicsInsertCall) Do() (*Topic, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.topic)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/topics")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Topic)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a new topic into the specified series.",
	//   "httpMethod": "POST",
	//   "id": "moderator.topics.insert",
	//   "parameterOrder": [
	//     "seriesId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series/{seriesId}/topics",
	//   "request": {
	//     "$ref": "Topic"
	//   },
	//   "response": {
	//     "$ref": "Topic"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.topics.list":

type TopicsListCall struct {
	s        *Service
	seriesId int64
	opt_     map[string]interface{}
}

// List: Searches the topics within the specified series and returns the
// search results.
func (r *TopicsService) List(seriesId int64) *TopicsListCall {
	c := &TopicsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	return c
}

// MaxResults sets the optional parameter "max-results": Maximum number
// of results to return.
func (c *TopicsListCall) MaxResults(maxResults int64) *TopicsListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// Mode sets the optional parameter "mode":
func (c *TopicsListCall) Mode(mode string) *TopicsListCall {
	c.opt_["mode"] = mode
	return c
}

// Q sets the optional parameter "q": Search query.
func (c *TopicsListCall) Q(q string) *TopicsListCall {
	c.opt_["q"] = q
	return c
}

// StartIndex sets the optional parameter "start-index": Index of the
// first result to be retrieved.
func (c *TopicsListCall) StartIndex(startIndex int64) *TopicsListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *TopicsListCall) Do() (*TopicList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["mode"]; ok {
		params.Set("mode", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["q"]; ok {
		params.Set("q", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/topics")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(TopicList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Searches the topics within the specified series and returns the search results.",
	//   "httpMethod": "GET",
	//   "id": "moderator.topics.list",
	//   "parameterOrder": [
	//     "seriesId"
	//   ],
	//   "parameters": {
	//     "max-results": {
	//       "description": "Maximum number of results to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "mode": {
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "q": {
	//       "description": "Search query.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "start-index": {
	//       "description": "Index of the first result to be retrieved.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series/{seriesId}/topics",
	//   "response": {
	//     "$ref": "TopicList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.topics.update":

type TopicsUpdateCall struct {
	s        *Service
	seriesId int64
	topicId  int64
	topic    *Topic
	opt_     map[string]interface{}
}

// Update: Updates the specified topic within the specified series.
func (r *TopicsService) Update(seriesId int64, topicId int64, topic *Topic) *TopicsUpdateCall {
	c := &TopicsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.topicId = topicId
	c.topic = topic
	return c
}

func (c *TopicsUpdateCall) Do() (*Topic, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.topic)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/topics/{topicId}")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{topicId}", strconv.FormatInt(c.topicId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Topic)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the specified topic within the specified series.",
	//   "httpMethod": "PUT",
	//   "id": "moderator.topics.update",
	//   "parameterOrder": [
	//     "seriesId",
	//     "topicId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "topicId": {
	//       "description": "The decimal ID of the Topic within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series/{seriesId}/topics/{topicId}",
	//   "request": {
	//     "$ref": "Topic"
	//   },
	//   "response": {
	//     "$ref": "Topic"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.votes.get":

type VotesGetCall struct {
	s            *Service
	seriesId     int64
	submissionId int64
	opt_         map[string]interface{}
}

// Get: Returns the votes by the authenticated user for the specified
// submission within the specified series.
func (r *VotesService) Get(seriesId int64, submissionId int64) *VotesGetCall {
	c := &VotesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.submissionId = submissionId
	return c
}

// UnauthToken sets the optional parameter "unauthToken": User
// identifier for unauthenticated usage mode
func (c *VotesGetCall) UnauthToken(unauthToken string) *VotesGetCall {
	c.opt_["unauthToken"] = unauthToken
	return c
}

// UserId sets the optional parameter "userId":
func (c *VotesGetCall) UserId(userId string) *VotesGetCall {
	c.opt_["userId"] = userId
	return c
}

func (c *VotesGetCall) Do() (*Vote, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["unauthToken"]; ok {
		params.Set("unauthToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["userId"]; ok {
		params.Set("userId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/submissions/{submissionId}/votes/@me")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{submissionId}", strconv.FormatInt(c.submissionId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Vote)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns the votes by the authenticated user for the specified submission within the specified series.",
	//   "httpMethod": "GET",
	//   "id": "moderator.votes.get",
	//   "parameterOrder": [
	//     "seriesId",
	//     "submissionId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "submissionId": {
	//       "description": "The decimal ID of the Submission within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "unauthToken": {
	//       "description": "User identifier for unauthenticated usage mode",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "series/{seriesId}/submissions/{submissionId}/votes/@me",
	//   "response": {
	//     "$ref": "Vote"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.votes.insert":

type VotesInsertCall struct {
	s            *Service
	seriesId     int64
	submissionId int64
	vote         *Vote
	opt_         map[string]interface{}
}

// Insert: Inserts a new vote by the authenticated user for the
// specified submission within the specified series.
func (r *VotesService) Insert(seriesId int64, submissionId int64, vote *Vote) *VotesInsertCall {
	c := &VotesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.submissionId = submissionId
	c.vote = vote
	return c
}

// UnauthToken sets the optional parameter "unauthToken": User
// identifier for unauthenticated usage mode
func (c *VotesInsertCall) UnauthToken(unauthToken string) *VotesInsertCall {
	c.opt_["unauthToken"] = unauthToken
	return c
}

func (c *VotesInsertCall) Do() (*Vote, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.vote)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["unauthToken"]; ok {
		params.Set("unauthToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/submissions/{submissionId}/votes/@me")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{submissionId}", strconv.FormatInt(c.submissionId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Vote)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a new vote by the authenticated user for the specified submission within the specified series.",
	//   "httpMethod": "POST",
	//   "id": "moderator.votes.insert",
	//   "parameterOrder": [
	//     "seriesId",
	//     "submissionId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "submissionId": {
	//       "description": "The decimal ID of the Submission within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "unauthToken": {
	//       "description": "User identifier for unauthenticated usage mode",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "series/{seriesId}/submissions/{submissionId}/votes/@me",
	//   "request": {
	//     "$ref": "Vote"
	//   },
	//   "response": {
	//     "$ref": "Vote"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.votes.list":

type VotesListCall struct {
	s        *Service
	seriesId int64
	opt_     map[string]interface{}
}

// List: Lists the votes by the authenticated user for the given series.
func (r *VotesService) List(seriesId int64) *VotesListCall {
	c := &VotesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	return c
}

// MaxResults sets the optional parameter "max-results": Maximum number
// of results to return.
func (c *VotesListCall) MaxResults(maxResults int64) *VotesListCall {
	c.opt_["max-results"] = maxResults
	return c
}

// StartIndex sets the optional parameter "start-index": Index of the
// first result to be retrieved.
func (c *VotesListCall) StartIndex(startIndex int64) *VotesListCall {
	c.opt_["start-index"] = startIndex
	return c
}

func (c *VotesListCall) Do() (*VoteList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["max-results"]; ok {
		params.Set("max-results", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["start-index"]; ok {
		params.Set("start-index", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/votes/@me")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(VoteList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists the votes by the authenticated user for the given series.",
	//   "httpMethod": "GET",
	//   "id": "moderator.votes.list",
	//   "parameterOrder": [
	//     "seriesId"
	//   ],
	//   "parameters": {
	//     "max-results": {
	//       "description": "Maximum number of results to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "start-index": {
	//       "description": "Index of the first result to be retrieved.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "series/{seriesId}/votes/@me",
	//   "response": {
	//     "$ref": "VoteList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.votes.patch":

type VotesPatchCall struct {
	s            *Service
	seriesId     int64
	submissionId int64
	vote         *Vote
	opt_         map[string]interface{}
}

// Patch: Updates the votes by the authenticated user for the specified
// submission within the specified series. This method supports patch
// semantics.
func (r *VotesService) Patch(seriesId int64, submissionId int64, vote *Vote) *VotesPatchCall {
	c := &VotesPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.submissionId = submissionId
	c.vote = vote
	return c
}

// UnauthToken sets the optional parameter "unauthToken": User
// identifier for unauthenticated usage mode
func (c *VotesPatchCall) UnauthToken(unauthToken string) *VotesPatchCall {
	c.opt_["unauthToken"] = unauthToken
	return c
}

// UserId sets the optional parameter "userId":
func (c *VotesPatchCall) UserId(userId string) *VotesPatchCall {
	c.opt_["userId"] = userId
	return c
}

func (c *VotesPatchCall) Do() (*Vote, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.vote)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["unauthToken"]; ok {
		params.Set("unauthToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["userId"]; ok {
		params.Set("userId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/submissions/{submissionId}/votes/@me")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{submissionId}", strconv.FormatInt(c.submissionId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Vote)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the votes by the authenticated user for the specified submission within the specified series. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "moderator.votes.patch",
	//   "parameterOrder": [
	//     "seriesId",
	//     "submissionId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "submissionId": {
	//       "description": "The decimal ID of the Submission within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "unauthToken": {
	//       "description": "User identifier for unauthenticated usage mode",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "series/{seriesId}/submissions/{submissionId}/votes/@me",
	//   "request": {
	//     "$ref": "Vote"
	//   },
	//   "response": {
	//     "$ref": "Vote"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

// method id "moderator.votes.update":

type VotesUpdateCall struct {
	s            *Service
	seriesId     int64
	submissionId int64
	vote         *Vote
	opt_         map[string]interface{}
}

// Update: Updates the votes by the authenticated user for the specified
// submission within the specified series.
func (r *VotesService) Update(seriesId int64, submissionId int64, vote *Vote) *VotesUpdateCall {
	c := &VotesUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.seriesId = seriesId
	c.submissionId = submissionId
	c.vote = vote
	return c
}

// UnauthToken sets the optional parameter "unauthToken": User
// identifier for unauthenticated usage mode
func (c *VotesUpdateCall) UnauthToken(unauthToken string) *VotesUpdateCall {
	c.opt_["unauthToken"] = unauthToken
	return c
}

// UserId sets the optional parameter "userId":
func (c *VotesUpdateCall) UserId(userId string) *VotesUpdateCall {
	c.opt_["userId"] = userId
	return c
}

func (c *VotesUpdateCall) Do() (*Vote, error) {
	var body io.Reader = nil
	body, err := googleapi.WithDataWrapper.JSONReader(c.vote)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["unauthToken"]; ok {
		params.Set("unauthToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["userId"]; ok {
		params.Set("userId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/moderator/v1/", "series/{seriesId}/submissions/{submissionId}/votes/@me")
	urls = strings.Replace(urls, "{seriesId}", strconv.FormatInt(c.seriesId, 10), 1)
	urls = strings.Replace(urls, "{submissionId}", strconv.FormatInt(c.submissionId, 10), 1)
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Vote)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the votes by the authenticated user for the specified submission within the specified series.",
	//   "httpMethod": "PUT",
	//   "id": "moderator.votes.update",
	//   "parameterOrder": [
	//     "seriesId",
	//     "submissionId"
	//   ],
	//   "parameters": {
	//     "seriesId": {
	//       "description": "The decimal ID of the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "submissionId": {
	//       "description": "The decimal ID of the Submission within the Series.",
	//       "format": "uint32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "unauthToken": {
	//       "description": "User identifier for unauthenticated usage mode",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "series/{seriesId}/submissions/{submissionId}/votes/@me",
	//   "request": {
	//     "$ref": "Vote"
	//   },
	//   "response": {
	//     "$ref": "Vote"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/moderator"
	//   ]
	// }

}

func cleanPathString(s string) string {
	return strings.Map(func(r rune) rune {
		if r >= 0x2d && r <= 0x7a || r == '~' {
			return r
		}
		return -1
	}, s)
}
