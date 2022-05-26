//
// Copyright 2021, Sander van Harmelen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

// Package gitlab implements a GitLab API client.
package gitlab

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"math/rand"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/go-querystring/query"
	"github.com/hashicorp/go-cleanhttp"
	retryablehttp "github.com/hashicorp/go-retryablehttp"
	"golang.org/x/oauth2"
	"golang.org/x/time/rate"
)

const (
	defaultBaseURL = "https://gitlab.com/"
	apiVersionPath = "api/v4/"
	userAgent      = "go-gitlab"

	headerRateLimit = "RateLimit-Limit"
	headerRateReset = "RateLimit-Reset"
)

// authType represents an authentication type within GitLab.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/
type authType int

// List of available authentication types.
//
// GitLab API docs: https://docs.gitlab.com/ce/api/
const (
	basicAuth authType = iota
	oAuthToken
	privateToken
)

// A Client manages communication with the GitLab API.
type Client struct {
	// HTTP client used to communicate with the API.
	client *retryablehttp.Client

	// Base URL for API requests. Defaults to the public GitLab API, but can be
	// set to a domain endpoint to use with a self hosted GitLab server. baseURL
	// should always be specified with a trailing slash.
	baseURL *url.URL

	// disableRetries is used to disable the default retry logic.
	disableRetries bool

	// configureLimiterOnce is used to make sure the limiter is configured exactly
	// once and block all other calls until the initial (one) call is done.
	configureLimiterOnce sync.Once

	// Limiter is used to limit API calls and prevent 429 responses.
	limiter RateLimiter

	// Token type used to make authenticated API calls.
	authType authType

	// Username and password used for basix authentication.
	username, password string

	// Token used to make authenticated API calls.
	token string

	// Protects the token field from concurrent read/write accesses.
	tokenLock sync.RWMutex

	// User agent used when communicating with the GitLab API.
	UserAgent string

	// Services used for talking to different parts of the GitLab API.
	AccessRequests        *AccessRequestsService
	Applications          *ApplicationsService
	AuditEvents           *AuditEventsService
	AwardEmoji            *AwardEmojiService
	Boards                *IssueBoardsService
	Branches              *BranchesService
	BroadcastMessage      *BroadcastMessagesService
	CIYMLTemplate         *CIYMLTemplatesService
	Commits               *CommitsService
	ContainerRegistry     *ContainerRegistryService
	CustomAttribute       *CustomAttributesService
	DeployKeys            *DeployKeysService
	DeployTokens          *DeployTokensService
	Deployments           *DeploymentsService
	Discussions           *DiscussionsService
	Environments          *EnvironmentsService
	EpicIssues            *EpicIssuesService
	Epics                 *EpicsService
	Events                *EventsService
	Features              *FeaturesService
	FreezePeriods         *FreezePeriodsService
	GitIgnoreTemplates    *GitIgnoreTemplatesService
	GroupBadges           *GroupBadgesService
	GroupCluster          *GroupClustersService
	GroupIssueBoards      *GroupIssueBoardsService
	GroupLabels           *GroupLabelsService
	GroupMembers          *GroupMembersService
	GroupMilestones       *GroupMilestonesService
	GroupVariables        *GroupVariablesService
	Groups                *GroupsService
	InstanceCluster       *InstanceClustersService
	InstanceVariables     *InstanceVariablesService
	Invites               *InvitesService
	IssueLinks            *IssueLinksService
	Issues                *IssuesService
	IssuesStatistics      *IssuesStatisticsService
	Jobs                  *JobsService
	Keys                  *KeysService
	Labels                *LabelsService
	License               *LicenseService
	LicenseTemplates      *LicenseTemplatesService
	MergeRequestApprovals *MergeRequestApprovalsService
	MergeRequests         *MergeRequestsService
	Milestones            *MilestonesService
	Namespaces            *NamespacesService
	Notes                 *NotesService
	NotificationSettings  *NotificationSettingsService
	Packages              *PackagesService
	PagesDomains          *PagesDomainsService
	PipelineSchedules     *PipelineSchedulesService
	PipelineTriggers      *PipelineTriggersService
	Pipelines             *PipelinesService
	ProjectBadges         *ProjectBadgesService
	ProjectCluster        *ProjectClustersService
	ProjectImportExport   *ProjectImportExportService
	ProjectMembers        *ProjectMembersService
	ProjectMirrors        *ProjectMirrorService
	ProjectSnippets       *ProjectSnippetsService
	ProjectVariables      *ProjectVariablesService
	Projects              *ProjectsService
	ProtectedBranches     *ProtectedBranchesService
	ProtectedTags         *ProtectedTagsService
	ReleaseLinks          *ReleaseLinksService
	Releases              *ReleasesService
	Repositories          *RepositoriesService
	RepositoryFiles       *RepositoryFilesService
	ResourceLabelEvents   *ResourceLabelEventsService
	Runners               *RunnersService
	Search                *SearchService
	Services              *ServicesService
	Settings              *SettingsService
	Sidekiq               *SidekiqService
	Snippets              *SnippetsService
	SystemHooks           *SystemHooksService
	Tags                  *TagsService
	Todos                 *TodosService
	Users                 *UsersService
	Validate              *ValidateService
	Version               *VersionService
	Wikis                 *WikisService
}

// ListOptions specifies the optional parameters to various List methods that
// support pagination.
type ListOptions struct {
	// For paginated result sets, page of results to retrieve.
	Page int `url:"page,omitempty" json:"page,omitempty"`

	// For paginated result sets, the number of results to include per page.
	PerPage int `url:"per_page,omitempty" json:"per_page,omitempty"`
}

// RateLimiter describes the interface that all (custom) rate limiters must implement.
type RateLimiter interface {
	Wait(context.Context) error
}

// NewClient returns a new GitLab API client. To use API methods which require
// authentication, provide a valid private or personal token.
func NewClient(token string, options ...ClientOptionFunc) (*Client, error) {
	client, err := newClient(options...)
	if err != nil {
		return nil, err
	}
	client.authType = privateToken
	client.token = token
	return client, nil
}

// NewBasicAuthClient returns a new GitLab API client. To use API methods which
// require authentication, provide a valid username and password.
func NewBasicAuthClient(username, password string, options ...ClientOptionFunc) (*Client, error) {
	client, err := newClient(options...)
	if err != nil {
		return nil, err
	}

	client.authType = basicAuth
	client.username = username
	client.password = password

	return client, nil
}

// NewOAuthClient returns a new GitLab API client. To use API methods which
// require authentication, provide a valid oauth token.
func NewOAuthClient(token string, options ...ClientOptionFunc) (*Client, error) {
	client, err := newClient(options...)
	if err != nil {
		return nil, err
	}
	client.authType = oAuthToken
	client.token = token
	return client, nil
}

func newClient(options ...ClientOptionFunc) (*Client, error) {
	c := &Client{UserAgent: userAgent}

	// Configure the HTTP client.
	c.client = &retryablehttp.Client{
		Backoff:      c.retryHTTPBackoff,
		CheckRetry:   c.retryHTTPCheck,
		ErrorHandler: retryablehttp.PassthroughErrorHandler,
		HTTPClient:   cleanhttp.DefaultPooledClient(),
		RetryWaitMin: 100 * time.Millisecond,
		RetryWaitMax: 400 * time.Millisecond,
		RetryMax:     5,
	}

	// Set the default base URL.
	c.setBaseURL(defaultBaseURL)

	// Apply any given client options.
	for _, fn := range options {
		if fn == nil {
			continue
		}
		if err := fn(c); err != nil {
			return nil, err
		}
	}

	// Create the internal timeStats service.
	timeStats := &timeStatsService{client: c}

	// Create all the public services.
	c.AccessRequests = &AccessRequestsService{client: c}
	c.Applications = &ApplicationsService{client: c}
	c.AuditEvents = &AuditEventsService{client: c}
	c.AwardEmoji = &AwardEmojiService{client: c}
	c.Boards = &IssueBoardsService{client: c}
	c.Branches = &BranchesService{client: c}
	c.BroadcastMessage = &BroadcastMessagesService{client: c}
	c.CIYMLTemplate = &CIYMLTemplatesService{client: c}
	c.Commits = &CommitsService{client: c}
	c.ContainerRegistry = &ContainerRegistryService{client: c}
	c.CustomAttribute = &CustomAttributesService{client: c}
	c.DeployKeys = &DeployKeysService{client: c}
	c.DeployTokens = &DeployTokensService{client: c}
	c.Deployments = &DeploymentsService{client: c}
	c.Discussions = &DiscussionsService{client: c}
	c.Environments = &EnvironmentsService{client: c}
	c.EpicIssues = &EpicIssuesService{client: c}
	c.Epics = &EpicsService{client: c}
	c.Events = &EventsService{client: c}
	c.Features = &FeaturesService{client: c}
	c.FreezePeriods = &FreezePeriodsService{client: c}
	c.GitIgnoreTemplates = &GitIgnoreTemplatesService{client: c}
	c.GroupBadges = &GroupBadgesService{client: c}
	c.GroupCluster = &GroupClustersService{client: c}
	c.GroupIssueBoards = &GroupIssueBoardsService{client: c}
	c.GroupLabels = &GroupLabelsService{client: c}
	c.GroupMembers = &GroupMembersService{client: c}
	c.GroupMilestones = &GroupMilestonesService{client: c}
	c.GroupVariables = &GroupVariablesService{client: c}
	c.Groups = &GroupsService{client: c}
	c.InstanceCluster = &InstanceClustersService{client: c}
	c.InstanceVariables = &InstanceVariablesService{client: c}
	c.Invites = &InvitesService{client: c}
	c.IssueLinks = &IssueLinksService{client: c}
	c.Issues = &IssuesService{client: c, timeStats: timeStats}
	c.IssuesStatistics = &IssuesStatisticsService{client: c}
	c.Jobs = &JobsService{client: c}
	c.Keys = &KeysService{client: c}
	c.Labels = &LabelsService{client: c}
	c.License = &LicenseService{client: c}
	c.LicenseTemplates = &LicenseTemplatesService{client: c}
	c.MergeRequestApprovals = &MergeRequestApprovalsService{client: c}
	c.MergeRequests = &MergeRequestsService{client: c, timeStats: timeStats}
	c.Milestones = &MilestonesService{client: c}
	c.Namespaces = &NamespacesService{client: c}
	c.Notes = &NotesService{client: c}
	c.NotificationSettings = &NotificationSettingsService{client: c}
	c.Packages = &PackagesService{client: c}
	c.PagesDomains = &PagesDomainsService{client: c}
	c.PipelineSchedules = &PipelineSchedulesService{client: c}
	c.PipelineTriggers = &PipelineTriggersService{client: c}
	c.Pipelines = &PipelinesService{client: c}
	c.ProjectBadges = &ProjectBadgesService{client: c}
	c.ProjectCluster = &ProjectClustersService{client: c}
	c.ProjectImportExport = &ProjectImportExportService{client: c}
	c.ProjectMembers = &ProjectMembersService{client: c}
	c.ProjectMirrors = &ProjectMirrorService{client: c}
	c.ProjectSnippets = &ProjectSnippetsService{client: c}
	c.ProjectVariables = &ProjectVariablesService{client: c}
	c.Projects = &ProjectsService{client: c}
	c.ProtectedBranches = &ProtectedBranchesService{client: c}
	c.ProtectedTags = &ProtectedTagsService{client: c}
	c.ReleaseLinks = &ReleaseLinksService{client: c}
	c.Releases = &ReleasesService{client: c}
	c.Repositories = &RepositoriesService{client: c}
	c.RepositoryFiles = &RepositoryFilesService{client: c}
	c.ResourceLabelEvents = &ResourceLabelEventsService{client: c}
	c.Runners = &RunnersService{client: c}
	c.Search = &SearchService{client: c}
	c.Services = &ServicesService{client: c}
	c.Settings = &SettingsService{client: c}
	c.Sidekiq = &SidekiqService{client: c}
	c.Snippets = &SnippetsService{client: c}
	c.SystemHooks = &SystemHooksService{client: c}
	c.Tags = &TagsService{client: c}
	c.Todos = &TodosService{client: c}
	c.Users = &UsersService{client: c}
	c.Validate = &ValidateService{client: c}
	c.Version = &VersionService{client: c}
	c.Wikis = &WikisService{client: c}

	return c, nil
}

// retryHTTPCheck provides a callback for Client.CheckRetry which
// will retry both rate limit (429) and server (>= 500) errors.
func (c *Client) retryHTTPCheck(ctx context.Context, resp *http.Response, err error) (bool, error) {
	if ctx.Err() != nil {
		return false, ctx.Err()
	}
	if err != nil {
		return false, err
	}
	if !c.disableRetries && (resp.StatusCode == 429 || resp.StatusCode >= 500) {
		return true, nil
	}
	return false, nil
}

// retryHTTPBackoff provides a generic callback for Client.Backoff which
// will pass through all calls based on the status code of the response.
func (c *Client) retryHTTPBackoff(min, max time.Duration, attemptNum int, resp *http.Response) time.Duration {
	// Use the rate limit backoff function when we are rate limited.
	if resp != nil && resp.StatusCode == 429 {
		return rateLimitBackoff(min, max, attemptNum, resp)
	}

	// Set custom duration's when we experience a service interruption.
	min = 700 * time.Millisecond
	max = 900 * time.Millisecond

	return retryablehttp.LinearJitterBackoff(min, max, attemptNum, resp)
}

// rateLimitBackoff provides a callback for Client.Backoff which will use the
// RateLimit-Reset header to determine the time to wait. We add some jitter
// to prevent a thundering herd.
//
// min and max are mainly used for bounding the jitter that will be added to
// the reset time retrieved from the headers. But if the final wait time is
// less then min, min will be used instead.
func rateLimitBackoff(min, max time.Duration, attemptNum int, resp *http.Response) time.Duration {
	// rnd is used to generate pseudo-random numbers.
	rnd := rand.New(rand.NewSource(time.Now().UnixNano()))

	// First create some jitter bounded by the min and max durations.
	jitter := time.Duration(rnd.Float64() * float64(max-min))

	if resp != nil {
		if v := resp.Header.Get(headerRateReset); v != "" {
			if reset, _ := strconv.ParseInt(v, 10, 64); reset > 0 {
				// Only update min if the given time to wait is longer.
				if wait := time.Until(time.Unix(reset, 0)); wait > min {
					min = wait
				}
			}
		}
	}

	return min + jitter
}

// configureLimiter configures the rate limiter.
func (c *Client) configureLimiter() error {
	// Set default values for when rate limiting is disabled.
	limit := rate.Inf
	burst := 0

	defer func() {
		// Create a new limiter using the calculated values.
		c.limiter = rate.NewLimiter(limit, burst)
	}()

	// Create a new request.
	req, err := http.NewRequest("GET", c.baseURL.String(), nil)
	if err != nil {
		return err
	}

	// Make a single request to retrieve the rate limit headers.
	resp, err := c.client.HTTPClient.Do(req)
	if err != nil {
		return err
	}
	resp.Body.Close()

	if v := resp.Header.Get(headerRateLimit); v != "" {
		if rateLimit, _ := strconv.ParseFloat(v, 64); rateLimit > 0 {
			// The rate limit is based on requests per minute, so for our limiter to
			// work correctly we divide the limit by 60 to get the limit per second.
			rateLimit /= 60
			// Configure the limit and burst using a split of 2/3 for the limit and
			// 1/3 for the burst. This enables clients to burst 1/3 of the allowed
			// calls before the limiter kicks in. The remaining calls will then be
			// spread out evenly using intervals of time.Second / limit which should
			// prevent hitting the rate limit.
			limit = rate.Limit(rateLimit * 0.66)
			burst = int(rateLimit * 0.33)
		}
	}

	return nil
}

// BaseURL return a copy of the baseURL.
func (c *Client) BaseURL() *url.URL {
	u := *c.baseURL
	return &u
}

// setBaseURL sets the base URL for API requests to a custom endpoint.
func (c *Client) setBaseURL(urlStr string) error {
	// Make sure the given URL end with a slash
	if !strings.HasSuffix(urlStr, "/") {
		urlStr += "/"
	}

	baseURL, err := url.Parse(urlStr)
	if err != nil {
		return err
	}

	if !strings.HasSuffix(baseURL.Path, apiVersionPath) {
		baseURL.Path += apiVersionPath
	}

	// Update the base URL of the client.
	c.baseURL = baseURL

	return nil
}

// NewRequest creates an API request. A relative URL path can be provided in
// path, in which case it is resolved relative to the base URL of the Client.
// Relative URL paths should always be specified without a preceding slash. If
// specified, the value pointed to by body is JSON encoded and included as the
// request body.
func (c *Client) NewRequest(method, path string, opt interface{}, options []RequestOptionFunc) (*retryablehttp.Request, error) {
	u := *c.baseURL
	unescaped, err := url.PathUnescape(path)
	if err != nil {
		return nil, err
	}

	// Set the encoded path data
	u.RawPath = c.baseURL.Path + path
	u.Path = c.baseURL.Path + unescaped

	// Create a request specific headers map.
	reqHeaders := make(http.Header)
	reqHeaders.Set("Accept", "application/json")

	if c.UserAgent != "" {
		reqHeaders.Set("User-Agent", c.UserAgent)
	}

	var body interface{}
	switch {
	case method == "POST" || method == "PUT":
		reqHeaders.Set("Content-Type", "application/json")

		if opt != nil {
			body, err = json.Marshal(opt)
			if err != nil {
				return nil, err
			}
		}
	case opt != nil:
		q, err := query.Values(opt)
		if err != nil {
			return nil, err
		}
		u.RawQuery = q.Encode()
	}

	req, err := retryablehttp.NewRequest(method, u.String(), body)
	if err != nil {
		return nil, err
	}

	for _, fn := range options {
		if fn == nil {
			continue
		}
		if err := fn(req); err != nil {
			return nil, err
		}
	}

	// Set the request specific headers.
	for k, v := range reqHeaders {
		req.Header[k] = v
	}

	return req, nil
}

// Response is a GitLab API response. This wraps the standard http.Response
// returned from GitLab and provides convenient access to things like
// pagination links.
type Response struct {
	*http.Response

	// These fields provide the page values for paginating through a set of
	// results. Any or all of these may be set to the zero value for
	// responses that are not part of a paginated set, or for which there
	// are no additional pages.
	TotalItems   int
	TotalPages   int
	ItemsPerPage int
	CurrentPage  int
	NextPage     int
	PreviousPage int
}

// newResponse creates a new Response for the provided http.Response.
func newResponse(r *http.Response) *Response {
	response := &Response{Response: r}
	response.populatePageValues()
	return response
}

const (
	xTotal      = "X-Total"
	xTotalPages = "X-Total-Pages"
	xPerPage    = "X-Per-Page"
	xPage       = "X-Page"
	xNextPage   = "X-Next-Page"
	xPrevPage   = "X-Prev-Page"
)

// populatePageValues parses the HTTP Link response headers and populates the
// various pagination link values in the Response.
func (r *Response) populatePageValues() {
	if totalItems := r.Response.Header.Get(xTotal); totalItems != "" {
		r.TotalItems, _ = strconv.Atoi(totalItems)
	}
	if totalPages := r.Response.Header.Get(xTotalPages); totalPages != "" {
		r.TotalPages, _ = strconv.Atoi(totalPages)
	}
	if itemsPerPage := r.Response.Header.Get(xPerPage); itemsPerPage != "" {
		r.ItemsPerPage, _ = strconv.Atoi(itemsPerPage)
	}
	if currentPage := r.Response.Header.Get(xPage); currentPage != "" {
		r.CurrentPage, _ = strconv.Atoi(currentPage)
	}
	if nextPage := r.Response.Header.Get(xNextPage); nextPage != "" {
		r.NextPage, _ = strconv.Atoi(nextPage)
	}
	if previousPage := r.Response.Header.Get(xPrevPage); previousPage != "" {
		r.PreviousPage, _ = strconv.Atoi(previousPage)
	}
}

// Do sends an API request and returns the API response. The API response is
// JSON decoded and stored in the value pointed to by v, or returned as an
// error if an API error has occurred. If v implements the io.Writer
// interface, the raw response body will be written to v, without attempting to
// first decode it.
func (c *Client) Do(req *retryablehttp.Request, v interface{}) (*Response, error) {
	// If not yet configured, try to configure the rate limiter. Fail
	// silently as the limiter will be disabled in case of an error.
	c.configureLimiterOnce.Do(func() { c.configureLimiter() })

	// Wait will block until the limiter can obtain a new token.
	err := c.limiter.Wait(req.Context())
	if err != nil {
		return nil, err
	}

	// Set the correct authentication header. If using basic auth, then check
	// if we already have a token and if not first authenticate and get one.
	var basicAuthToken string
	switch c.authType {
	case basicAuth:
		c.tokenLock.RLock()
		basicAuthToken = c.token
		c.tokenLock.RUnlock()
		if basicAuthToken == "" {
			// If we don't have a token yet, we first need to request one.
			basicAuthToken, err = c.requestOAuthToken(req.Context(), basicAuthToken)
			if err != nil {
				return nil, err
			}
		}
		req.Header.Set("Authorization", "Bearer "+basicAuthToken)
	case oAuthToken:
		req.Header.Set("Authorization", "Bearer "+c.token)
	case privateToken:
		req.Header.Set("PRIVATE-TOKEN", c.token)
	}

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode == http.StatusUnauthorized && c.authType == basicAuth {
		resp.Body.Close()
		// The token most likely expired, so we need to request a new one and try again.
		if _, err := c.requestOAuthToken(req.Context(), basicAuthToken); err != nil {
			return nil, err
		}
		return c.Do(req, v)
	}
	defer resp.Body.Close()

	response := newResponse(resp)

	err = CheckResponse(resp)
	if err != nil {
		// Even though there was an error, we still return the response
		// in case the caller wants to inspect it further.
		return response, err
	}

	if v != nil {
		if w, ok := v.(io.Writer); ok {
			_, err = io.Copy(w, resp.Body)
		} else {
			err = json.NewDecoder(resp.Body).Decode(v)
		}
	}

	return response, err
}

func (c *Client) requestOAuthToken(ctx context.Context, token string) (string, error) {
	c.tokenLock.Lock()
	defer c.tokenLock.Unlock()

	// Return early if the token was updated while waiting for the lock.
	if c.token != token {
		return c.token, nil
	}

	config := &oauth2.Config{
		Endpoint: oauth2.Endpoint{
			AuthURL:  strings.TrimSuffix(c.baseURL.String(), apiVersionPath) + "oauth/authorize",
			TokenURL: strings.TrimSuffix(c.baseURL.String(), apiVersionPath) + "oauth/token",
		},
	}

	ctx = context.WithValue(ctx, oauth2.HTTPClient, c.client.HTTPClient)
	t, err := config.PasswordCredentialsToken(ctx, c.username, c.password)
	if err != nil {
		return "", err
	}
	c.token = t.AccessToken

	return c.token, nil
}

// Helper function to accept and format both the project ID or name as project
// identifier for all API calls.
func parseID(id interface{}) (string, error) {
	switch v := id.(type) {
	case int:
		return strconv.Itoa(v), nil
	case string:
		return v, nil
	default:
		return "", fmt.Errorf("invalid ID type %#v, the ID must be an int or a string", id)
	}
}

// Helper function to escape a project identifier.
func pathEscape(s string) string {
	return strings.Replace(url.PathEscape(s), ".", "%2E", -1)
}

// An ErrorResponse reports one or more errors caused by an API request.
//
// GitLab API docs:
// https://docs.gitlab.com/ce/api/README.html#data-validation-and-error-reporting
type ErrorResponse struct {
	Body     []byte
	Response *http.Response
	Message  string
}

func (e *ErrorResponse) Error() string {
	path, _ := url.QueryUnescape(e.Response.Request.URL.Path)
	u := fmt.Sprintf("%s://%s%s", e.Response.Request.URL.Scheme, e.Response.Request.URL.Host, path)
	return fmt.Sprintf("%s %s: %d %s", e.Response.Request.Method, u, e.Response.StatusCode, e.Message)
}

// CheckResponse checks the API response for errors, and returns them if present.
func CheckResponse(r *http.Response) error {
	switch r.StatusCode {
	case 200, 201, 202, 204, 304:
		return nil
	}

	errorResponse := &ErrorResponse{Response: r}
	data, err := ioutil.ReadAll(r.Body)
	if err == nil && data != nil {
		errorResponse.Body = data

		var raw interface{}
		if err := json.Unmarshal(data, &raw); err != nil {
			errorResponse.Message = "failed to parse unknown error format"
		} else {
			errorResponse.Message = parseError(raw)
		}
	}

	return errorResponse
}

// Format:
// {
//     "message": {
//         "<property-name>": [
//             "<error-message>",
//             "<error-message>",
//             ...
//         ],
//         "<embed-entity>": {
//             "<property-name>": [
//                 "<error-message>",
//                 "<error-message>",
//                 ...
//             ],
//         }
//     },
//     "error": "<error-message>"
// }
func parseError(raw interface{}) string {
	switch raw := raw.(type) {
	case string:
		return raw

	case []interface{}:
		var errs []string
		for _, v := range raw {
			errs = append(errs, parseError(v))
		}
		return fmt.Sprintf("[%s]", strings.Join(errs, ", "))

	case map[string]interface{}:
		var errs []string
		for k, v := range raw {
			errs = append(errs, fmt.Sprintf("{%s: %s}", k, parseError(v)))
		}
		sort.Strings(errs)
		return strings.Join(errs, ", ")

	default:
		return fmt.Sprintf("failed to parse unexpected error type: %T", raw)
	}
}
