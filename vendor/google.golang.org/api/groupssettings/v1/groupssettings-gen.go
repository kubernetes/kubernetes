// Package groupssettings provides access to the Groups Settings API.
//
// See https://developers.google.com/google-apps/groups-settings/get_started
//
// Usage example:
//
//   import "google.golang.org/api/groupssettings/v1"
//   ...
//   groupssettingsService, err := groupssettings.New(oauthHttpClient)
package groupssettings // import "google.golang.org/api/groupssettings/v1"

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

const apiId = "groupssettings:v1"
const apiName = "groupssettings"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/groups/v1/groups/"

// OAuth2 scopes used by this API.
const (
	// View and manage the settings of a Google Apps Group
	AppsGroupsSettingsScope = "https://www.googleapis.com/auth/apps.groups.settings"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Groups = NewGroupsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Groups *GroupsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewGroupsService(s *Service) *GroupsService {
	rs := &GroupsService{s: s}
	return rs
}

type GroupsService struct {
	s *Service
}

// Groups: JSON template for Group resource
type Groups struct {
	// AllowExternalMembers: Are external members allowed to join the group.
	AllowExternalMembers string `json:"allowExternalMembers,omitempty"`

	// AllowGoogleCommunication: Is google allowed to contact admins.
	AllowGoogleCommunication string `json:"allowGoogleCommunication,omitempty"`

	// AllowWebPosting: If posting from web is allowed.
	AllowWebPosting string `json:"allowWebPosting,omitempty"`

	// ArchiveOnly: If the group is archive only
	ArchiveOnly string `json:"archiveOnly,omitempty"`

	// CustomReplyTo: Default email to which reply to any message should go.
	CustomReplyTo string `json:"customReplyTo,omitempty"`

	// DefaultMessageDenyNotificationText: Default message deny notification
	// message
	DefaultMessageDenyNotificationText string `json:"defaultMessageDenyNotificationText,omitempty"`

	// Description: Description of the group
	Description string `json:"description,omitempty"`

	// Email: Email id of the group
	Email string `json:"email,omitempty"`

	// IncludeInGlobalAddressList: If this groups should be included in
	// global address list or not.
	IncludeInGlobalAddressList string `json:"includeInGlobalAddressList,omitempty"`

	// IsArchived: If the contents of the group are archived.
	IsArchived string `json:"isArchived,omitempty"`

	// Kind: The type of the resource.
	Kind string `json:"kind,omitempty"`

	// MaxMessageBytes: Maximum message size allowed.
	MaxMessageBytes int64 `json:"maxMessageBytes,omitempty"`

	// MembersCanPostAsTheGroup: Can members post using the group email
	// address.
	MembersCanPostAsTheGroup string `json:"membersCanPostAsTheGroup,omitempty"`

	// MessageDisplayFont: Default message display font. Possible values
	// are: DEFAULT_FONT FIXED_WIDTH_FONT
	MessageDisplayFont string `json:"messageDisplayFont,omitempty"`

	// MessageModerationLevel: Moderation level for messages. Possible
	// values are: MODERATE_ALL_MESSAGES MODERATE_NON_MEMBERS
	// MODERATE_NEW_MEMBERS MODERATE_NONE
	MessageModerationLevel string `json:"messageModerationLevel,omitempty"`

	// Name: Name of the Group
	Name string `json:"name,omitempty"`

	// PrimaryLanguage: Primary language for the group.
	PrimaryLanguage string `json:"primaryLanguage,omitempty"`

	// ReplyTo: Whome should the default reply to a message go to. Possible
	// values are: REPLY_TO_CUSTOM REPLY_TO_SENDER REPLY_TO_LIST
	// REPLY_TO_OWNER REPLY_TO_IGNORE REPLY_TO_MANAGERS
	ReplyTo string `json:"replyTo,omitempty"`

	// SendMessageDenyNotification: Should the member be notified if his
	// message is denied by owner.
	SendMessageDenyNotification string `json:"sendMessageDenyNotification,omitempty"`

	// ShowInGroupDirectory: Is the group listed in groups directory
	ShowInGroupDirectory string `json:"showInGroupDirectory,omitempty"`

	// SpamModerationLevel: Moderation level for messages detected as spam.
	// Possible values are: ALLOW MODERATE SILENTLY_MODERATE REJECT
	SpamModerationLevel string `json:"spamModerationLevel,omitempty"`

	// WhoCanContactOwner: Permission to contact owner of the group via web
	// UI. Possbile values are: ANYONE_CAN_CONTACT ALL_IN_DOMAIN_CAN_CONTACT
	// ALL_MEMBERS_CAN_CONTACT ALL_MANAGERS_CAN_CONTACT
	WhoCanContactOwner string `json:"whoCanContactOwner,omitempty"`

	// WhoCanInvite: Permissions to invite members. Possbile values are:
	// ALL_MEMBERS_CAN_INVITE ALL_MANAGERS_CAN_INVITE
	WhoCanInvite string `json:"whoCanInvite,omitempty"`

	// WhoCanJoin: Permissions to join the group. Possible values are:
	// ANYONE_CAN_JOIN ALL_IN_DOMAIN_CAN_JOIN INVITED_CAN_JOIN
	// CAN_REQUEST_TO_JOIN
	WhoCanJoin string `json:"whoCanJoin,omitempty"`

	// WhoCanLeaveGroup: Permission to leave the group. Possible values are:
	// ALL_MANAGERS_CAN_LEAVE ALL_MEMBERS_CAN_LEAVE
	WhoCanLeaveGroup string `json:"whoCanLeaveGroup,omitempty"`

	// WhoCanPostMessage: Permissions to post messages to the group.
	// Possible values are: NONE_CAN_POST ALL_MANAGERS_CAN_POST
	// ALL_MEMBERS_CAN_POST ALL_IN_DOMAIN_CAN_POST ANYONE_CAN_POST
	WhoCanPostMessage string `json:"whoCanPostMessage,omitempty"`

	// WhoCanViewGroup: Permissions to view group. Possbile values are:
	// ANYONE_CAN_VIEW ALL_IN_DOMAIN_CAN_VIEW ALL_MEMBERS_CAN_VIEW
	// ALL_MANAGERS_CAN_VIEW
	WhoCanViewGroup string `json:"whoCanViewGroup,omitempty"`

	// WhoCanViewMembership: Permissions to view membership. Possbile values
	// are: ALL_IN_DOMAIN_CAN_VIEW ALL_MEMBERS_CAN_VIEW
	// ALL_MANAGERS_CAN_VIEW
	WhoCanViewMembership string `json:"whoCanViewMembership,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g.
	// "AllowExternalMembers") to unconditionally include in API requests.
	// By default, fields with empty values are omitted from API requests.
	// However, any non-pointer, non-interface field appearing in
	// ForceSendFields will be sent to the server regardless of whether the
	// field is empty or not. This may be used to include empty fields in
	// Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Groups) MarshalJSON() ([]byte, error) {
	type noMethod Groups
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "groupsSettings.groups.get":

type GroupsGetCall struct {
	s             *Service
	groupUniqueId string
	urlParams_    gensupport.URLParams
	ifNoneMatch_  string
	ctx_          context.Context
}

// Get: Gets one resource by id.
func (r *GroupsService) Get(groupUniqueId string) *GroupsGetCall {
	c := &GroupsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.groupUniqueId = groupUniqueId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GroupsGetCall) QuotaUser(quotaUser string) *GroupsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GroupsGetCall) UserIP(userIP string) *GroupsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsGetCall) Fields(s ...googleapi.Field) *GroupsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *GroupsGetCall) IfNoneMatch(entityTag string) *GroupsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsGetCall) Context(ctx context.Context) *GroupsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *GroupsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{groupUniqueId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"groupUniqueId": c.groupUniqueId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "groupsSettings.groups.get" call.
// Exactly one of *Groups or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Groups.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *GroupsGetCall) Do() (*Groups, error) {
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
	ret := &Groups{
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
	//   "description": "Gets one resource by id.",
	//   "httpMethod": "GET",
	//   "id": "groupsSettings.groups.get",
	//   "parameterOrder": [
	//     "groupUniqueId"
	//   ],
	//   "parameters": {
	//     "groupUniqueId": {
	//       "description": "The resource ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{groupUniqueId}",
	//   "response": {
	//     "$ref": "Groups"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.groups.settings"
	//   ]
	// }

}

// method id "groupsSettings.groups.patch":

type GroupsPatchCall struct {
	s             *Service
	groupUniqueId string
	groups        *Groups
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Patch: Updates an existing resource. This method supports patch
// semantics.
func (r *GroupsService) Patch(groupUniqueId string, groups *Groups) *GroupsPatchCall {
	c := &GroupsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.groupUniqueId = groupUniqueId
	c.groups = groups
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GroupsPatchCall) QuotaUser(quotaUser string) *GroupsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GroupsPatchCall) UserIP(userIP string) *GroupsPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsPatchCall) Fields(s ...googleapi.Field) *GroupsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsPatchCall) Context(ctx context.Context) *GroupsPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *GroupsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.groups)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{groupUniqueId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"groupUniqueId": c.groupUniqueId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "groupsSettings.groups.patch" call.
// Exactly one of *Groups or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Groups.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *GroupsPatchCall) Do() (*Groups, error) {
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
	ret := &Groups{
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
	//   "description": "Updates an existing resource. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "groupsSettings.groups.patch",
	//   "parameterOrder": [
	//     "groupUniqueId"
	//   ],
	//   "parameters": {
	//     "groupUniqueId": {
	//       "description": "The resource ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{groupUniqueId}",
	//   "request": {
	//     "$ref": "Groups"
	//   },
	//   "response": {
	//     "$ref": "Groups"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.groups.settings"
	//   ]
	// }

}

// method id "groupsSettings.groups.update":

type GroupsUpdateCall struct {
	s             *Service
	groupUniqueId string
	groups        *Groups
	urlParams_    gensupport.URLParams
	ctx_          context.Context
}

// Update: Updates an existing resource.
func (r *GroupsService) Update(groupUniqueId string, groups *Groups) *GroupsUpdateCall {
	c := &GroupsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.groupUniqueId = groupUniqueId
	c.groups = groups
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *GroupsUpdateCall) QuotaUser(quotaUser string) *GroupsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *GroupsUpdateCall) UserIP(userIP string) *GroupsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *GroupsUpdateCall) Fields(s ...googleapi.Field) *GroupsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *GroupsUpdateCall) Context(ctx context.Context) *GroupsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *GroupsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.groups)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{groupUniqueId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"groupUniqueId": c.groupUniqueId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "groupsSettings.groups.update" call.
// Exactly one of *Groups or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Groups.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *GroupsUpdateCall) Do() (*Groups, error) {
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
	ret := &Groups{
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
	//   "description": "Updates an existing resource.",
	//   "httpMethod": "PUT",
	//   "id": "groupsSettings.groups.update",
	//   "parameterOrder": [
	//     "groupUniqueId"
	//   ],
	//   "parameters": {
	//     "groupUniqueId": {
	//       "description": "The resource ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{groupUniqueId}",
	//   "request": {
	//     "$ref": "Groups"
	//   },
	//   "response": {
	//     "$ref": "Groups"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/apps.groups.settings"
	//   ]
	// }

}
