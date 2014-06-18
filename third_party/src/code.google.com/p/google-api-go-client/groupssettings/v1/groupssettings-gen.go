// Package groupssettings provides access to the Groups Settings API.
//
// See https://developers.google.com/google-apps/groups-settings/get_started
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/groupssettings/v1"
//   ...
//   groupssettingsService, err := groupssettings.New(oauthHttpClient)
package groupssettings

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
	client   *http.Client
	BasePath string // API endpoint base URL

	Groups *GroupsService
}

func NewGroupsService(s *Service) *GroupsService {
	rs := &GroupsService{s: s}
	return rs
}

type GroupsService struct {
	s *Service
}

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
}

// method id "groupsSettings.groups.get":

type GroupsGetCall struct {
	s             *Service
	groupUniqueId string
	opt_          map[string]interface{}
}

// Get: Gets one resource by id.
func (r *GroupsService) Get(groupUniqueId string) *GroupsGetCall {
	c := &GroupsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.groupUniqueId = groupUniqueId
	return c
}

func (c *GroupsGetCall) Do() (*Groups, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{groupUniqueId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{groupUniqueId}", url.QueryEscape(c.groupUniqueId), 1)
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
	ret := new(Groups)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	opt_          map[string]interface{}
}

// Patch: Updates an existing resource. This method supports patch
// semantics.
func (r *GroupsService) Patch(groupUniqueId string, groups *Groups) *GroupsPatchCall {
	c := &GroupsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.groupUniqueId = groupUniqueId
	c.groups = groups
	return c
}

func (c *GroupsPatchCall) Do() (*Groups, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.groups)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{groupUniqueId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{groupUniqueId}", url.QueryEscape(c.groupUniqueId), 1)
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
	ret := new(Groups)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
	opt_          map[string]interface{}
}

// Update: Updates an existing resource.
func (r *GroupsService) Update(groupUniqueId string, groups *Groups) *GroupsUpdateCall {
	c := &GroupsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.groupUniqueId = groupUniqueId
	c.groups = groups
	return c
}

func (c *GroupsUpdateCall) Do() (*Groups, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.groups)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{groupUniqueId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{groupUniqueId}", url.QueryEscape(c.groupUniqueId), 1)
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
	ret := new(Groups)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
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
