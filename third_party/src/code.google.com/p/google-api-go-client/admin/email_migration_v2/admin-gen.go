// Package admin provides access to the Email Migration API v2.
//
// See https://developers.google.com/admin-sdk/email-migration/v2/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/admin/email_migration_v2"
//   ...
//   adminService, err := admin.New(oauthHttpClient)
package admin

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

const apiId = "admin:email_migration_v2"
const apiName = "admin"
const apiVersion = "email_migration_v2"
const basePath = "https://www.googleapis.com/email/v2/users/"

// OAuth2 scopes used by this API.
const (
	// Manage email messages of users on your domain
	EmailMigrationScope = "https://www.googleapis.com/auth/email.migration"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Mail = NewMailService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Mail *MailService
}

func NewMailService(s *Service) *MailService {
	rs := &MailService{s: s}
	return rs
}

type MailService struct {
	s *Service
}

type MailItem struct {
	// IsDeleted: Boolean indicating if the mail is deleted (used in Vault)
	IsDeleted bool `json:"isDeleted,omitempty"`

	// IsDraft: Boolean indicating if the mail is draft
	IsDraft bool `json:"isDraft,omitempty"`

	// IsInbox: Boolean indicating if the mail is in inbox
	IsInbox bool `json:"isInbox,omitempty"`

	// IsSent: Boolean indicating if the mail is in 'sent mails'
	IsSent bool `json:"isSent,omitempty"`

	// IsStarred: Boolean indicating if the mail is starred
	IsStarred bool `json:"isStarred,omitempty"`

	// IsTrash: Boolean indicating if the mail is in trash
	IsTrash bool `json:"isTrash,omitempty"`

	// IsUnread: Boolean indicating if the mail is unread
	IsUnread bool `json:"isUnread,omitempty"`

	// Kind: Kind of resource this is.
	Kind string `json:"kind,omitempty"`

	// Labels: List of labels (strings)
	Labels []string `json:"labels,omitempty"`
}

// method id "emailMigration.mail.insert":

type MailInsertCall struct {
	s        *Service
	userKey  string
	mailitem *MailItem
	opt_     map[string]interface{}
	media_   io.Reader
}

// Insert: Insert Mail into Google's Gmail backends
func (r *MailService) Insert(userKey string, mailitem *MailItem) *MailInsertCall {
	c := &MailInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.userKey = userKey
	c.mailitem = mailitem
	return c
}
func (c *MailInsertCall) Media(r io.Reader) *MailInsertCall {
	c.media_ = r
	return c
}

func (c *MailInsertCall) Do() error {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.mailitem)
	if err != nil {
		return err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userKey}/mail")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userKey}", url.QueryEscape(c.userKey), 1)
	googleapi.SetOpaque(req.URL)
	if hasMedia_ {
		req.ContentLength = contentLength_
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Insert Mail into Google's Gmail backends",
	//   "httpMethod": "POST",
	//   "id": "emailMigration.mail.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "message/rfc822"
	//     ],
	//     "maxSize": "35MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/email/v2/users/{userKey}/mail"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/email/v2/users/{userKey}/mail"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "userKey"
	//   ],
	//   "parameters": {
	//     "userKey": {
	//       "description": "The email or immutable id of the user",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userKey}/mail",
	//   "request": {
	//     "$ref": "MailItem"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/email.migration"
	//   ],
	//   "supportsMediaUpload": true
	// }

}
