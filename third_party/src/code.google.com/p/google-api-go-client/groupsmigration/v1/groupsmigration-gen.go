// Package groupsmigration provides access to the Groups Migration API.
//
// See https://developers.google.com/google-apps/groups-migration/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/groupsmigration/v1"
//   ...
//   groupsmigrationService, err := groupsmigration.New(oauthHttpClient)
package groupsmigration

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

const apiId = "groupsmigration:v1"
const apiName = "groupsmigration"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/groups/v1/groups/"

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Archive = NewArchiveService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Archive *ArchiveService
}

func NewArchiveService(s *Service) *ArchiveService {
	rs := &ArchiveService{s: s}
	return rs
}

type ArchiveService struct {
	s *Service
}

type Groups struct {
	// Kind: The kind of insert resource this is.
	Kind string `json:"kind,omitempty"`

	// ResponseCode: The status of the insert request.
	ResponseCode string `json:"responseCode,omitempty"`
}

// method id "groupsmigration.archive.insert":

type ArchiveInsertCall struct {
	s       *Service
	groupId string
	opt_    map[string]interface{}
	media_  io.Reader
}

// Insert: Inserts a new mail into the archive of the Google group.
func (r *ArchiveService) Insert(groupId string) *ArchiveInsertCall {
	c := &ArchiveInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.groupId = groupId
	return c
}
func (c *ArchiveInsertCall) Media(r io.Reader) *ArchiveInsertCall {
	c.media_ = r
	return c
}

func (c *ArchiveInsertCall) Do() (*Groups, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "{groupId}/archive")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	body = new(bytes.Buffer)
	ctype := "application/json"
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{groupId}", url.QueryEscape(c.groupId), 1)
	googleapi.SetOpaque(req.URL)
	if hasMedia_ {
		req.ContentLength = contentLength_
	}
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
	//   "description": "Inserts a new mail into the archive of the Google group.",
	//   "httpMethod": "POST",
	//   "id": "groupsmigration.archive.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "message/rfc822"
	//     ],
	//     "maxSize": "16MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/groups/v1/groups/{groupId}/archive"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/groups/v1/groups/{groupId}/archive"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "groupId"
	//   ],
	//   "parameters": {
	//     "groupId": {
	//       "description": "The group ID",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{groupId}/archive",
	//   "response": {
	//     "$ref": "Groups"
	//   },
	//   "supportsMediaUpload": true
	// }

}
