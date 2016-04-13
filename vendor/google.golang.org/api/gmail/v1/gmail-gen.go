// Package gmail provides access to the Gmail API.
//
// See https://developers.google.com/gmail/api/
//
// Usage example:
//
//   import "google.golang.org/api/gmail/v1"
//   ...
//   gmailService, err := gmail.New(oauthHttpClient)
package gmail // import "google.golang.org/api/gmail/v1"

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

const apiId = "gmail:v1"
const apiName = "gmail"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/gmail/v1/users/"

// OAuth2 scopes used by this API.
const (
	// View and manage your mail
	MailGoogleComScope = "https://mail.google.com/"

	// Manage drafts and send emails
	GmailComposeScope = "https://www.googleapis.com/auth/gmail.compose"

	// Insert mail into your mailbox
	GmailInsertScope = "https://www.googleapis.com/auth/gmail.insert"

	// Manage mailbox labels
	GmailLabelsScope = "https://www.googleapis.com/auth/gmail.labels"

	// View and modify but not delete your email
	GmailModifyScope = "https://www.googleapis.com/auth/gmail.modify"

	// View your emails messages and settings
	GmailReadonlyScope = "https://www.googleapis.com/auth/gmail.readonly"

	// Send email on your behalf
	GmailSendScope = "https://www.googleapis.com/auth/gmail.send"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Users = NewUsersService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Users *UsersService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewUsersService(s *Service) *UsersService {
	rs := &UsersService{s: s}
	rs.Drafts = NewUsersDraftsService(s)
	rs.History = NewUsersHistoryService(s)
	rs.Labels = NewUsersLabelsService(s)
	rs.Messages = NewUsersMessagesService(s)
	rs.Threads = NewUsersThreadsService(s)
	return rs
}

type UsersService struct {
	s *Service

	Drafts *UsersDraftsService

	History *UsersHistoryService

	Labels *UsersLabelsService

	Messages *UsersMessagesService

	Threads *UsersThreadsService
}

func NewUsersDraftsService(s *Service) *UsersDraftsService {
	rs := &UsersDraftsService{s: s}
	return rs
}

type UsersDraftsService struct {
	s *Service
}

func NewUsersHistoryService(s *Service) *UsersHistoryService {
	rs := &UsersHistoryService{s: s}
	return rs
}

type UsersHistoryService struct {
	s *Service
}

func NewUsersLabelsService(s *Service) *UsersLabelsService {
	rs := &UsersLabelsService{s: s}
	return rs
}

type UsersLabelsService struct {
	s *Service
}

func NewUsersMessagesService(s *Service) *UsersMessagesService {
	rs := &UsersMessagesService{s: s}
	rs.Attachments = NewUsersMessagesAttachmentsService(s)
	return rs
}

type UsersMessagesService struct {
	s *Service

	Attachments *UsersMessagesAttachmentsService
}

func NewUsersMessagesAttachmentsService(s *Service) *UsersMessagesAttachmentsService {
	rs := &UsersMessagesAttachmentsService{s: s}
	return rs
}

type UsersMessagesAttachmentsService struct {
	s *Service
}

func NewUsersThreadsService(s *Service) *UsersThreadsService {
	rs := &UsersThreadsService{s: s}
	return rs
}

type UsersThreadsService struct {
	s *Service
}

// Draft: A draft email in the user's mailbox.
type Draft struct {
	// Id: The immutable ID of the draft.
	Id string `json:"id,omitempty"`

	// Message: The message content of the draft.
	Message *Message `json:"message,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Draft) MarshalJSON() ([]byte, error) {
	type noMethod Draft
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// History: A record of a change to the user's mailbox. Each history
// change may affect multiple messages in multiple ways.
type History struct {
	// Id: The mailbox sequence ID.
	Id uint64 `json:"id,omitempty,string"`

	// LabelsAdded: Labels added to messages in this history record.
	LabelsAdded []*HistoryLabelAdded `json:"labelsAdded,omitempty"`

	// LabelsRemoved: Labels removed from messages in this history record.
	LabelsRemoved []*HistoryLabelRemoved `json:"labelsRemoved,omitempty"`

	// Messages: List of messages changed in this history record. The fields
	// for specific change types, such as messagesAdded may duplicate
	// messages in this field. We recommend using the specific change-type
	// fields instead of this.
	Messages []*Message `json:"messages,omitempty"`

	// MessagesAdded: Messages added to the mailbox in this history record.
	MessagesAdded []*HistoryMessageAdded `json:"messagesAdded,omitempty"`

	// MessagesDeleted: Messages deleted (not Trashed) from the mailbox in
	// this history record.
	MessagesDeleted []*HistoryMessageDeleted `json:"messagesDeleted,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *History) MarshalJSON() ([]byte, error) {
	type noMethod History
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type HistoryLabelAdded struct {
	// LabelIds: Label IDs added to the message.
	LabelIds []string `json:"labelIds,omitempty"`

	Message *Message `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LabelIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *HistoryLabelAdded) MarshalJSON() ([]byte, error) {
	type noMethod HistoryLabelAdded
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type HistoryLabelRemoved struct {
	// LabelIds: Label IDs removed from the message.
	LabelIds []string `json:"labelIds,omitempty"`

	Message *Message `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LabelIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *HistoryLabelRemoved) MarshalJSON() ([]byte, error) {
	type noMethod HistoryLabelRemoved
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type HistoryMessageAdded struct {
	Message *Message `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Message") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *HistoryMessageAdded) MarshalJSON() ([]byte, error) {
	type noMethod HistoryMessageAdded
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type HistoryMessageDeleted struct {
	Message *Message `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Message") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *HistoryMessageDeleted) MarshalJSON() ([]byte, error) {
	type noMethod HistoryMessageDeleted
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Label: Labels are used to categorize messages and threads within the
// user's mailbox.
type Label struct {
	// Id: The immutable ID of the label.
	Id string `json:"id,omitempty"`

	// LabelListVisibility: The visibility of the label in the label list in
	// the Gmail web interface.
	//
	// Possible values:
	//   "labelHide"
	//   "labelShow"
	//   "labelShowIfUnread"
	LabelListVisibility string `json:"labelListVisibility,omitempty"`

	// MessageListVisibility: The visibility of the label in the message
	// list in the Gmail web interface.
	//
	// Possible values:
	//   "hide"
	//   "show"
	MessageListVisibility string `json:"messageListVisibility,omitempty"`

	// MessagesTotal: The total number of messages with the label.
	MessagesTotal int64 `json:"messagesTotal,omitempty"`

	// MessagesUnread: The number of unread messages with the label.
	MessagesUnread int64 `json:"messagesUnread,omitempty"`

	// Name: The display name of the label.
	Name string `json:"name,omitempty"`

	// ThreadsTotal: The total number of threads with the label.
	ThreadsTotal int64 `json:"threadsTotal,omitempty"`

	// ThreadsUnread: The number of unread threads with the label.
	ThreadsUnread int64 `json:"threadsUnread,omitempty"`

	// Type: The owner type for the label. User labels are created by the
	// user and can be modified and deleted by the user and can be applied
	// to any message or thread. System labels are internally created and
	// cannot be added, modified, or deleted. System labels may be able to
	// be applied to or removed from messages and threads under some
	// circumstances but this is not guaranteed. For example, users can
	// apply and remove the INBOX and UNREAD labels from messages and
	// threads, but cannot apply or remove the DRAFTS or SENT labels from
	// messages or threads.
	//
	// Possible values:
	//   "system"
	//   "user"
	Type string `json:"type,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Label) MarshalJSON() ([]byte, error) {
	type noMethod Label
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ListDraftsResponse struct {
	// Drafts: List of drafts.
	Drafts []*Draft `json:"drafts,omitempty"`

	// NextPageToken: Token to retrieve the next page of results in the
	// list.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ResultSizeEstimate: Estimated total number of results.
	ResultSizeEstimate int64 `json:"resultSizeEstimate,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Drafts") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListDraftsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListDraftsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ListHistoryResponse struct {
	// History: List of history records. Any messages contained in the
	// response will typically only have id and threadId fields populated.
	History []*History `json:"history,omitempty"`

	// HistoryId: The ID of the mailbox's current history record.
	HistoryId uint64 `json:"historyId,omitempty,string"`

	// NextPageToken: Page token to retrieve the next page of results in the
	// list.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "History") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListHistoryResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListHistoryResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ListLabelsResponse struct {
	// Labels: List of labels.
	Labels []*Label `json:"labels,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Labels") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListLabelsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListLabelsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ListMessagesResponse struct {
	// Messages: List of messages.
	Messages []*Message `json:"messages,omitempty"`

	// NextPageToken: Token to retrieve the next page of results in the
	// list.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ResultSizeEstimate: Estimated total number of results.
	ResultSizeEstimate int64 `json:"resultSizeEstimate,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Messages") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListMessagesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListMessagesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ListThreadsResponse struct {
	// NextPageToken: Page token to retrieve the next page of results in the
	// list.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ResultSizeEstimate: Estimated total number of results.
	ResultSizeEstimate int64 `json:"resultSizeEstimate,omitempty"`

	// Threads: List of threads.
	Threads []*Thread `json:"threads,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "NextPageToken") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListThreadsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListThreadsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Message: An email message.
type Message struct {
	// HistoryId: The ID of the last history record that modified this
	// message.
	HistoryId uint64 `json:"historyId,omitempty,string"`

	// Id: The immutable ID of the message.
	Id string `json:"id,omitempty"`

	// InternalDate: The internal message creation timestamp (epoch ms),
	// which determines ordering in the inbox. For normal SMTP-received
	// email, this represents the time the message was originally accepted
	// by Google, which is more reliable than the Date header. However, for
	// API-migrated mail, it can be configured by client to be based on the
	// Date header.
	InternalDate int64 `json:"internalDate,omitempty,string"`

	// LabelIds: List of IDs of labels applied to this message.
	LabelIds []string `json:"labelIds,omitempty"`

	// Payload: The parsed email structure in the message parts.
	Payload *MessagePart `json:"payload,omitempty"`

	// Raw: The entire email message in an RFC 2822 formatted and base64url
	// encoded string. Returned in messages.get and drafts.get responses
	// when the format=RAW parameter is supplied.
	Raw string `json:"raw,omitempty"`

	// SizeEstimate: Estimated size in bytes of the message.
	SizeEstimate int64 `json:"sizeEstimate,omitempty"`

	// Snippet: A short part of the message text.
	Snippet string `json:"snippet,omitempty"`

	// ThreadId: The ID of the thread the message belongs to. To add a
	// message or draft to a thread, the following criteria must be met:
	// - The requested threadId must be specified on the Message or
	// Draft.Message you supply with your request.
	// - The References and In-Reply-To headers must be set in compliance
	// with the RFC 2822 standard.
	// - The Subject headers must match.
	ThreadId string `json:"threadId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "HistoryId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Message) MarshalJSON() ([]byte, error) {
	type noMethod Message
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// MessagePart: A single MIME message part.
type MessagePart struct {
	// Body: The message part body for this part, which may be empty for
	// container MIME message parts.
	Body *MessagePartBody `json:"body,omitempty"`

	// Filename: The filename of the attachment. Only present if this
	// message part represents an attachment.
	Filename string `json:"filename,omitempty"`

	// Headers: List of headers on this message part. For the top-level
	// message part, representing the entire message payload, it will
	// contain the standard RFC 2822 email headers such as To, From, and
	// Subject.
	Headers []*MessagePartHeader `json:"headers,omitempty"`

	// MimeType: The MIME type of the message part.
	MimeType string `json:"mimeType,omitempty"`

	// PartId: The immutable ID of the message part.
	PartId string `json:"partId,omitempty"`

	// Parts: The child MIME message parts of this part. This only applies
	// to container MIME message parts, for example multipart/*. For non-
	// container MIME message part types, such as text/plain, this field is
	// empty. For more information, see RFC 1521.
	Parts []*MessagePart `json:"parts,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Body") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *MessagePart) MarshalJSON() ([]byte, error) {
	type noMethod MessagePart
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// MessagePartBody: The body of a single MIME message part.
type MessagePartBody struct {
	// AttachmentId: When present, contains the ID of an external attachment
	// that can be retrieved in a separate messages.attachments.get request.
	// When not present, the entire content of the message part body is
	// contained in the data field.
	AttachmentId string `json:"attachmentId,omitempty"`

	// Data: The body data of a MIME message part. May be empty for MIME
	// container types that have no message body or when the body data is
	// sent as a separate attachment. An attachment ID is present if the
	// body data is contained in a separate attachment.
	Data string `json:"data,omitempty"`

	// Size: Total number of bytes in the body of the message part.
	Size int64 `json:"size,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AttachmentId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *MessagePartBody) MarshalJSON() ([]byte, error) {
	type noMethod MessagePartBody
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type MessagePartHeader struct {
	// Name: The name of the header before the : separator. For example, To.
	Name string `json:"name,omitempty"`

	// Value: The value of the header after the : separator. For example,
	// someuser@example.com.
	Value string `json:"value,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Name") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *MessagePartHeader) MarshalJSON() ([]byte, error) {
	type noMethod MessagePartHeader
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ModifyMessageRequest struct {
	// AddLabelIds: A list of IDs of labels to add to this message.
	AddLabelIds []string `json:"addLabelIds,omitempty"`

	// RemoveLabelIds: A list IDs of labels to remove from this message.
	RemoveLabelIds []string `json:"removeLabelIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AddLabelIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ModifyMessageRequest) MarshalJSON() ([]byte, error) {
	type noMethod ModifyMessageRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

type ModifyThreadRequest struct {
	// AddLabelIds: A list of IDs of labels to add to this thread.
	AddLabelIds []string `json:"addLabelIds,omitempty"`

	// RemoveLabelIds: A list of IDs of labels to remove from this thread.
	RemoveLabelIds []string `json:"removeLabelIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AddLabelIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ModifyThreadRequest) MarshalJSON() ([]byte, error) {
	type noMethod ModifyThreadRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Profile: Profile for a Gmail user.
type Profile struct {
	// EmailAddress: The user's email address.
	EmailAddress string `json:"emailAddress,omitempty"`

	// HistoryId: The ID of the mailbox's current history record.
	HistoryId uint64 `json:"historyId,omitempty,string"`

	// MessagesTotal: The total number of messages in the mailbox.
	MessagesTotal int64 `json:"messagesTotal,omitempty"`

	// ThreadsTotal: The total number of threads in the mailbox.
	ThreadsTotal int64 `json:"threadsTotal,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "EmailAddress") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Profile) MarshalJSON() ([]byte, error) {
	type noMethod Profile
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Thread: A collection of messages representing a conversation.
type Thread struct {
	// HistoryId: The ID of the last history record that modified this
	// thread.
	HistoryId uint64 `json:"historyId,omitempty,string"`

	// Id: The unique ID of the thread.
	Id string `json:"id,omitempty"`

	// Messages: The list of messages in the thread.
	Messages []*Message `json:"messages,omitempty"`

	// Snippet: A short part of the message text.
	Snippet string `json:"snippet,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "HistoryId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Thread) MarshalJSON() ([]byte, error) {
	type noMethod Thread
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// WatchRequest: Set up or update a new push notification watch on this
// user's mailbox.
type WatchRequest struct {
	// LabelFilterAction: Filtering behavior of labelIds list specified.
	//
	// Possible values:
	//   "exclude"
	//   "include"
	LabelFilterAction string `json:"labelFilterAction,omitempty"`

	// LabelIds: List of label_ids to restrict notifications about. By
	// default, if unspecified, all changes are pushed out. If specified
	// then dictates which labels are required for a push notification to be
	// generated.
	LabelIds []string `json:"labelIds,omitempty"`

	// TopicName: A fully qualified Google Cloud Pub/Sub API topic name to
	// publish the events to. This topic name **must** already exist in
	// Cloud Pub/Sub and you **must** have already granted gmail "publish"
	// permission on it. For example,
	// "projects/my-project-identifier/topics/my-topic-name" (using the
	// Cloud Pub/Sub "v1" topic naming format).
	//
	// Note that the "my-project-identifier" portion must exactly match your
	// Google developer project id (the one executing this watch request).
	TopicName string `json:"topicName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "LabelFilterAction")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *WatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod WatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// WatchResponse: Push notification watch response.
type WatchResponse struct {
	// Expiration: When Gmail will stop sending notifications for mailbox
	// updates (epoch millis). Call watch again before this time to renew
	// the watch.
	Expiration int64 `json:"expiration,omitempty,string"`

	// HistoryId: The ID of the mailbox's current history record.
	HistoryId uint64 `json:"historyId,omitempty,string"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Expiration") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *WatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod WatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "gmail.users.getProfile":

type UsersGetProfileCall struct {
	s            *Service
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// GetProfile: Gets the current user's Gmail profile.
func (r *UsersService) GetProfile(userId string) *UsersGetProfileCall {
	c := &UsersGetProfileCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersGetProfileCall) QuotaUser(quotaUser string) *UsersGetProfileCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersGetProfileCall) UserIP(userIP string) *UsersGetProfileCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersGetProfileCall) Fields(s ...googleapi.Field) *UsersGetProfileCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersGetProfileCall) IfNoneMatch(entityTag string) *UsersGetProfileCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersGetProfileCall) Context(ctx context.Context) *UsersGetProfileCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersGetProfileCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/profile")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
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

// Do executes the "gmail.users.getProfile" call.
// Exactly one of *Profile or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Profile.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersGetProfileCall) Do() (*Profile, error) {
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
	ret := &Profile{
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
	//   "description": "Gets the current user's Gmail profile.",
	//   "httpMethod": "GET",
	//   "id": "gmail.users.getProfile",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/profile",
	//   "response": {
	//     "$ref": "Profile"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.compose",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.stop":

type UsersStopCall struct {
	s          *Service
	userId     string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Stop: Stop receiving push notifications for the given user mailbox.
func (r *UsersService) Stop(userId string) *UsersStopCall {
	c := &UsersStopCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersStopCall) QuotaUser(quotaUser string) *UsersStopCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersStopCall) UserIP(userIP string) *UsersStopCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersStopCall) Fields(s ...googleapi.Field) *UsersStopCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersStopCall) Context(ctx context.Context) *UsersStopCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersStopCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/stop")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.stop" call.
func (c *UsersStopCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Stop receiving push notifications for the given user mailbox.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.stop",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/stop",
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.watch":

type UsersWatchCall struct {
	s            *Service
	userId       string
	watchrequest *WatchRequest
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Watch: Set up or update a push notification watch on the given user
// mailbox.
func (r *UsersService) Watch(userId string, watchrequest *WatchRequest) *UsersWatchCall {
	c := &UsersWatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.watchrequest = watchrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersWatchCall) QuotaUser(quotaUser string) *UsersWatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersWatchCall) UserIP(userIP string) *UsersWatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersWatchCall) Fields(s ...googleapi.Field) *UsersWatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersWatchCall) Context(ctx context.Context) *UsersWatchCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersWatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.watchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/watch")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.watch" call.
// Exactly one of *WatchResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *WatchResponse.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersWatchCall) Do() (*WatchResponse, error) {
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
	ret := &WatchResponse{
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
	//   "description": "Set up or update a push notification watch on the given user mailbox.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.watch",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/watch",
	//   "request": {
	//     "$ref": "WatchRequest"
	//   },
	//   "response": {
	//     "$ref": "WatchResponse"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.drafts.create":

type UsersDraftsCreateCall struct {
	s                *Service
	userId           string
	draft            *Draft
	urlParams_       gensupport.URLParams
	media_           io.Reader
	resumable_       googleapi.SizeReaderAt
	mediaType_       string
	protocol_        string
	progressUpdater_ googleapi.ProgressUpdater
	ctx_             context.Context
}

// Create: Creates a new draft with the DRAFT label.
func (r *UsersDraftsService) Create(userId string, draft *Draft) *UsersDraftsCreateCall {
	c := &UsersDraftsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.draft = draft
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersDraftsCreateCall) QuotaUser(quotaUser string) *UsersDraftsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersDraftsCreateCall) UserIP(userIP string) *UsersDraftsCreateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Media specifies the media to upload in a single chunk. At most one of
// Media and ResumableMedia may be set.
func (c *UsersDraftsCreateCall) Media(r io.Reader) *UsersDraftsCreateCall {
	c.media_ = r
	c.protocol_ = "multipart"
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx. At most one of Media and ResumableMedia may be
// set. mediaType identifies the MIME media type of the upload, such as
// "image/png". If mediaType is "", it will be auto-detected. The
// provided ctx will supersede any context previously provided to the
// Context method.
func (c *UsersDraftsCreateCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *UsersDraftsCreateCall {
	c.ctx_ = ctx
	c.resumable_ = io.NewSectionReader(r, 0, size)
	c.mediaType_ = mediaType
	c.protocol_ = "resumable"
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *UsersDraftsCreateCall) ProgressUpdater(pu googleapi.ProgressUpdater) *UsersDraftsCreateCall {
	c.progressUpdater_ = pu
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDraftsCreateCall) Fields(s ...googleapi.Field) *UsersDraftsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *UsersDraftsCreateCall) Context(ctx context.Context) *UsersDraftsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersDraftsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.draft)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/drafts")
	if c.media_ != nil || c.resumable_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.protocol_)
	}
	urls += "?" + c.urlParams_.Encode()
	if c.protocol_ != "resumable" && c.media_ != nil {
		cancel := gensupport.IncludeMedia(c.media_, &body, &ctype)
		defer cancel()
	}
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	if c.protocol_ == "resumable" {
		if c.mediaType_ == "" {
			c.mediaType_ = gensupport.DetectMediaType(c.resumable_)
		}
		req.Header.Set("X-Upload-Content-Type", c.mediaType_)
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.drafts.create" call.
// Exactly one of *Draft or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Draft.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersDraftsCreateCall) Do() (*Draft, error) {
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
	if c.protocol_ == "resumable" {
		loc := res.Header.Get("Location")
		rx := &googleapi.ResumableUpload{
			Client:        c.s.client,
			UserAgent:     c.s.userAgent(),
			URI:           loc,
			Media:         c.resumable_,
			MediaType:     c.mediaType_,
			ContentLength: c.resumable_.Size(),
			Callback: func(curr int64) {
				if c.progressUpdater_ != nil {
					c.progressUpdater_(curr, c.resumable_.Size())
				}
			},
		}
		res, err = rx.Upload(c.ctx_)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
	}
	ret := &Draft{
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
	//   "description": "Creates a new draft with the DRAFT label.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.drafts.create",
	//   "mediaUpload": {
	//     "accept": [
	//       "message/rfc822"
	//     ],
	//     "maxSize": "35MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/gmail/v1/users/{userId}/drafts"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/gmail/v1/users/{userId}/drafts"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/drafts",
	//   "request": {
	//     "$ref": "Draft"
	//   },
	//   "response": {
	//     "$ref": "Draft"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.compose",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "gmail.users.drafts.delete":

type UsersDraftsDeleteCall struct {
	s          *Service
	userId     string
	id         string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Immediately and permanently deletes the specified draft. Does
// not simply trash it.
func (r *UsersDraftsService) Delete(userId string, id string) *UsersDraftsDeleteCall {
	c := &UsersDraftsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersDraftsDeleteCall) QuotaUser(quotaUser string) *UsersDraftsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersDraftsDeleteCall) UserIP(userIP string) *UsersDraftsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDraftsDeleteCall) Fields(s ...googleapi.Field) *UsersDraftsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDraftsDeleteCall) Context(ctx context.Context) *UsersDraftsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersDraftsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/drafts/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.drafts.delete" call.
func (c *UsersDraftsDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Immediately and permanently deletes the specified draft. Does not simply trash it.",
	//   "httpMethod": "DELETE",
	//   "id": "gmail.users.drafts.delete",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the draft to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/drafts/{id}",
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.compose",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ]
	// }

}

// method id "gmail.users.drafts.get":

type UsersDraftsGetCall struct {
	s            *Service
	userId       string
	id           string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the specified draft.
func (r *UsersDraftsService) Get(userId string, id string) *UsersDraftsGetCall {
	c := &UsersDraftsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	return c
}

// Format sets the optional parameter "format": The format to return the
// draft in.
//
// Possible values:
//   "full" (default)
//   "metadata"
//   "minimal"
//   "raw"
func (c *UsersDraftsGetCall) Format(format string) *UsersDraftsGetCall {
	c.urlParams_.Set("format", format)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersDraftsGetCall) QuotaUser(quotaUser string) *UsersDraftsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersDraftsGetCall) UserIP(userIP string) *UsersDraftsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDraftsGetCall) Fields(s ...googleapi.Field) *UsersDraftsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersDraftsGetCall) IfNoneMatch(entityTag string) *UsersDraftsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDraftsGetCall) Context(ctx context.Context) *UsersDraftsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersDraftsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/drafts/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
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

// Do executes the "gmail.users.drafts.get" call.
// Exactly one of *Draft or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Draft.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersDraftsGetCall) Do() (*Draft, error) {
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
	ret := &Draft{
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
	//   "description": "Gets the specified draft.",
	//   "httpMethod": "GET",
	//   "id": "gmail.users.drafts.get",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "format": {
	//       "default": "full",
	//       "description": "The format to return the draft in.",
	//       "enum": [
	//         "full",
	//         "metadata",
	//         "minimal",
	//         "raw"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "The ID of the draft to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/drafts/{id}",
	//   "response": {
	//     "$ref": "Draft"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.compose",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.drafts.list":

type UsersDraftsListCall struct {
	s            *Service
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the drafts in the user's mailbox.
func (r *UsersDraftsService) List(userId string) *UsersDraftsListCall {
	c := &UsersDraftsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of drafts to return.
func (c *UsersDraftsListCall) MaxResults(maxResults int64) *UsersDraftsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Page token to
// retrieve a specific page of results in the list.
func (c *UsersDraftsListCall) PageToken(pageToken string) *UsersDraftsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersDraftsListCall) QuotaUser(quotaUser string) *UsersDraftsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersDraftsListCall) UserIP(userIP string) *UsersDraftsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDraftsListCall) Fields(s ...googleapi.Field) *UsersDraftsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersDraftsListCall) IfNoneMatch(entityTag string) *UsersDraftsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersDraftsListCall) Context(ctx context.Context) *UsersDraftsListCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersDraftsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/drafts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
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

// Do executes the "gmail.users.drafts.list" call.
// Exactly one of *ListDraftsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListDraftsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersDraftsListCall) Do() (*ListDraftsResponse, error) {
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
	ret := &ListDraftsResponse{
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
	//   "description": "Lists the drafts in the user's mailbox.",
	//   "httpMethod": "GET",
	//   "id": "gmail.users.drafts.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Maximum number of drafts to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Page token to retrieve a specific page of results in the list.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/drafts",
	//   "response": {
	//     "$ref": "ListDraftsResponse"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.compose",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.drafts.send":

type UsersDraftsSendCall struct {
	s                *Service
	userId           string
	draft            *Draft
	urlParams_       gensupport.URLParams
	media_           io.Reader
	resumable_       googleapi.SizeReaderAt
	mediaType_       string
	protocol_        string
	progressUpdater_ googleapi.ProgressUpdater
	ctx_             context.Context
}

// Send: Sends the specified, existing draft to the recipients in the
// To, Cc, and Bcc headers.
func (r *UsersDraftsService) Send(userId string, draft *Draft) *UsersDraftsSendCall {
	c := &UsersDraftsSendCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.draft = draft
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersDraftsSendCall) QuotaUser(quotaUser string) *UsersDraftsSendCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersDraftsSendCall) UserIP(userIP string) *UsersDraftsSendCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Media specifies the media to upload in a single chunk. At most one of
// Media and ResumableMedia may be set.
func (c *UsersDraftsSendCall) Media(r io.Reader) *UsersDraftsSendCall {
	c.media_ = r
	c.protocol_ = "multipart"
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx. At most one of Media and ResumableMedia may be
// set. mediaType identifies the MIME media type of the upload, such as
// "image/png". If mediaType is "", it will be auto-detected. The
// provided ctx will supersede any context previously provided to the
// Context method.
func (c *UsersDraftsSendCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *UsersDraftsSendCall {
	c.ctx_ = ctx
	c.resumable_ = io.NewSectionReader(r, 0, size)
	c.mediaType_ = mediaType
	c.protocol_ = "resumable"
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *UsersDraftsSendCall) ProgressUpdater(pu googleapi.ProgressUpdater) *UsersDraftsSendCall {
	c.progressUpdater_ = pu
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDraftsSendCall) Fields(s ...googleapi.Field) *UsersDraftsSendCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *UsersDraftsSendCall) Context(ctx context.Context) *UsersDraftsSendCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersDraftsSendCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.draft)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/drafts/send")
	if c.media_ != nil || c.resumable_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.protocol_)
	}
	urls += "?" + c.urlParams_.Encode()
	if c.protocol_ != "resumable" && c.media_ != nil {
		cancel := gensupport.IncludeMedia(c.media_, &body, &ctype)
		defer cancel()
	}
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	if c.protocol_ == "resumable" {
		if c.mediaType_ == "" {
			c.mediaType_ = gensupport.DetectMediaType(c.resumable_)
		}
		req.Header.Set("X-Upload-Content-Type", c.mediaType_)
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.drafts.send" call.
// Exactly one of *Message or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Message.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersDraftsSendCall) Do() (*Message, error) {
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
	if c.protocol_ == "resumable" {
		loc := res.Header.Get("Location")
		rx := &googleapi.ResumableUpload{
			Client:        c.s.client,
			UserAgent:     c.s.userAgent(),
			URI:           loc,
			Media:         c.resumable_,
			MediaType:     c.mediaType_,
			ContentLength: c.resumable_.Size(),
			Callback: func(curr int64) {
				if c.progressUpdater_ != nil {
					c.progressUpdater_(curr, c.resumable_.Size())
				}
			},
		}
		res, err = rx.Upload(c.ctx_)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
	}
	ret := &Message{
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
	//   "description": "Sends the specified, existing draft to the recipients in the To, Cc, and Bcc headers.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.drafts.send",
	//   "mediaUpload": {
	//     "accept": [
	//       "message/rfc822"
	//     ],
	//     "maxSize": "35MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/gmail/v1/users/{userId}/drafts/send"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/gmail/v1/users/{userId}/drafts/send"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/drafts/send",
	//   "request": {
	//     "$ref": "Draft"
	//   },
	//   "response": {
	//     "$ref": "Message"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.compose",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "gmail.users.drafts.update":

type UsersDraftsUpdateCall struct {
	s                *Service
	userId           string
	id               string
	draft            *Draft
	urlParams_       gensupport.URLParams
	media_           io.Reader
	resumable_       googleapi.SizeReaderAt
	mediaType_       string
	protocol_        string
	progressUpdater_ googleapi.ProgressUpdater
	ctx_             context.Context
}

// Update: Replaces a draft's content.
func (r *UsersDraftsService) Update(userId string, id string, draft *Draft) *UsersDraftsUpdateCall {
	c := &UsersDraftsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	c.draft = draft
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersDraftsUpdateCall) QuotaUser(quotaUser string) *UsersDraftsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersDraftsUpdateCall) UserIP(userIP string) *UsersDraftsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Media specifies the media to upload in a single chunk. At most one of
// Media and ResumableMedia may be set.
func (c *UsersDraftsUpdateCall) Media(r io.Reader) *UsersDraftsUpdateCall {
	c.media_ = r
	c.protocol_ = "multipart"
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx. At most one of Media and ResumableMedia may be
// set. mediaType identifies the MIME media type of the upload, such as
// "image/png". If mediaType is "", it will be auto-detected. The
// provided ctx will supersede any context previously provided to the
// Context method.
func (c *UsersDraftsUpdateCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *UsersDraftsUpdateCall {
	c.ctx_ = ctx
	c.resumable_ = io.NewSectionReader(r, 0, size)
	c.mediaType_ = mediaType
	c.protocol_ = "resumable"
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *UsersDraftsUpdateCall) ProgressUpdater(pu googleapi.ProgressUpdater) *UsersDraftsUpdateCall {
	c.progressUpdater_ = pu
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersDraftsUpdateCall) Fields(s ...googleapi.Field) *UsersDraftsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *UsersDraftsUpdateCall) Context(ctx context.Context) *UsersDraftsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersDraftsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.draft)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/drafts/{id}")
	if c.media_ != nil || c.resumable_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.protocol_)
	}
	urls += "?" + c.urlParams_.Encode()
	if c.protocol_ != "resumable" && c.media_ != nil {
		cancel := gensupport.IncludeMedia(c.media_, &body, &ctype)
		defer cancel()
	}
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	if c.protocol_ == "resumable" {
		if c.mediaType_ == "" {
			c.mediaType_ = gensupport.DetectMediaType(c.resumable_)
		}
		req.Header.Set("X-Upload-Content-Type", c.mediaType_)
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.drafts.update" call.
// Exactly one of *Draft or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Draft.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersDraftsUpdateCall) Do() (*Draft, error) {
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
	if c.protocol_ == "resumable" {
		loc := res.Header.Get("Location")
		rx := &googleapi.ResumableUpload{
			Client:        c.s.client,
			UserAgent:     c.s.userAgent(),
			URI:           loc,
			Media:         c.resumable_,
			MediaType:     c.mediaType_,
			ContentLength: c.resumable_.Size(),
			Callback: func(curr int64) {
				if c.progressUpdater_ != nil {
					c.progressUpdater_(curr, c.resumable_.Size())
				}
			},
		}
		res, err = rx.Upload(c.ctx_)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
	}
	ret := &Draft{
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
	//   "description": "Replaces a draft's content.",
	//   "httpMethod": "PUT",
	//   "id": "gmail.users.drafts.update",
	//   "mediaUpload": {
	//     "accept": [
	//       "message/rfc822"
	//     ],
	//     "maxSize": "35MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/gmail/v1/users/{userId}/drafts/{id}"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/gmail/v1/users/{userId}/drafts/{id}"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the draft to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/drafts/{id}",
	//   "request": {
	//     "$ref": "Draft"
	//   },
	//   "response": {
	//     "$ref": "Draft"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.compose",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "gmail.users.history.list":

type UsersHistoryListCall struct {
	s            *Service
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the history of all changes to the given mailbox. History
// results are returned in chronological order (increasing historyId).
func (r *UsersHistoryService) List(userId string) *UsersHistoryListCall {
	c := &UsersHistoryListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	return c
}

// LabelId sets the optional parameter "labelId": Only return messages
// with a label matching the ID.
func (c *UsersHistoryListCall) LabelId(labelId string) *UsersHistoryListCall {
	c.urlParams_.Set("labelId", labelId)
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of history records to return.
func (c *UsersHistoryListCall) MaxResults(maxResults int64) *UsersHistoryListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Page token to
// retrieve a specific page of results in the list.
func (c *UsersHistoryListCall) PageToken(pageToken string) *UsersHistoryListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersHistoryListCall) QuotaUser(quotaUser string) *UsersHistoryListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// StartHistoryId sets the optional parameter "startHistoryId":
// Required. Returns history records after the specified startHistoryId.
// The supplied startHistoryId should be obtained from the historyId of
// a message, thread, or previous list response. History IDs increase
// chronologically but are not contiguous with random gaps in between
// valid IDs. Supplying an invalid or out of date startHistoryId
// typically returns an HTTP 404 error code. A historyId is typically
// valid for at least a week, but in some rare circumstances may be
// valid for only a few hours. If you receive an HTTP 404 error
// response, your application should perform a full sync. If you receive
// no nextPageToken in the response, there are no updates to retrieve
// and you can store the returned historyId for a future request.
func (c *UsersHistoryListCall) StartHistoryId(startHistoryId uint64) *UsersHistoryListCall {
	c.urlParams_.Set("startHistoryId", fmt.Sprint(startHistoryId))
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersHistoryListCall) UserIP(userIP string) *UsersHistoryListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersHistoryListCall) Fields(s ...googleapi.Field) *UsersHistoryListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersHistoryListCall) IfNoneMatch(entityTag string) *UsersHistoryListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersHistoryListCall) Context(ctx context.Context) *UsersHistoryListCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersHistoryListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/history")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
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

// Do executes the "gmail.users.history.list" call.
// Exactly one of *ListHistoryResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListHistoryResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersHistoryListCall) Do() (*ListHistoryResponse, error) {
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
	ret := &ListHistoryResponse{
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
	//   "description": "Lists the history of all changes to the given mailbox. History results are returned in chronological order (increasing historyId).",
	//   "httpMethod": "GET",
	//   "id": "gmail.users.history.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "labelId": {
	//       "description": "Only return messages with a label matching the ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "The maximum number of history records to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Page token to retrieve a specific page of results in the list.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startHistoryId": {
	//       "description": "Required. Returns history records after the specified startHistoryId. The supplied startHistoryId should be obtained from the historyId of a message, thread, or previous list response. History IDs increase chronologically but are not contiguous with random gaps in between valid IDs. Supplying an invalid or out of date startHistoryId typically returns an HTTP 404 error code. A historyId is typically valid for at least a week, but in some rare circumstances may be valid for only a few hours. If you receive an HTTP 404 error response, your application should perform a full sync. If you receive no nextPageToken in the response, there are no updates to retrieve and you can store the returned historyId for a future request.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/history",
	//   "response": {
	//     "$ref": "ListHistoryResponse"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.labels.create":

type UsersLabelsCreateCall struct {
	s          *Service
	userId     string
	label      *Label
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Creates a new label.
func (r *UsersLabelsService) Create(userId string, label *Label) *UsersLabelsCreateCall {
	c := &UsersLabelsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.label = label
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersLabelsCreateCall) QuotaUser(quotaUser string) *UsersLabelsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersLabelsCreateCall) UserIP(userIP string) *UsersLabelsCreateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersLabelsCreateCall) Fields(s ...googleapi.Field) *UsersLabelsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersLabelsCreateCall) Context(ctx context.Context) *UsersLabelsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersLabelsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.label)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/labels")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.labels.create" call.
// Exactly one of *Label or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Label.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersLabelsCreateCall) Do() (*Label, error) {
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
	ret := &Label{
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
	//   "description": "Creates a new label.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.labels.create",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/labels",
	//   "request": {
	//     "$ref": "Label"
	//   },
	//   "response": {
	//     "$ref": "Label"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.labels",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ]
	// }

}

// method id "gmail.users.labels.delete":

type UsersLabelsDeleteCall struct {
	s          *Service
	userId     string
	id         string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Immediately and permanently deletes the specified label and
// removes it from any messages and threads that it is applied to.
func (r *UsersLabelsService) Delete(userId string, id string) *UsersLabelsDeleteCall {
	c := &UsersLabelsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersLabelsDeleteCall) QuotaUser(quotaUser string) *UsersLabelsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersLabelsDeleteCall) UserIP(userIP string) *UsersLabelsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersLabelsDeleteCall) Fields(s ...googleapi.Field) *UsersLabelsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersLabelsDeleteCall) Context(ctx context.Context) *UsersLabelsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersLabelsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/labels/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.labels.delete" call.
func (c *UsersLabelsDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Immediately and permanently deletes the specified label and removes it from any messages and threads that it is applied to.",
	//   "httpMethod": "DELETE",
	//   "id": "gmail.users.labels.delete",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the label to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/labels/{id}",
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.labels",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ]
	// }

}

// method id "gmail.users.labels.get":

type UsersLabelsGetCall struct {
	s            *Service
	userId       string
	id           string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the specified label.
func (r *UsersLabelsService) Get(userId string, id string) *UsersLabelsGetCall {
	c := &UsersLabelsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersLabelsGetCall) QuotaUser(quotaUser string) *UsersLabelsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersLabelsGetCall) UserIP(userIP string) *UsersLabelsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersLabelsGetCall) Fields(s ...googleapi.Field) *UsersLabelsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersLabelsGetCall) IfNoneMatch(entityTag string) *UsersLabelsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersLabelsGetCall) Context(ctx context.Context) *UsersLabelsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersLabelsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/labels/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
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

// Do executes the "gmail.users.labels.get" call.
// Exactly one of *Label or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Label.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersLabelsGetCall) Do() (*Label, error) {
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
	ret := &Label{
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
	//   "description": "Gets the specified label.",
	//   "httpMethod": "GET",
	//   "id": "gmail.users.labels.get",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the label to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/labels/{id}",
	//   "response": {
	//     "$ref": "Label"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.labels",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.labels.list":

type UsersLabelsListCall struct {
	s            *Service
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists all labels in the user's mailbox.
func (r *UsersLabelsService) List(userId string) *UsersLabelsListCall {
	c := &UsersLabelsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersLabelsListCall) QuotaUser(quotaUser string) *UsersLabelsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersLabelsListCall) UserIP(userIP string) *UsersLabelsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersLabelsListCall) Fields(s ...googleapi.Field) *UsersLabelsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersLabelsListCall) IfNoneMatch(entityTag string) *UsersLabelsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersLabelsListCall) Context(ctx context.Context) *UsersLabelsListCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersLabelsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/labels")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
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

// Do executes the "gmail.users.labels.list" call.
// Exactly one of *ListLabelsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListLabelsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersLabelsListCall) Do() (*ListLabelsResponse, error) {
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
	ret := &ListLabelsResponse{
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
	//   "description": "Lists all labels in the user's mailbox.",
	//   "httpMethod": "GET",
	//   "id": "gmail.users.labels.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/labels",
	//   "response": {
	//     "$ref": "ListLabelsResponse"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.labels",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.labels.patch":

type UsersLabelsPatchCall struct {
	s          *Service
	userId     string
	id         string
	label      *Label
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Patch: Updates the specified label. This method supports patch
// semantics.
func (r *UsersLabelsService) Patch(userId string, id string, label *Label) *UsersLabelsPatchCall {
	c := &UsersLabelsPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	c.label = label
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersLabelsPatchCall) QuotaUser(quotaUser string) *UsersLabelsPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersLabelsPatchCall) UserIP(userIP string) *UsersLabelsPatchCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersLabelsPatchCall) Fields(s ...googleapi.Field) *UsersLabelsPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersLabelsPatchCall) Context(ctx context.Context) *UsersLabelsPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersLabelsPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.label)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/labels/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.labels.patch" call.
// Exactly one of *Label or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Label.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersLabelsPatchCall) Do() (*Label, error) {
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
	ret := &Label{
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
	//   "description": "Updates the specified label. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "gmail.users.labels.patch",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the label to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/labels/{id}",
	//   "request": {
	//     "$ref": "Label"
	//   },
	//   "response": {
	//     "$ref": "Label"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.labels",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ]
	// }

}

// method id "gmail.users.labels.update":

type UsersLabelsUpdateCall struct {
	s          *Service
	userId     string
	id         string
	label      *Label
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Updates the specified label.
func (r *UsersLabelsService) Update(userId string, id string, label *Label) *UsersLabelsUpdateCall {
	c := &UsersLabelsUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	c.label = label
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersLabelsUpdateCall) QuotaUser(quotaUser string) *UsersLabelsUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersLabelsUpdateCall) UserIP(userIP string) *UsersLabelsUpdateCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersLabelsUpdateCall) Fields(s ...googleapi.Field) *UsersLabelsUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersLabelsUpdateCall) Context(ctx context.Context) *UsersLabelsUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersLabelsUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.label)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/labels/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.labels.update" call.
// Exactly one of *Label or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Label.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersLabelsUpdateCall) Do() (*Label, error) {
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
	ret := &Label{
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
	//   "description": "Updates the specified label.",
	//   "httpMethod": "PUT",
	//   "id": "gmail.users.labels.update",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the label to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/labels/{id}",
	//   "request": {
	//     "$ref": "Label"
	//   },
	//   "response": {
	//     "$ref": "Label"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.labels",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ]
	// }

}

// method id "gmail.users.messages.delete":

type UsersMessagesDeleteCall struct {
	s          *Service
	userId     string
	id         string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Immediately and permanently deletes the specified message.
// This operation cannot be undone. Prefer messages.trash instead.
func (r *UsersMessagesService) Delete(userId string, id string) *UsersMessagesDeleteCall {
	c := &UsersMessagesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersMessagesDeleteCall) QuotaUser(quotaUser string) *UsersMessagesDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersMessagesDeleteCall) UserIP(userIP string) *UsersMessagesDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersMessagesDeleteCall) Fields(s ...googleapi.Field) *UsersMessagesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersMessagesDeleteCall) Context(ctx context.Context) *UsersMessagesDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersMessagesDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/messages/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.messages.delete" call.
func (c *UsersMessagesDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Immediately and permanently deletes the specified message. This operation cannot be undone. Prefer messages.trash instead.",
	//   "httpMethod": "DELETE",
	//   "id": "gmail.users.messages.delete",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the message to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/messages/{id}",
	//   "scopes": [
	//     "https://mail.google.com/"
	//   ]
	// }

}

// method id "gmail.users.messages.get":

type UsersMessagesGetCall struct {
	s            *Service
	userId       string
	id           string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the specified message.
func (r *UsersMessagesService) Get(userId string, id string) *UsersMessagesGetCall {
	c := &UsersMessagesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	return c
}

// Format sets the optional parameter "format": The format to return the
// message in.
//
// Possible values:
//   "full" (default)
//   "metadata"
//   "minimal"
//   "raw"
func (c *UsersMessagesGetCall) Format(format string) *UsersMessagesGetCall {
	c.urlParams_.Set("format", format)
	return c
}

// MetadataHeaders sets the optional parameter "metadataHeaders": When
// given and format is METADATA, only include headers specified.
func (c *UsersMessagesGetCall) MetadataHeaders(metadataHeaders ...string) *UsersMessagesGetCall {
	c.urlParams_.SetMulti("metadataHeaders", append([]string{}, metadataHeaders...))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersMessagesGetCall) QuotaUser(quotaUser string) *UsersMessagesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersMessagesGetCall) UserIP(userIP string) *UsersMessagesGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersMessagesGetCall) Fields(s ...googleapi.Field) *UsersMessagesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersMessagesGetCall) IfNoneMatch(entityTag string) *UsersMessagesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersMessagesGetCall) Context(ctx context.Context) *UsersMessagesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersMessagesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/messages/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
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

// Do executes the "gmail.users.messages.get" call.
// Exactly one of *Message or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Message.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersMessagesGetCall) Do() (*Message, error) {
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
	ret := &Message{
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
	//   "description": "Gets the specified message.",
	//   "httpMethod": "GET",
	//   "id": "gmail.users.messages.get",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "format": {
	//       "default": "full",
	//       "description": "The format to return the message in.",
	//       "enum": [
	//         "full",
	//         "metadata",
	//         "minimal",
	//         "raw"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "The ID of the message to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "metadataHeaders": {
	//       "description": "When given and format is METADATA, only include headers specified.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/messages/{id}",
	//   "response": {
	//     "$ref": "Message"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.messages.import":

type UsersMessagesImportCall struct {
	s                *Service
	userId           string
	message          *Message
	urlParams_       gensupport.URLParams
	media_           io.Reader
	resumable_       googleapi.SizeReaderAt
	mediaType_       string
	protocol_        string
	progressUpdater_ googleapi.ProgressUpdater
	ctx_             context.Context
}

// Import: Imports a message into only this user's mailbox, with
// standard email delivery scanning and classification similar to
// receiving via SMTP. Does not send a message.
func (r *UsersMessagesService) Import(userId string, message *Message) *UsersMessagesImportCall {
	c := &UsersMessagesImportCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.message = message
	return c
}

// Deleted sets the optional parameter "deleted": Mark the email as
// permanently deleted (not TRASH) and only visible in Google Apps Vault
// to a Vault administrator. Only used for Google Apps for Work
// accounts.
func (c *UsersMessagesImportCall) Deleted(deleted bool) *UsersMessagesImportCall {
	c.urlParams_.Set("deleted", fmt.Sprint(deleted))
	return c
}

// InternalDateSource sets the optional parameter "internalDateSource":
// Source for Gmail's internal date of the message.
//
// Possible values:
//   "dateHeader" (default)
//   "receivedTime"
func (c *UsersMessagesImportCall) InternalDateSource(internalDateSource string) *UsersMessagesImportCall {
	c.urlParams_.Set("internalDateSource", internalDateSource)
	return c
}

// NeverMarkSpam sets the optional parameter "neverMarkSpam": Ignore the
// Gmail spam classifier decision and never mark this email as SPAM in
// the mailbox.
func (c *UsersMessagesImportCall) NeverMarkSpam(neverMarkSpam bool) *UsersMessagesImportCall {
	c.urlParams_.Set("neverMarkSpam", fmt.Sprint(neverMarkSpam))
	return c
}

// ProcessForCalendar sets the optional parameter "processForCalendar":
// Process calendar invites in the email and add any extracted meetings
// to the Google Calendar for this user.
func (c *UsersMessagesImportCall) ProcessForCalendar(processForCalendar bool) *UsersMessagesImportCall {
	c.urlParams_.Set("processForCalendar", fmt.Sprint(processForCalendar))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersMessagesImportCall) QuotaUser(quotaUser string) *UsersMessagesImportCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersMessagesImportCall) UserIP(userIP string) *UsersMessagesImportCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Media specifies the media to upload in a single chunk. At most one of
// Media and ResumableMedia may be set.
func (c *UsersMessagesImportCall) Media(r io.Reader) *UsersMessagesImportCall {
	c.media_ = r
	c.protocol_ = "multipart"
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx. At most one of Media and ResumableMedia may be
// set. mediaType identifies the MIME media type of the upload, such as
// "image/png". If mediaType is "", it will be auto-detected. The
// provided ctx will supersede any context previously provided to the
// Context method.
func (c *UsersMessagesImportCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *UsersMessagesImportCall {
	c.ctx_ = ctx
	c.resumable_ = io.NewSectionReader(r, 0, size)
	c.mediaType_ = mediaType
	c.protocol_ = "resumable"
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *UsersMessagesImportCall) ProgressUpdater(pu googleapi.ProgressUpdater) *UsersMessagesImportCall {
	c.progressUpdater_ = pu
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersMessagesImportCall) Fields(s ...googleapi.Field) *UsersMessagesImportCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *UsersMessagesImportCall) Context(ctx context.Context) *UsersMessagesImportCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersMessagesImportCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.message)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/messages/import")
	if c.media_ != nil || c.resumable_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.protocol_)
	}
	urls += "?" + c.urlParams_.Encode()
	if c.protocol_ != "resumable" && c.media_ != nil {
		cancel := gensupport.IncludeMedia(c.media_, &body, &ctype)
		defer cancel()
	}
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	if c.protocol_ == "resumable" {
		if c.mediaType_ == "" {
			c.mediaType_ = gensupport.DetectMediaType(c.resumable_)
		}
		req.Header.Set("X-Upload-Content-Type", c.mediaType_)
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.messages.import" call.
// Exactly one of *Message or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Message.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersMessagesImportCall) Do() (*Message, error) {
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
	if c.protocol_ == "resumable" {
		loc := res.Header.Get("Location")
		rx := &googleapi.ResumableUpload{
			Client:        c.s.client,
			UserAgent:     c.s.userAgent(),
			URI:           loc,
			Media:         c.resumable_,
			MediaType:     c.mediaType_,
			ContentLength: c.resumable_.Size(),
			Callback: func(curr int64) {
				if c.progressUpdater_ != nil {
					c.progressUpdater_(curr, c.resumable_.Size())
				}
			},
		}
		res, err = rx.Upload(c.ctx_)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
	}
	ret := &Message{
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
	//   "description": "Imports a message into only this user's mailbox, with standard email delivery scanning and classification similar to receiving via SMTP. Does not send a message.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.messages.import",
	//   "mediaUpload": {
	//     "accept": [
	//       "message/rfc822"
	//     ],
	//     "maxSize": "35MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/gmail/v1/users/{userId}/messages/import"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/gmail/v1/users/{userId}/messages/import"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "deleted": {
	//       "default": "false",
	//       "description": "Mark the email as permanently deleted (not TRASH) and only visible in Google Apps Vault to a Vault administrator. Only used for Google Apps for Work accounts.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "internalDateSource": {
	//       "default": "dateHeader",
	//       "description": "Source for Gmail's internal date of the message.",
	//       "enum": [
	//         "dateHeader",
	//         "receivedTime"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "neverMarkSpam": {
	//       "default": "false",
	//       "description": "Ignore the Gmail spam classifier decision and never mark this email as SPAM in the mailbox.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "processForCalendar": {
	//       "default": "false",
	//       "description": "Process calendar invites in the email and add any extracted meetings to the Google Calendar for this user.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/messages/import",
	//   "request": {
	//     "$ref": "Message"
	//   },
	//   "response": {
	//     "$ref": "Message"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.insert",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "gmail.users.messages.insert":

type UsersMessagesInsertCall struct {
	s                *Service
	userId           string
	message          *Message
	urlParams_       gensupport.URLParams
	media_           io.Reader
	resumable_       googleapi.SizeReaderAt
	mediaType_       string
	protocol_        string
	progressUpdater_ googleapi.ProgressUpdater
	ctx_             context.Context
}

// Insert: Directly inserts a message into only this user's mailbox
// similar to IMAP APPEND, bypassing most scanning and classification.
// Does not send a message.
func (r *UsersMessagesService) Insert(userId string, message *Message) *UsersMessagesInsertCall {
	c := &UsersMessagesInsertCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.message = message
	return c
}

// Deleted sets the optional parameter "deleted": Mark the email as
// permanently deleted (not TRASH) and only visible in Google Apps Vault
// to a Vault administrator. Only used for Google Apps for Work
// accounts.
func (c *UsersMessagesInsertCall) Deleted(deleted bool) *UsersMessagesInsertCall {
	c.urlParams_.Set("deleted", fmt.Sprint(deleted))
	return c
}

// InternalDateSource sets the optional parameter "internalDateSource":
// Source for Gmail's internal date of the message.
//
// Possible values:
//   "dateHeader"
//   "receivedTime" (default)
func (c *UsersMessagesInsertCall) InternalDateSource(internalDateSource string) *UsersMessagesInsertCall {
	c.urlParams_.Set("internalDateSource", internalDateSource)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersMessagesInsertCall) QuotaUser(quotaUser string) *UsersMessagesInsertCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersMessagesInsertCall) UserIP(userIP string) *UsersMessagesInsertCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Media specifies the media to upload in a single chunk. At most one of
// Media and ResumableMedia may be set.
func (c *UsersMessagesInsertCall) Media(r io.Reader) *UsersMessagesInsertCall {
	c.media_ = r
	c.protocol_ = "multipart"
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx. At most one of Media and ResumableMedia may be
// set. mediaType identifies the MIME media type of the upload, such as
// "image/png". If mediaType is "", it will be auto-detected. The
// provided ctx will supersede any context previously provided to the
// Context method.
func (c *UsersMessagesInsertCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *UsersMessagesInsertCall {
	c.ctx_ = ctx
	c.resumable_ = io.NewSectionReader(r, 0, size)
	c.mediaType_ = mediaType
	c.protocol_ = "resumable"
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *UsersMessagesInsertCall) ProgressUpdater(pu googleapi.ProgressUpdater) *UsersMessagesInsertCall {
	c.progressUpdater_ = pu
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersMessagesInsertCall) Fields(s ...googleapi.Field) *UsersMessagesInsertCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *UsersMessagesInsertCall) Context(ctx context.Context) *UsersMessagesInsertCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersMessagesInsertCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.message)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/messages")
	if c.media_ != nil || c.resumable_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.protocol_)
	}
	urls += "?" + c.urlParams_.Encode()
	if c.protocol_ != "resumable" && c.media_ != nil {
		cancel := gensupport.IncludeMedia(c.media_, &body, &ctype)
		defer cancel()
	}
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	if c.protocol_ == "resumable" {
		if c.mediaType_ == "" {
			c.mediaType_ = gensupport.DetectMediaType(c.resumable_)
		}
		req.Header.Set("X-Upload-Content-Type", c.mediaType_)
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.messages.insert" call.
// Exactly one of *Message or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Message.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersMessagesInsertCall) Do() (*Message, error) {
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
	if c.protocol_ == "resumable" {
		loc := res.Header.Get("Location")
		rx := &googleapi.ResumableUpload{
			Client:        c.s.client,
			UserAgent:     c.s.userAgent(),
			URI:           loc,
			Media:         c.resumable_,
			MediaType:     c.mediaType_,
			ContentLength: c.resumable_.Size(),
			Callback: func(curr int64) {
				if c.progressUpdater_ != nil {
					c.progressUpdater_(curr, c.resumable_.Size())
				}
			},
		}
		res, err = rx.Upload(c.ctx_)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
	}
	ret := &Message{
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
	//   "description": "Directly inserts a message into only this user's mailbox similar to IMAP APPEND, bypassing most scanning and classification. Does not send a message.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.messages.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "message/rfc822"
	//     ],
	//     "maxSize": "35MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/gmail/v1/users/{userId}/messages"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/gmail/v1/users/{userId}/messages"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "deleted": {
	//       "default": "false",
	//       "description": "Mark the email as permanently deleted (not TRASH) and only visible in Google Apps Vault to a Vault administrator. Only used for Google Apps for Work accounts.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "internalDateSource": {
	//       "default": "receivedTime",
	//       "description": "Source for Gmail's internal date of the message.",
	//       "enum": [
	//         "dateHeader",
	//         "receivedTime"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/messages",
	//   "request": {
	//     "$ref": "Message"
	//   },
	//   "response": {
	//     "$ref": "Message"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.insert",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "gmail.users.messages.list":

type UsersMessagesListCall struct {
	s            *Service
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the messages in the user's mailbox.
func (r *UsersMessagesService) List(userId string) *UsersMessagesListCall {
	c := &UsersMessagesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	return c
}

// IncludeSpamTrash sets the optional parameter "includeSpamTrash":
// Include messages from SPAM and TRASH in the results.
func (c *UsersMessagesListCall) IncludeSpamTrash(includeSpamTrash bool) *UsersMessagesListCall {
	c.urlParams_.Set("includeSpamTrash", fmt.Sprint(includeSpamTrash))
	return c
}

// LabelIds sets the optional parameter "labelIds": Only return messages
// with labels that match all of the specified label IDs.
func (c *UsersMessagesListCall) LabelIds(labelIds ...string) *UsersMessagesListCall {
	c.urlParams_.SetMulti("labelIds", append([]string{}, labelIds...))
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of messages to return.
func (c *UsersMessagesListCall) MaxResults(maxResults int64) *UsersMessagesListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Page token to
// retrieve a specific page of results in the list.
func (c *UsersMessagesListCall) PageToken(pageToken string) *UsersMessagesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Q sets the optional parameter "q": Only return messages matching the
// specified query. Supports the same query format as the Gmail search
// box. For example, "from:someuser@example.com rfc822msgid: is:unread".
func (c *UsersMessagesListCall) Q(q string) *UsersMessagesListCall {
	c.urlParams_.Set("q", q)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersMessagesListCall) QuotaUser(quotaUser string) *UsersMessagesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersMessagesListCall) UserIP(userIP string) *UsersMessagesListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersMessagesListCall) Fields(s ...googleapi.Field) *UsersMessagesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersMessagesListCall) IfNoneMatch(entityTag string) *UsersMessagesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersMessagesListCall) Context(ctx context.Context) *UsersMessagesListCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersMessagesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/messages")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
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

// Do executes the "gmail.users.messages.list" call.
// Exactly one of *ListMessagesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListMessagesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersMessagesListCall) Do() (*ListMessagesResponse, error) {
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
	ret := &ListMessagesResponse{
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
	//   "description": "Lists the messages in the user's mailbox.",
	//   "httpMethod": "GET",
	//   "id": "gmail.users.messages.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "includeSpamTrash": {
	//       "default": "false",
	//       "description": "Include messages from SPAM and TRASH in the results.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "labelIds": {
	//       "description": "Only return messages with labels that match all of the specified label IDs.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Maximum number of messages to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Page token to retrieve a specific page of results in the list.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "q": {
	//       "description": "Only return messages matching the specified query. Supports the same query format as the Gmail search box. For example, \"from:someuser@example.com rfc822msgid: is:unread\".",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/messages",
	//   "response": {
	//     "$ref": "ListMessagesResponse"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.messages.modify":

type UsersMessagesModifyCall struct {
	s                    *Service
	userId               string
	id                   string
	modifymessagerequest *ModifyMessageRequest
	urlParams_           gensupport.URLParams
	ctx_                 context.Context
}

// Modify: Modifies the labels on the specified message.
func (r *UsersMessagesService) Modify(userId string, id string, modifymessagerequest *ModifyMessageRequest) *UsersMessagesModifyCall {
	c := &UsersMessagesModifyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	c.modifymessagerequest = modifymessagerequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersMessagesModifyCall) QuotaUser(quotaUser string) *UsersMessagesModifyCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersMessagesModifyCall) UserIP(userIP string) *UsersMessagesModifyCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersMessagesModifyCall) Fields(s ...googleapi.Field) *UsersMessagesModifyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersMessagesModifyCall) Context(ctx context.Context) *UsersMessagesModifyCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersMessagesModifyCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.modifymessagerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/messages/{id}/modify")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.messages.modify" call.
// Exactly one of *Message or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Message.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersMessagesModifyCall) Do() (*Message, error) {
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
	ret := &Message{
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
	//   "description": "Modifies the labels on the specified message.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.messages.modify",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the message to modify.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/messages/{id}/modify",
	//   "request": {
	//     "$ref": "ModifyMessageRequest"
	//   },
	//   "response": {
	//     "$ref": "Message"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ]
	// }

}

// method id "gmail.users.messages.send":

type UsersMessagesSendCall struct {
	s                *Service
	userId           string
	message          *Message
	urlParams_       gensupport.URLParams
	media_           io.Reader
	resumable_       googleapi.SizeReaderAt
	mediaType_       string
	protocol_        string
	progressUpdater_ googleapi.ProgressUpdater
	ctx_             context.Context
}

// Send: Sends the specified message to the recipients in the To, Cc,
// and Bcc headers.
func (r *UsersMessagesService) Send(userId string, message *Message) *UsersMessagesSendCall {
	c := &UsersMessagesSendCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.message = message
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersMessagesSendCall) QuotaUser(quotaUser string) *UsersMessagesSendCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersMessagesSendCall) UserIP(userIP string) *UsersMessagesSendCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Media specifies the media to upload in a single chunk. At most one of
// Media and ResumableMedia may be set.
func (c *UsersMessagesSendCall) Media(r io.Reader) *UsersMessagesSendCall {
	c.media_ = r
	c.protocol_ = "multipart"
	return c
}

// ResumableMedia specifies the media to upload in chunks and can be
// canceled with ctx. At most one of Media and ResumableMedia may be
// set. mediaType identifies the MIME media type of the upload, such as
// "image/png". If mediaType is "", it will be auto-detected. The
// provided ctx will supersede any context previously provided to the
// Context method.
func (c *UsersMessagesSendCall) ResumableMedia(ctx context.Context, r io.ReaderAt, size int64, mediaType string) *UsersMessagesSendCall {
	c.ctx_ = ctx
	c.resumable_ = io.NewSectionReader(r, 0, size)
	c.mediaType_ = mediaType
	c.protocol_ = "resumable"
	return c
}

// ProgressUpdater provides a callback function that will be called
// after every chunk. It should be a low-latency function in order to
// not slow down the upload operation. This should only be called when
// using ResumableMedia (as opposed to Media).
func (c *UsersMessagesSendCall) ProgressUpdater(pu googleapi.ProgressUpdater) *UsersMessagesSendCall {
	c.progressUpdater_ = pu
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersMessagesSendCall) Fields(s ...googleapi.Field) *UsersMessagesSendCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
// This context will supersede any context previously provided to the
// ResumableMedia method.
func (c *UsersMessagesSendCall) Context(ctx context.Context) *UsersMessagesSendCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersMessagesSendCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.message)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/messages/send")
	if c.media_ != nil || c.resumable_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		c.urlParams_.Set("uploadType", c.protocol_)
	}
	urls += "?" + c.urlParams_.Encode()
	if c.protocol_ != "resumable" && c.media_ != nil {
		cancel := gensupport.IncludeMedia(c.media_, &body, &ctype)
		defer cancel()
	}
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	if c.protocol_ == "resumable" {
		if c.mediaType_ == "" {
			c.mediaType_ = gensupport.DetectMediaType(c.resumable_)
		}
		req.Header.Set("X-Upload-Content-Type", c.mediaType_)
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.messages.send" call.
// Exactly one of *Message or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Message.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersMessagesSendCall) Do() (*Message, error) {
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
	if c.protocol_ == "resumable" {
		loc := res.Header.Get("Location")
		rx := &googleapi.ResumableUpload{
			Client:        c.s.client,
			UserAgent:     c.s.userAgent(),
			URI:           loc,
			Media:         c.resumable_,
			MediaType:     c.mediaType_,
			ContentLength: c.resumable_.Size(),
			Callback: func(curr int64) {
				if c.progressUpdater_ != nil {
					c.progressUpdater_(curr, c.resumable_.Size())
				}
			},
		}
		res, err = rx.Upload(c.ctx_)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
	}
	ret := &Message{
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
	//   "description": "Sends the specified message to the recipients in the To, Cc, and Bcc headers.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.messages.send",
	//   "mediaUpload": {
	//     "accept": [
	//       "message/rfc822"
	//     ],
	//     "maxSize": "35MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/gmail/v1/users/{userId}/messages/send"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/gmail/v1/users/{userId}/messages/send"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/messages/send",
	//   "request": {
	//     "$ref": "Message"
	//   },
	//   "response": {
	//     "$ref": "Message"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.compose",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.send"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "gmail.users.messages.trash":

type UsersMessagesTrashCall struct {
	s          *Service
	userId     string
	id         string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Trash: Moves the specified message to the trash.
func (r *UsersMessagesService) Trash(userId string, id string) *UsersMessagesTrashCall {
	c := &UsersMessagesTrashCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersMessagesTrashCall) QuotaUser(quotaUser string) *UsersMessagesTrashCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersMessagesTrashCall) UserIP(userIP string) *UsersMessagesTrashCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersMessagesTrashCall) Fields(s ...googleapi.Field) *UsersMessagesTrashCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersMessagesTrashCall) Context(ctx context.Context) *UsersMessagesTrashCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersMessagesTrashCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/messages/{id}/trash")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.messages.trash" call.
// Exactly one of *Message or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Message.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersMessagesTrashCall) Do() (*Message, error) {
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
	ret := &Message{
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
	//   "description": "Moves the specified message to the trash.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.messages.trash",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the message to Trash.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/messages/{id}/trash",
	//   "response": {
	//     "$ref": "Message"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ]
	// }

}

// method id "gmail.users.messages.untrash":

type UsersMessagesUntrashCall struct {
	s          *Service
	userId     string
	id         string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Untrash: Removes the specified message from the trash.
func (r *UsersMessagesService) Untrash(userId string, id string) *UsersMessagesUntrashCall {
	c := &UsersMessagesUntrashCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersMessagesUntrashCall) QuotaUser(quotaUser string) *UsersMessagesUntrashCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersMessagesUntrashCall) UserIP(userIP string) *UsersMessagesUntrashCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersMessagesUntrashCall) Fields(s ...googleapi.Field) *UsersMessagesUntrashCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersMessagesUntrashCall) Context(ctx context.Context) *UsersMessagesUntrashCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersMessagesUntrashCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/messages/{id}/untrash")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.messages.untrash" call.
// Exactly one of *Message or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Message.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersMessagesUntrashCall) Do() (*Message, error) {
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
	ret := &Message{
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
	//   "description": "Removes the specified message from the trash.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.messages.untrash",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the message to remove from Trash.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/messages/{id}/untrash",
	//   "response": {
	//     "$ref": "Message"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ]
	// }

}

// method id "gmail.users.messages.attachments.get":

type UsersMessagesAttachmentsGetCall struct {
	s            *Service
	userId       string
	messageId    string
	id           string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the specified message attachment.
func (r *UsersMessagesAttachmentsService) Get(userId string, messageId string, id string) *UsersMessagesAttachmentsGetCall {
	c := &UsersMessagesAttachmentsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.messageId = messageId
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersMessagesAttachmentsGetCall) QuotaUser(quotaUser string) *UsersMessagesAttachmentsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersMessagesAttachmentsGetCall) UserIP(userIP string) *UsersMessagesAttachmentsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersMessagesAttachmentsGetCall) Fields(s ...googleapi.Field) *UsersMessagesAttachmentsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersMessagesAttachmentsGetCall) IfNoneMatch(entityTag string) *UsersMessagesAttachmentsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersMessagesAttachmentsGetCall) Context(ctx context.Context) *UsersMessagesAttachmentsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersMessagesAttachmentsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/messages/{messageId}/attachments/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId":    c.userId,
		"messageId": c.messageId,
		"id":        c.id,
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

// Do executes the "gmail.users.messages.attachments.get" call.
// Exactly one of *MessagePartBody or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *MessagePartBody.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersMessagesAttachmentsGetCall) Do() (*MessagePartBody, error) {
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
	ret := &MessagePartBody{
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
	//   "description": "Gets the specified message attachment.",
	//   "httpMethod": "GET",
	//   "id": "gmail.users.messages.attachments.get",
	//   "parameterOrder": [
	//     "userId",
	//     "messageId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the attachment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "messageId": {
	//       "description": "The ID of the message containing the attachment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/messages/{messageId}/attachments/{id}",
	//   "response": {
	//     "$ref": "MessagePartBody"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.threads.delete":

type UsersThreadsDeleteCall struct {
	s          *Service
	userId     string
	id         string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Immediately and permanently deletes the specified thread.
// This operation cannot be undone. Prefer threads.trash instead.
func (r *UsersThreadsService) Delete(userId string, id string) *UsersThreadsDeleteCall {
	c := &UsersThreadsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersThreadsDeleteCall) QuotaUser(quotaUser string) *UsersThreadsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersThreadsDeleteCall) UserIP(userIP string) *UsersThreadsDeleteCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersThreadsDeleteCall) Fields(s ...googleapi.Field) *UsersThreadsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersThreadsDeleteCall) Context(ctx context.Context) *UsersThreadsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersThreadsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/threads/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.threads.delete" call.
func (c *UsersThreadsDeleteCall) Do() error {
	res, err := c.doRequest("json")
	if err != nil {
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Immediately and permanently deletes the specified thread. This operation cannot be undone. Prefer threads.trash instead.",
	//   "httpMethod": "DELETE",
	//   "id": "gmail.users.threads.delete",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "ID of the Thread to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/threads/{id}",
	//   "scopes": [
	//     "https://mail.google.com/"
	//   ]
	// }

}

// method id "gmail.users.threads.get":

type UsersThreadsGetCall struct {
	s            *Service
	userId       string
	id           string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the specified thread.
func (r *UsersThreadsService) Get(userId string, id string) *UsersThreadsGetCall {
	c := &UsersThreadsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	return c
}

// Format sets the optional parameter "format": The format to return the
// messages in.
//
// Possible values:
//   "full" (default)
//   "metadata"
//   "minimal"
func (c *UsersThreadsGetCall) Format(format string) *UsersThreadsGetCall {
	c.urlParams_.Set("format", format)
	return c
}

// MetadataHeaders sets the optional parameter "metadataHeaders": When
// given and format is METADATA, only include headers specified.
func (c *UsersThreadsGetCall) MetadataHeaders(metadataHeaders ...string) *UsersThreadsGetCall {
	c.urlParams_.SetMulti("metadataHeaders", append([]string{}, metadataHeaders...))
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersThreadsGetCall) QuotaUser(quotaUser string) *UsersThreadsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersThreadsGetCall) UserIP(userIP string) *UsersThreadsGetCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersThreadsGetCall) Fields(s ...googleapi.Field) *UsersThreadsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersThreadsGetCall) IfNoneMatch(entityTag string) *UsersThreadsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersThreadsGetCall) Context(ctx context.Context) *UsersThreadsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersThreadsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/threads/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
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

// Do executes the "gmail.users.threads.get" call.
// Exactly one of *Thread or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Thread.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersThreadsGetCall) Do() (*Thread, error) {
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
	ret := &Thread{
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
	//   "description": "Gets the specified thread.",
	//   "httpMethod": "GET",
	//   "id": "gmail.users.threads.get",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "format": {
	//       "default": "full",
	//       "description": "The format to return the messages in.",
	//       "enum": [
	//         "full",
	//         "metadata",
	//         "minimal"
	//       ],
	//       "enumDescriptions": [
	//         "",
	//         "",
	//         ""
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "The ID of the thread to retrieve.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "metadataHeaders": {
	//       "description": "When given and format is METADATA, only include headers specified.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/threads/{id}",
	//   "response": {
	//     "$ref": "Thread"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.threads.list":

type UsersThreadsListCall struct {
	s            *Service
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the threads in the user's mailbox.
func (r *UsersThreadsService) List(userId string) *UsersThreadsListCall {
	c := &UsersThreadsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	return c
}

// IncludeSpamTrash sets the optional parameter "includeSpamTrash":
// Include threads from SPAM and TRASH in the results.
func (c *UsersThreadsListCall) IncludeSpamTrash(includeSpamTrash bool) *UsersThreadsListCall {
	c.urlParams_.Set("includeSpamTrash", fmt.Sprint(includeSpamTrash))
	return c
}

// LabelIds sets the optional parameter "labelIds": Only return threads
// with labels that match all of the specified label IDs.
func (c *UsersThreadsListCall) LabelIds(labelIds ...string) *UsersThreadsListCall {
	c.urlParams_.SetMulti("labelIds", append([]string{}, labelIds...))
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of threads to return.
func (c *UsersThreadsListCall) MaxResults(maxResults int64) *UsersThreadsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Page token to
// retrieve a specific page of results in the list.
func (c *UsersThreadsListCall) PageToken(pageToken string) *UsersThreadsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Q sets the optional parameter "q": Only return threads matching the
// specified query. Supports the same query format as the Gmail search
// box. For example, "from:someuser@example.com rfc822msgid: is:unread".
func (c *UsersThreadsListCall) Q(q string) *UsersThreadsListCall {
	c.urlParams_.Set("q", q)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersThreadsListCall) QuotaUser(quotaUser string) *UsersThreadsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersThreadsListCall) UserIP(userIP string) *UsersThreadsListCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersThreadsListCall) Fields(s ...googleapi.Field) *UsersThreadsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersThreadsListCall) IfNoneMatch(entityTag string) *UsersThreadsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersThreadsListCall) Context(ctx context.Context) *UsersThreadsListCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersThreadsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/threads")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
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

// Do executes the "gmail.users.threads.list" call.
// Exactly one of *ListThreadsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListThreadsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *UsersThreadsListCall) Do() (*ListThreadsResponse, error) {
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
	ret := &ListThreadsResponse{
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
	//   "description": "Lists the threads in the user's mailbox.",
	//   "httpMethod": "GET",
	//   "id": "gmail.users.threads.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "includeSpamTrash": {
	//       "default": "false",
	//       "description": "Include threads from SPAM and TRASH in the results.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "labelIds": {
	//       "description": "Only return threads with labels that match all of the specified label IDs.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "Maximum number of threads to return.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Page token to retrieve a specific page of results in the list.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "q": {
	//       "description": "Only return threads matching the specified query. Supports the same query format as the Gmail search box. For example, \"from:someuser@example.com rfc822msgid: is:unread\".",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/threads",
	//   "response": {
	//     "$ref": "ListThreadsResponse"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify",
	//     "https://www.googleapis.com/auth/gmail.readonly"
	//   ]
	// }

}

// method id "gmail.users.threads.modify":

type UsersThreadsModifyCall struct {
	s                   *Service
	userId              string
	id                  string
	modifythreadrequest *ModifyThreadRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
}

// Modify: Modifies the labels applied to the thread. This applies to
// all messages in the thread.
func (r *UsersThreadsService) Modify(userId string, id string, modifythreadrequest *ModifyThreadRequest) *UsersThreadsModifyCall {
	c := &UsersThreadsModifyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	c.modifythreadrequest = modifythreadrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersThreadsModifyCall) QuotaUser(quotaUser string) *UsersThreadsModifyCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersThreadsModifyCall) UserIP(userIP string) *UsersThreadsModifyCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersThreadsModifyCall) Fields(s ...googleapi.Field) *UsersThreadsModifyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersThreadsModifyCall) Context(ctx context.Context) *UsersThreadsModifyCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersThreadsModifyCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.modifythreadrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/threads/{id}/modify")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.threads.modify" call.
// Exactly one of *Thread or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Thread.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersThreadsModifyCall) Do() (*Thread, error) {
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
	ret := &Thread{
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
	//   "description": "Modifies the labels applied to the thread. This applies to all messages in the thread.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.threads.modify",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the thread to modify.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/threads/{id}/modify",
	//   "request": {
	//     "$ref": "ModifyThreadRequest"
	//   },
	//   "response": {
	//     "$ref": "Thread"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ]
	// }

}

// method id "gmail.users.threads.trash":

type UsersThreadsTrashCall struct {
	s          *Service
	userId     string
	id         string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Trash: Moves the specified thread to the trash.
func (r *UsersThreadsService) Trash(userId string, id string) *UsersThreadsTrashCall {
	c := &UsersThreadsTrashCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersThreadsTrashCall) QuotaUser(quotaUser string) *UsersThreadsTrashCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersThreadsTrashCall) UserIP(userIP string) *UsersThreadsTrashCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersThreadsTrashCall) Fields(s ...googleapi.Field) *UsersThreadsTrashCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersThreadsTrashCall) Context(ctx context.Context) *UsersThreadsTrashCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersThreadsTrashCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/threads/{id}/trash")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.threads.trash" call.
// Exactly one of *Thread or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Thread.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersThreadsTrashCall) Do() (*Thread, error) {
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
	ret := &Thread{
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
	//   "description": "Moves the specified thread to the trash.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.threads.trash",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the thread to Trash.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/threads/{id}/trash",
	//   "response": {
	//     "$ref": "Thread"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ]
	// }

}

// method id "gmail.users.threads.untrash":

type UsersThreadsUntrashCall struct {
	s          *Service
	userId     string
	id         string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Untrash: Removes the specified thread from the trash.
func (r *UsersThreadsService) Untrash(userId string, id string) *UsersThreadsUntrashCall {
	c := &UsersThreadsUntrashCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
// Overrides userIp if both are provided.
func (c *UsersThreadsUntrashCall) QuotaUser(quotaUser string) *UsersThreadsUntrashCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserIP sets the optional parameter "userIp": IP address of the site
// where the request originates. Use this if you want to enforce
// per-user limits.
func (c *UsersThreadsUntrashCall) UserIP(userIP string) *UsersThreadsUntrashCall {
	c.urlParams_.Set("userIp", userIP)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersThreadsUntrashCall) Fields(s ...googleapi.Field) *UsersThreadsUntrashCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersThreadsUntrashCall) Context(ctx context.Context) *UsersThreadsUntrashCall {
	c.ctx_ = ctx
	return c
}

func (c *UsersThreadsUntrashCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "{userId}/threads/{id}/untrash")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
		"id":     c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "gmail.users.threads.untrash" call.
// Exactly one of *Thread or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Thread.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *UsersThreadsUntrashCall) Do() (*Thread, error) {
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
	ret := &Thread{
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
	//   "description": "Removes the specified thread from the trash.",
	//   "httpMethod": "POST",
	//   "id": "gmail.users.threads.untrash",
	//   "parameterOrder": [
	//     "userId",
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the thread to remove from Trash.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "default": "me",
	//       "description": "The user's email address. The special value me can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "{userId}/threads/{id}/untrash",
	//   "response": {
	//     "$ref": "Thread"
	//   },
	//   "scopes": [
	//     "https://mail.google.com/",
	//     "https://www.googleapis.com/auth/gmail.modify"
	//   ]
	// }

}
