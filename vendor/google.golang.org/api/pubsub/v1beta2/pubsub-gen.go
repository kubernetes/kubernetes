// Package pubsub provides access to the Google Cloud Pub/Sub API.
//
// See https://cloud.google.com/pubsub/docs
//
// Usage example:
//
//   import "google.golang.org/api/pubsub/v1beta2"
//   ...
//   pubsubService, err := pubsub.New(oauthHttpClient)
package pubsub // import "google.golang.org/api/pubsub/v1beta2"

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

const apiId = "pubsub:v1beta2"
const apiName = "pubsub"
const apiVersion = "v1beta2"
const basePath = "https://pubsub.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// View and manage your data across Google Cloud Platform services
	CloudPlatformScope = "https://www.googleapis.com/auth/cloud-platform"

	// View and manage Pub/Sub topics and subscriptions
	PubsubScope = "https://www.googleapis.com/auth/pubsub"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Projects = NewProjectsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Projects *ProjectsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewProjectsService(s *Service) *ProjectsService {
	rs := &ProjectsService{s: s}
	rs.Subscriptions = NewProjectsSubscriptionsService(s)
	rs.Topics = NewProjectsTopicsService(s)
	return rs
}

type ProjectsService struct {
	s *Service

	Subscriptions *ProjectsSubscriptionsService

	Topics *ProjectsTopicsService
}

func NewProjectsSubscriptionsService(s *Service) *ProjectsSubscriptionsService {
	rs := &ProjectsSubscriptionsService{s: s}
	return rs
}

type ProjectsSubscriptionsService struct {
	s *Service
}

func NewProjectsTopicsService(s *Service) *ProjectsTopicsService {
	rs := &ProjectsTopicsService{s: s}
	rs.Subscriptions = NewProjectsTopicsSubscriptionsService(s)
	return rs
}

type ProjectsTopicsService struct {
	s *Service

	Subscriptions *ProjectsTopicsSubscriptionsService
}

func NewProjectsTopicsSubscriptionsService(s *Service) *ProjectsTopicsSubscriptionsService {
	rs := &ProjectsTopicsSubscriptionsService{s: s}
	return rs
}

type ProjectsTopicsSubscriptionsService struct {
	s *Service
}

// AcknowledgeRequest: Request for the Acknowledge method.
type AcknowledgeRequest struct {
	// AckIds: The acknowledgment ID for the messages being acknowledged
	// that was returned by the Pub/Sub system in the `Pull` response. Must
	// not be empty.
	AckIds []string `json:"ackIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AckIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *AcknowledgeRequest) MarshalJSON() ([]byte, error) {
	type noMethod AcknowledgeRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Binding: Associates `members` with a `role`.
type Binding struct {
	// Members: Specifies the identities requesting access for a Cloud
	// Platform resource. `members` can have the following formats: *
	// `allUsers`: A special identifier that represents anyone who is on the
	// internet; with or without a Google account. *
	// `allAuthenticatedUsers`: A special identifier that represents anyone
	// who is authenticated with a Google account or a service account. *
	// `user:{emailid}`: An email address that represents a specific Google
	// account. For example, `alice@gmail.com` or `joe@example.com`. *
	// `serviceAccount:{emailid}`: An email address that represents a
	// service account. For example,
	// `my-other-app@appspot.gserviceaccount.com`. * `group:{emailid}`: An
	// email address that represents a Google group. For example,
	// `admins@example.com`. * `domain:{domain}`: A Google Apps domain name
	// that represents all the users of that domain. For example,
	// `google.com` or `example.com`.
	Members []string `json:"members,omitempty"`

	// Role: Role that is assigned to `members`. For example,
	// `roles/viewer`, `roles/editor`, or `roles/owner`. Required
	Role string `json:"role,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Members") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Binding) MarshalJSON() ([]byte, error) {
	type noMethod Binding
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Empty: A generic empty message that you can re-use to avoid defining
// duplicated empty messages in your APIs. A typical example is to use
// it as the request or the response type of an API method. For
// instance: service Foo { rpc Bar(google.protobuf.Empty) returns
// (google.protobuf.Empty); } The JSON representation for `Empty` is
// empty JSON object `{}`.
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// ListSubscriptionsResponse: Response for the `ListSubscriptions`
// method.
type ListSubscriptionsResponse struct {
	// NextPageToken: If not empty, indicates that there may be more
	// subscriptions that match the request; this value should be passed in
	// a new `ListSubscriptionsRequest` to get more subscriptions.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Subscriptions: The subscriptions that match the request.
	Subscriptions []*Subscription `json:"subscriptions,omitempty"`

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

func (s *ListSubscriptionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListSubscriptionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListTopicSubscriptionsResponse: Response for the
// `ListTopicSubscriptions` method.
type ListTopicSubscriptionsResponse struct {
	// NextPageToken: If not empty, indicates that there may be more
	// subscriptions that match the request; this value should be passed in
	// a new `ListTopicSubscriptionsRequest` to get more subscriptions.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Subscriptions: The names of the subscriptions that match the request.
	Subscriptions []string `json:"subscriptions,omitempty"`

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

func (s *ListTopicSubscriptionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListTopicSubscriptionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListTopicsResponse: Response for the `ListTopics` method.
type ListTopicsResponse struct {
	// NextPageToken: If not empty, indicates that there may be more topics
	// that match the request; this value should be passed in a new
	// `ListTopicsRequest`.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Topics: The resulting topics.
	Topics []*Topic `json:"topics,omitempty"`

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

func (s *ListTopicsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListTopicsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ModifyAckDeadlineRequest: Request for the ModifyAckDeadline method.
type ModifyAckDeadlineRequest struct {
	// AckDeadlineSeconds: The new ack deadline with respect to the time
	// this request was sent to the Pub/Sub system. Must be >= 0. For
	// example, if the value is 10, the new ack deadline will expire 10
	// seconds after the `ModifyAckDeadline` call was made. Specifying zero
	// may immediately make the message available for another pull request.
	AckDeadlineSeconds int64 `json:"ackDeadlineSeconds,omitempty"`

	// AckId: The acknowledgment ID. Either this or ack_ids must be
	// populated, but not both.
	AckId string `json:"ackId,omitempty"`

	// AckIds: List of acknowledgment IDs.
	AckIds []string `json:"ackIds,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AckDeadlineSeconds")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ModifyAckDeadlineRequest) MarshalJSON() ([]byte, error) {
	type noMethod ModifyAckDeadlineRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ModifyPushConfigRequest: Request for the ModifyPushConfig method.
type ModifyPushConfigRequest struct {
	// PushConfig: The push configuration for future deliveries. An empty
	// `pushConfig` indicates that the Pub/Sub system should stop pushing
	// messages from the given subscription and allow messages to be pulled
	// and acknowledged - effectively pausing the subscription if `Pull` is
	// not called.
	PushConfig *PushConfig `json:"pushConfig,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PushConfig") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ModifyPushConfigRequest) MarshalJSON() ([]byte, error) {
	type noMethod ModifyPushConfigRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Policy: Defines an Identity and Access Management (IAM) policy. It is
// used to specify access control policies for Cloud Platform resources.
// A `Policy` consists of a list of `bindings`. A `Binding` binds a list
// of `members` to a `role`, where the members can be user accounts,
// Google groups, Google domains, and service accounts. A `role` is a
// named list of permissions defined by IAM. **Example** { "bindings": [
// { "role": "roles/owner", "members": [ "user:mike@example.com",
// "group:admins@example.com", "domain:google.com",
// "serviceAccount:my-other-app@appspot.gserviceaccount.com"] }, {
// "role": "roles/viewer", "members": ["user:sean@example.com"] } ] }
// For a description of IAM and its features, see the [IAM developer's
// guide](https://cloud.google.com/iam).
type Policy struct {
	// Bindings: Associates a list of `members` to a `role`. Multiple
	// `bindings` must not be specified for the same `role`. `bindings` with
	// no members will result in an error.
	Bindings []*Binding `json:"bindings,omitempty"`

	// Etag: Can be used to perform a read-modify-write.
	Etag string `json:"etag,omitempty"`

	// Version: Version of the `Policy`. The default version is 0.
	Version int64 `json:"version,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Bindings") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Policy) MarshalJSON() ([]byte, error) {
	type noMethod Policy
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PublishRequest: Request for the Publish method.
type PublishRequest struct {
	// Messages: The messages to publish.
	Messages []*PubsubMessage `json:"messages,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Messages") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PublishRequest) MarshalJSON() ([]byte, error) {
	type noMethod PublishRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PublishResponse: Response for the `Publish` method.
type PublishResponse struct {
	// MessageIds: The server-assigned ID of each published message, in the
	// same order as the messages in the request. IDs are guaranteed to be
	// unique within the topic.
	MessageIds []string `json:"messageIds,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "MessageIds") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PublishResponse) MarshalJSON() ([]byte, error) {
	type noMethod PublishResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PubsubMessage: A message data and its attributes. The message payload
// must not be empty; it must contain either a non-empty data field, or
// at least one attribute.
type PubsubMessage struct {
	// Attributes: Optional attributes for this message.
	Attributes map[string]string `json:"attributes,omitempty"`

	// Data: The message payload. For JSON requests, the value of this field
	// must be base64-encoded.
	Data string `json:"data,omitempty"`

	// MessageId: ID of this message, assigned by the server when the
	// message is published. Guaranteed to be unique within the topic. This
	// value may be read by a subscriber that receives a `PubsubMessage` via
	// a `Pull` call or a push delivery. It must not be populated by the
	// publisher in a `Publish` call.
	MessageId string `json:"messageId,omitempty"`

	// PublishTime: The time at which the message was published, populated
	// by the server when it receives the `Publish` call. It must not be
	// populated by the publisher in a `Publish` call.
	PublishTime string `json:"publishTime,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Attributes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PubsubMessage) MarshalJSON() ([]byte, error) {
	type noMethod PubsubMessage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PullRequest: Request for the `Pull` method.
type PullRequest struct {
	// MaxMessages: The maximum number of messages returned for this
	// request. The Pub/Sub system may return fewer than the number
	// specified.
	MaxMessages int64 `json:"maxMessages,omitempty"`

	// ReturnImmediately: If this is specified as true the system will
	// respond immediately even if it is not able to return a message in the
	// `Pull` response. Otherwise the system is allowed to wait until at
	// least one message is available rather than returning no messages. The
	// client may cancel the request if it does not wish to wait any longer
	// for the response.
	ReturnImmediately bool `json:"returnImmediately,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxMessages") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PullRequest) MarshalJSON() ([]byte, error) {
	type noMethod PullRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PullResponse: Response for the `Pull` method.
type PullResponse struct {
	// ReceivedMessages: Received Pub/Sub messages. The Pub/Sub system will
	// return zero messages if there are no more available in the backlog.
	// The Pub/Sub system may return fewer than the `maxMessages` requested
	// even if there are more messages available in the backlog.
	ReceivedMessages []*ReceivedMessage `json:"receivedMessages,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "ReceivedMessages") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PullResponse) MarshalJSON() ([]byte, error) {
	type noMethod PullResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PushConfig: Configuration for a push delivery endpoint.
type PushConfig struct {
	// Attributes: Endpoint configuration attributes. Every endpoint has a
	// set of API supported attributes that can be used to control different
	// aspects of the message delivery. The currently supported attribute is
	// `x-goog-version`, which you can use to change the format of the push
	// message. This attribute indicates the version of the data expected by
	// the endpoint. This controls the shape of the envelope (i.e. its
	// fields and metadata). The endpoint version is based on the version of
	// the Pub/Sub API. If not present during the `CreateSubscription` call,
	// it will default to the version of the API used to make such call. If
	// not present during a `ModifyPushConfig` call, its value will not be
	// changed. `GetSubscription` calls will always return a valid version,
	// even if the subscription was created without this attribute. The
	// possible values for this attribute are: * `v1beta1`: uses the push
	// format defined in the v1beta1 Pub/Sub API. * `v1` or `v1beta2`: uses
	// the push format defined in the v1 Pub/Sub API.
	Attributes map[string]string `json:"attributes,omitempty"`

	// PushEndpoint: A URL locating the endpoint to which messages should be
	// pushed. For example, a Webhook endpoint might use
	// "https://example.com/push".
	PushEndpoint string `json:"pushEndpoint,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Attributes") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PushConfig) MarshalJSON() ([]byte, error) {
	type noMethod PushConfig
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ReceivedMessage: A message and its corresponding acknowledgment ID.
type ReceivedMessage struct {
	// AckId: This ID can be used to acknowledge the received message.
	AckId string `json:"ackId,omitempty"`

	// Message: The message.
	Message *PubsubMessage `json:"message,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AckId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ReceivedMessage) MarshalJSON() ([]byte, error) {
	type noMethod ReceivedMessage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// SetIamPolicyRequest: Request message for `SetIamPolicy` method.
type SetIamPolicyRequest struct {
	// Policy: REQUIRED: The complete policy to be applied to the
	// `resource`. The size of the policy is limited to a few 10s of KB. An
	// empty policy is a valid policy but certain Cloud Platform services
	// (such as Projects) might reject them.
	Policy *Policy `json:"policy,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Policy") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *SetIamPolicyRequest) MarshalJSON() ([]byte, error) {
	type noMethod SetIamPolicyRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Subscription: A subscription resource.
type Subscription struct {
	// AckDeadlineSeconds: This value is the maximum time after a subscriber
	// receives a message before the subscriber should acknowledge the
	// message. After message delivery but before the ack deadline expires
	// and before the message is acknowledged, it is an outstanding message
	// and will not be delivered again during that time (on a best-effort
	// basis). For pull delivery this value is used as the initial value for
	// the ack deadline. To override this value for a given message, call
	// `ModifyAckDeadline` with the corresponding `ack_id`. For push
	// delivery, this value is also used to set the request timeout for the
	// call to the push endpoint. If the subscriber never acknowledges the
	// message, the Pub/Sub system will eventually redeliver the message. If
	// this parameter is not set, the default value of 10 seconds is used.
	AckDeadlineSeconds int64 `json:"ackDeadlineSeconds,omitempty"`

	// Name: The name of the subscription. It must have the format
	// "projects/{project}/subscriptions/{subscription}". `{subscription}`
	// must start with a letter, and contain only letters (`[A-Za-z]`),
	// numbers (`[0-9]`), dashes (`-`), underscores (`_`), periods (`.`),
	// tildes (`~`), plus (`+`) or percent signs (`%`). It must be between 3
	// and 255 characters in length, and it must not start with "goog".
	Name string `json:"name,omitempty"`

	// PushConfig: If push delivery is used with this subscription, this
	// field is used to configure it. An empty `pushConfig` signifies that
	// the subscriber will pull and ack messages using API methods.
	PushConfig *PushConfig `json:"pushConfig,omitempty"`

	// Topic: The name of the topic from which this subscription is
	// receiving messages. The value of this field will be `_deleted-topic_`
	// if the topic has been deleted.
	Topic string `json:"topic,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AckDeadlineSeconds")
	// to unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Subscription) MarshalJSON() ([]byte, error) {
	type noMethod Subscription
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TestIamPermissionsRequest: Request message for `TestIamPermissions`
// method.
type TestIamPermissionsRequest struct {
	// Permissions: The set of permissions to check for the `resource`.
	// Permissions with wildcards (such as '*' or 'storage.*') are not
	// allowed.
	Permissions []string `json:"permissions,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Permissions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TestIamPermissionsRequest) MarshalJSON() ([]byte, error) {
	type noMethod TestIamPermissionsRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// TestIamPermissionsResponse: Response message for `TestIamPermissions`
// method.
type TestIamPermissionsResponse struct {
	// Permissions: A subset of `TestPermissionsRequest.permissions` that
	// the caller is allowed.
	Permissions []string `json:"permissions,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Permissions") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *TestIamPermissionsResponse) MarshalJSON() ([]byte, error) {
	type noMethod TestIamPermissionsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Topic: A topic resource.
type Topic struct {
	// Name: The name of the topic. It must have the format
	// "projects/{project}/topics/{topic}". `{topic}` must start with a
	// letter, and contain only letters (`[A-Za-z]`), numbers (`[0-9]`),
	// dashes (`-`), underscores (`_`), periods (`.`), tildes (`~`), plus
	// (`+`) or percent signs (`%`). It must be between 3 and 255 characters
	// in length, and it must not start with "goog".
	Name string `json:"name,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Name") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Topic) MarshalJSON() ([]byte, error) {
	type noMethod Topic
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "pubsub.projects.subscriptions.acknowledge":

type ProjectsSubscriptionsAcknowledgeCall struct {
	s                  *Service
	subscription       string
	acknowledgerequest *AcknowledgeRequest
	urlParams_         gensupport.URLParams
	ctx_               context.Context
}

// Acknowledge: Acknowledges the messages associated with the `ack_ids`
// in the `AcknowledgeRequest`. The Pub/Sub system can remove the
// relevant messages from the subscription. Acknowledging a message
// whose ack deadline has expired may succeed, but such a message may be
// redelivered later. Acknowledging a message more than once will not
// result in an error.
func (r *ProjectsSubscriptionsService) Acknowledge(subscription string, acknowledgerequest *AcknowledgeRequest) *ProjectsSubscriptionsAcknowledgeCall {
	c := &ProjectsSubscriptionsAcknowledgeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.subscription = subscription
	c.acknowledgerequest = acknowledgerequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSubscriptionsAcknowledgeCall) QuotaUser(quotaUser string) *ProjectsSubscriptionsAcknowledgeCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSubscriptionsAcknowledgeCall) Fields(s ...googleapi.Field) *ProjectsSubscriptionsAcknowledgeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSubscriptionsAcknowledgeCall) Context(ctx context.Context) *ProjectsSubscriptionsAcknowledgeCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsSubscriptionsAcknowledgeCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.acknowledgerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+subscription}:acknowledge")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"subscription": c.subscription,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.subscriptions.acknowledge" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSubscriptionsAcknowledgeCall) Do() (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Acknowledges the messages associated with the `ack_ids` in the `AcknowledgeRequest`. The Pub/Sub system can remove the relevant messages from the subscription. Acknowledging a message whose ack deadline has expired may succeed, but such a message may be redelivered later. Acknowledging a message more than once will not result in an error.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.projects.subscriptions.acknowledge",
	//   "parameterOrder": [
	//     "subscription"
	//   ],
	//   "parameters": {
	//     "subscription": {
	//       "description": "The subscription whose message is being acknowledged.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/subscriptions/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+subscription}:acknowledge",
	//   "request": {
	//     "$ref": "AcknowledgeRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.subscriptions.create":

type ProjectsSubscriptionsCreateCall struct {
	s            *Service
	name         string
	subscription *Subscription
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Create: Creates a subscription to a given topic for a given
// subscriber. If the subscription already exists, returns
// `ALREADY_EXISTS`. If the corresponding topic doesn't exist, returns
// `NOT_FOUND`. If the name is not provided in the request, the server
// will assign a random name for this subscription on the same project
// as the topic.
func (r *ProjectsSubscriptionsService) Create(name string, subscription *Subscription) *ProjectsSubscriptionsCreateCall {
	c := &ProjectsSubscriptionsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.subscription = subscription
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSubscriptionsCreateCall) QuotaUser(quotaUser string) *ProjectsSubscriptionsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSubscriptionsCreateCall) Fields(s ...googleapi.Field) *ProjectsSubscriptionsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSubscriptionsCreateCall) Context(ctx context.Context) *ProjectsSubscriptionsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsSubscriptionsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.subscription)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.subscriptions.create" call.
// Exactly one of *Subscription or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscription.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsSubscriptionsCreateCall) Do() (*Subscription, error) {
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
	ret := &Subscription{
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
	//   "description": "Creates a subscription to a given topic for a given subscriber. If the subscription already exists, returns `ALREADY_EXISTS`. If the corresponding topic doesn't exist, returns `NOT_FOUND`. If the name is not provided in the request, the server will assign a random name for this subscription on the same project as the topic.",
	//   "httpMethod": "PUT",
	//   "id": "pubsub.projects.subscriptions.create",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the subscription. It must have the format `\"projects/{project}/subscriptions/{subscription}\"`. `{subscription}` must start with a letter, and contain only letters (`[A-Za-z]`), numbers (`[0-9]`), dashes (`-`), underscores (`_`), periods (`.`), tildes (`~`), plus (`+`) or percent signs (`%`). It must be between 3 and 255 characters in length, and it must not start with `\"goog\"`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/subscriptions/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+name}",
	//   "request": {
	//     "$ref": "Subscription"
	//   },
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.subscriptions.delete":

type ProjectsSubscriptionsDeleteCall struct {
	s            *Service
	subscription string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Delete: Deletes an existing subscription. All pending messages in the
// subscription are immediately dropped. Calls to `Pull` after deletion
// will return `NOT_FOUND`. After a subscription is deleted, a new one
// may be created with the same name, but the new one has no association
// with the old subscription, or its topic unless the same topic is
// specified.
func (r *ProjectsSubscriptionsService) Delete(subscription string) *ProjectsSubscriptionsDeleteCall {
	c := &ProjectsSubscriptionsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.subscription = subscription
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSubscriptionsDeleteCall) QuotaUser(quotaUser string) *ProjectsSubscriptionsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSubscriptionsDeleteCall) Fields(s ...googleapi.Field) *ProjectsSubscriptionsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSubscriptionsDeleteCall) Context(ctx context.Context) *ProjectsSubscriptionsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsSubscriptionsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+subscription}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"subscription": c.subscription,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.subscriptions.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSubscriptionsDeleteCall) Do() (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Deletes an existing subscription. All pending messages in the subscription are immediately dropped. Calls to `Pull` after deletion will return `NOT_FOUND`. After a subscription is deleted, a new one may be created with the same name, but the new one has no association with the old subscription, or its topic unless the same topic is specified.",
	//   "httpMethod": "DELETE",
	//   "id": "pubsub.projects.subscriptions.delete",
	//   "parameterOrder": [
	//     "subscription"
	//   ],
	//   "parameters": {
	//     "subscription": {
	//       "description": "The subscription to delete.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/subscriptions/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+subscription}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.subscriptions.get":

type ProjectsSubscriptionsGetCall struct {
	s            *Service
	subscription string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the configuration details of a subscription.
func (r *ProjectsSubscriptionsService) Get(subscription string) *ProjectsSubscriptionsGetCall {
	c := &ProjectsSubscriptionsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.subscription = subscription
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSubscriptionsGetCall) QuotaUser(quotaUser string) *ProjectsSubscriptionsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSubscriptionsGetCall) Fields(s ...googleapi.Field) *ProjectsSubscriptionsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsSubscriptionsGetCall) IfNoneMatch(entityTag string) *ProjectsSubscriptionsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSubscriptionsGetCall) Context(ctx context.Context) *ProjectsSubscriptionsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsSubscriptionsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+subscription}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"subscription": c.subscription,
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

// Do executes the "pubsub.projects.subscriptions.get" call.
// Exactly one of *Subscription or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscription.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsSubscriptionsGetCall) Do() (*Subscription, error) {
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
	ret := &Subscription{
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
	//   "description": "Gets the configuration details of a subscription.",
	//   "httpMethod": "GET",
	//   "id": "pubsub.projects.subscriptions.get",
	//   "parameterOrder": [
	//     "subscription"
	//   ],
	//   "parameters": {
	//     "subscription": {
	//       "description": "The name of the subscription to get.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/subscriptions/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+subscription}",
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.subscriptions.getIamPolicy":

type ProjectsSubscriptionsGetIamPolicyCall struct {
	s            *Service
	resource     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// GetIamPolicy: Gets the access control policy for a `resource`. Is
// empty if the policy or the resource does not exist.
func (r *ProjectsSubscriptionsService) GetIamPolicy(resource string) *ProjectsSubscriptionsGetIamPolicyCall {
	c := &ProjectsSubscriptionsGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSubscriptionsGetIamPolicyCall) QuotaUser(quotaUser string) *ProjectsSubscriptionsGetIamPolicyCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSubscriptionsGetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsSubscriptionsGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsSubscriptionsGetIamPolicyCall) IfNoneMatch(entityTag string) *ProjectsSubscriptionsGetIamPolicyCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSubscriptionsGetIamPolicyCall) Context(ctx context.Context) *ProjectsSubscriptionsGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsSubscriptionsGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+resource}:getIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
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

// Do executes the "pubsub.projects.subscriptions.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSubscriptionsGetIamPolicyCall) Do() (*Policy, error) {
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
	ret := &Policy{
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
	//   "description": "Gets the access control policy for a `resource`. Is empty if the policy or the resource does not exist.",
	//   "httpMethod": "GET",
	//   "id": "pubsub.projects.subscriptions.getIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which policy is being requested. `resource` is usually specified as a path, such as, `projects/{project}/zones/{zone}/disks/{disk}`. The format for the path specified in this value is resource specific and is specified in the documentation for the respective GetIamPolicy rpc.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/subscriptions/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+resource}:getIamPolicy",
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.subscriptions.list":

type ProjectsSubscriptionsListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists matching subscriptions.
func (r *ProjectsSubscriptionsService) List(project string) *ProjectsSubscriptionsListCall {
	c := &ProjectsSubscriptionsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum number of
// subscriptions to return.
func (c *ProjectsSubscriptionsListCall) PageSize(pageSize int64) *ProjectsSubscriptionsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The value returned
// by the last `ListSubscriptionsResponse`; indicates that this is a
// continuation of a prior `ListSubscriptions` call, and that the system
// should return the next page of data.
func (c *ProjectsSubscriptionsListCall) PageToken(pageToken string) *ProjectsSubscriptionsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSubscriptionsListCall) QuotaUser(quotaUser string) *ProjectsSubscriptionsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSubscriptionsListCall) Fields(s ...googleapi.Field) *ProjectsSubscriptionsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsSubscriptionsListCall) IfNoneMatch(entityTag string) *ProjectsSubscriptionsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSubscriptionsListCall) Context(ctx context.Context) *ProjectsSubscriptionsListCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsSubscriptionsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+project}/subscriptions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
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

// Do executes the "pubsub.projects.subscriptions.list" call.
// Exactly one of *ListSubscriptionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListSubscriptionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsSubscriptionsListCall) Do() (*ListSubscriptionsResponse, error) {
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
	ret := &ListSubscriptionsResponse{
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
	//   "description": "Lists matching subscriptions.",
	//   "httpMethod": "GET",
	//   "id": "pubsub.projects.subscriptions.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Maximum number of subscriptions to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value returned by the last `ListSubscriptionsResponse`; indicates that this is a continuation of a prior `ListSubscriptions` call, and that the system should return the next page of data.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The name of the cloud project that subscriptions belong to.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+project}/subscriptions",
	//   "response": {
	//     "$ref": "ListSubscriptionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.subscriptions.modifyAckDeadline":

type ProjectsSubscriptionsModifyAckDeadlineCall struct {
	s                        *Service
	subscription             string
	modifyackdeadlinerequest *ModifyAckDeadlineRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
}

// ModifyAckDeadline: Modifies the ack deadline for a specific message.
// This method is useful to indicate that more time is needed to process
// a message by the subscriber, or to make the message available for
// redelivery if the processing was interrupted.
func (r *ProjectsSubscriptionsService) ModifyAckDeadline(subscription string, modifyackdeadlinerequest *ModifyAckDeadlineRequest) *ProjectsSubscriptionsModifyAckDeadlineCall {
	c := &ProjectsSubscriptionsModifyAckDeadlineCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.subscription = subscription
	c.modifyackdeadlinerequest = modifyackdeadlinerequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSubscriptionsModifyAckDeadlineCall) QuotaUser(quotaUser string) *ProjectsSubscriptionsModifyAckDeadlineCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSubscriptionsModifyAckDeadlineCall) Fields(s ...googleapi.Field) *ProjectsSubscriptionsModifyAckDeadlineCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSubscriptionsModifyAckDeadlineCall) Context(ctx context.Context) *ProjectsSubscriptionsModifyAckDeadlineCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsSubscriptionsModifyAckDeadlineCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.modifyackdeadlinerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+subscription}:modifyAckDeadline")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"subscription": c.subscription,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.subscriptions.modifyAckDeadline" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSubscriptionsModifyAckDeadlineCall) Do() (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Modifies the ack deadline for a specific message. This method is useful to indicate that more time is needed to process a message by the subscriber, or to make the message available for redelivery if the processing was interrupted.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.projects.subscriptions.modifyAckDeadline",
	//   "parameterOrder": [
	//     "subscription"
	//   ],
	//   "parameters": {
	//     "subscription": {
	//       "description": "The name of the subscription.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/subscriptions/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+subscription}:modifyAckDeadline",
	//   "request": {
	//     "$ref": "ModifyAckDeadlineRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.subscriptions.modifyPushConfig":

type ProjectsSubscriptionsModifyPushConfigCall struct {
	s                       *Service
	subscription            string
	modifypushconfigrequest *ModifyPushConfigRequest
	urlParams_              gensupport.URLParams
	ctx_                    context.Context
}

// ModifyPushConfig: Modifies the `PushConfig` for a specified
// subscription. This may be used to change a push subscription to a
// pull one (signified by an empty `PushConfig`) or vice versa, or
// change the endpoint URL and other attributes of a push subscription.
// Messages will accumulate for delivery continuously through the call
// regardless of changes to the `PushConfig`.
func (r *ProjectsSubscriptionsService) ModifyPushConfig(subscription string, modifypushconfigrequest *ModifyPushConfigRequest) *ProjectsSubscriptionsModifyPushConfigCall {
	c := &ProjectsSubscriptionsModifyPushConfigCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.subscription = subscription
	c.modifypushconfigrequest = modifypushconfigrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSubscriptionsModifyPushConfigCall) QuotaUser(quotaUser string) *ProjectsSubscriptionsModifyPushConfigCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSubscriptionsModifyPushConfigCall) Fields(s ...googleapi.Field) *ProjectsSubscriptionsModifyPushConfigCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSubscriptionsModifyPushConfigCall) Context(ctx context.Context) *ProjectsSubscriptionsModifyPushConfigCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsSubscriptionsModifyPushConfigCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.modifypushconfigrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+subscription}:modifyPushConfig")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"subscription": c.subscription,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.subscriptions.modifyPushConfig" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSubscriptionsModifyPushConfigCall) Do() (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Modifies the `PushConfig` for a specified subscription. This may be used to change a push subscription to a pull one (signified by an empty `PushConfig`) or vice versa, or change the endpoint URL and other attributes of a push subscription. Messages will accumulate for delivery continuously through the call regardless of changes to the `PushConfig`.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.projects.subscriptions.modifyPushConfig",
	//   "parameterOrder": [
	//     "subscription"
	//   ],
	//   "parameters": {
	//     "subscription": {
	//       "description": "The name of the subscription.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/subscriptions/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+subscription}:modifyPushConfig",
	//   "request": {
	//     "$ref": "ModifyPushConfigRequest"
	//   },
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.subscriptions.pull":

type ProjectsSubscriptionsPullCall struct {
	s            *Service
	subscription string
	pullrequest  *PullRequest
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Pull: Pulls messages from the server. Returns an empty list if there
// are no messages available in the backlog. The server may return
// `UNAVAILABLE` if there are too many concurrent pull requests pending
// for the given subscription.
func (r *ProjectsSubscriptionsService) Pull(subscription string, pullrequest *PullRequest) *ProjectsSubscriptionsPullCall {
	c := &ProjectsSubscriptionsPullCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.subscription = subscription
	c.pullrequest = pullrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSubscriptionsPullCall) QuotaUser(quotaUser string) *ProjectsSubscriptionsPullCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSubscriptionsPullCall) Fields(s ...googleapi.Field) *ProjectsSubscriptionsPullCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSubscriptionsPullCall) Context(ctx context.Context) *ProjectsSubscriptionsPullCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsSubscriptionsPullCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pullrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+subscription}:pull")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"subscription": c.subscription,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.subscriptions.pull" call.
// Exactly one of *PullResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *PullResponse.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *ProjectsSubscriptionsPullCall) Do() (*PullResponse, error) {
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
	ret := &PullResponse{
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
	//   "description": "Pulls messages from the server. Returns an empty list if there are no messages available in the backlog. The server may return `UNAVAILABLE` if there are too many concurrent pull requests pending for the given subscription.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.projects.subscriptions.pull",
	//   "parameterOrder": [
	//     "subscription"
	//   ],
	//   "parameters": {
	//     "subscription": {
	//       "description": "The subscription from which messages should be pulled.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/subscriptions/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+subscription}:pull",
	//   "request": {
	//     "$ref": "PullRequest"
	//   },
	//   "response": {
	//     "$ref": "PullResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.subscriptions.setIamPolicy":

type ProjectsSubscriptionsSetIamPolicyCall struct {
	s                   *Service
	resource            string
	setiampolicyrequest *SetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
}

// SetIamPolicy: Sets the access control policy on the specified
// resource. Replaces any existing policy.
func (r *ProjectsSubscriptionsService) SetIamPolicy(resource string, setiampolicyrequest *SetIamPolicyRequest) *ProjectsSubscriptionsSetIamPolicyCall {
	c := &ProjectsSubscriptionsSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.setiampolicyrequest = setiampolicyrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSubscriptionsSetIamPolicyCall) QuotaUser(quotaUser string) *ProjectsSubscriptionsSetIamPolicyCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSubscriptionsSetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsSubscriptionsSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSubscriptionsSetIamPolicyCall) Context(ctx context.Context) *ProjectsSubscriptionsSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsSubscriptionsSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.setiampolicyrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+resource}:setIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.subscriptions.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsSubscriptionsSetIamPolicyCall) Do() (*Policy, error) {
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
	ret := &Policy{
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
	//   "description": "Sets the access control policy on the specified resource. Replaces any existing policy.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.projects.subscriptions.setIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which policy is being specified. `resource` is usually specified as a path, such as, `projects/{project}/zones/{zone}/disks/{disk}`. The format for the path specified in this value is resource specific and is specified in the documentation for the respective SetIamPolicy rpc.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/subscriptions/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+resource}:setIamPolicy",
	//   "request": {
	//     "$ref": "SetIamPolicyRequest"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.subscriptions.testIamPermissions":

type ProjectsSubscriptionsTestIamPermissionsCall struct {
	s                         *Service
	resource                  string
	testiampermissionsrequest *TestIamPermissionsRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
}

// TestIamPermissions: Returns permissions that a caller has on the
// specified resource.
func (r *ProjectsSubscriptionsService) TestIamPermissions(resource string, testiampermissionsrequest *TestIamPermissionsRequest) *ProjectsSubscriptionsTestIamPermissionsCall {
	c := &ProjectsSubscriptionsTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.testiampermissionsrequest = testiampermissionsrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsSubscriptionsTestIamPermissionsCall) QuotaUser(quotaUser string) *ProjectsSubscriptionsTestIamPermissionsCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsSubscriptionsTestIamPermissionsCall) Fields(s ...googleapi.Field) *ProjectsSubscriptionsTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsSubscriptionsTestIamPermissionsCall) Context(ctx context.Context) *ProjectsSubscriptionsTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsSubscriptionsTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.testiampermissionsrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+resource}:testIamPermissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.subscriptions.testIamPermissions" call.
// Exactly one of *TestIamPermissionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *TestIamPermissionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsSubscriptionsTestIamPermissionsCall) Do() (*TestIamPermissionsResponse, error) {
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
	ret := &TestIamPermissionsResponse{
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
	//   "description": "Returns permissions that a caller has on the specified resource.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.projects.subscriptions.testIamPermissions",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which policy detail is being requested. `resource` is usually specified as a path, such as, `projects/{project}/zones/{zone}/disks/{disk}`. The format for the path specified in this value is resource specific and is specified in the documentation for the respective TestIamPermissions rpc.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/subscriptions/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+resource}:testIamPermissions",
	//   "request": {
	//     "$ref": "TestIamPermissionsRequest"
	//   },
	//   "response": {
	//     "$ref": "TestIamPermissionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.topics.create":

type ProjectsTopicsCreateCall struct {
	s          *Service
	name       string
	topic      *Topic
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Creates the given topic with the given name.
func (r *ProjectsTopicsService) Create(name string, topic *Topic) *ProjectsTopicsCreateCall {
	c := &ProjectsTopicsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.name = name
	c.topic = topic
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsTopicsCreateCall) QuotaUser(quotaUser string) *ProjectsTopicsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsTopicsCreateCall) Fields(s ...googleapi.Field) *ProjectsTopicsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsTopicsCreateCall) Context(ctx context.Context) *ProjectsTopicsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsTopicsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.topic)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+name}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"name": c.name,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.topics.create" call.
// Exactly one of *Topic or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Topic.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsTopicsCreateCall) Do() (*Topic, error) {
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
	ret := &Topic{
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
	//   "description": "Creates the given topic with the given name.",
	//   "httpMethod": "PUT",
	//   "id": "pubsub.projects.topics.create",
	//   "parameterOrder": [
	//     "name"
	//   ],
	//   "parameters": {
	//     "name": {
	//       "description": "The name of the topic. It must have the format `\"projects/{project}/topics/{topic}\"`. `{topic}` must start with a letter, and contain only letters (`[A-Za-z]`), numbers (`[0-9]`), dashes (`-`), underscores (`_`), periods (`.`), tildes (`~`), plus (`+`) or percent signs (`%`). It must be between 3 and 255 characters in length, and it must not start with `\"goog\"`.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/topics/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+name}",
	//   "request": {
	//     "$ref": "Topic"
	//   },
	//   "response": {
	//     "$ref": "Topic"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.topics.delete":

type ProjectsTopicsDeleteCall struct {
	s          *Service
	topic      string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes the topic with the given name. Returns `NOT_FOUND` if
// the topic does not exist. After a topic is deleted, a new topic may
// be created with the same name; this is an entirely new topic with
// none of the old configuration or subscriptions. Existing
// subscriptions to this topic are not deleted, but their `topic` field
// is set to `_deleted-topic_`.
func (r *ProjectsTopicsService) Delete(topic string) *ProjectsTopicsDeleteCall {
	c := &ProjectsTopicsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.topic = topic
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsTopicsDeleteCall) QuotaUser(quotaUser string) *ProjectsTopicsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsTopicsDeleteCall) Fields(s ...googleapi.Field) *ProjectsTopicsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsTopicsDeleteCall) Context(ctx context.Context) *ProjectsTopicsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsTopicsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+topic}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"topic": c.topic,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.topics.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsTopicsDeleteCall) Do() (*Empty, error) {
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
	ret := &Empty{
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
	//   "description": "Deletes the topic with the given name. Returns `NOT_FOUND` if the topic does not exist. After a topic is deleted, a new topic may be created with the same name; this is an entirely new topic with none of the old configuration or subscriptions. Existing subscriptions to this topic are not deleted, but their `topic` field is set to `_deleted-topic_`.",
	//   "httpMethod": "DELETE",
	//   "id": "pubsub.projects.topics.delete",
	//   "parameterOrder": [
	//     "topic"
	//   ],
	//   "parameters": {
	//     "topic": {
	//       "description": "Name of the topic to delete.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/topics/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+topic}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.topics.get":

type ProjectsTopicsGetCall struct {
	s            *Service
	topic        string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the configuration of a topic.
func (r *ProjectsTopicsService) Get(topic string) *ProjectsTopicsGetCall {
	c := &ProjectsTopicsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.topic = topic
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsTopicsGetCall) QuotaUser(quotaUser string) *ProjectsTopicsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsTopicsGetCall) Fields(s ...googleapi.Field) *ProjectsTopicsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsTopicsGetCall) IfNoneMatch(entityTag string) *ProjectsTopicsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsTopicsGetCall) Context(ctx context.Context) *ProjectsTopicsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsTopicsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+topic}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"topic": c.topic,
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

// Do executes the "pubsub.projects.topics.get" call.
// Exactly one of *Topic or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Topic.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsTopicsGetCall) Do() (*Topic, error) {
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
	ret := &Topic{
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
	//   "description": "Gets the configuration of a topic.",
	//   "httpMethod": "GET",
	//   "id": "pubsub.projects.topics.get",
	//   "parameterOrder": [
	//     "topic"
	//   ],
	//   "parameters": {
	//     "topic": {
	//       "description": "The name of the topic to get.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/topics/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+topic}",
	//   "response": {
	//     "$ref": "Topic"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.topics.getIamPolicy":

type ProjectsTopicsGetIamPolicyCall struct {
	s            *Service
	resource     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// GetIamPolicy: Gets the access control policy for a `resource`. Is
// empty if the policy or the resource does not exist.
func (r *ProjectsTopicsService) GetIamPolicy(resource string) *ProjectsTopicsGetIamPolicyCall {
	c := &ProjectsTopicsGetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsTopicsGetIamPolicyCall) QuotaUser(quotaUser string) *ProjectsTopicsGetIamPolicyCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsTopicsGetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsTopicsGetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsTopicsGetIamPolicyCall) IfNoneMatch(entityTag string) *ProjectsTopicsGetIamPolicyCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsTopicsGetIamPolicyCall) Context(ctx context.Context) *ProjectsTopicsGetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsTopicsGetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+resource}:getIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
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

// Do executes the "pubsub.projects.topics.getIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsTopicsGetIamPolicyCall) Do() (*Policy, error) {
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
	ret := &Policy{
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
	//   "description": "Gets the access control policy for a `resource`. Is empty if the policy or the resource does not exist.",
	//   "httpMethod": "GET",
	//   "id": "pubsub.projects.topics.getIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which policy is being requested. `resource` is usually specified as a path, such as, `projects/{project}/zones/{zone}/disks/{disk}`. The format for the path specified in this value is resource specific and is specified in the documentation for the respective GetIamPolicy rpc.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/topics/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+resource}:getIamPolicy",
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.topics.list":

type ProjectsTopicsListCall struct {
	s            *Service
	project      string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists matching topics.
func (r *ProjectsTopicsService) List(project string) *ProjectsTopicsListCall {
	c := &ProjectsTopicsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.project = project
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum number of
// topics to return.
func (c *ProjectsTopicsListCall) PageSize(pageSize int64) *ProjectsTopicsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The value returned
// by the last `ListTopicsResponse`; indicates that this is a
// continuation of a prior `ListTopics` call, and that the system should
// return the next page of data.
func (c *ProjectsTopicsListCall) PageToken(pageToken string) *ProjectsTopicsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsTopicsListCall) QuotaUser(quotaUser string) *ProjectsTopicsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsTopicsListCall) Fields(s ...googleapi.Field) *ProjectsTopicsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsTopicsListCall) IfNoneMatch(entityTag string) *ProjectsTopicsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsTopicsListCall) Context(ctx context.Context) *ProjectsTopicsListCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsTopicsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+project}/topics")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"project": c.project,
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

// Do executes the "pubsub.projects.topics.list" call.
// Exactly one of *ListTopicsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListTopicsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsTopicsListCall) Do() (*ListTopicsResponse, error) {
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
	ret := &ListTopicsResponse{
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
	//   "description": "Lists matching topics.",
	//   "httpMethod": "GET",
	//   "id": "pubsub.projects.topics.list",
	//   "parameterOrder": [
	//     "project"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Maximum number of topics to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value returned by the last `ListTopicsResponse`; indicates that this is a continuation of a prior `ListTopics` call, and that the system should return the next page of data.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "project": {
	//       "description": "The name of the cloud project that topics belong to.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+project}/topics",
	//   "response": {
	//     "$ref": "ListTopicsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.topics.publish":

type ProjectsTopicsPublishCall struct {
	s              *Service
	topic          string
	publishrequest *PublishRequest
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Publish: Adds one or more messages to the topic. Returns `NOT_FOUND`
// if the topic does not exist. The message payload must not be empty;
// it must contain either a non-empty data field, or at least one
// attribute.
func (r *ProjectsTopicsService) Publish(topic string, publishrequest *PublishRequest) *ProjectsTopicsPublishCall {
	c := &ProjectsTopicsPublishCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.topic = topic
	c.publishrequest = publishrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsTopicsPublishCall) QuotaUser(quotaUser string) *ProjectsTopicsPublishCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsTopicsPublishCall) Fields(s ...googleapi.Field) *ProjectsTopicsPublishCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsTopicsPublishCall) Context(ctx context.Context) *ProjectsTopicsPublishCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsTopicsPublishCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.publishrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+topic}:publish")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"topic": c.topic,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.topics.publish" call.
// Exactly one of *PublishResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *PublishResponse.ServerResponse.Header or (if a response was returned
// at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsTopicsPublishCall) Do() (*PublishResponse, error) {
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
	ret := &PublishResponse{
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
	//   "description": "Adds one or more messages to the topic. Returns `NOT_FOUND` if the topic does not exist. The message payload must not be empty; it must contain either a non-empty data field, or at least one attribute.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.projects.topics.publish",
	//   "parameterOrder": [
	//     "topic"
	//   ],
	//   "parameters": {
	//     "topic": {
	//       "description": "The messages in the request will be published on this topic.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/topics/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+topic}:publish",
	//   "request": {
	//     "$ref": "PublishRequest"
	//   },
	//   "response": {
	//     "$ref": "PublishResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.topics.setIamPolicy":

type ProjectsTopicsSetIamPolicyCall struct {
	s                   *Service
	resource            string
	setiampolicyrequest *SetIamPolicyRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
}

// SetIamPolicy: Sets the access control policy on the specified
// resource. Replaces any existing policy.
func (r *ProjectsTopicsService) SetIamPolicy(resource string, setiampolicyrequest *SetIamPolicyRequest) *ProjectsTopicsSetIamPolicyCall {
	c := &ProjectsTopicsSetIamPolicyCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.setiampolicyrequest = setiampolicyrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsTopicsSetIamPolicyCall) QuotaUser(quotaUser string) *ProjectsTopicsSetIamPolicyCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsTopicsSetIamPolicyCall) Fields(s ...googleapi.Field) *ProjectsTopicsSetIamPolicyCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsTopicsSetIamPolicyCall) Context(ctx context.Context) *ProjectsTopicsSetIamPolicyCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsTopicsSetIamPolicyCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.setiampolicyrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+resource}:setIamPolicy")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.topics.setIamPolicy" call.
// Exactly one of *Policy or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Policy.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *ProjectsTopicsSetIamPolicyCall) Do() (*Policy, error) {
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
	ret := &Policy{
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
	//   "description": "Sets the access control policy on the specified resource. Replaces any existing policy.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.projects.topics.setIamPolicy",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which policy is being specified. `resource` is usually specified as a path, such as, `projects/{project}/zones/{zone}/disks/{disk}`. The format for the path specified in this value is resource specific and is specified in the documentation for the respective SetIamPolicy rpc.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/topics/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+resource}:setIamPolicy",
	//   "request": {
	//     "$ref": "SetIamPolicyRequest"
	//   },
	//   "response": {
	//     "$ref": "Policy"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.topics.testIamPermissions":

type ProjectsTopicsTestIamPermissionsCall struct {
	s                         *Service
	resource                  string
	testiampermissionsrequest *TestIamPermissionsRequest
	urlParams_                gensupport.URLParams
	ctx_                      context.Context
}

// TestIamPermissions: Returns permissions that a caller has on the
// specified resource.
func (r *ProjectsTopicsService) TestIamPermissions(resource string, testiampermissionsrequest *TestIamPermissionsRequest) *ProjectsTopicsTestIamPermissionsCall {
	c := &ProjectsTopicsTestIamPermissionsCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.resource = resource
	c.testiampermissionsrequest = testiampermissionsrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsTopicsTestIamPermissionsCall) QuotaUser(quotaUser string) *ProjectsTopicsTestIamPermissionsCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsTopicsTestIamPermissionsCall) Fields(s ...googleapi.Field) *ProjectsTopicsTestIamPermissionsCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsTopicsTestIamPermissionsCall) Context(ctx context.Context) *ProjectsTopicsTestIamPermissionsCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsTopicsTestIamPermissionsCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.testiampermissionsrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+resource}:testIamPermissions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"resource": c.resource,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.projects.topics.testIamPermissions" call.
// Exactly one of *TestIamPermissionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *TestIamPermissionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsTopicsTestIamPermissionsCall) Do() (*TestIamPermissionsResponse, error) {
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
	ret := &TestIamPermissionsResponse{
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
	//   "description": "Returns permissions that a caller has on the specified resource.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.projects.topics.testIamPermissions",
	//   "parameterOrder": [
	//     "resource"
	//   ],
	//   "parameters": {
	//     "resource": {
	//       "description": "REQUIRED: The resource for which policy detail is being requested. `resource` is usually specified as a path, such as, `projects/{project}/zones/{zone}/disks/{disk}`. The format for the path specified in this value is resource specific and is specified in the documentation for the respective TestIamPermissions rpc.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/topics/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+resource}:testIamPermissions",
	//   "request": {
	//     "$ref": "TestIamPermissionsRequest"
	//   },
	//   "response": {
	//     "$ref": "TestIamPermissionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.projects.topics.subscriptions.list":

type ProjectsTopicsSubscriptionsListCall struct {
	s            *Service
	topic        string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists the name of the subscriptions for this topic.
func (r *ProjectsTopicsSubscriptionsService) List(topic string) *ProjectsTopicsSubscriptionsListCall {
	c := &ProjectsTopicsSubscriptionsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.topic = topic
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum number of
// subscription names to return.
func (c *ProjectsTopicsSubscriptionsListCall) PageSize(pageSize int64) *ProjectsTopicsSubscriptionsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": The value returned
// by the last `ListTopicSubscriptionsResponse`; indicates that this is
// a continuation of a prior `ListTopicSubscriptions` call, and that the
// system should return the next page of data.
func (c *ProjectsTopicsSubscriptionsListCall) PageToken(pageToken string) *ProjectsTopicsSubscriptionsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *ProjectsTopicsSubscriptionsListCall) QuotaUser(quotaUser string) *ProjectsTopicsSubscriptionsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *ProjectsTopicsSubscriptionsListCall) Fields(s ...googleapi.Field) *ProjectsTopicsSubscriptionsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *ProjectsTopicsSubscriptionsListCall) IfNoneMatch(entityTag string) *ProjectsTopicsSubscriptionsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *ProjectsTopicsSubscriptionsListCall) Context(ctx context.Context) *ProjectsTopicsSubscriptionsListCall {
	c.ctx_ = ctx
	return c
}

func (c *ProjectsTopicsSubscriptionsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta2/{+topic}/subscriptions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"topic": c.topic,
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

// Do executes the "pubsub.projects.topics.subscriptions.list" call.
// Exactly one of *ListTopicSubscriptionsResponse or error will be
// non-nil. Any non-2xx status code is an error. Response headers are in
// either *ListTopicSubscriptionsResponse.ServerResponse.Header or (if a
// response was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *ProjectsTopicsSubscriptionsListCall) Do() (*ListTopicSubscriptionsResponse, error) {
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
	ret := &ListTopicSubscriptionsResponse{
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
	//   "description": "Lists the name of the subscriptions for this topic.",
	//   "httpMethod": "GET",
	//   "id": "pubsub.projects.topics.subscriptions.list",
	//   "parameterOrder": [
	//     "topic"
	//   ],
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Maximum number of subscription names to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value returned by the last `ListTopicSubscriptionsResponse`; indicates that this is a continuation of a prior `ListTopicSubscriptions` call, and that the system should return the next page of data.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "topic": {
	//       "description": "The name of the topic that subscriptions are attached to.",
	//       "location": "path",
	//       "pattern": "^projects/[^/]*/topics/[^/]*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta2/{+topic}/subscriptions",
	//   "response": {
	//     "$ref": "ListTopicSubscriptionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}
