// Package pubsub provides access to the Google Cloud Pub/Sub API.
//
// See https://cloud.google.com/pubsub/docs
//
// Usage example:
//
//   import "google.golang.org/api/pubsub/v1beta1a"
//   ...
//   pubsubService, err := pubsub.New(oauthHttpClient)
package pubsub // import "google.golang.org/api/pubsub/v1beta1a"

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

const apiId = "pubsub:v1beta1a"
const apiName = "pubsub"
const apiVersion = "v1beta1a"
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
	s.Subscriptions = NewSubscriptionsService(s)
	s.Topics = NewTopicsService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Subscriptions *SubscriptionsService

	Topics *TopicsService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewSubscriptionsService(s *Service) *SubscriptionsService {
	rs := &SubscriptionsService{s: s}
	return rs
}

type SubscriptionsService struct {
	s *Service
}

func NewTopicsService(s *Service) *TopicsService {
	rs := &TopicsService{s: s}
	return rs
}

type TopicsService struct {
	s *Service
}

// AcknowledgeRequest: Request for the Acknowledge method.
type AcknowledgeRequest struct {
	// AckId: The acknowledgment ID for the message being acknowledged. This
	// was returned by the Pub/Sub system in the Pull response.
	AckId []string `json:"ackId,omitempty"`

	// Subscription: The subscription whose message is being acknowledged.
	Subscription string `json:"subscription,omitempty"`

	// ForceSendFields is a list of field names (e.g. "AckId") to
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

// Empty: An empty message that you can re-use to avoid defining
// duplicated empty messages in your project. A typical example is to
// use it as argument or the return value of a service API. For
// instance: service Foo { rpc Bar (proto2.Empty) returns (proto2.Empty)
// { }; }; BEGIN GOOGLE-INTERNAL The difference between this one and
// net/rpc/empty-message.proto is that 1) The generated message here is
// in proto2 C++ API. 2) The proto2.Empty has minimum dependencies (no
// message_set or net/rpc dependencies) END GOOGLE-INTERNAL
type Empty struct {
	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`
}

// Label: A key-value pair applied to a given object.
type Label struct {
	// Key: The key of a label is a syntactically valid URL (as per RFC
	// 1738) with the "scheme" and initial slashes omitted and with the
	// additional restrictions noted below. Each key should be globally
	// unique. The "host" portion is called the "namespace" and is not
	// necessarily resolvable to a network endpoint. Instead, the namespace
	// indicates what system or entity defines the semantics of the label.
	// Namespaces do not restrict the set of objects to which a label may be
	// associated. Keys are defined by the following grammar: key = hostname
	// "/" kpath kpath = ksegment *[ "/" ksegment ] ksegment = alphadigit |
	// *[ alphadigit | "-" | "_" | "." ] where "hostname" and "alphadigit"
	// are defined as in RFC 1738. Example key: spanner.google.com/universe
	Key string `json:"key,omitempty"`

	// NumValue: An integer value.
	NumValue int64 `json:"numValue,omitempty,string"`

	// StrValue: A string value.
	StrValue string `json:"strValue,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Key") to
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

// ListSubscriptionsResponse: Response for the ListSubscriptions method.
type ListSubscriptionsResponse struct {
	// NextPageToken: If not empty, indicates that there are more
	// subscriptions that match the request and this value should be passed
	// to the next ListSubscriptionsRequest to continue.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Subscription: The subscriptions that match the request.
	Subscription []*Subscription `json:"subscription,omitempty"`

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

// ListTopicsResponse: Response for the ListTopics method.
type ListTopicsResponse struct {
	// NextPageToken: If not empty, indicates that there are more topics
	// that match the request, and this value should be passed to the next
	// ListTopicsRequest to continue.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Topic: The resulting topics.
	Topic []*Topic `json:"topic,omitempty"`

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
	// seconds after the ModifyAckDeadline call was made. Specifying zero
	// may immediately make the message available for another pull request.
	AckDeadlineSeconds int64 `json:"ackDeadlineSeconds,omitempty"`

	// AckId: The acknowledgment ID. Either this or ack_ids must be
	// populated, not both.
	AckId string `json:"ackId,omitempty"`

	// AckIds: List of acknowledgment IDs. Either this field or ack_id
	// should be populated, not both.
	AckIds []string `json:"ackIds,omitempty"`

	// Subscription: Next Index: 5 The name of the subscription from which
	// messages are being pulled.
	Subscription string `json:"subscription,omitempty"`

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
	// PushConfig: An empty push_config indicates that the Pub/Sub system
	// should pause pushing messages from the given subscription.
	PushConfig *PushConfig `json:"pushConfig,omitempty"`

	// Subscription: The name of the subscription.
	Subscription string `json:"subscription,omitempty"`

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

// PublishBatchRequest: Request for the PublishBatch method.
type PublishBatchRequest struct {
	// Messages: The messages to publish.
	Messages []*PubsubMessage `json:"messages,omitempty"`

	// Topic: The messages in the request will be published on this topic.
	Topic string `json:"topic,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Messages") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PublishBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod PublishBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PublishBatchResponse: Response for the PublishBatch method.
type PublishBatchResponse struct {
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

func (s *PublishBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod PublishBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PublishRequest: Request for the Publish method.
type PublishRequest struct {
	// Message: The message to publish.
	Message *PubsubMessage `json:"message,omitempty"`

	// Topic: The message in the request will be published on this topic.
	Topic string `json:"topic,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Message") to
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

// PubsubEvent: An event indicating a received message or truncation
// event.
type PubsubEvent struct {
	// Deleted: Indicates that this subscription has been deleted. (Note
	// that pull subscribers will always receive NOT_FOUND in response in
	// their pull request on the subscription, rather than seeing this
	// boolean.)
	Deleted bool `json:"deleted,omitempty"`

	// Message: A received message.
	Message *PubsubMessage `json:"message,omitempty"`

	// Subscription: The subscription that received the event.
	Subscription string `json:"subscription,omitempty"`

	// Truncated: Indicates that this subscription has been truncated.
	Truncated bool `json:"truncated,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Deleted") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PubsubEvent) MarshalJSON() ([]byte, error) {
	type noMethod PubsubEvent
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PubsubMessage: A message data and its labels.
type PubsubMessage struct {
	// Data: The message payload.
	Data string `json:"data,omitempty"`

	// Label: Optional list of labels for this message. Keys in this
	// collection must be unique.
	Label []*Label `json:"label,omitempty"`

	// MessageId: ID of this message assigned by the server at publication
	// time. Guaranteed to be unique within the topic. This value may be
	// read by a subscriber that receives a PubsubMessage via a Pull call or
	// a push delivery. It must not be populated by a publisher in a Publish
	// call.
	MessageId string `json:"messageId,omitempty"`

	// PublishTime: The time at which the message was published. The time is
	// milliseconds since the UNIX epoch.
	PublishTime int64 `json:"publishTime,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Data") to
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

// PullBatchRequest: Request for the PullBatch method.
type PullBatchRequest struct {
	// MaxEvents: The maximum number of PubsubEvents returned for this
	// request. The Pub/Sub system may return fewer than the number of
	// events specified.
	MaxEvents int64 `json:"maxEvents,omitempty"`

	// ReturnImmediately: If this is specified as true the system will
	// respond immediately even if it is not able to return a message in the
	// Pull response. Otherwise the system is allowed to wait until at least
	// one message is available rather than returning no messages. The
	// client may cancel the request if it does not wish to wait any longer
	// for the response.
	ReturnImmediately bool `json:"returnImmediately,omitempty"`

	// Subscription: The subscription from which messages should be pulled.
	Subscription string `json:"subscription,omitempty"`

	// ForceSendFields is a list of field names (e.g. "MaxEvents") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PullBatchRequest) MarshalJSON() ([]byte, error) {
	type noMethod PullBatchRequest
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PullBatchResponse: Response for the PullBatch method.
type PullBatchResponse struct {
	// PullResponses: Received Pub/Sub messages or status events. The
	// Pub/Sub system will return zero messages if there are no more
	// messages available in the backlog. The Pub/Sub system may return
	// fewer than the max_events requested even if there are more messages
	// available in the backlog.
	PullResponses []*PullResponse `json:"pullResponses,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "PullResponses") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *PullBatchResponse) MarshalJSON() ([]byte, error) {
	type noMethod PullBatchResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// PullRequest: Request for the Pull method.
type PullRequest struct {
	// ReturnImmediately: If this is specified as true the system will
	// respond immediately even if it is not able to return a message in the
	// Pull response. Otherwise the system is allowed to wait until at least
	// one message is available rather than returning FAILED_PRECONDITION.
	// The client may cancel the request if it does not wish to wait any
	// longer for the response.
	ReturnImmediately bool `json:"returnImmediately,omitempty"`

	// Subscription: The subscription from which a message should be pulled.
	Subscription string `json:"subscription,omitempty"`

	// ForceSendFields is a list of field names (e.g. "ReturnImmediately")
	// to unconditionally include in API requests. By default, fields with
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

// PullResponse: Either a PubsubMessage or a truncation event. One of
// these two must be populated.
type PullResponse struct {
	// AckId: This ID must be used to acknowledge the received event or
	// message.
	AckId string `json:"ackId,omitempty"`

	// PubsubEvent: A pubsub message or truncation event.
	PubsubEvent *PubsubEvent `json:"pubsubEvent,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AckId") to
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
	// PushEndpoint: A URL locating the endpoint to which messages should be
	// pushed. For example, a Webhook endpoint might use
	// "https://example.com/push".
	PushEndpoint string `json:"pushEndpoint,omitempty"`

	// ForceSendFields is a list of field names (e.g. "PushEndpoint") to
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

// Subscription: A subscription resource.
type Subscription struct {
	// AckDeadlineSeconds: For either push or pull delivery, the value is
	// the maximum time after a subscriber receives a message before the
	// subscriber should acknowledge or Nack the message. If the Ack
	// deadline for a message passes without an Ack or a Nack, the Pub/Sub
	// system will eventually redeliver the message. If a subscriber
	// acknowledges after the deadline, the Pub/Sub system may accept the
	// Ack, but it is possible that the message has been already delivered
	// again. Multiple Acks to the message are allowed and will succeed. For
	// push delivery, this value is used to set the request timeout for the
	// call to the push endpoint. For pull delivery, this value is used as
	// the initial value for the Ack deadline. It may be overridden for each
	// message using its corresponding ack_id with ModifyAckDeadline. While
	// a message is outstanding (i.e. it has been delivered to a pull
	// subscriber and the subscriber has not yet Acked or Nacked), the
	// Pub/Sub system will not deliver that message to another pull
	// subscriber (on a best-effort basis).
	AckDeadlineSeconds int64 `json:"ackDeadlineSeconds,omitempty"`

	// Name: Name of the subscription.
	Name string `json:"name,omitempty"`

	// PushConfig: If push delivery is used with this subscription, this
	// field is used to configure it.
	PushConfig *PushConfig `json:"pushConfig,omitempty"`

	// Topic: The name of the topic from which this subscription is
	// receiving messages.
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

// Topic: A topic resource.
type Topic struct {
	// Name: Name of the topic.
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

// method id "pubsub.subscriptions.acknowledge":

type SubscriptionsAcknowledgeCall struct {
	s                  *Service
	acknowledgerequest *AcknowledgeRequest
	urlParams_         gensupport.URLParams
	ctx_               context.Context
}

// Acknowledge: Acknowledges a particular received message: the Pub/Sub
// system can remove the given message from the subscription.
// Acknowledging a message whose Ack deadline has expired may succeed,
// but the message could have been already redelivered. Acknowledging a
// message more than once will not result in an error. This is only used
// for messages received via pull.
func (r *SubscriptionsService) Acknowledge(acknowledgerequest *AcknowledgeRequest) *SubscriptionsAcknowledgeCall {
	c := &SubscriptionsAcknowledgeCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.acknowledgerequest = acknowledgerequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *SubscriptionsAcknowledgeCall) QuotaUser(quotaUser string) *SubscriptionsAcknowledgeCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsAcknowledgeCall) Fields(s ...googleapi.Field) *SubscriptionsAcknowledgeCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsAcknowledgeCall) Context(ctx context.Context) *SubscriptionsAcknowledgeCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsAcknowledgeCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.acknowledgerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/subscriptions/acknowledge")
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

// Do executes the "pubsub.subscriptions.acknowledge" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *SubscriptionsAcknowledgeCall) Do() (*Empty, error) {
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
	//   "description": "Acknowledges a particular received message: the Pub/Sub system can remove the given message from the subscription. Acknowledging a message whose Ack deadline has expired may succeed, but the message could have been already redelivered. Acknowledging a message more than once will not result in an error. This is only used for messages received via pull.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.subscriptions.acknowledge",
	//   "path": "v1beta1a/subscriptions/acknowledge",
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

// method id "pubsub.subscriptions.create":

type SubscriptionsCreateCall struct {
	s            *Service
	subscription *Subscription
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Create: Creates a subscription on a given topic for a given
// subscriber. If the subscription already exists, returns
// ALREADY_EXISTS. If the corresponding topic doesn't exist, returns
// NOT_FOUND. If the name is not provided in the request, the server
// will assign a random name for this subscription on the same project
// as the topic.
func (r *SubscriptionsService) Create(subscription *Subscription) *SubscriptionsCreateCall {
	c := &SubscriptionsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.subscription = subscription
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *SubscriptionsCreateCall) QuotaUser(quotaUser string) *SubscriptionsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsCreateCall) Fields(s ...googleapi.Field) *SubscriptionsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsCreateCall) Context(ctx context.Context) *SubscriptionsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.subscription)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/subscriptions")
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

// Do executes the "pubsub.subscriptions.create" call.
// Exactly one of *Subscription or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscription.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SubscriptionsCreateCall) Do() (*Subscription, error) {
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
	//   "description": "Creates a subscription on a given topic for a given subscriber. If the subscription already exists, returns ALREADY_EXISTS. If the corresponding topic doesn't exist, returns NOT_FOUND. If the name is not provided in the request, the server will assign a random name for this subscription on the same project as the topic.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.subscriptions.create",
	//   "path": "v1beta1a/subscriptions",
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

// method id "pubsub.subscriptions.delete":

type SubscriptionsDeleteCall struct {
	s            *Service
	subscription string
	urlParams_   gensupport.URLParams
	ctx_         context.Context
}

// Delete: Deletes an existing subscription. All pending messages in the
// subscription are immediately dropped. Calls to Pull after deletion
// will return NOT_FOUND.
func (r *SubscriptionsService) Delete(subscription string) *SubscriptionsDeleteCall {
	c := &SubscriptionsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.subscription = subscription
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *SubscriptionsDeleteCall) QuotaUser(quotaUser string) *SubscriptionsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsDeleteCall) Fields(s ...googleapi.Field) *SubscriptionsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsDeleteCall) Context(ctx context.Context) *SubscriptionsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/subscriptions/{+subscription}")
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

// Do executes the "pubsub.subscriptions.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *SubscriptionsDeleteCall) Do() (*Empty, error) {
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
	//   "description": "Deletes an existing subscription. All pending messages in the subscription are immediately dropped. Calls to Pull after deletion will return NOT_FOUND.",
	//   "httpMethod": "DELETE",
	//   "id": "pubsub.subscriptions.delete",
	//   "parameterOrder": [
	//     "subscription"
	//   ],
	//   "parameters": {
	//     "subscription": {
	//       "description": "The subscription to delete.",
	//       "location": "path",
	//       "pattern": "^.*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1a/subscriptions/{+subscription}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.subscriptions.get":

type SubscriptionsGetCall struct {
	s            *Service
	subscription string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the configuration details of a subscription.
func (r *SubscriptionsService) Get(subscription string) *SubscriptionsGetCall {
	c := &SubscriptionsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.subscription = subscription
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *SubscriptionsGetCall) QuotaUser(quotaUser string) *SubscriptionsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsGetCall) Fields(s ...googleapi.Field) *SubscriptionsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *SubscriptionsGetCall) IfNoneMatch(entityTag string) *SubscriptionsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsGetCall) Context(ctx context.Context) *SubscriptionsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/subscriptions/{+subscription}")
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

// Do executes the "pubsub.subscriptions.get" call.
// Exactly one of *Subscription or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Subscription.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SubscriptionsGetCall) Do() (*Subscription, error) {
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
	//   "id": "pubsub.subscriptions.get",
	//   "parameterOrder": [
	//     "subscription"
	//   ],
	//   "parameters": {
	//     "subscription": {
	//       "description": "The name of the subscription to get.",
	//       "location": "path",
	//       "pattern": "^.*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1a/subscriptions/{+subscription}",
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.subscriptions.list":

type SubscriptionsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists matching subscriptions.
func (r *SubscriptionsService) List() *SubscriptionsListCall {
	c := &SubscriptionsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of subscriptions to return.
func (c *SubscriptionsListCall) MaxResults(maxResults int64) *SubscriptionsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The value obtained
// in the last ListSubscriptionsResponse for continuation.
func (c *SubscriptionsListCall) PageToken(pageToken string) *SubscriptionsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Query sets the optional parameter "query": A valid label query
// expression.
func (c *SubscriptionsListCall) Query(query string) *SubscriptionsListCall {
	c.urlParams_.Set("query", query)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *SubscriptionsListCall) QuotaUser(quotaUser string) *SubscriptionsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsListCall) Fields(s ...googleapi.Field) *SubscriptionsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *SubscriptionsListCall) IfNoneMatch(entityTag string) *SubscriptionsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsListCall) Context(ctx context.Context) *SubscriptionsListCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/subscriptions")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.subscriptions.list" call.
// Exactly one of *ListSubscriptionsResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListSubscriptionsResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SubscriptionsListCall) Do() (*ListSubscriptionsResponse, error) {
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
	//   "id": "pubsub.subscriptions.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of subscriptions to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value obtained in the last ListSubscriptionsResponse for continuation.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "query": {
	//       "description": "A valid label query expression.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1a/subscriptions",
	//   "response": {
	//     "$ref": "ListSubscriptionsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.subscriptions.modifyAckDeadline":

type SubscriptionsModifyAckDeadlineCall struct {
	s                        *Service
	modifyackdeadlinerequest *ModifyAckDeadlineRequest
	urlParams_               gensupport.URLParams
	ctx_                     context.Context
}

// ModifyAckDeadline: Modifies the Ack deadline for a message received
// from a pull request.
func (r *SubscriptionsService) ModifyAckDeadline(modifyackdeadlinerequest *ModifyAckDeadlineRequest) *SubscriptionsModifyAckDeadlineCall {
	c := &SubscriptionsModifyAckDeadlineCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.modifyackdeadlinerequest = modifyackdeadlinerequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *SubscriptionsModifyAckDeadlineCall) QuotaUser(quotaUser string) *SubscriptionsModifyAckDeadlineCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsModifyAckDeadlineCall) Fields(s ...googleapi.Field) *SubscriptionsModifyAckDeadlineCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsModifyAckDeadlineCall) Context(ctx context.Context) *SubscriptionsModifyAckDeadlineCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsModifyAckDeadlineCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.modifyackdeadlinerequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/subscriptions/modifyAckDeadline")
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

// Do executes the "pubsub.subscriptions.modifyAckDeadline" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *SubscriptionsModifyAckDeadlineCall) Do() (*Empty, error) {
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
	//   "description": "Modifies the Ack deadline for a message received from a pull request.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.subscriptions.modifyAckDeadline",
	//   "path": "v1beta1a/subscriptions/modifyAckDeadline",
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

// method id "pubsub.subscriptions.modifyPushConfig":

type SubscriptionsModifyPushConfigCall struct {
	s                       *Service
	modifypushconfigrequest *ModifyPushConfigRequest
	urlParams_              gensupport.URLParams
	ctx_                    context.Context
}

// ModifyPushConfig: Modifies the PushConfig for a specified
// subscription. This method can be used to suspend the flow of messages
// to an endpoint by clearing the PushConfig field in the request.
// Messages will be accumulated for delivery even if no push
// configuration is defined or while the configuration is modified.
func (r *SubscriptionsService) ModifyPushConfig(modifypushconfigrequest *ModifyPushConfigRequest) *SubscriptionsModifyPushConfigCall {
	c := &SubscriptionsModifyPushConfigCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.modifypushconfigrequest = modifypushconfigrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *SubscriptionsModifyPushConfigCall) QuotaUser(quotaUser string) *SubscriptionsModifyPushConfigCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsModifyPushConfigCall) Fields(s ...googleapi.Field) *SubscriptionsModifyPushConfigCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsModifyPushConfigCall) Context(ctx context.Context) *SubscriptionsModifyPushConfigCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsModifyPushConfigCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.modifypushconfigrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/subscriptions/modifyPushConfig")
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

// Do executes the "pubsub.subscriptions.modifyPushConfig" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *SubscriptionsModifyPushConfigCall) Do() (*Empty, error) {
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
	//   "description": "Modifies the PushConfig for a specified subscription. This method can be used to suspend the flow of messages to an endpoint by clearing the PushConfig field in the request. Messages will be accumulated for delivery even if no push configuration is defined or while the configuration is modified.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.subscriptions.modifyPushConfig",
	//   "path": "v1beta1a/subscriptions/modifyPushConfig",
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

// method id "pubsub.subscriptions.pull":

type SubscriptionsPullCall struct {
	s           *Service
	pullrequest *PullRequest
	urlParams_  gensupport.URLParams
	ctx_        context.Context
}

// Pull: Pulls a single message from the server. If return_immediately
// is true, and no messages are available in the subscription, this
// method returns FAILED_PRECONDITION. The system is free to return an
// UNAVAILABLE error if no messages are available in a reasonable amount
// of time (to reduce system load).
func (r *SubscriptionsService) Pull(pullrequest *PullRequest) *SubscriptionsPullCall {
	c := &SubscriptionsPullCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.pullrequest = pullrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *SubscriptionsPullCall) QuotaUser(quotaUser string) *SubscriptionsPullCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsPullCall) Fields(s ...googleapi.Field) *SubscriptionsPullCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsPullCall) Context(ctx context.Context) *SubscriptionsPullCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsPullCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pullrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/subscriptions/pull")
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

// Do executes the "pubsub.subscriptions.pull" call.
// Exactly one of *PullResponse or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *PullResponse.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *SubscriptionsPullCall) Do() (*PullResponse, error) {
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
	//   "description": "Pulls a single message from the server. If return_immediately is true, and no messages are available in the subscription, this method returns FAILED_PRECONDITION. The system is free to return an UNAVAILABLE error if no messages are available in a reasonable amount of time (to reduce system load).",
	//   "httpMethod": "POST",
	//   "id": "pubsub.subscriptions.pull",
	//   "path": "v1beta1a/subscriptions/pull",
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

// method id "pubsub.subscriptions.pullBatch":

type SubscriptionsPullBatchCall struct {
	s                *Service
	pullbatchrequest *PullBatchRequest
	urlParams_       gensupport.URLParams
	ctx_             context.Context
}

// PullBatch: Pulls messages from the server. Returns an empty list if
// there are no messages available in the backlog. The system is free to
// return UNAVAILABLE if there are too many pull requests outstanding
// for the given subscription.
func (r *SubscriptionsService) PullBatch(pullbatchrequest *PullBatchRequest) *SubscriptionsPullBatchCall {
	c := &SubscriptionsPullBatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.pullbatchrequest = pullbatchrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *SubscriptionsPullBatchCall) QuotaUser(quotaUser string) *SubscriptionsPullBatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *SubscriptionsPullBatchCall) Fields(s ...googleapi.Field) *SubscriptionsPullBatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *SubscriptionsPullBatchCall) Context(ctx context.Context) *SubscriptionsPullBatchCall {
	c.ctx_ = ctx
	return c
}

func (c *SubscriptionsPullBatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.pullbatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/subscriptions/pullBatch")
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

// Do executes the "pubsub.subscriptions.pullBatch" call.
// Exactly one of *PullBatchResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *PullBatchResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *SubscriptionsPullBatchCall) Do() (*PullBatchResponse, error) {
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
	ret := &PullBatchResponse{
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
	//   "description": "Pulls messages from the server. Returns an empty list if there are no messages available in the backlog. The system is free to return UNAVAILABLE if there are too many pull requests outstanding for the given subscription.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.subscriptions.pullBatch",
	//   "path": "v1beta1a/subscriptions/pullBatch",
	//   "request": {
	//     "$ref": "PullBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "PullBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.topics.create":

type TopicsCreateCall struct {
	s          *Service
	topic      *Topic
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Creates the given topic with the given name.
func (r *TopicsService) Create(topic *Topic) *TopicsCreateCall {
	c := &TopicsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.topic = topic
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *TopicsCreateCall) QuotaUser(quotaUser string) *TopicsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TopicsCreateCall) Fields(s ...googleapi.Field) *TopicsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TopicsCreateCall) Context(ctx context.Context) *TopicsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *TopicsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.topic)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/topics")
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

// Do executes the "pubsub.topics.create" call.
// Exactly one of *Topic or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Topic.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TopicsCreateCall) Do() (*Topic, error) {
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
	//   "httpMethod": "POST",
	//   "id": "pubsub.topics.create",
	//   "path": "v1beta1a/topics",
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

// method id "pubsub.topics.delete":

type TopicsDeleteCall struct {
	s          *Service
	topic      string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes the topic with the given name. Returns NOT_FOUND if
// the topic does not exist. After a topic is deleted, a new topic may
// be created with the same name.
func (r *TopicsService) Delete(topic string) *TopicsDeleteCall {
	c := &TopicsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.topic = topic
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *TopicsDeleteCall) QuotaUser(quotaUser string) *TopicsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TopicsDeleteCall) Fields(s ...googleapi.Field) *TopicsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TopicsDeleteCall) Context(ctx context.Context) *TopicsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *TopicsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/topics/{+topic}")
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

// Do executes the "pubsub.topics.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TopicsDeleteCall) Do() (*Empty, error) {
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
	//   "description": "Deletes the topic with the given name. Returns NOT_FOUND if the topic does not exist. After a topic is deleted, a new topic may be created with the same name.",
	//   "httpMethod": "DELETE",
	//   "id": "pubsub.topics.delete",
	//   "parameterOrder": [
	//     "topic"
	//   ],
	//   "parameters": {
	//     "topic": {
	//       "description": "Name of the topic to delete.",
	//       "location": "path",
	//       "pattern": "^.*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1a/topics/{+topic}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.topics.get":

type TopicsGetCall struct {
	s            *Service
	topic        string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Gets the configuration of a topic. Since the topic only has the
// name attribute, this method is only useful to check the existence of
// a topic. If other attributes are added in the future, they will be
// returned here.
func (r *TopicsService) Get(topic string) *TopicsGetCall {
	c := &TopicsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.topic = topic
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *TopicsGetCall) QuotaUser(quotaUser string) *TopicsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TopicsGetCall) Fields(s ...googleapi.Field) *TopicsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TopicsGetCall) IfNoneMatch(entityTag string) *TopicsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TopicsGetCall) Context(ctx context.Context) *TopicsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *TopicsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/topics/{+topic}")
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

// Do executes the "pubsub.topics.get" call.
// Exactly one of *Topic or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Topic.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TopicsGetCall) Do() (*Topic, error) {
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
	//   "description": "Gets the configuration of a topic. Since the topic only has the name attribute, this method is only useful to check the existence of a topic. If other attributes are added in the future, they will be returned here.",
	//   "httpMethod": "GET",
	//   "id": "pubsub.topics.get",
	//   "parameterOrder": [
	//     "topic"
	//   ],
	//   "parameters": {
	//     "topic": {
	//       "description": "The name of the topic to get.",
	//       "location": "path",
	//       "pattern": "^.*$",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1a/topics/{+topic}",
	//   "response": {
	//     "$ref": "Topic"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.topics.list":

type TopicsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Lists matching topics.
func (r *TopicsService) List() *TopicsListCall {
	c := &TopicsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of topics to return.
func (c *TopicsListCall) MaxResults(maxResults int64) *TopicsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": The value obtained
// in the last ListTopicsResponse for continuation.
func (c *TopicsListCall) PageToken(pageToken string) *TopicsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// Query sets the optional parameter "query": A valid label query
// expression.
func (c *TopicsListCall) Query(query string) *TopicsListCall {
	c.urlParams_.Set("query", query)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *TopicsListCall) QuotaUser(quotaUser string) *TopicsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TopicsListCall) Fields(s ...googleapi.Field) *TopicsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *TopicsListCall) IfNoneMatch(entityTag string) *TopicsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TopicsListCall) Context(ctx context.Context) *TopicsListCall {
	c.ctx_ = ctx
	return c
}

func (c *TopicsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/topics")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		req.Header.Set("If-None-Match", c.ifNoneMatch_)
	}
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "pubsub.topics.list" call.
// Exactly one of *ListTopicsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListTopicsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TopicsListCall) Do() (*ListTopicsResponse, error) {
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
	//   "id": "pubsub.topics.list",
	//   "parameters": {
	//     "maxResults": {
	//       "description": "Maximum number of topics to return.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The value obtained in the last ListTopicsResponse for continuation.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "query": {
	//       "description": "A valid label query expression.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1beta1a/topics",
	//   "response": {
	//     "$ref": "ListTopicsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}

// method id "pubsub.topics.publish":

type TopicsPublishCall struct {
	s              *Service
	publishrequest *PublishRequest
	urlParams_     gensupport.URLParams
	ctx_           context.Context
}

// Publish: Adds a message to the topic. Returns NOT_FOUND if the topic
// does not exist.
func (r *TopicsService) Publish(publishrequest *PublishRequest) *TopicsPublishCall {
	c := &TopicsPublishCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.publishrequest = publishrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *TopicsPublishCall) QuotaUser(quotaUser string) *TopicsPublishCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TopicsPublishCall) Fields(s ...googleapi.Field) *TopicsPublishCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TopicsPublishCall) Context(ctx context.Context) *TopicsPublishCall {
	c.ctx_ = ctx
	return c
}

func (c *TopicsPublishCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.publishrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/topics/publish")
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

// Do executes the "pubsub.topics.publish" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *TopicsPublishCall) Do() (*Empty, error) {
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
	//   "description": "Adds a message to the topic. Returns NOT_FOUND if the topic does not exist.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.topics.publish",
	//   "path": "v1beta1a/topics/publish",
	//   "request": {
	//     "$ref": "PublishRequest"
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

// method id "pubsub.topics.publishBatch":

type TopicsPublishBatchCall struct {
	s                   *Service
	publishbatchrequest *PublishBatchRequest
	urlParams_          gensupport.URLParams
	ctx_                context.Context
}

// PublishBatch: Adds one or more messages to the topic. Returns
// NOT_FOUND if the topic does not exist.
func (r *TopicsService) PublishBatch(publishbatchrequest *PublishBatchRequest) *TopicsPublishBatchCall {
	c := &TopicsPublishBatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.publishbatchrequest = publishbatchrequest
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *TopicsPublishBatchCall) QuotaUser(quotaUser string) *TopicsPublishBatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *TopicsPublishBatchCall) Fields(s ...googleapi.Field) *TopicsPublishBatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *TopicsPublishBatchCall) Context(ctx context.Context) *TopicsPublishBatchCall {
	c.ctx_ = ctx
	return c
}

func (c *TopicsPublishBatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.publishbatchrequest)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1beta1a/topics/publishBatch")
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

// Do executes the "pubsub.topics.publishBatch" call.
// Exactly one of *PublishBatchResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *PublishBatchResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *TopicsPublishBatchCall) Do() (*PublishBatchResponse, error) {
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
	ret := &PublishBatchResponse{
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
	//   "description": "Adds one or more messages to the topic. Returns NOT_FOUND if the topic does not exist.",
	//   "httpMethod": "POST",
	//   "id": "pubsub.topics.publishBatch",
	//   "path": "v1beta1a/topics/publishBatch",
	//   "request": {
	//     "$ref": "PublishBatchRequest"
	//   },
	//   "response": {
	//     "$ref": "PublishBatchResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/cloud-platform",
	//     "https://www.googleapis.com/auth/pubsub"
	//   ]
	// }

}
