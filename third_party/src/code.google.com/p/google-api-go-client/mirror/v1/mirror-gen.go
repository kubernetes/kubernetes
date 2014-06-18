// Package mirror provides access to the Google Mirror API.
//
// See https://developers.google.com/glass
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/mirror/v1"
//   ...
//   mirrorService, err := mirror.New(oauthHttpClient)
package mirror

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

const apiId = "mirror:v1"
const apiName = "mirror"
const apiVersion = "v1"
const basePath = "https://www.googleapis.com/mirror/v1/"

// OAuth2 scopes used by this API.
const (
	// View your location
	GlassLocationScope = "https://www.googleapis.com/auth/glass.location"

	// View and manage your Glass timeline
	GlassTimelineScope = "https://www.googleapis.com/auth/glass.timeline"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Accounts = NewAccountsService(s)
	s.Contacts = NewContactsService(s)
	s.Locations = NewLocationsService(s)
	s.Subscriptions = NewSubscriptionsService(s)
	s.Timeline = NewTimelineService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Accounts *AccountsService

	Contacts *ContactsService

	Locations *LocationsService

	Subscriptions *SubscriptionsService

	Timeline *TimelineService
}

func NewAccountsService(s *Service) *AccountsService {
	rs := &AccountsService{s: s}
	return rs
}

type AccountsService struct {
	s *Service
}

func NewContactsService(s *Service) *ContactsService {
	rs := &ContactsService{s: s}
	return rs
}

type ContactsService struct {
	s *Service
}

func NewLocationsService(s *Service) *LocationsService {
	rs := &LocationsService{s: s}
	return rs
}

type LocationsService struct {
	s *Service
}

func NewSubscriptionsService(s *Service) *SubscriptionsService {
	rs := &SubscriptionsService{s: s}
	return rs
}

type SubscriptionsService struct {
	s *Service
}

func NewTimelineService(s *Service) *TimelineService {
	rs := &TimelineService{s: s}
	rs.Attachments = NewTimelineAttachmentsService(s)
	return rs
}

type TimelineService struct {
	s *Service

	Attachments *TimelineAttachmentsService
}

func NewTimelineAttachmentsService(s *Service) *TimelineAttachmentsService {
	rs := &TimelineAttachmentsService{s: s}
	return rs
}

type TimelineAttachmentsService struct {
	s *Service
}

type Account struct {
	AuthTokens []*AuthToken `json:"authTokens,omitempty"`

	Features []string `json:"features,omitempty"`

	Password string `json:"password,omitempty"`

	UserData []*UserData `json:"userData,omitempty"`
}

type Attachment struct {
	// ContentType: The MIME type of the attachment.
	ContentType string `json:"contentType,omitempty"`

	// ContentUrl: The URL for the content.
	ContentUrl string `json:"contentUrl,omitempty"`

	// Id: The ID of the attachment.
	Id string `json:"id,omitempty"`

	// IsProcessingContent: Indicates that the contentUrl is not available
	// because the attachment content is still being processed. If the
	// caller wishes to retrieve the content, it should try again later.
	IsProcessingContent bool `json:"isProcessingContent,omitempty"`
}

type AttachmentsListResponse struct {
	// Items: The list of attachments.
	Items []*Attachment `json:"items,omitempty"`

	// Kind: The type of resource. This is always mirror#attachmentsList.
	Kind string `json:"kind,omitempty"`
}

type AuthToken struct {
	AuthToken string `json:"authToken,omitempty"`

	Type string `json:"type,omitempty"`
}

type Command struct {
	// Type: The type of operation this command corresponds to. Allowed
	// values are:
	// - TAKE_A_NOTE - Shares a timeline item with the
	// transcription of user speech from the "Take a note" voice menu
	// command.
	// - POST_AN_UPDATE - Shares a timeline item with the
	// transcription of user speech from the "Post an update" voice menu
	// command.
	Type string `json:"type,omitempty"`
}

type Contact struct {
	// AcceptCommands: A list of voice menu commands that a contact can
	// handle. Glass shows up to three contacts for each voice menu command.
	// If there are more than that, the three contacts with the highest
	// priority are shown for that particular command.
	AcceptCommands []*Command `json:"acceptCommands,omitempty"`

	// AcceptTypes: A list of MIME types that a contact supports. The
	// contact will be shown to the user if any of its acceptTypes matches
	// any of the types of the attachments on the item. If no acceptTypes
	// are given, the contact will be shown for all items.
	AcceptTypes []string `json:"acceptTypes,omitempty"`

	// DisplayName: The name to display for this contact.
	DisplayName string `json:"displayName,omitempty"`

	// Id: An ID for this contact. This is generated by the application and
	// is treated as an opaque token.
	Id string `json:"id,omitempty"`

	// ImageUrls: Set of image URLs to display for a contact. Most contacts
	// will have a single image, but a "group" contact may include up to 8
	// image URLs and they will be resized and cropped into a mosaic on the
	// client.
	ImageUrls []string `json:"imageUrls,omitempty"`

	// Kind: The type of resource. This is always mirror#contact.
	Kind string `json:"kind,omitempty"`

	// PhoneNumber: Primary phone number for the contact. This can be a
	// fully-qualified number, with country calling code and area code, or a
	// local number.
	PhoneNumber string `json:"phoneNumber,omitempty"`

	// Priority: Priority for the contact to determine ordering in a list of
	// contacts. Contacts with higher priorities will be shown before ones
	// with lower priorities.
	Priority int64 `json:"priority,omitempty"`

	// SharingFeatures: A list of sharing features that a contact can
	// handle. Allowed values are:
	// - ADD_CAPTION
	SharingFeatures []string `json:"sharingFeatures,omitempty"`

	// Source: The ID of the application that created this contact. This is
	// populated by the API
	Source string `json:"source,omitempty"`

	// SpeakableName: Name of this contact as it should be pronounced. If
	// this contact's name must be spoken as part of a voice disambiguation
	// menu, this name is used as the expected pronunciation. This is useful
	// for contact names with unpronounceable characters or whose display
	// spelling is otherwise not phonetic.
	SpeakableName string `json:"speakableName,omitempty"`

	// Type: The type for this contact. This is used for sorting in UIs.
	// Allowed values are:
	// - INDIVIDUAL - Represents a single person. This
	// is the default.
	// - GROUP - Represents more than a single person.
	Type string `json:"type,omitempty"`
}

type ContactsListResponse struct {
	// Items: Contact list.
	Items []*Contact `json:"items,omitempty"`

	// Kind: The type of resource. This is always mirror#contacts.
	Kind string `json:"kind,omitempty"`
}

type Location struct {
	// Accuracy: The accuracy of the location fix in meters.
	Accuracy float64 `json:"accuracy,omitempty"`

	// Address: The full address of the location.
	Address string `json:"address,omitempty"`

	// DisplayName: The name to be displayed. This may be a business name or
	// a user-defined place, such as "Home".
	DisplayName string `json:"displayName,omitempty"`

	// Id: The ID of the location.
	Id string `json:"id,omitempty"`

	// Kind: The type of resource. This is always mirror#location.
	Kind string `json:"kind,omitempty"`

	// Latitude: The latitude, in degrees.
	Latitude float64 `json:"latitude,omitempty"`

	// Longitude: The longitude, in degrees.
	Longitude float64 `json:"longitude,omitempty"`

	// Timestamp: The time at which this location was captured, formatted
	// according to RFC 3339.
	Timestamp string `json:"timestamp,omitempty"`
}

type LocationsListResponse struct {
	// Items: The list of locations.
	Items []*Location `json:"items,omitempty"`

	// Kind: The type of resource. This is always mirror#locationsList.
	Kind string `json:"kind,omitempty"`
}

type MenuItem struct {
	// Action: Controls the behavior when the user picks the menu option.
	// Allowed values are:
	// - CUSTOM - Custom action set by the service.
	// When the user selects this menuItem, the API triggers a notification
	// to your callbackUrl with the userActions.type set to CUSTOM and the
	// userActions.payload set to the ID of this menu item. This is the
	// default value.
	// - Built-in actions:
	// - REPLY - Initiate a reply to
	// the timeline item using the voice recording UI. The creator attribute
	// must be set in the timeline item for this menu to be available.
	// -
	// REPLY_ALL - Same behavior as REPLY. The original timeline item's
	// recipients will be added to the reply item.
	// - DELETE - Delete the
	// timeline item.
	// - SHARE - Share the timeline item with the available
	// contacts.
	// - READ_ALOUD - Read the timeline item's speakableText
	// aloud; if this field is not set, read the text field; if none of
	// those fields are set, this menu item is ignored.
	// - GET_MEDIA_INPUT -
	// Allow users to provide media payloads to Glassware from a menu item
	// (currently, only transcribed text from voice input is supported).
	// Subscribe to notifications when users invoke this menu item to
	// receive the timeline item ID. Retrieve the media from the timeline
	// item in the payload property.
	// - VOICE_CALL - Initiate a phone call
	// using the timeline item's creator.phoneNumber attribute as recipient.
	//
	// - NAVIGATE - Navigate to the timeline item's location.
	// -
	// TOGGLE_PINNED - Toggle the isPinned state of the timeline item.
	// -
	// OPEN_URI - Open the payload of the menu item in the browser.
	// -
	// PLAY_VIDEO - Open the payload of the menu item in the Glass video
	// player.
	// - SEND_MESSAGE - Initiate sending a message to the timeline
	// item's creator:
	// - If the creator.phoneNumber is set and Glass is
	// connected to an Android phone, the message is an SMS.
	// - Otherwise,
	// if the creator.email is set, the message is an email.
	Action string `json:"action,omitempty"`

	// Id: The ID for this menu item. This is generated by the application
	// and is treated as an opaque token.
	Id string `json:"id,omitempty"`

	// Payload: A generic payload whose meaning changes depending on this
	// MenuItem's action.
	// - When the action is OPEN_URI, the payload is
	// the URL of the website to view.
	// - When the action is PLAY_VIDEO, the
	// payload is the streaming URL of the video
	// - When the action is
	// GET_MEDIA_INPUT, the payload is the text transcription of a user's
	// speech input
	Payload string `json:"payload,omitempty"`

	// RemoveWhenSelected: If set to true on a CUSTOM menu item, that item
	// will be removed from the menu after it is selected.
	RemoveWhenSelected bool `json:"removeWhenSelected,omitempty"`

	// Values: For CUSTOM items, a list of values controlling the appearance
	// of the menu item in each of its states. A value for the DEFAULT state
	// must be provided. If the PENDING or CONFIRMED states are missing,
	// they will not be shown.
	Values []*MenuValue `json:"values,omitempty"`
}

type MenuValue struct {
	// DisplayName: The name to display for the menu item. If you specify
	// this property for a built-in menu item, the default contextual voice
	// command for that menu item is not shown.
	DisplayName string `json:"displayName,omitempty"`

	// IconUrl: URL of an icon to display with the menu item.
	IconUrl string `json:"iconUrl,omitempty"`

	// State: The state that this value applies to. Allowed values are:
	// -
	// DEFAULT - Default value shown when displayed in the menuItems list.
	//
	// - PENDING - Value shown when the menuItem has been selected by the
	// user but can still be cancelled.
	// - CONFIRMED - Value shown when the
	// menuItem has been selected by the user and can no longer be
	// cancelled.
	State string `json:"state,omitempty"`
}

type Notification struct {
	// Collection: The collection that generated the notification.
	Collection string `json:"collection,omitempty"`

	// ItemId: The ID of the item that generated the notification.
	ItemId string `json:"itemId,omitempty"`

	// Operation: The type of operation that generated the notification.
	Operation string `json:"operation,omitempty"`

	// UserActions: A list of actions taken by the user that triggered the
	// notification.
	UserActions []*UserAction `json:"userActions,omitempty"`

	// UserToken: The user token provided by the service when it subscribed
	// for notifications.
	UserToken string `json:"userToken,omitempty"`

	// VerifyToken: The secret verify token provided by the service when it
	// subscribed for notifications.
	VerifyToken string `json:"verifyToken,omitempty"`
}

type NotificationConfig struct {
	// DeliveryTime: The time at which the notification should be delivered.
	DeliveryTime string `json:"deliveryTime,omitempty"`

	// Level: Describes how important the notification is. Allowed values
	// are:
	// - DEFAULT - Notifications of default importance. A chime will
	// be played to alert users.
	Level string `json:"level,omitempty"`
}

type Subscription struct {
	// CallbackUrl: The URL where notifications should be delivered (must
	// start with https://).
	CallbackUrl string `json:"callbackUrl,omitempty"`

	// Collection: The collection to subscribe to. Allowed values are:
	// -
	// timeline - Changes in the timeline including insertion, deletion, and
	// updates.
	// - locations - Location updates.
	Collection string `json:"collection,omitempty"`

	// Id: The ID of the subscription.
	Id string `json:"id,omitempty"`

	// Kind: The type of resource. This is always mirror#subscription.
	Kind string `json:"kind,omitempty"`

	// Notification: Container object for notifications. This is not
	// populated in the Subscription resource.
	Notification *Notification `json:"notification,omitempty"`

	// Operation: A list of operations that should be subscribed to. An
	// empty list indicates that all operations on the collection should be
	// subscribed to. Allowed values are:
	// - UPDATE - The item has been
	// updated.
	// - INSERT - A new item has been inserted.
	// - DELETE - The
	// item has been deleted.
	// - MENU_ACTION - A custom menu item has been
	// triggered by the user.
	Operation []string `json:"operation,omitempty"`

	// Updated: The time at which this subscription was last modified,
	// formatted according to RFC 3339.
	Updated string `json:"updated,omitempty"`

	// UserToken: An opaque token sent to the subscriber in notifications so
	// that it can determine the ID of the user.
	UserToken string `json:"userToken,omitempty"`

	// VerifyToken: A secret token sent to the subscriber in notifications
	// so that it can verify that the notification was generated by Google.
	VerifyToken string `json:"verifyToken,omitempty"`
}

type SubscriptionsListResponse struct {
	// Items: The list of subscriptions.
	Items []*Subscription `json:"items,omitempty"`

	// Kind: The type of resource. This is always mirror#subscriptionsList.
	Kind string `json:"kind,omitempty"`
}

type TimelineItem struct {
	// Attachments: A list of media attachments associated with this item.
	// As a convenience, you can refer to attachments in your HTML payloads
	// with the attachment or cid scheme. For example:
	// - attachment: <img
	// src="attachment:attachment_index"> where attachment_index is the
	// 0-based index of this array.
	// - cid: <img src="cid:attachment_id">
	// where attachment_id is the ID of the attachment.
	Attachments []*Attachment `json:"attachments,omitempty"`

	// BundleId: The bundle ID for this item. Services can specify a
	// bundleId to group many items together. They appear under a single
	// top-level item on the device.
	BundleId string `json:"bundleId,omitempty"`

	// CanonicalUrl: A canonical URL pointing to the canonical/high quality
	// version of the data represented by the timeline item.
	CanonicalUrl string `json:"canonicalUrl,omitempty"`

	// Created: The time at which this item was created, formatted according
	// to RFC 3339.
	Created string `json:"created,omitempty"`

	// Creator: The user or group that created this item.
	Creator *Contact `json:"creator,omitempty"`

	// DisplayTime: The time that should be displayed when this item is
	// viewed in the timeline, formatted according to RFC 3339. This user's
	// timeline is sorted chronologically on display time, so this will also
	// determine where the item is displayed in the timeline. If not set by
	// the service, the display time defaults to the updated time.
	DisplayTime string `json:"displayTime,omitempty"`

	// Etag: ETag for this item.
	Etag string `json:"etag,omitempty"`

	// Html: HTML content for this item. If both text and html are provided
	// for an item, the html will be rendered in the timeline.
	// Allowed HTML
	// elements - You can use these elements in your timeline cards.
	//
	// -
	// Headers: h1, h2, h3, h4, h5, h6
	// - Images: img
	// - Lists: li, ol, ul
	//
	// - HTML5 semantics: article, aside, details, figure, figcaption,
	// footer, header, nav, section, summary, time
	// - Structural:
	// blockquote, br, div, hr, p, span
	// - Style: b, big, center, em, i, u,
	// s, small, strike, strong, style, sub, sup
	// - Tables: table, tbody,
	// td, tfoot, th, thead, tr
	// Blocked HTML elements: These elements and
	// their contents are removed from HTML payloads.
	//
	// - Document headers:
	// head, title
	// - Embeds: audio, embed, object, source, video
	// - Frames:
	// frame, frameset
	// - Scripting: applet, script
	// Other elements: Any
	// elements that aren't listed are removed, but their contents are
	// preserved.
	Html string `json:"html,omitempty"`

	// Id: The ID of the timeline item. This is unique within a user's
	// timeline.
	Id string `json:"id,omitempty"`

	// InReplyTo: If this item was generated as a reply to another item,
	// this field will be set to the ID of the item being replied to. This
	// can be used to attach a reply to the appropriate conversation or
	// post.
	InReplyTo string `json:"inReplyTo,omitempty"`

	// IsBundleCover: Whether this item is a bundle cover.
	//
	// If an item is
	// marked as a bundle cover, it will be the entry point to the bundle of
	// items that have the same bundleId as that item. It will be shown only
	// on the main timeline — not within the opened bundle.
	//
	// On the main
	// timeline, items that are shown are:
	// - Items that have isBundleCover
	// set to true
	// - Items that do not have a bundleId  In a bundle
	// sub-timeline, items that are shown are:
	// - Items that have the
	// bundleId in question AND isBundleCover set to false
	IsBundleCover bool `json:"isBundleCover,omitempty"`

	// IsDeleted: When true, indicates this item is deleted, and only the ID
	// property is set.
	IsDeleted bool `json:"isDeleted,omitempty"`

	// IsPinned: When true, indicates this item is pinned, which means it's
	// grouped alongside "active" items like navigation and hangouts, on the
	// opposite side of the home screen from historical (non-pinned)
	// timeline items. You can allow the user to toggle the value of this
	// property with the TOGGLE_PINNED built-in menu item.
	IsPinned bool `json:"isPinned,omitempty"`

	// Kind: The type of resource. This is always mirror#timelineItem.
	Kind string `json:"kind,omitempty"`

	// Location: The geographic location associated with this item.
	Location *Location `json:"location,omitempty"`

	// MenuItems: A list of menu items that will be presented to the user
	// when this item is selected in the timeline.
	MenuItems []*MenuItem `json:"menuItems,omitempty"`

	// Notification: Controls how notifications for this item are presented
	// on the device. If this is missing, no notification will be generated.
	Notification *NotificationConfig `json:"notification,omitempty"`

	// PinScore: For pinned items, this determines the order in which the
	// item is displayed in the timeline, with a higher score appearing
	// closer to the clock. Note: setting this field is currently not
	// supported.
	PinScore int64 `json:"pinScore,omitempty"`

	// Recipients: A list of users or groups that this item has been shared
	// with.
	Recipients []*Contact `json:"recipients,omitempty"`

	// SelfLink: A URL that can be used to retrieve this item.
	SelfLink string `json:"selfLink,omitempty"`

	// SourceItemId: Opaque string you can use to map a timeline item to
	// data in your own service.
	SourceItemId string `json:"sourceItemId,omitempty"`

	// SpeakableText: The speakable version of the content of this item.
	// Along with the READ_ALOUD menu item, use this field to provide text
	// that would be clearer when read aloud, or to provide extended
	// information to what is displayed visually on Glass.
	//
	// Glassware should
	// also specify the speakableType field, which will be spoken before
	// this text in cases where the additional context is useful, for
	// example when the user requests that the item be read aloud following
	// a notification.
	SpeakableText string `json:"speakableText,omitempty"`

	// SpeakableType: A speakable description of the type of this item. This
	// will be announced to the user prior to reading the content of the
	// item in cases where the additional context is useful, for example
	// when the user requests that the item be read aloud following a
	// notification.
	//
	// This should be a short, simple noun phrase such as
	// "Email", "Text message", or "Daily Planet News Update".
	//
	// Glassware
	// are encouraged to populate this field for every timeline item, even
	// if the item does not contain speakableText or text so that the user
	// can learn the type of the item without looking at the screen.
	SpeakableType string `json:"speakableType,omitempty"`

	// Text: Text content of this item.
	Text string `json:"text,omitempty"`

	// Title: The title of this item.
	Title string `json:"title,omitempty"`

	// Updated: The time at which this item was last modified, formatted
	// according to RFC 3339.
	Updated string `json:"updated,omitempty"`
}

type TimelineListResponse struct {
	// Items: Items in the timeline.
	Items []*TimelineItem `json:"items,omitempty"`

	// Kind: The type of resource. This is always mirror#timeline.
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The next page token. Provide this as the pageToken
	// parameter in the request to retrieve the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type UserAction struct {
	// Payload: An optional payload for the action.
	//
	// For actions of type
	// CUSTOM, this is the ID of the custom menu item that was selected.
	Payload string `json:"payload,omitempty"`

	// Type: The type of action. The value of this can be:
	// - SHARE - the
	// user shared an item.
	// - REPLY - the user replied to an item.
	// -
	// REPLY_ALL - the user replied to all recipients of an item.
	// - CUSTOM
	// - the user selected a custom menu item on the timeline item.
	// -
	// DELETE - the user deleted the item.
	// - PIN - the user pinned the
	// item.
	// - UNPIN - the user unpinned the item.
	// - LAUNCH - the user
	// initiated a voice command.  In the future, additional types may be
	// added. UserActions with unrecognized types should be ignored.
	Type string `json:"type,omitempty"`
}

type UserData struct {
	Key string `json:"key,omitempty"`

	Value string `json:"value,omitempty"`
}

// method id "mirror.accounts.insert":

type AccountsInsertCall struct {
	s           *Service
	userToken   string
	accountType string
	accountName string
	account     *Account
	opt_        map[string]interface{}
}

// Insert: Inserts a new account for a user
func (r *AccountsService) Insert(userToken string, accountType string, accountName string, account *Account) *AccountsInsertCall {
	c := &AccountsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.userToken = userToken
	c.accountType = accountType
	c.accountName = accountName
	c.account = account
	return c
}

func (c *AccountsInsertCall) Do() (*Account, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.account)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "accounts/{userToken}/{accountType}/{accountName}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userToken}", url.QueryEscape(c.userToken), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountType}", url.QueryEscape(c.accountType), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{accountName}", url.QueryEscape(c.accountName), 1)
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
	ret := new(Account)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a new account for a user",
	//   "httpMethod": "POST",
	//   "id": "mirror.accounts.insert",
	//   "parameterOrder": [
	//     "userToken",
	//     "accountType",
	//     "accountName"
	//   ],
	//   "parameters": {
	//     "accountName": {
	//       "description": "The name of the account to be passed to the Android Account Manager.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "accountType": {
	//       "description": "Account type to be passed to Android Account Manager.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userToken": {
	//       "description": "The ID for the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "accounts/{userToken}/{accountType}/{accountName}",
	//   "request": {
	//     "$ref": "Account"
	//   },
	//   "response": {
	//     "$ref": "Account"
	//   }
	// }

}

// method id "mirror.contacts.delete":

type ContactsDeleteCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Delete: Deletes a contact.
func (r *ContactsService) Delete(id string) *ContactsDeleteCall {
	c := &ContactsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *ContactsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "contacts/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
	googleapi.SetOpaque(req.URL)
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
	//   "description": "Deletes a contact.",
	//   "httpMethod": "DELETE",
	//   "id": "mirror.contacts.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the contact.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "contacts/{id}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.contacts.get":

type ContactsGetCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Get: Gets a single contact by ID.
func (r *ContactsService) Get(id string) *ContactsGetCall {
	c := &ContactsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *ContactsGetCall) Do() (*Contact, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "contacts/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	ret := new(Contact)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a single contact by ID.",
	//   "httpMethod": "GET",
	//   "id": "mirror.contacts.get",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the contact.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "contacts/{id}",
	//   "response": {
	//     "$ref": "Contact"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.contacts.insert":

type ContactsInsertCall struct {
	s       *Service
	contact *Contact
	opt_    map[string]interface{}
}

// Insert: Inserts a new contact.
func (r *ContactsService) Insert(contact *Contact) *ContactsInsertCall {
	c := &ContactsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.contact = contact
	return c
}

func (c *ContactsInsertCall) Do() (*Contact, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.contact)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "contacts")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(Contact)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a new contact.",
	//   "httpMethod": "POST",
	//   "id": "mirror.contacts.insert",
	//   "path": "contacts",
	//   "request": {
	//     "$ref": "Contact"
	//   },
	//   "response": {
	//     "$ref": "Contact"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.contacts.list":

type ContactsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieves a list of contacts for the authenticated user.
func (r *ContactsService) List() *ContactsListCall {
	c := &ContactsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *ContactsListCall) Do() (*ContactsListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "contacts")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
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
	ret := new(ContactsListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of contacts for the authenticated user.",
	//   "httpMethod": "GET",
	//   "id": "mirror.contacts.list",
	//   "path": "contacts",
	//   "response": {
	//     "$ref": "ContactsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.contacts.patch":

type ContactsPatchCall struct {
	s       *Service
	id      string
	contact *Contact
	opt_    map[string]interface{}
}

// Patch: Updates a contact in place. This method supports patch
// semantics.
func (r *ContactsService) Patch(id string, contact *Contact) *ContactsPatchCall {
	c := &ContactsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.contact = contact
	return c
}

func (c *ContactsPatchCall) Do() (*Contact, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.contact)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "contacts/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	ret := new(Contact)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a contact in place. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "mirror.contacts.patch",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the contact.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "contacts/{id}",
	//   "request": {
	//     "$ref": "Contact"
	//   },
	//   "response": {
	//     "$ref": "Contact"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.contacts.update":

type ContactsUpdateCall struct {
	s       *Service
	id      string
	contact *Contact
	opt_    map[string]interface{}
}

// Update: Updates a contact in place.
func (r *ContactsService) Update(id string, contact *Contact) *ContactsUpdateCall {
	c := &ContactsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.contact = contact
	return c
}

func (c *ContactsUpdateCall) Do() (*Contact, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.contact)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "contacts/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	ret := new(Contact)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a contact in place.",
	//   "httpMethod": "PUT",
	//   "id": "mirror.contacts.update",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the contact.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "contacts/{id}",
	//   "request": {
	//     "$ref": "Contact"
	//   },
	//   "response": {
	//     "$ref": "Contact"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.locations.get":

type LocationsGetCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Get: Gets a single location by ID.
func (r *LocationsService) Get(id string) *LocationsGetCall {
	c := &LocationsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *LocationsGetCall) Do() (*Location, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "locations/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	ret := new(Location)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a single location by ID.",
	//   "httpMethod": "GET",
	//   "id": "mirror.locations.get",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the location or latest for the last known location.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "locations/{id}",
	//   "response": {
	//     "$ref": "Location"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.location",
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.locations.list":

type LocationsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieves a list of locations for the user.
func (r *LocationsService) List() *LocationsListCall {
	c := &LocationsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *LocationsListCall) Do() (*LocationsListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "locations")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
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
	ret := new(LocationsListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of locations for the user.",
	//   "httpMethod": "GET",
	//   "id": "mirror.locations.list",
	//   "path": "locations",
	//   "response": {
	//     "$ref": "LocationsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.location",
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.subscriptions.delete":

type SubscriptionsDeleteCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Delete: Deletes a subscription.
func (r *SubscriptionsService) Delete(id string) *SubscriptionsDeleteCall {
	c := &SubscriptionsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *SubscriptionsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "subscriptions/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
	googleapi.SetOpaque(req.URL)
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
	//   "description": "Deletes a subscription.",
	//   "httpMethod": "DELETE",
	//   "id": "mirror.subscriptions.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the subscription.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "subscriptions/{id}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.subscriptions.insert":

type SubscriptionsInsertCall struct {
	s            *Service
	subscription *Subscription
	opt_         map[string]interface{}
}

// Insert: Creates a new subscription.
func (r *SubscriptionsService) Insert(subscription *Subscription) *SubscriptionsInsertCall {
	c := &SubscriptionsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.subscription = subscription
	return c
}

func (c *SubscriptionsInsertCall) Do() (*Subscription, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.subscription)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "subscriptions")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(Subscription)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a new subscription.",
	//   "httpMethod": "POST",
	//   "id": "mirror.subscriptions.insert",
	//   "path": "subscriptions",
	//   "request": {
	//     "$ref": "Subscription"
	//   },
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.subscriptions.list":

type SubscriptionsListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieves a list of subscriptions for the authenticated user
// and service.
func (r *SubscriptionsService) List() *SubscriptionsListCall {
	c := &SubscriptionsListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

func (c *SubscriptionsListCall) Do() (*SubscriptionsListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "subscriptions")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
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
	ret := new(SubscriptionsListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of subscriptions for the authenticated user and service.",
	//   "httpMethod": "GET",
	//   "id": "mirror.subscriptions.list",
	//   "path": "subscriptions",
	//   "response": {
	//     "$ref": "SubscriptionsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.subscriptions.update":

type SubscriptionsUpdateCall struct {
	s            *Service
	id           string
	subscription *Subscription
	opt_         map[string]interface{}
}

// Update: Updates an existing subscription in place.
func (r *SubscriptionsService) Update(id string, subscription *Subscription) *SubscriptionsUpdateCall {
	c := &SubscriptionsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.subscription = subscription
	return c
}

func (c *SubscriptionsUpdateCall) Do() (*Subscription, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.subscription)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "subscriptions/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	ret := new(Subscription)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates an existing subscription in place.",
	//   "httpMethod": "PUT",
	//   "id": "mirror.subscriptions.update",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the subscription.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "subscriptions/{id}",
	//   "request": {
	//     "$ref": "Subscription"
	//   },
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.timeline.delete":

type TimelineDeleteCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Delete: Deletes a timeline item.
func (r *TimelineService) Delete(id string) *TimelineDeleteCall {
	c := &TimelineDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *TimelineDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "timeline/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
	googleapi.SetOpaque(req.URL)
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
	//   "description": "Deletes a timeline item.",
	//   "httpMethod": "DELETE",
	//   "id": "mirror.timeline.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the timeline item.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "timeline/{id}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.location",
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.timeline.get":

type TimelineGetCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Get: Gets a single timeline item by ID.
func (r *TimelineService) Get(id string) *TimelineGetCall {
	c := &TimelineGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *TimelineGetCall) Do() (*TimelineItem, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "timeline/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	ret := new(TimelineItem)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets a single timeline item by ID.",
	//   "httpMethod": "GET",
	//   "id": "mirror.timeline.get",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the timeline item.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "timeline/{id}",
	//   "response": {
	//     "$ref": "TimelineItem"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.location",
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.timeline.insert":

type TimelineInsertCall struct {
	s            *Service
	timelineitem *TimelineItem
	opt_         map[string]interface{}
	media_       io.Reader
}

// Insert: Inserts a new item into the timeline.
func (r *TimelineService) Insert(timelineitem *TimelineItem) *TimelineInsertCall {
	c := &TimelineInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.timelineitem = timelineitem
	return c
}
func (c *TimelineInsertCall) Media(r io.Reader) *TimelineInsertCall {
	c.media_ = r
	return c
}

func (c *TimelineInsertCall) Do() (*TimelineItem, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.timelineitem)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "timeline")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(TimelineItem)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a new item into the timeline.",
	//   "httpMethod": "POST",
	//   "id": "mirror.timeline.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "audio/*",
	//       "image/*",
	//       "video/*"
	//     ],
	//     "maxSize": "10MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/mirror/v1/timeline"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/mirror/v1/timeline"
	//       }
	//     }
	//   },
	//   "path": "timeline",
	//   "request": {
	//     "$ref": "TimelineItem"
	//   },
	//   "response": {
	//     "$ref": "TimelineItem"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.location",
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "mirror.timeline.list":

type TimelineListCall struct {
	s    *Service
	opt_ map[string]interface{}
}

// List: Retrieves a list of timeline items for the authenticated user.
func (r *TimelineService) List() *TimelineListCall {
	c := &TimelineListCall{s: r.s, opt_: make(map[string]interface{})}
	return c
}

// BundleId sets the optional parameter "bundleId": If provided, only
// items with the given bundleId will be returned.
func (c *TimelineListCall) BundleId(bundleId string) *TimelineListCall {
	c.opt_["bundleId"] = bundleId
	return c
}

// IncludeDeleted sets the optional parameter "includeDeleted": If true,
// tombstone records for deleted items will be returned.
func (c *TimelineListCall) IncludeDeleted(includeDeleted bool) *TimelineListCall {
	c.opt_["includeDeleted"] = includeDeleted
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of items to include in the response, used for paging.
func (c *TimelineListCall) MaxResults(maxResults int64) *TimelineListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// OrderBy sets the optional parameter "orderBy": Controls the order in
// which timeline items are returned.
func (c *TimelineListCall) OrderBy(orderBy string) *TimelineListCall {
	c.opt_["orderBy"] = orderBy
	return c
}

// PageToken sets the optional parameter "pageToken": Token for the page
// of results to return.
func (c *TimelineListCall) PageToken(pageToken string) *TimelineListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// PinnedOnly sets the optional parameter "pinnedOnly": If true, only
// pinned items will be returned.
func (c *TimelineListCall) PinnedOnly(pinnedOnly bool) *TimelineListCall {
	c.opt_["pinnedOnly"] = pinnedOnly
	return c
}

// SourceItemId sets the optional parameter "sourceItemId": If provided,
// only items with the given sourceItemId will be returned.
func (c *TimelineListCall) SourceItemId(sourceItemId string) *TimelineListCall {
	c.opt_["sourceItemId"] = sourceItemId
	return c
}

func (c *TimelineListCall) Do() (*TimelineListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["bundleId"]; ok {
		params.Set("bundleId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["includeDeleted"]; ok {
		params.Set("includeDeleted", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["orderBy"]; ok {
		params.Set("orderBy", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pinnedOnly"]; ok {
		params.Set("pinnedOnly", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sourceItemId"]; ok {
		params.Set("sourceItemId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "timeline")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
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
	ret := new(TimelineListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of timeline items for the authenticated user.",
	//   "httpMethod": "GET",
	//   "id": "mirror.timeline.list",
	//   "parameters": {
	//     "bundleId": {
	//       "description": "If provided, only items with the given bundleId will be returned.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "includeDeleted": {
	//       "description": "If true, tombstone records for deleted items will be returned.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of items to include in the response, used for paging.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "orderBy": {
	//       "description": "Controls the order in which timeline items are returned.",
	//       "enum": [
	//         "displayTime",
	//         "writeTime"
	//       ],
	//       "enumDescriptions": [
	//         "Results will be ordered by displayTime (default). This is the same ordering as is used in the timeline on the device.",
	//         "Results will be ordered by the time at which they were last written to the data store."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "Token for the page of results to return.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pinnedOnly": {
	//       "description": "If true, only pinned items will be returned.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "sourceItemId": {
	//       "description": "If provided, only items with the given sourceItemId will be returned.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "timeline",
	//   "response": {
	//     "$ref": "TimelineListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.location",
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.timeline.patch":

type TimelinePatchCall struct {
	s            *Service
	id           string
	timelineitem *TimelineItem
	opt_         map[string]interface{}
}

// Patch: Updates a timeline item in place. This method supports patch
// semantics.
func (r *TimelineService) Patch(id string, timelineitem *TimelineItem) *TimelinePatchCall {
	c := &TimelinePatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.timelineitem = timelineitem
	return c
}

func (c *TimelinePatchCall) Do() (*TimelineItem, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.timelineitem)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "timeline/{id}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	ret := new(TimelineItem)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a timeline item in place. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "mirror.timeline.patch",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the timeline item.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "timeline/{id}",
	//   "request": {
	//     "$ref": "TimelineItem"
	//   },
	//   "response": {
	//     "$ref": "TimelineItem"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.location",
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.timeline.update":

type TimelineUpdateCall struct {
	s            *Service
	id           string
	timelineitem *TimelineItem
	opt_         map[string]interface{}
	media_       io.Reader
}

// Update: Updates a timeline item in place.
func (r *TimelineService) Update(id string, timelineitem *TimelineItem) *TimelineUpdateCall {
	c := &TimelineUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.timelineitem = timelineitem
	return c
}
func (c *TimelineUpdateCall) Media(r io.Reader) *TimelineUpdateCall {
	c.media_ = r
	return c
}

func (c *TimelineUpdateCall) Do() (*TimelineItem, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.timelineitem)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "timeline/{id}")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{id}", url.QueryEscape(c.id), 1)
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
	ret := new(TimelineItem)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a timeline item in place.",
	//   "httpMethod": "PUT",
	//   "id": "mirror.timeline.update",
	//   "mediaUpload": {
	//     "accept": [
	//       "audio/*",
	//       "image/*",
	//       "video/*"
	//     ],
	//     "maxSize": "10MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/mirror/v1/timeline/{id}"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/mirror/v1/timeline/{id}"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The ID of the timeline item.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "timeline/{id}",
	//   "request": {
	//     "$ref": "TimelineItem"
	//   },
	//   "response": {
	//     "$ref": "TimelineItem"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.location",
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "mirror.timeline.attachments.delete":

type TimelineAttachmentsDeleteCall struct {
	s            *Service
	itemId       string
	attachmentId string
	opt_         map[string]interface{}
}

// Delete: Deletes an attachment from a timeline item.
func (r *TimelineAttachmentsService) Delete(itemId string, attachmentId string) *TimelineAttachmentsDeleteCall {
	c := &TimelineAttachmentsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.itemId = itemId
	c.attachmentId = attachmentId
	return c
}

func (c *TimelineAttachmentsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "timeline/{itemId}/attachments/{attachmentId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{itemId}", url.QueryEscape(c.itemId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{attachmentId}", url.QueryEscape(c.attachmentId), 1)
	googleapi.SetOpaque(req.URL)
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
	//   "description": "Deletes an attachment from a timeline item.",
	//   "httpMethod": "DELETE",
	//   "id": "mirror.timeline.attachments.delete",
	//   "parameterOrder": [
	//     "itemId",
	//     "attachmentId"
	//   ],
	//   "parameters": {
	//     "attachmentId": {
	//       "description": "The ID of the attachment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "itemId": {
	//       "description": "The ID of the timeline item the attachment belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "timeline/{itemId}/attachments/{attachmentId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}

// method id "mirror.timeline.attachments.get":

type TimelineAttachmentsGetCall struct {
	s            *Service
	itemId       string
	attachmentId string
	opt_         map[string]interface{}
}

// Get: Retrieves an attachment on a timeline item by item ID and
// attachment ID.
func (r *TimelineAttachmentsService) Get(itemId string, attachmentId string) *TimelineAttachmentsGetCall {
	c := &TimelineAttachmentsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.itemId = itemId
	c.attachmentId = attachmentId
	return c
}

func (c *TimelineAttachmentsGetCall) Do() (*Attachment, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "timeline/{itemId}/attachments/{attachmentId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{itemId}", url.QueryEscape(c.itemId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{attachmentId}", url.QueryEscape(c.attachmentId), 1)
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
	ret := new(Attachment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves an attachment on a timeline item by item ID and attachment ID.",
	//   "httpMethod": "GET",
	//   "id": "mirror.timeline.attachments.get",
	//   "parameterOrder": [
	//     "itemId",
	//     "attachmentId"
	//   ],
	//   "parameters": {
	//     "attachmentId": {
	//       "description": "The ID of the attachment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "itemId": {
	//       "description": "The ID of the timeline item the attachment belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "timeline/{itemId}/attachments/{attachmentId}",
	//   "response": {
	//     "$ref": "Attachment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ],
	//   "supportsMediaDownload": true
	// }

}

// method id "mirror.timeline.attachments.insert":

type TimelineAttachmentsInsertCall struct {
	s      *Service
	itemId string
	opt_   map[string]interface{}
	media_ io.Reader
}

// Insert: Adds a new attachment to a timeline item.
func (r *TimelineAttachmentsService) Insert(itemId string) *TimelineAttachmentsInsertCall {
	c := &TimelineAttachmentsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.itemId = itemId
	return c
}
func (c *TimelineAttachmentsInsertCall) Media(r io.Reader) *TimelineAttachmentsInsertCall {
	c.media_ = r
	return c
}

func (c *TimelineAttachmentsInsertCall) Do() (*Attachment, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "timeline/{itemId}/attachments")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	body = new(bytes.Buffer)
	ctype := "application/json"
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{itemId}", url.QueryEscape(c.itemId), 1)
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
	ret := new(Attachment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds a new attachment to a timeline item.",
	//   "httpMethod": "POST",
	//   "id": "mirror.timeline.attachments.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "audio/*",
	//       "image/*",
	//       "video/*"
	//     ],
	//     "maxSize": "10MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/mirror/v1/timeline/{itemId}/attachments"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/mirror/v1/timeline/{itemId}/attachments"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "itemId"
	//   ],
	//   "parameters": {
	//     "itemId": {
	//       "description": "The ID of the timeline item the attachment belongs to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "timeline/{itemId}/attachments",
	//   "response": {
	//     "$ref": "Attachment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "mirror.timeline.attachments.list":

type TimelineAttachmentsListCall struct {
	s      *Service
	itemId string
	opt_   map[string]interface{}
}

// List: Returns a list of attachments for a timeline item.
func (r *TimelineAttachmentsService) List(itemId string) *TimelineAttachmentsListCall {
	c := &TimelineAttachmentsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.itemId = itemId
	return c
}

func (c *TimelineAttachmentsListCall) Do() (*AttachmentsListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "timeline/{itemId}/attachments")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{itemId}", url.QueryEscape(c.itemId), 1)
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
	ret := new(AttachmentsListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of attachments for a timeline item.",
	//   "httpMethod": "GET",
	//   "id": "mirror.timeline.attachments.list",
	//   "parameterOrder": [
	//     "itemId"
	//   ],
	//   "parameters": {
	//     "itemId": {
	//       "description": "The ID of the timeline item whose attachments should be listed.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "timeline/{itemId}/attachments",
	//   "response": {
	//     "$ref": "AttachmentsListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/glass.timeline"
	//   ]
	// }

}
