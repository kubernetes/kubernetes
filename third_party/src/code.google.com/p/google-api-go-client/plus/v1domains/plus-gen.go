// Package plus provides access to the Google+ API.
//
// See https://developers.google.com/+/api/
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/plus/v1domains"
//   ...
//   plusService, err := plus.New(oauthHttpClient)
package plus

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

const apiId = "plus:v1domains"
const apiName = "plus"
const apiVersion = "v1domains"
const basePath = "https://www.googleapis.com/plus/v1domains/"

// OAuth2 scopes used by this API.
const (
	// View your circles and people in them
	PlusCirclesReadScope = "https://www.googleapis.com/auth/plus.circles.read"

	// Manage your circles and add people
	PlusCirclesWriteScope = "https://www.googleapis.com/auth/plus.circles.write"

	// Know your name, basic info, and list of people you're connected to on
	// Google+
	PlusLoginScope = "https://www.googleapis.com/auth/plus.login"

	// Know who you are on Google
	PlusMeScope = "https://www.googleapis.com/auth/plus.me"

	// Send your photos and videos to Google+
	PlusMediaUploadScope = "https://www.googleapis.com/auth/plus.media.upload"

	// View your own Google+ profile and profiles shared with you
	PlusProfilesReadScope = "https://www.googleapis.com/auth/plus.profiles.read"

	// View your posts, comments, and stream
	PlusStreamReadScope = "https://www.googleapis.com/auth/plus.stream.read"

	// Manage your posts, comments, and stream
	PlusStreamWriteScope = "https://www.googleapis.com/auth/plus.stream.write"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client}
	s.Activities = NewActivitiesService(s)
	s.Audiences = NewAudiencesService(s)
	s.Circles = NewCirclesService(s)
	s.Comments = NewCommentsService(s)
	s.Media = NewMediaService(s)
	s.People = NewPeopleService(s)
	return s, nil
}

type Service struct {
	client *http.Client

	Activities *ActivitiesService

	Audiences *AudiencesService

	Circles *CirclesService

	Comments *CommentsService

	Media *MediaService

	People *PeopleService
}

func NewActivitiesService(s *Service) *ActivitiesService {
	rs := &ActivitiesService{s: s}
	return rs
}

type ActivitiesService struct {
	s *Service
}

func NewAudiencesService(s *Service) *AudiencesService {
	rs := &AudiencesService{s: s}
	return rs
}

type AudiencesService struct {
	s *Service
}

func NewCirclesService(s *Service) *CirclesService {
	rs := &CirclesService{s: s}
	return rs
}

type CirclesService struct {
	s *Service
}

func NewCommentsService(s *Service) *CommentsService {
	rs := &CommentsService{s: s}
	return rs
}

type CommentsService struct {
	s *Service
}

func NewMediaService(s *Service) *MediaService {
	rs := &MediaService{s: s}
	return rs
}

type MediaService struct {
	s *Service
}

func NewPeopleService(s *Service) *PeopleService {
	rs := &PeopleService{s: s}
	return rs
}

type PeopleService struct {
	s *Service
}

type Acl struct {
	// Description: Description of the access granted, suitable for display.
	Description string `json:"description,omitempty"`

	// DomainRestricted: Whether access is restricted to the domain.
	DomainRestricted bool `json:"domainRestricted,omitempty"`

	// Items: The list of access entries.
	Items []*PlusAclentryResource `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of access controls.
	// Value: "plus#acl".
	Kind string `json:"kind,omitempty"`
}

type Activity struct {
	// Access: Identifies who has access to see this activity.
	Access *Acl `json:"access,omitempty"`

	// Actor: The person who performed this activity.
	Actor *ActivityActor `json:"actor,omitempty"`

	// Address: Street address where this activity occurred.
	Address string `json:"address,omitempty"`

	// Annotation: Additional content added by the person who shared this
	// activity, applicable only when resharing an activity.
	Annotation string `json:"annotation,omitempty"`

	// CrosspostSource: If this activity is a crosspost from another system,
	// this property specifies the ID of the original activity.
	CrosspostSource string `json:"crosspostSource,omitempty"`

	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Geocode: Latitude and longitude where this activity occurred. Format
	// is latitude followed by longitude, space separated.
	Geocode string `json:"geocode,omitempty"`

	// Id: The ID of this activity.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this resource as an activity. Value:
	// "plus#activity".
	Kind string `json:"kind,omitempty"`

	// Location: The location where this activity occurred.
	Location *Place `json:"location,omitempty"`

	// Object: The object of this activity.
	Object *ActivityObject `json:"object,omitempty"`

	// PlaceId: ID of the place where this activity occurred.
	PlaceId string `json:"placeId,omitempty"`

	// PlaceName: Name of the place where this activity occurred.
	PlaceName string `json:"placeName,omitempty"`

	// Provider: The service provider that initially published this
	// activity.
	Provider *ActivityProvider `json:"provider,omitempty"`

	// Published: The time at which this activity was initially published.
	// Formatted as an RFC 3339 timestamp.
	Published string `json:"published,omitempty"`

	// Radius: Radius, in meters, of the region where this activity
	// occurred, centered at the latitude and longitude identified in
	// geocode.
	Radius string `json:"radius,omitempty"`

	// Title: Title of this activity.
	Title string `json:"title,omitempty"`

	// Updated: The time at which this activity was last updated. Formatted
	// as an RFC 3339 timestamp.
	Updated string `json:"updated,omitempty"`

	// Url: The link to this activity.
	Url string `json:"url,omitempty"`

	// Verb: This activity's verb, which indicates the action that was
	// performed. Possible values include, but are not limited to, the
	// following values:
	// - "post" - Publish content to the stream.
	// -
	// "share" - Reshare an activity.
	Verb string `json:"verb,omitempty"`
}

type ActivityActor struct {
	// DisplayName: The name of the actor, suitable for display.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The ID of the actor's Person resource.
	Id string `json:"id,omitempty"`

	// Image: The image representation of the actor.
	Image *ActivityActorImage `json:"image,omitempty"`

	// Name: An object representation of the individual components of name.
	Name *ActivityActorName `json:"name,omitempty"`

	// Url: The link to the actor's Google profile.
	Url string `json:"url,omitempty"`
}

type ActivityActorImage struct {
	// Url: The URL of the actor's profile photo. To resize the image and
	// crop it to a square, append the query string ?sz=x, where x is the
	// dimension in pixels of each side.
	Url string `json:"url,omitempty"`
}

type ActivityActorName struct {
	// FamilyName: The family name ("last name") of the actor.
	FamilyName string `json:"familyName,omitempty"`

	// GivenName: The given name ("first name") of the actor.
	GivenName string `json:"givenName,omitempty"`
}

type ActivityObject struct {
	// Actor: If this activity's object is itself another activity, such as
	// when a person reshares an activity, this property specifies the
	// original activity's actor.
	Actor *ActivityObjectActor `json:"actor,omitempty"`

	// Attachments: The media objects attached to this activity.
	Attachments []*ActivityObjectAttachments `json:"attachments,omitempty"`

	// Content: The HTML-formatted content, which is suitable for display.
	Content string `json:"content,omitempty"`

	// Id: The ID of the object. When resharing an activity, this is the ID
	// of the activity that is being reshared.
	Id string `json:"id,omitempty"`

	// ObjectType: The type of the object. Possible values include, but are
	// not limited to, the following values:
	// - "note" - Textual content.
	//
	// - "activity" - A Google+ activity.
	ObjectType string `json:"objectType,omitempty"`

	// OriginalContent: The content (text) as provided by the author, which
	// is stored without any HTML formatting. When creating or updating an
	// activity, this value must be supplied as plain text in the request.
	OriginalContent string `json:"originalContent,omitempty"`

	// Plusoners: People who +1'd this activity.
	Plusoners *ActivityObjectPlusoners `json:"plusoners,omitempty"`

	// Replies: Comments in reply to this activity.
	Replies *ActivityObjectReplies `json:"replies,omitempty"`

	// Resharers: People who reshared this activity.
	Resharers *ActivityObjectResharers `json:"resharers,omitempty"`

	// StatusForViewer: Status of the activity as seen by the viewer.
	StatusForViewer *ActivityObjectStatusForViewer `json:"statusForViewer,omitempty"`

	// Url: The URL that points to the linked resource.
	Url string `json:"url,omitempty"`
}

type ActivityObjectActor struct {
	// DisplayName: The original actor's name, which is suitable for
	// display.
	DisplayName string `json:"displayName,omitempty"`

	// Id: ID of the original actor.
	Id string `json:"id,omitempty"`

	// Image: The image representation of the original actor.
	Image *ActivityObjectActorImage `json:"image,omitempty"`

	// Url: A link to the original actor's Google profile.
	Url string `json:"url,omitempty"`
}

type ActivityObjectActorImage struct {
	// Url: A URL that points to a thumbnail photo of the original actor.
	Url string `json:"url,omitempty"`
}

type ActivityObjectAttachments struct {
	// Content: If the attachment is an article, this property contains a
	// snippet of text from the article. It can also include descriptions
	// for other types.
	Content string `json:"content,omitempty"`

	// DisplayName: The title of the attachment, such as a photo caption or
	// an article title.
	DisplayName string `json:"displayName,omitempty"`

	// Embed: If the attachment is a video, the embeddable link.
	Embed *ActivityObjectAttachmentsEmbed `json:"embed,omitempty"`

	// FullImage: The full image URL for photo attachments.
	FullImage *ActivityObjectAttachmentsFullImage `json:"fullImage,omitempty"`

	// Id: The ID of the attachment.
	Id string `json:"id,omitempty"`

	// Image: The preview image for photos or videos.
	Image *ActivityObjectAttachmentsImage `json:"image,omitempty"`

	// ObjectType: The type of media object. Possible values include, but
	// are not limited to, the following values:
	// - "photo" - A photo.
	// -
	// "album" - A photo album.
	// - "video" - A video.
	// - "article" - An
	// article, specified by a link.
	ObjectType string `json:"objectType,omitempty"`

	// PreviewThumbnails: When previewing, these are the optional thumbnails
	// for the post. When posting an article, choose one by setting the
	// attachment.image.url property. If you don't choose one, one will be
	// chosen for you.
	PreviewThumbnails []*ActivityObjectAttachmentsPreviewThumbnails `json:"previewThumbnails,omitempty"`

	// Thumbnails: If the attachment is an album, this property is a list of
	// potential additional thumbnails from the album.
	Thumbnails []*ActivityObjectAttachmentsThumbnails `json:"thumbnails,omitempty"`

	// Url: The link to the attachment; should be of type text/html.
	Url string `json:"url,omitempty"`
}

type ActivityObjectAttachmentsEmbed struct {
	// Type: Media type of the link.
	Type string `json:"type,omitempty"`

	// Url: URL of the link.
	Url string `json:"url,omitempty"`
}

type ActivityObjectAttachmentsFullImage struct {
	// Height: The height, in pixels, of the linked resource.
	Height int64 `json:"height,omitempty"`

	// Type: Media type of the link.
	Type string `json:"type,omitempty"`

	// Url: URL of the image.
	Url string `json:"url,omitempty"`

	// Width: The width, in pixels, of the linked resource.
	Width int64 `json:"width,omitempty"`
}

type ActivityObjectAttachmentsImage struct {
	// Height: The height, in pixels, of the linked resource.
	Height int64 `json:"height,omitempty"`

	// Type: Media type of the link.
	Type string `json:"type,omitempty"`

	// Url: Image URL.
	Url string `json:"url,omitempty"`

	// Width: The width, in pixels, of the linked resource.
	Width int64 `json:"width,omitempty"`
}

type ActivityObjectAttachmentsPreviewThumbnails struct {
	// Url: URL of the thumbnail image.
	Url string `json:"url,omitempty"`
}

type ActivityObjectAttachmentsThumbnails struct {
	// Description: Potential name of the thumbnail.
	Description string `json:"description,omitempty"`

	// Image: Image resource.
	Image *ActivityObjectAttachmentsThumbnailsImage `json:"image,omitempty"`

	// Url: URL of the webpage containing the image.
	Url string `json:"url,omitempty"`
}

type ActivityObjectAttachmentsThumbnailsImage struct {
	// Height: The height, in pixels, of the linked resource.
	Height int64 `json:"height,omitempty"`

	// Type: Media type of the link.
	Type string `json:"type,omitempty"`

	// Url: Image url.
	Url string `json:"url,omitempty"`

	// Width: The width, in pixels, of the linked resource.
	Width int64 `json:"width,omitempty"`
}

type ActivityObjectPlusoners struct {
	// SelfLink: The URL for the collection of people who +1'd this
	// activity.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalItems: Total number of people who +1'd this activity.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type ActivityObjectReplies struct {
	// SelfLink: The URL for the collection of comments in reply to this
	// activity.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalItems: Total number of comments on this activity.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type ActivityObjectResharers struct {
	// SelfLink: The URL for the collection of resharers.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalItems: Total number of people who reshared this activity.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type ActivityObjectStatusForViewer struct {
	// CanComment: Whether the viewer can comment on the activity.
	CanComment bool `json:"canComment,omitempty"`

	// CanPlusone: Whether the viewer can +1 the activity.
	CanPlusone bool `json:"canPlusone,omitempty"`

	// IsPlusOned: Whether the viewer has +1'd the activity.
	IsPlusOned bool `json:"isPlusOned,omitempty"`

	// ResharingDisabled: Whether reshares are disabled for the activity.
	ResharingDisabled bool `json:"resharingDisabled,omitempty"`
}

type ActivityProvider struct {
	// Title: Name of the service provider.
	Title string `json:"title,omitempty"`
}

type ActivityFeed struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Id: The ID of this collection of activities. Deprecated.
	Id string `json:"id,omitempty"`

	// Items: The activities in this page of results.
	Items []*Activity `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of activities. Value:
	// "plus#activityFeed".
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to the next page of activities.
	NextLink string `json:"nextLink,omitempty"`

	// NextPageToken: The continuation token, which is used to page through
	// large result sets. Provide this value in a subsequent request to
	// return the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Link to this activities resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Title: The title of this collection of activities, which is a
	// truncated portion of the content.
	Title string `json:"title,omitempty"`

	// Updated: The time at which this collection of activities was last
	// updated. Formatted as an RFC 3339 timestamp.
	Updated string `json:"updated,omitempty"`
}

type Audience struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Item: The access control list entry.
	Item *PlusAclentryResource `json:"item,omitempty"`

	// Kind: Identifies this resource as an audience. Value:
	// "plus#audience".
	Kind string `json:"kind,omitempty"`

	// Visibility: The circle members' visibility as chosen by the owner of
	// the circle. This only applies for items with "item.type" equals
	// "circle". Possible values are:
	// - "public" - Members are visible to
	// the public.
	// - "limited" - Members are visible to a limited audience.
	//
	// - "private" - Members are visible to the owner only.
	Visibility string `json:"visibility,omitempty"`
}

type AudiencesFeed struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The audiences in this result.
	Items []*Audience `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of audiences. Value:
	// "plus#audienceFeed".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The continuation token, which is used to page through
	// large result sets. Provide this value in a subsequent request to
	// return the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// TotalItems: The total number of ACL entries. The number of entries in
	// this response may be smaller due to paging.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type Circle struct {
	// Description: The description of this circle.
	Description string `json:"description,omitempty"`

	// DisplayName: The circle name.
	DisplayName string `json:"displayName,omitempty"`

	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Id: The ID of the circle.
	Id string `json:"id,omitempty"`

	// Kind: Identifies this resource as a circle. Value: "plus#circle".
	Kind string `json:"kind,omitempty"`

	// People: The people in this circle.
	People *CirclePeople `json:"people,omitempty"`

	// SelfLink: Link to this circle resource
	SelfLink string `json:"selfLink,omitempty"`
}

type CirclePeople struct {
	// TotalItems: The total number of people in this circle.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type CircleFeed struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The circles in this page of results.
	Items []*Circle `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of circles. Value:
	// "plus#circleFeed".
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to the next page of circles.
	NextLink string `json:"nextLink,omitempty"`

	// NextPageToken: The continuation token, which is used to page through
	// large result sets. Provide this value in a subsequent request to
	// return the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Link to this page of circles.
	SelfLink string `json:"selfLink,omitempty"`

	// Title: The title of this list of resources.
	Title string `json:"title,omitempty"`

	// TotalItems: The total number of circles. The number of circles in
	// this response may be smaller due to paging.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type Comment struct {
	// Actor: The person who posted this comment.
	Actor *CommentActor `json:"actor,omitempty"`

	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Id: The ID of this comment.
	Id string `json:"id,omitempty"`

	// InReplyTo: The activity this comment replied to.
	InReplyTo []*CommentInReplyTo `json:"inReplyTo,omitempty"`

	// Kind: Identifies this resource as a comment. Value: "plus#comment".
	Kind string `json:"kind,omitempty"`

	// Object: The object of this comment.
	Object *CommentObject `json:"object,omitempty"`

	// Plusoners: People who +1'd this comment.
	Plusoners *CommentPlusoners `json:"plusoners,omitempty"`

	// Published: The time at which this comment was initially published.
	// Formatted as an RFC 3339 timestamp.
	Published string `json:"published,omitempty"`

	// SelfLink: Link to this comment resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Updated: The time at which this comment was last updated. Formatted
	// as an RFC 3339 timestamp.
	Updated string `json:"updated,omitempty"`

	// Verb: This comment's verb, indicating what action was performed.
	// Possible values are:
	// - "post" - Publish content to the stream.
	Verb string `json:"verb,omitempty"`
}

type CommentActor struct {
	// DisplayName: The name of this actor, suitable for display.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The ID of the actor.
	Id string `json:"id,omitempty"`

	// Image: The image representation of this actor.
	Image *CommentActorImage `json:"image,omitempty"`

	// Url: A link to the Person resource for this actor.
	Url string `json:"url,omitempty"`
}

type CommentActorImage struct {
	// Url: The URL of the actor's profile photo. To resize the image and
	// crop it to a square, append the query string ?sz=x, where x is the
	// dimension in pixels of each side.
	Url string `json:"url,omitempty"`
}

type CommentInReplyTo struct {
	// Id: The ID of the activity.
	Id string `json:"id,omitempty"`

	// Url: The URL of the activity.
	Url string `json:"url,omitempty"`
}

type CommentObject struct {
	// Content: The HTML-formatted content, suitable for display.
	Content string `json:"content,omitempty"`

	// ObjectType: The object type of this comment. Possible values are:
	// -
	// "comment" - A comment in reply to an activity.
	ObjectType string `json:"objectType,omitempty"`

	// OriginalContent: The content (text) as provided by the author, stored
	// without any HTML formatting. When creating or updating a comment,
	// this value must be supplied as plain text in the request.
	OriginalContent string `json:"originalContent,omitempty"`
}

type CommentPlusoners struct {
	// TotalItems: Total number of people who +1'd this comment.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type CommentFeed struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Id: The ID of this collection of comments.
	Id string `json:"id,omitempty"`

	// Items: The comments in this page of results.
	Items []*Comment `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of comments. Value:
	// "plus#commentFeed".
	Kind string `json:"kind,omitempty"`

	// NextLink: Link to the next page of activities.
	NextLink string `json:"nextLink,omitempty"`

	// NextPageToken: The continuation token, which is used to page through
	// large result sets. Provide this value in a subsequent request to
	// return the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Title: The title of this collection of comments.
	Title string `json:"title,omitempty"`

	// Updated: The time at which this collection of comments was last
	// updated. Formatted as an RFC 3339 timestamp.
	Updated string `json:"updated,omitempty"`
}

type Media struct {
	// Author: The person who uploaded this media.
	Author *MediaAuthor `json:"author,omitempty"`

	// DisplayName: The display name for this media.
	DisplayName string `json:"displayName,omitempty"`

	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Exif: Exif information of the media item.
	Exif *MediaExif `json:"exif,omitempty"`

	// Height: The height in pixels of the original image.
	Height int64 `json:"height,omitempty"`

	// Id: ID of this media, which is generated by the API.
	Id string `json:"id,omitempty"`

	// Kind: The type of resource.
	Kind string `json:"kind,omitempty"`

	// MediaUrl: The URL of this photo or video's still image.
	MediaUrl string `json:"mediaUrl,omitempty"`

	// Published: The time at which this media was uploaded. Formatted as an
	// RFC 3339 timestamp.
	Published string `json:"published,omitempty"`

	// SizeBytes: The size in bytes of this video.
	SizeBytes int64 `json:"sizeBytes,omitempty,string"`

	// Streams: The list of video streams for this video. There might be
	// several different streams available for a single video, either Flash
	// or MPEG, of various sizes
	Streams []*Videostream `json:"streams,omitempty"`

	// Summary: A description, or caption, for this media.
	Summary string `json:"summary,omitempty"`

	// Updated: The time at which this media was last updated. This includes
	// changes to media metadata. Formatted as an RFC 3339 timestamp.
	Updated string `json:"updated,omitempty"`

	// Url: The URL for the page that hosts this media.
	Url string `json:"url,omitempty"`

	// VideoDuration: The duration in milliseconds of this video.
	VideoDuration int64 `json:"videoDuration,omitempty,string"`

	// VideoStatus: The encoding status of this video. Possible values are:
	//
	// - "PENDING" - Video not yet processed.
	// - "FAILED" - Video
	// processing failed.
	// - "READY" - A single video stream is playable.
	// -
	// "FINAL" - All video streams are playable.
	VideoStatus string `json:"videoStatus,omitempty"`

	// Width: The width in pixels of the original image.
	Width int64 `json:"width,omitempty"`
}

type MediaAuthor struct {
	// DisplayName: The author's name.
	DisplayName string `json:"displayName,omitempty"`

	// Id: ID of the author.
	Id string `json:"id,omitempty"`

	// Image: The author's Google profile image.
	Image *MediaAuthorImage `json:"image,omitempty"`

	// Url: A link to the author's Google profile.
	Url string `json:"url,omitempty"`
}

type MediaAuthorImage struct {
	// Url: The URL of the author's profile photo. To resize the image and
	// crop it to a square, append the query string ?sz=x, where x is the
	// dimension in pixels of each side.
	Url string `json:"url,omitempty"`
}

type MediaExif struct {
	// Time: The time the media was captured. Formatted as an RFC 3339
	// timestamp.
	Time string `json:"time,omitempty"`
}

type PeopleFeed struct {
	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Items: The people in this page of results. Each item includes the id,
	// displayName, image, and url for the person. To retrieve additional
	// profile data, see the people.get method.
	Items []*Person `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of people. Value:
	// "plus#peopleFeed".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The continuation token, which is used to page through
	// large result sets. Provide this value in a subsequent request to
	// return the next page of results.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// SelfLink: Link to this resource.
	SelfLink string `json:"selfLink,omitempty"`

	// Title: The title of this collection of people.
	Title string `json:"title,omitempty"`

	// TotalItems: The total number of people available in this list. The
	// number of people in a response might be smaller due to paging. This
	// might not be set for all collections.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type Person struct {
	// AboutMe: A short biography for this person.
	AboutMe string `json:"aboutMe,omitempty"`

	// Birthday: The person's date of birth, represented as YYYY-MM-DD.
	Birthday string `json:"birthday,omitempty"`

	// BraggingRights: The "bragging rights" line of this person.
	BraggingRights string `json:"braggingRights,omitempty"`

	// CircledByCount: If a Google+ Page and for followers who are visible,
	// the number of people who have added this page to a circle.
	CircledByCount int64 `json:"circledByCount,omitempty"`

	// Cover: The cover photo content.
	Cover *PersonCover `json:"cover,omitempty"`

	// CurrentLocation: The current location for this person.
	CurrentLocation string `json:"currentLocation,omitempty"`

	// DisplayName: The name of this person, which is suitable for display.
	DisplayName string `json:"displayName,omitempty"`

	// Etag: ETag of this response for caching purposes.
	Etag string `json:"etag,omitempty"`

	// Gender: The person's gender. Possible values include, but are not
	// limited to, the following values:
	// - "male" - Male gender.
	// -
	// "female" - Female gender.
	// - "other" - Other.
	Gender string `json:"gender,omitempty"`

	// Id: The ID of this person.
	Id string `json:"id,omitempty"`

	// Image: The representation of the person's profile photo.
	Image *PersonImage `json:"image,omitempty"`

	// IsPlusUser: Whether this user has signed up for Google+.
	IsPlusUser bool `json:"isPlusUser,omitempty"`

	// Kind: Identifies this resource as a person. Value: "plus#person".
	Kind string `json:"kind,omitempty"`

	// Name: An object representation of the individual components of a
	// person's name.
	Name *PersonName `json:"name,omitempty"`

	// Nickname: The nickname of this person.
	Nickname string `json:"nickname,omitempty"`

	// ObjectType: Type of person within Google+. Possible values include,
	// but are not limited to, the following values:
	// - "person" -
	// represents an actual person.
	// - "page" - represents a page.
	ObjectType string `json:"objectType,omitempty"`

	// Organizations: A list of current or past organizations with which
	// this person is associated.
	Organizations []*PersonOrganizations `json:"organizations,omitempty"`

	// PlacesLived: A list of places where this person has lived.
	PlacesLived []*PersonPlacesLived `json:"placesLived,omitempty"`

	// PlusOneCount: If a Google+ Page, the number of people who have +1'd
	// this page.
	PlusOneCount int64 `json:"plusOneCount,omitempty"`

	// RelationshipStatus: The person's relationship status. Possible values
	// include, but are not limited to, the following values:
	// - "single" -
	// Person is single.
	// - "in_a_relationship" - Person is in a
	// relationship.
	// - "engaged" - Person is engaged.
	// - "married" - Person
	// is married.
	// - "its_complicated" - The relationship is complicated.
	//
	// - "open_relationship" - Person is in an open relationship.
	// -
	// "widowed" - Person is widowed.
	// - "in_domestic_partnership" - Person
	// is in a domestic partnership.
	// - "in_civil_union" - Person is in a
	// civil union.
	RelationshipStatus string `json:"relationshipStatus,omitempty"`

	// Tagline: The brief description (tagline) of this person.
	Tagline string `json:"tagline,omitempty"`

	// Url: The URL of this person's profile.
	Url string `json:"url,omitempty"`

	// Urls: A list of URLs for this person.
	Urls []*PersonUrls `json:"urls,omitempty"`

	// Verified: Whether the person or Google+ Page has been verified.
	Verified bool `json:"verified,omitempty"`
}

type PersonCover struct {
	// CoverInfo: Extra information about the cover photo.
	CoverInfo *PersonCoverCoverInfo `json:"coverInfo,omitempty"`

	// CoverPhoto: The person's primary cover image.
	CoverPhoto *PersonCoverCoverPhoto `json:"coverPhoto,omitempty"`

	// Layout: The layout of the cover art. Possible values include, but are
	// not limited to, the following values:
	// - "banner" - One large image
	// banner.
	Layout string `json:"layout,omitempty"`
}

type PersonCoverCoverInfo struct {
	// LeftImageOffset: The difference between the left position of the
	// cover image and the actual displayed cover image. Only valid for
	// banner layout.
	LeftImageOffset int64 `json:"leftImageOffset,omitempty"`

	// TopImageOffset: The difference between the top position of the cover
	// image and the actual displayed cover image. Only valid for banner
	// layout.
	TopImageOffset int64 `json:"topImageOffset,omitempty"`
}

type PersonCoverCoverPhoto struct {
	// Height: The height of the image.
	Height int64 `json:"height,omitempty"`

	// Url: The URL of the image.
	Url string `json:"url,omitempty"`

	// Width: The width of the image.
	Width int64 `json:"width,omitempty"`
}

type PersonImage struct {
	// Url: The URL of the person's profile photo. To resize the image and
	// crop it to a square, append the query string ?sz=x, where x is the
	// dimension in pixels of each side.
	Url string `json:"url,omitempty"`
}

type PersonName struct {
	// FamilyName: The family name (last name) of this person.
	FamilyName string `json:"familyName,omitempty"`

	// Formatted: The full name of this person, including middle names,
	// suffixes, etc.
	Formatted string `json:"formatted,omitempty"`

	// GivenName: The given name (first name) of this person.
	GivenName string `json:"givenName,omitempty"`

	// HonorificPrefix: The honorific prefixes (such as "Dr." or "Mrs.") for
	// this person.
	HonorificPrefix string `json:"honorificPrefix,omitempty"`

	// HonorificSuffix: The honorific suffixes (such as "Jr.") for this
	// person.
	HonorificSuffix string `json:"honorificSuffix,omitempty"`

	// MiddleName: The middle name of this person.
	MiddleName string `json:"middleName,omitempty"`
}

type PersonOrganizations struct {
	// Department: The department within the organization. Deprecated.
	Department string `json:"department,omitempty"`

	// Description: A short description of the person's role in this
	// organization. Deprecated.
	Description string `json:"description,omitempty"`

	// EndDate: The date that the person left this organization.
	EndDate string `json:"endDate,omitempty"`

	// Location: The location of this organization. Deprecated.
	Location string `json:"location,omitempty"`

	// Name: The name of the organization.
	Name string `json:"name,omitempty"`

	// Primary: If "true", indicates this organization is the person's
	// primary one, which is typically interpreted as the current one.
	Primary bool `json:"primary,omitempty"`

	// StartDate: The date that the person joined this organization.
	StartDate string `json:"startDate,omitempty"`

	// Title: The person's job title or role within the organization.
	Title string `json:"title,omitempty"`

	// Type: The type of organization. Possible values include, but are not
	// limited to, the following values:
	// - "work" - Work.
	// - "school" -
	// School.
	Type string `json:"type,omitempty"`
}

type PersonPlacesLived struct {
	// Primary: If "true", this place of residence is this person's primary
	// residence.
	Primary bool `json:"primary,omitempty"`

	// Value: A place where this person has lived. For example: "Seattle,
	// WA", "Near Toronto".
	Value string `json:"value,omitempty"`
}

type PersonUrls struct {
	// Label: The label of the URL.
	Label string `json:"label,omitempty"`

	// Type: The type of URL. Possible values include, but are not limited
	// to, the following values:
	// - "otherProfile" - URL for another
	// profile.
	// - "contributor" - URL to a site for which this person is a
	// contributor.
	// - "website" - URL for this Google+ Page's primary
	// website.
	// - "other" - Other URL.
	Type string `json:"type,omitempty"`

	// Value: The URL value.
	Value string `json:"value,omitempty"`
}

type Place struct {
	// Address: The physical address of the place.
	Address *PlaceAddress `json:"address,omitempty"`

	// DisplayName: The display name of the place.
	DisplayName string `json:"displayName,omitempty"`

	// Kind: Identifies this resource as a place. Value: "plus#place".
	Kind string `json:"kind,omitempty"`

	// Position: The position of the place.
	Position *PlacePosition `json:"position,omitempty"`
}

type PlaceAddress struct {
	// Formatted: The formatted address for display.
	Formatted string `json:"formatted,omitempty"`
}

type PlacePosition struct {
	// Latitude: The latitude of this position.
	Latitude float64 `json:"latitude,omitempty"`

	// Longitude: The longitude of this position.
	Longitude float64 `json:"longitude,omitempty"`
}

type PlusAclentryResource struct {
	// DisplayName: A descriptive name for this entry. Suitable for display.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The ID of the entry. For entries of type "person" or "circle",
	// this is the ID of the resource. For other types, this property is not
	// set.
	Id string `json:"id,omitempty"`

	// Type: The type of entry describing to whom access is granted.
	// Possible values are:
	// - "person" - Access to an individual.
	// -
	// "circle" - Access to members of a circle.
	// - "myCircles" - Access to
	// members of all the person's circles.
	// - "extendedCircles" - Access to
	// members of all the person's circles, plus all of the people in their
	// circles.
	// - "domain" - Access to members of the person's Google Apps
	// domain.
	// - "public" - Access to anyone on the web.
	Type string `json:"type,omitempty"`
}

type Videostream struct {
	// Height: The height, in pixels, of the video resource.
	Height int64 `json:"height,omitempty"`

	// Type: MIME type of the video stream.
	Type string `json:"type,omitempty"`

	// Url: URL of the video stream.
	Url string `json:"url,omitempty"`

	// Width: The width, in pixels, of the video resource.
	Width int64 `json:"width,omitempty"`
}

// method id "plus.activities.get":

type ActivitiesGetCall struct {
	s          *Service
	activityId string
	opt_       map[string]interface{}
}

// Get: Get an activity.
func (r *ActivitiesService) Get(activityId string) *ActivitiesGetCall {
	c := &ActivitiesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.activityId = activityId
	return c
}

func (c *ActivitiesGetCall) Do() (*Activity, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "activities/{activityId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{activityId}", url.QueryEscape(c.activityId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Activity)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get an activity.",
	//   "httpMethod": "GET",
	//   "id": "plus.activities.get",
	//   "parameterOrder": [
	//     "activityId"
	//   ],
	//   "parameters": {
	//     "activityId": {
	//       "description": "The ID of the activity to get.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "activities/{activityId}",
	//   "response": {
	//     "$ref": "Activity"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.me",
	//     "https://www.googleapis.com/auth/plus.stream.read"
	//   ]
	// }

}

// method id "plus.activities.insert":

type ActivitiesInsertCall struct {
	s        *Service
	userId   string
	activity *Activity
	opt_     map[string]interface{}
}

// Insert: Create a new activity for the authenticated user.
func (r *ActivitiesService) Insert(userId string, activity *Activity) *ActivitiesInsertCall {
	c := &ActivitiesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	c.activity = activity
	return c
}

// Preview sets the optional parameter "preview": If "true", extract the
// potential media attachments for a URL. The response will include all
// possible attachments for a URL, including video, photos, and articles
// based on the content of the page.
func (c *ActivitiesInsertCall) Preview(preview bool) *ActivitiesInsertCall {
	c.opt_["preview"] = preview
	return c
}

func (c *ActivitiesInsertCall) Do() (*Activity, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.activity)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["preview"]; ok {
		params.Set("preview", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "people/{userId}/activities")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Activity)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Create a new activity for the authenticated user.",
	//   "httpMethod": "POST",
	//   "id": "plus.activities.insert",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "preview": {
	//       "description": "If \"true\", extract the potential media attachments for a URL. The response will include all possible attachments for a URL, including video, photos, and articles based on the content of the page.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "userId": {
	//       "description": "The ID of the user to create the activity on behalf of. Its value should be \"me\", to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}/activities",
	//   "request": {
	//     "$ref": "Activity"
	//   },
	//   "response": {
	//     "$ref": "Activity"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.me",
	//     "https://www.googleapis.com/auth/plus.stream.write"
	//   ]
	// }

}

// method id "plus.activities.list":

type ActivitiesListCall struct {
	s          *Service
	userId     string
	collection string
	opt_       map[string]interface{}
}

// List: List all of the activities in the specified collection for a
// particular user.
func (r *ActivitiesService) List(userId string, collection string) *ActivitiesListCall {
	c := &ActivitiesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	c.collection = collection
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of activities to include in the response, which is used for
// paging. For any response, the actual number returned might be less
// than the specified maxResults.
func (c *ActivitiesListCall) MaxResults(maxResults int64) *ActivitiesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The continuation
// token, which is used to page through large result sets. To get the
// next page of results, set this parameter to the value of
// "nextPageToken" from the previous response.
func (c *ActivitiesListCall) PageToken(pageToken string) *ActivitiesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *ActivitiesListCall) Do() (*ActivityFeed, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "people/{userId}/activities/{collection}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{collection}", url.QueryEscape(c.collection), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(ActivityFeed)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all of the activities in the specified collection for a particular user.",
	//   "httpMethod": "GET",
	//   "id": "plus.activities.list",
	//   "parameterOrder": [
	//     "userId",
	//     "collection"
	//   ],
	//   "parameters": {
	//     "collection": {
	//       "description": "The collection of activities to list.",
	//       "enum": [
	//         "user"
	//       ],
	//       "enumDescriptions": [
	//         "All activities created by the specified user that the authenticated user is authorized to view."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "20",
	//       "description": "The maximum number of activities to include in the response, which is used for paging. For any response, the actual number returned might be less than the specified maxResults.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The continuation token, which is used to page through large result sets. To get the next page of results, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user to get activities for. The special value \"me\" can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}/activities/{collection}",
	//   "response": {
	//     "$ref": "ActivityFeed"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.me",
	//     "https://www.googleapis.com/auth/plus.stream.read"
	//   ]
	// }

}

// method id "plus.audiences.list":

type AudiencesListCall struct {
	s      *Service
	userId string
	opt_   map[string]interface{}
}

// List: List all of the audiences to which a user can share.
func (r *AudiencesService) List(userId string) *AudiencesListCall {
	c := &AudiencesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of circles to include in the response, which is used for
// paging. For any response, the actual number returned might be less
// than the specified maxResults.
func (c *AudiencesListCall) MaxResults(maxResults int64) *AudiencesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The continuation
// token, which is used to page through large result sets. To get the
// next page of results, set this parameter to the value of
// "nextPageToken" from the previous response.
func (c *AudiencesListCall) PageToken(pageToken string) *AudiencesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *AudiencesListCall) Do() (*AudiencesFeed, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "people/{userId}/audiences")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(AudiencesFeed)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all of the audiences to which a user can share.",
	//   "httpMethod": "GET",
	//   "id": "plus.audiences.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "20",
	//       "description": "The maximum number of circles to include in the response, which is used for paging. For any response, the actual number returned might be less than the specified maxResults.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The continuation token, which is used to page through large result sets. To get the next page of results, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user to get audiences for. The special value \"me\" can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}/audiences",
	//   "response": {
	//     "$ref": "AudiencesFeed"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.circles.read",
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.me"
	//   ]
	// }

}

// method id "plus.circles.addPeople":

type CirclesAddPeopleCall struct {
	s        *Service
	circleId string
	opt_     map[string]interface{}
}

// AddPeople: Add a person to a circle. Google+ limits certain circle
// operations, including the number of circle adds. Learn More.
func (r *CirclesService) AddPeople(circleId string) *CirclesAddPeopleCall {
	c := &CirclesAddPeopleCall{s: r.s, opt_: make(map[string]interface{})}
	c.circleId = circleId
	return c
}

// Email sets the optional parameter "email": Email of the people to add
// to the circle. Optional, can be repeated.
func (c *CirclesAddPeopleCall) Email(email string) *CirclesAddPeopleCall {
	c.opt_["email"] = email
	return c
}

// UserId sets the optional parameter "userId": IDs of the people to add
// to the circle. Optional, can be repeated.
func (c *CirclesAddPeopleCall) UserId(userId string) *CirclesAddPeopleCall {
	c.opt_["userId"] = userId
	return c
}

func (c *CirclesAddPeopleCall) Do() (*Circle, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["email"]; ok {
		params.Set("email", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["userId"]; ok {
		params.Set("userId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "circles/{circleId}/people")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{circleId}", url.QueryEscape(c.circleId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Circle)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Add a person to a circle. Google+ limits certain circle operations, including the number of circle adds. Learn More.",
	//   "httpMethod": "PUT",
	//   "id": "plus.circles.addPeople",
	//   "parameterOrder": [
	//     "circleId"
	//   ],
	//   "parameters": {
	//     "circleId": {
	//       "description": "The ID of the circle to add the person to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "email": {
	//       "description": "Email of the people to add to the circle. Optional, can be repeated.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "IDs of the people to add to the circle. Optional, can be repeated.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "circles/{circleId}/people",
	//   "response": {
	//     "$ref": "Circle"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.circles.write",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "plus.circles.get":

type CirclesGetCall struct {
	s        *Service
	circleId string
	opt_     map[string]interface{}
}

// Get: Get a circle.
func (r *CirclesService) Get(circleId string) *CirclesGetCall {
	c := &CirclesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.circleId = circleId
	return c
}

func (c *CirclesGetCall) Do() (*Circle, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "circles/{circleId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{circleId}", url.QueryEscape(c.circleId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Circle)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get a circle.",
	//   "httpMethod": "GET",
	//   "id": "plus.circles.get",
	//   "parameterOrder": [
	//     "circleId"
	//   ],
	//   "parameters": {
	//     "circleId": {
	//       "description": "The ID of the circle to get.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "circles/{circleId}",
	//   "response": {
	//     "$ref": "Circle"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.circles.read",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "plus.circles.insert":

type CirclesInsertCall struct {
	s      *Service
	userId string
	circle *Circle
	opt_   map[string]interface{}
}

// Insert: Create a new circle for the authenticated user.
func (r *CirclesService) Insert(userId string, circle *Circle) *CirclesInsertCall {
	c := &CirclesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	c.circle = circle
	return c
}

func (c *CirclesInsertCall) Do() (*Circle, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.circle)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "people/{userId}/circles")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Circle)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Create a new circle for the authenticated user.",
	//   "httpMethod": "POST",
	//   "id": "plus.circles.insert",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "description": "The ID of the user to create the circle on behalf of. The value \"me\" can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}/circles",
	//   "request": {
	//     "$ref": "Circle"
	//   },
	//   "response": {
	//     "$ref": "Circle"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.circles.write",
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.me"
	//   ]
	// }

}

// method id "plus.circles.list":

type CirclesListCall struct {
	s      *Service
	userId string
	opt_   map[string]interface{}
}

// List: List all of the circles for a user.
func (r *CirclesService) List(userId string) *CirclesListCall {
	c := &CirclesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of circles to include in the response, which is used for
// paging. For any response, the actual number returned might be less
// than the specified maxResults.
func (c *CirclesListCall) MaxResults(maxResults int64) *CirclesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The continuation
// token, which is used to page through large result sets. To get the
// next page of results, set this parameter to the value of
// "nextPageToken" from the previous response.
func (c *CirclesListCall) PageToken(pageToken string) *CirclesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *CirclesListCall) Do() (*CircleFeed, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "people/{userId}/circles")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(CircleFeed)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all of the circles for a user.",
	//   "httpMethod": "GET",
	//   "id": "plus.circles.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "20",
	//       "description": "The maximum number of circles to include in the response, which is used for paging. For any response, the actual number returned might be less than the specified maxResults.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The continuation token, which is used to page through large result sets. To get the next page of results, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user to get circles for. The special value \"me\" can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}/circles",
	//   "response": {
	//     "$ref": "CircleFeed"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.circles.read",
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.me"
	//   ]
	// }

}

// method id "plus.circles.patch":

type CirclesPatchCall struct {
	s        *Service
	circleId string
	circle   *Circle
	opt_     map[string]interface{}
}

// Patch: Update a circle's description. This method supports patch
// semantics.
func (r *CirclesService) Patch(circleId string, circle *Circle) *CirclesPatchCall {
	c := &CirclesPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.circleId = circleId
	c.circle = circle
	return c
}

func (c *CirclesPatchCall) Do() (*Circle, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.circle)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "circles/{circleId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{circleId}", url.QueryEscape(c.circleId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Circle)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update a circle's description. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "plus.circles.patch",
	//   "parameterOrder": [
	//     "circleId"
	//   ],
	//   "parameters": {
	//     "circleId": {
	//       "description": "The ID of the circle to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "circles/{circleId}",
	//   "request": {
	//     "$ref": "Circle"
	//   },
	//   "response": {
	//     "$ref": "Circle"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.circles.write",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "plus.circles.remove":

type CirclesRemoveCall struct {
	s        *Service
	circleId string
	opt_     map[string]interface{}
}

// Remove: Delete a circle.
func (r *CirclesService) Remove(circleId string) *CirclesRemoveCall {
	c := &CirclesRemoveCall{s: r.s, opt_: make(map[string]interface{})}
	c.circleId = circleId
	return c
}

func (c *CirclesRemoveCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "circles/{circleId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{circleId}", url.QueryEscape(c.circleId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Delete a circle.",
	//   "httpMethod": "DELETE",
	//   "id": "plus.circles.remove",
	//   "parameterOrder": [
	//     "circleId"
	//   ],
	//   "parameters": {
	//     "circleId": {
	//       "description": "The ID of the circle to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "circles/{circleId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.circles.write",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "plus.circles.removePeople":

type CirclesRemovePeopleCall struct {
	s        *Service
	circleId string
	opt_     map[string]interface{}
}

// RemovePeople: Remove a person from a circle.
func (r *CirclesService) RemovePeople(circleId string) *CirclesRemovePeopleCall {
	c := &CirclesRemovePeopleCall{s: r.s, opt_: make(map[string]interface{})}
	c.circleId = circleId
	return c
}

// Email sets the optional parameter "email": Email of the people to add
// to the circle. Optional, can be repeated.
func (c *CirclesRemovePeopleCall) Email(email string) *CirclesRemovePeopleCall {
	c.opt_["email"] = email
	return c
}

// UserId sets the optional parameter "userId": IDs of the people to
// remove from the circle. Optional, can be repeated.
func (c *CirclesRemovePeopleCall) UserId(userId string) *CirclesRemovePeopleCall {
	c.opt_["userId"] = userId
	return c
}

func (c *CirclesRemovePeopleCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["email"]; ok {
		params.Set("email", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["userId"]; ok {
		params.Set("userId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "circles/{circleId}/people")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{circleId}", url.QueryEscape(c.circleId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Remove a person from a circle.",
	//   "httpMethod": "DELETE",
	//   "id": "plus.circles.removePeople",
	//   "parameterOrder": [
	//     "circleId"
	//   ],
	//   "parameters": {
	//     "circleId": {
	//       "description": "The ID of the circle to remove the person from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "email": {
	//       "description": "Email of the people to add to the circle. Optional, can be repeated.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "IDs of the people to remove from the circle. Optional, can be repeated.",
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "circles/{circleId}/people",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.circles.write",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "plus.circles.update":

type CirclesUpdateCall struct {
	s        *Service
	circleId string
	circle   *Circle
	opt_     map[string]interface{}
}

// Update: Update a circle's description.
func (r *CirclesService) Update(circleId string, circle *Circle) *CirclesUpdateCall {
	c := &CirclesUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.circleId = circleId
	c.circle = circle
	return c
}

func (c *CirclesUpdateCall) Do() (*Circle, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.circle)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "circles/{circleId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{circleId}", url.QueryEscape(c.circleId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Circle)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update a circle's description.",
	//   "httpMethod": "PUT",
	//   "id": "plus.circles.update",
	//   "parameterOrder": [
	//     "circleId"
	//   ],
	//   "parameters": {
	//     "circleId": {
	//       "description": "The ID of the circle to update.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "circles/{circleId}",
	//   "request": {
	//     "$ref": "Circle"
	//   },
	//   "response": {
	//     "$ref": "Circle"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.circles.write",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}

// method id "plus.comments.get":

type CommentsGetCall struct {
	s         *Service
	commentId string
	opt_      map[string]interface{}
}

// Get: Get a comment.
func (r *CommentsService) Get(commentId string) *CommentsGetCall {
	c := &CommentsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.commentId = commentId
	return c
}

func (c *CommentsGetCall) Do() (*Comment, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "comments/{commentId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{commentId}", url.QueryEscape(c.commentId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Comment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get a comment.",
	//   "httpMethod": "GET",
	//   "id": "plus.comments.get",
	//   "parameterOrder": [
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "commentId": {
	//       "description": "The ID of the comment to get.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "comments/{commentId}",
	//   "response": {
	//     "$ref": "Comment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.stream.read"
	//   ]
	// }

}

// method id "plus.comments.insert":

type CommentsInsertCall struct {
	s          *Service
	activityId string
	comment    *Comment
	opt_       map[string]interface{}
}

// Insert: Create a new comment in reply to an activity.
func (r *CommentsService) Insert(activityId string, comment *Comment) *CommentsInsertCall {
	c := &CommentsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.activityId = activityId
	c.comment = comment
	return c
}

func (c *CommentsInsertCall) Do() (*Comment, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.comment)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "activities/{activityId}/comments")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{activityId}", url.QueryEscape(c.activityId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Comment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Create a new comment in reply to an activity.",
	//   "httpMethod": "POST",
	//   "id": "plus.comments.insert",
	//   "parameterOrder": [
	//     "activityId"
	//   ],
	//   "parameters": {
	//     "activityId": {
	//       "description": "The ID of the activity to reply to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "activities/{activityId}/comments",
	//   "request": {
	//     "$ref": "Comment"
	//   },
	//   "response": {
	//     "$ref": "Comment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.stream.write"
	//   ]
	// }

}

// method id "plus.comments.list":

type CommentsListCall struct {
	s          *Service
	activityId string
	opt_       map[string]interface{}
}

// List: List all of the comments for an activity.
func (r *CommentsService) List(activityId string) *CommentsListCall {
	c := &CommentsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.activityId = activityId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of comments to include in the response, which is used for
// paging. For any response, the actual number returned might be less
// than the specified maxResults.
func (c *CommentsListCall) MaxResults(maxResults int64) *CommentsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The continuation
// token, which is used to page through large result sets. To get the
// next page of results, set this parameter to the value of
// "nextPageToken" from the previous response.
func (c *CommentsListCall) PageToken(pageToken string) *CommentsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// SortOrder sets the optional parameter "sortOrder": The order in which
// to sort the list of comments.
func (c *CommentsListCall) SortOrder(sortOrder string) *CommentsListCall {
	c.opt_["sortOrder"] = sortOrder
	return c
}

func (c *CommentsListCall) Do() (*CommentFeed, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["sortOrder"]; ok {
		params.Set("sortOrder", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "activities/{activityId}/comments")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{activityId}", url.QueryEscape(c.activityId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(CommentFeed)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all of the comments for an activity.",
	//   "httpMethod": "GET",
	//   "id": "plus.comments.list",
	//   "parameterOrder": [
	//     "activityId"
	//   ],
	//   "parameters": {
	//     "activityId": {
	//       "description": "The ID of the activity to get comments for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "20",
	//       "description": "The maximum number of comments to include in the response, which is used for paging. For any response, the actual number returned might be less than the specified maxResults.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "500",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The continuation token, which is used to page through large result sets. To get the next page of results, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "sortOrder": {
	//       "default": "ascending",
	//       "description": "The order in which to sort the list of comments.",
	//       "enum": [
	//         "ascending",
	//         "descending"
	//       ],
	//       "enumDescriptions": [
	//         "Sort oldest comments first.",
	//         "Sort newest comments first."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "activities/{activityId}/comments",
	//   "response": {
	//     "$ref": "CommentFeed"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.stream.read"
	//   ]
	// }

}

// method id "plus.media.insert":

type MediaInsertCall struct {
	s          *Service
	userId     string
	collection string
	media      *Media
	opt_       map[string]interface{}
	media_     io.Reader
}

// Insert: Add a new media item to an album. The current upload size
// limitations are 36MB for a photo and 1GB for a video. Uploads do not
// count against quota if photos are less than 2048 pixels on their
// longest side or videos are less than 15 minutes in length.
func (r *MediaService) Insert(userId string, collection string, media *Media) *MediaInsertCall {
	c := &MediaInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	c.collection = collection
	c.media = media
	return c
}
func (c *MediaInsertCall) Media(r io.Reader) *MediaInsertCall {
	c.media_ = r
	return c
}

func (c *MediaInsertCall) Do() (*Media, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.media)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "people/{userId}/media/{collection}")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{collection}", url.QueryEscape(c.collection), 1)
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
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Media)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Add a new media item to an album. The current upload size limitations are 36MB for a photo and 1GB for a video. Uploads do not count against quota if photos are less than 2048 pixels on their longest side or videos are less than 15 minutes in length.",
	//   "httpMethod": "POST",
	//   "id": "plus.media.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "image/*",
	//       "video/*"
	//     ],
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/plus/v1domains/people/{userId}/media/{collection}"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/plus/v1domains/people/{userId}/media/{collection}"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "userId",
	//     "collection"
	//   ],
	//   "parameters": {
	//     "collection": {
	//       "enum": [
	//         "cloud"
	//       ],
	//       "enumDescriptions": [
	//         "Upload the media to share on Google+."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user to create the activity on behalf of.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}/media/{collection}",
	//   "request": {
	//     "$ref": "Media"
	//   },
	//   "response": {
	//     "$ref": "Media"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.media.upload"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "plus.people.get":

type PeopleGetCall struct {
	s      *Service
	userId string
	opt_   map[string]interface{}
}

// Get: Get a person's profile.
func (r *PeopleService) Get(userId string) *PeopleGetCall {
	c := &PeopleGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	return c
}

func (c *PeopleGetCall) Do() (*Person, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "people/{userId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Person)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get a person's profile.",
	//   "httpMethod": "GET",
	//   "id": "plus.people.get",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "description": "The ID of the person to get the profile for. The special value \"me\" can be used to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}",
	//   "response": {
	//     "$ref": "Person"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.me",
	//     "https://www.googleapis.com/auth/plus.profiles.read"
	//   ]
	// }

}

// method id "plus.people.list":

type PeopleListCall struct {
	s          *Service
	userId     string
	collection string
	opt_       map[string]interface{}
}

// List: List all of the people in the specified collection.
func (r *PeopleService) List(userId string, collection string) *PeopleListCall {
	c := &PeopleListCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	c.collection = collection
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of people to include in the response, which is used for
// paging. For any response, the actual number returned might be less
// than the specified maxResults.
func (c *PeopleListCall) MaxResults(maxResults int64) *PeopleListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// OrderBy sets the optional parameter "orderBy": The order to return
// people in.
func (c *PeopleListCall) OrderBy(orderBy string) *PeopleListCall {
	c.opt_["orderBy"] = orderBy
	return c
}

// PageToken sets the optional parameter "pageToken": The continuation
// token, which is used to page through large result sets. To get the
// next page of results, set this parameter to the value of
// "nextPageToken" from the previous response.
func (c *PeopleListCall) PageToken(pageToken string) *PeopleListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *PeopleListCall) Do() (*PeopleFeed, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["orderBy"]; ok {
		params.Set("orderBy", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "people/{userId}/people/{collection}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{collection}", url.QueryEscape(c.collection), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PeopleFeed)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all of the people in the specified collection.",
	//   "httpMethod": "GET",
	//   "id": "plus.people.list",
	//   "parameterOrder": [
	//     "userId",
	//     "collection"
	//   ],
	//   "parameters": {
	//     "collection": {
	//       "description": "The collection of people to list.",
	//       "enum": [
	//         "circled"
	//       ],
	//       "enumDescriptions": [
	//         "The list of people who this user has added to one or more circles."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "100",
	//       "description": "The maximum number of people to include in the response, which is used for paging. For any response, the actual number returned might be less than the specified maxResults.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "orderBy": {
	//       "description": "The order to return people in.",
	//       "enum": [
	//         "alphabetical",
	//         "best"
	//       ],
	//       "enumDescriptions": [
	//         "Order the people by their display name.",
	//         "Order people based on the relevence to the viewer."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The continuation token, which is used to page through large result sets. To get the next page of results, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Get the collection of people for the person identified. Use \"me\" to indicate the authenticated user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}/people/{collection}",
	//   "response": {
	//     "$ref": "PeopleFeed"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.circles.read",
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.me"
	//   ]
	// }

}

// method id "plus.people.listByActivity":

type PeopleListByActivityCall struct {
	s          *Service
	activityId string
	collection string
	opt_       map[string]interface{}
}

// ListByActivity: List all of the people in the specified collection
// for a particular activity.
func (r *PeopleService) ListByActivity(activityId string, collection string) *PeopleListByActivityCall {
	c := &PeopleListByActivityCall{s: r.s, opt_: make(map[string]interface{})}
	c.activityId = activityId
	c.collection = collection
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of people to include in the response, which is used for
// paging. For any response, the actual number returned might be less
// than the specified maxResults.
func (c *PeopleListByActivityCall) MaxResults(maxResults int64) *PeopleListByActivityCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The continuation
// token, which is used to page through large result sets. To get the
// next page of results, set this parameter to the value of
// "nextPageToken" from the previous response.
func (c *PeopleListByActivityCall) PageToken(pageToken string) *PeopleListByActivityCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *PeopleListByActivityCall) Do() (*PeopleFeed, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "activities/{activityId}/people/{collection}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{activityId}", url.QueryEscape(c.activityId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{collection}", url.QueryEscape(c.collection), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PeopleFeed)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all of the people in the specified collection for a particular activity.",
	//   "httpMethod": "GET",
	//   "id": "plus.people.listByActivity",
	//   "parameterOrder": [
	//     "activityId",
	//     "collection"
	//   ],
	//   "parameters": {
	//     "activityId": {
	//       "description": "The ID of the activity to get the list of people for.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "collection": {
	//       "description": "The collection of people to list.",
	//       "enum": [
	//         "plusoners",
	//         "resharers",
	//         "sharedto"
	//       ],
	//       "enumDescriptions": [
	//         "List all people who have +1'd this activity.",
	//         "List all people who have reshared this activity.",
	//         "List all people who this activity was shared to."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "20",
	//       "description": "The maximum number of people to include in the response, which is used for paging. For any response, the actual number returned might be less than the specified maxResults.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The continuation token, which is used to page through large result sets. To get the next page of results, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "activities/{activityId}/people/{collection}",
	//   "response": {
	//     "$ref": "PeopleFeed"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.login",
	//     "https://www.googleapis.com/auth/plus.stream.read"
	//   ]
	// }

}

// method id "plus.people.listByCircle":

type PeopleListByCircleCall struct {
	s        *Service
	circleId string
	opt_     map[string]interface{}
}

// ListByCircle: List all of the people who are members of a circle.
func (r *PeopleService) ListByCircle(circleId string) *PeopleListByCircleCall {
	c := &PeopleListByCircleCall{s: r.s, opt_: make(map[string]interface{})}
	c.circleId = circleId
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of people to include in the response, which is used for
// paging. For any response, the actual number returned might be less
// than the specified maxResults.
func (c *PeopleListByCircleCall) MaxResults(maxResults int64) *PeopleListByCircleCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": The continuation
// token, which is used to page through large result sets. To get the
// next page of results, set this parameter to the value of
// "nextPageToken" from the previous response.
func (c *PeopleListByCircleCall) PageToken(pageToken string) *PeopleListByCircleCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *PeopleListByCircleCall) Do() (*PeopleFeed, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/plus/v1domains/", "circles/{circleId}/people")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{circleId}", url.QueryEscape(c.circleId), 1)
	googleapi.SetOpaque(req.URL)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PeopleFeed)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "List all of the people who are members of a circle.",
	//   "httpMethod": "GET",
	//   "id": "plus.people.listByCircle",
	//   "parameterOrder": [
	//     "circleId"
	//   ],
	//   "parameters": {
	//     "circleId": {
	//       "description": "The ID of the circle to get the members of.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "20",
	//       "description": "The maximum number of people to include in the response, which is used for paging. For any response, the actual number returned might be less than the specified maxResults.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "The continuation token, which is used to page through large result sets. To get the next page of results, set this parameter to the value of \"nextPageToken\" from the previous response.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "circles/{circleId}/people",
	//   "response": {
	//     "$ref": "PeopleFeed"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/plus.circles.read",
	//     "https://www.googleapis.com/auth/plus.login"
	//   ]
	// }

}
