// Package youtube provides access to the YouTube API.
//
// See https://developers.google.com/youtube
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/youtube/v3alpha"
//   ...
//   youtubeService, err := youtube.New(oauthHttpClient)
package youtube

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

var _ = bytes.NewBuffer
var _ = strconv.Itoa
var _ = fmt.Sprintf
var _ = json.NewDecoder
var _ = io.Copy
var _ = url.Parse
var _ = googleapi.Version
var _ = errors.New

const apiId = "youtube:v3alpha"
const apiName = "youtube"
const apiVersion = "v3alpha"
const basePath = "https://www.googleapis.com/youtube/v3alpha/"

// OAuth2 scopes used by this API.
const (
	// Manage your YouTube account
	YoutubeScope = "https://www.googleapis.com/auth/youtube"

	// View your YouTube account
	YoutubeReadonlyScope = "https://www.googleapis.com/auth/youtube.readonly"

	// Manage your YouTube videos
	YoutubeUploadScope = "https://www.googleapis.com/auth/youtube.upload"

	// View and manage your assets and associated content on YouTube
	YoutubepartnerScope = "https://www.googleapis.com/auth/youtubepartner"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client}
	s.Activities = &ActivitiesService{s: s}
	s.Channels = &ChannelsService{s: s}
	s.PlaylistItems = &PlaylistItemsService{s: s}
	s.Playlists = &PlaylistsService{s: s}
	s.Search = &SearchService{s: s}
	s.Videos = &VideosService{s: s}
	return s, nil
}

type Service struct {
	client *http.Client

	Activities *ActivitiesService

	Channels *ChannelsService

	PlaylistItems *PlaylistItemsService

	Playlists *PlaylistsService

	Search *SearchService

	Videos *VideosService
}

type ActivitiesService struct {
	s *Service
}

type ChannelsService struct {
	s *Service
}

type PlaylistItemsService struct {
	s *Service
}

type PlaylistsService struct {
	s *Service
}

type SearchService struct {
	s *Service
}

type VideosService struct {
	s *Service
}

type Activity struct {
	// ContentDetails: Type specific information about the activity.
	ContentDetails *ActivityContentDetails `json:"contentDetails,omitempty"`

	// Etag: The eTag of the activity.
	Etag string `json:"etag,omitempty"`

	// Id: The unique ID of the activity.
	Id string `json:"id,omitempty"`

	// Snippet: Basic details about the activity: title, description,
	// thumbnails.
	Snippet *ActivitySnippet `json:"snippet,omitempty"`
}

type ActivityContentDetails struct {
	// BulletinPosted: Only present if the type is "bulletinPosted".
	BulletinPosted *ActivityContentDetailsBulletinPosted `json:"bulletinPosted,omitempty"`

	// SubscriptionAdded: Only present if the type is "subscriptionAdded".
	SubscriptionAdded *ActivityContentDetailsSubscriptionAdded `json:"subscriptionAdded,omitempty"`

	// VideoAddedToPlaylist: Only present if the type is
	// "videoAddedToPlaylist".
	VideoAddedToPlaylist *ActivityContentDetailsVideoAddedToPlaylist `json:"videoAddedToPlaylist,omitempty"`

	// VideoCommented: Only present if the type is "videoCommented".
	VideoCommented *ActivityContentDetailsVideoCommented `json:"videoCommented,omitempty"`

	// VideoFavorited: Only present if the type is "videoFavorited".
	VideoFavorited *ActivityContentDetailsVideoFavorited `json:"videoFavorited,omitempty"`

	// VideoRated: Only present if the type is "videoRated".
	VideoRated *ActivityContentDetailsVideoRated `json:"videoRated,omitempty"`

	// VideoRecommended: Only set if the type is "videoRecommended".
	VideoRecommended *ActivityContentDetailsVideoRecommended `json:"videoRecommended,omitempty"`

	// VideoUploaded: Only present if the type is "videoUploaded".
	VideoUploaded *ActivityContentDetailsVideoUploaded `json:"videoUploaded,omitempty"`
}

type ActivityContentDetailsBulletinPosted struct {
	// BulletinText: Text if the posted bulletin.
	BulletinText string `json:"bulletinText,omitempty"`

	// PlaylistId: ID of the playlist this bulletin is about.
	PlaylistId string `json:"playlistId,omitempty"`

	// VideoId: ID of the video this bulletin is about.
	VideoId string `json:"videoId,omitempty"`
}

type ActivityContentDetailsSubscriptionAdded struct {
	// ChannelId: ID of the channel subscribed to.
	ChannelId string `json:"channelId,omitempty"`
}

type ActivityContentDetailsVideoAddedToPlaylist struct {
	// PlaylistId: ID of the playlist the video was added to.
	PlaylistId string `json:"playlistId,omitempty"`

	// VideoId: ID of the video added to the playlist.
	VideoId string `json:"videoId,omitempty"`
}

type ActivityContentDetailsVideoCommented struct {
	// VideoId: ID of the commented video.
	VideoId string `json:"videoId,omitempty"`
}

type ActivityContentDetailsVideoFavorited struct {
	// VideoId: ID of the favorited video.
	VideoId string `json:"videoId,omitempty"`
}

type ActivityContentDetailsVideoRated struct {
	// VideoId: ID of the rated video.
	VideoId string `json:"videoId,omitempty"`
}

type ActivityContentDetailsVideoRecommended struct {
	// RecommendationReason: Reason for which the video was recommended.
	RecommendationReason string `json:"recommendationReason,omitempty"`

	// SeedVideoId: ID of the video that caused this recommendation.
	SeedVideoId string `json:"seedVideoId,omitempty"`

	// VideoId: ID of the recommended video.
	VideoId string `json:"videoId,omitempty"`
}

type ActivityContentDetailsVideoUploaded struct {
	// VideoId: ID of the uploaded video.
	VideoId string `json:"videoId,omitempty"`
}

type ActivityListResponse struct {
	// Activities: Map of activities matching the request criteria, keyed by
	// activity id.
	Activities *ActivityListResponseActivities `json:"activities,omitempty"`

	// Etag: The eTag of the response.
	Etag string `json:"etag,omitempty"`

	// Kind: The type of this API response.
	Kind string `json:"kind,omitempty"`
}

type ActivityListResponseActivities struct {
}

type ActivitySnippet struct {
	// ChannelId: Channel responsible for this activity.
	ChannelId string `json:"channelId,omitempty"`

	// Description: Description of the main resource.
	Description string `json:"description,omitempty"`

	// GroupId: Id of the group that this activity is part of.
	GroupId string `json:"groupId,omitempty"`

	// PublishedAt: Time when this activity was created.
	PublishedAt string `json:"publishedAt,omitempty"`

	// Thumbnails: Thumbnails of the main resource.
	Thumbnails *ActivitySnippetThumbnails `json:"thumbnails,omitempty"`

	// Title: Title of the main resource.
	Title string `json:"title,omitempty"`

	// Type: Category of the activity activity.
	Type string `json:"type,omitempty"`
}

type ActivitySnippetThumbnails struct {
}

type Channel struct {
	// ContentDetails: Information about the channel content: upload
	// playlist id, privacy status.
	ContentDetails *ChannelContentDetails `json:"contentDetails,omitempty"`

	// Etag: The eTag of the channel.
	Etag string `json:"etag,omitempty"`

	// Id: The unique ID of the channel.
	Id string `json:"id,omitempty"`

	// Kind: The type of this API resource.
	Kind string `json:"kind,omitempty"`

	// Snippet: Basic details about the channel: title, description, and
	// thumbnails.
	Snippet *ChannelSnippet `json:"snippet,omitempty"`

	// Statistics: Statistics about the channel: number of subscribers,
	// views, and comments.
	Statistics *ChannelStatistics `json:"statistics,omitempty"`

	// TopicDetails: Information about channel topics
	TopicDetails *ChannelTopicDetails `json:"topicDetails,omitempty"`
}

type ChannelContentDetails struct {
	// PrivacyStatus: Privacy status of the channel.
	PrivacyStatus string `json:"privacyStatus,omitempty"`

	// Uploads: The ID of the playlist containing the uploads of this
	// channel.
	Uploads string `json:"uploads,omitempty"`
}

type ChannelListResponse struct {
	// Channels: Map of channels matching the request criteria, keyed by
	// channel id.
	Channels *ChannelListResponseChannels `json:"channels,omitempty"`

	// Etag: The eTag of the response.
	Etag string `json:"etag,omitempty"`

	// Kind: The type of this API response.
	Kind string `json:"kind,omitempty"`
}

type ChannelListResponseChannels struct {
}

type ChannelSnippet struct {
	// Description: Description of the channel.
	Description string `json:"description,omitempty"`

	// Thumbnails: Channel thumbnails.
	Thumbnails *ChannelSnippetThumbnails `json:"thumbnails,omitempty"`

	// Title: Title of the channel.
	Title string `json:"title,omitempty"`
}

type ChannelSnippetThumbnails struct {
}

type ChannelStatistics struct {
	// CommentCount: Number of comments for this channel.
	CommentCount uint64 `json:"commentCount,omitempty,string"`

	// SubscriberCount: Number of subscribers to this channel.
	SubscriberCount uint64 `json:"subscriberCount,omitempty,string"`

	// VideoCount: Number of videos in the channel.
	VideoCount uint64 `json:"videoCount,omitempty,string"`

	// ViewCount: Number of times the channel has been viewed.
	ViewCount uint64 `json:"viewCount,omitempty,string"`
}

type ChannelTopicDetails struct {
	// Topics: List of topic ids for this channel *
	Topics []string `json:"topics,omitempty"`
}

type PageInfo struct {
	// ResultPerPage: The number of results to display for each page.
	ResultPerPage int64 `json:"resultPerPage,omitempty"`

	// StartIndex: The index position of the first result to display.
	StartIndex int64 `json:"startIndex,omitempty"`

	// TotalResults: The total number of results.
	TotalResults int64 `json:"totalResults,omitempty"`
}

type Playlist struct {
	// Etag: The eTag of the playlist.
	Etag string `json:"etag,omitempty"`

	// Id: The unique id of the playlist.
	Id string `json:"id,omitempty"`

	// Kind: The type of this API resource.
	Kind string `json:"kind,omitempty"`

	// Snippet: Basic details about the playlist: title, description,
	// thumbnails.
	Snippet *PlaylistSnippet `json:"snippet,omitempty"`
}

type PlaylistItem struct {
	// ContentDetails: Content details about the playlist item: start and
	// end clipping time.
	ContentDetails *PlaylistItemContentDetails `json:"contentDetails,omitempty"`

	// Etag: The eTag of the playlist item.
	Etag string `json:"etag,omitempty"`

	// Id: The unique id of the playlist item.
	Id string `json:"id,omitempty"`

	// Kind: The type of this API resource.
	Kind string `json:"kind,omitempty"`

	// Snippet: Basic details about the playlist item: title, description,
	// thumbnails.
	Snippet *PlaylistItemSnippet `json:"snippet,omitempty"`
}

type PlaylistItemContentDetails struct {
	// EndAt: The time video playback ends.
	EndAt string `json:"endAt,omitempty"`

	// Note: The user-generated note for this item.
	Note string `json:"note,omitempty"`

	// StartAt: The time video playback begins.
	StartAt string `json:"startAt,omitempty"`

	// VideoId: ID of the video.
	VideoId string `json:"videoId,omitempty"`
}

type PlaylistItemListResponse struct {
	// Etag: The eTag of the response.
	Etag string `json:"etag,omitempty"`

	// Kind: The type of this API response.
	Kind string `json:"kind,omitempty"`

	// PlaylistItems: Map of playlist items matching the request criteria,
	// keyed by id.
	PlaylistItems *PlaylistItemListResponsePlaylistItems `json:"playlistItems,omitempty"`
}

type PlaylistItemListResponsePlaylistItems struct {
}

type PlaylistItemSnippet struct {
	// ChannelId: Author of the playlist item.
	ChannelId string `json:"channelId,omitempty"`

	// Description: Description of the playlist item.
	Description string `json:"description,omitempty"`

	// PlaylistId: The playlist the item is part of.
	PlaylistId string `json:"playlistId,omitempty"`

	// Position: The position of the item within the playlist.
	Position int64 `json:"position,omitempty"`

	// PublishedAt: The date and time the playlist item was created.
	PublishedAt string `json:"publishedAt,omitempty"`

	// ResourceId: The ID of the resource referenced by the playlist item.
	ResourceId *ResourceId `json:"resourceId,omitempty"`

	// Title: Title of the playlist item.
	Title string `json:"title,omitempty"`
}

type PlaylistListResponse struct {
	// Etag: The eTag of the response.
	Etag string `json:"etag,omitempty"`

	// Kind: The type of this API response.
	Kind string `json:"kind,omitempty"`

	// Playlists: Map of playlists matching the request criteria, keyed by
	// id.
	Playlists *PlaylistListResponsePlaylists `json:"playlists,omitempty"`
}

type PlaylistListResponsePlaylists struct {
}

type PlaylistSnippet struct {
	// ChannelId: Author of the playlist.
	ChannelId string `json:"channelId,omitempty"`

	// Description: Description of the playlist.
	Description string `json:"description,omitempty"`

	// PublishedAt: The date and time the playlist was created.
	PublishedAt string `json:"publishedAt,omitempty"`

	// Tags: Textual tags associated with the playlist.
	Tags []string `json:"tags,omitempty"`

	// Title: Title of the playlist.
	Title string `json:"title,omitempty"`
}

type ResourceId struct {
	// ChannelId: ID of the referred channel. Present only when type is
	// "CHANNEL".
	ChannelId string `json:"channelId,omitempty"`

	// Kind: The kind of the referred resource.
	Kind string `json:"kind,omitempty"`

	// PlaylistId: ID of the referred playlist. Present only when type is
	// "PLAYLIST".
	PlaylistId string `json:"playlistId,omitempty"`

	// VideoId: ID of the referred video. Present only when type is "VIDEO".
	VideoId string `json:"videoId,omitempty"`
}

type SearchListResponse struct {
	// Etag: The eTag of the response.
	Etag string `json:"etag,omitempty"`

	// Kind: The type of this API response.
	Kind string `json:"kind,omitempty"`

	// PageInfo: Paging information for the search result.
	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// SearchResults: List of results matching the request criteria.
	SearchResults []*SearchResult `json:"searchResults,omitempty"`

	// TokenPagination: Pagination information for the next and previous
	// page.
	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`
}

type SearchResult struct {
	// Etag: The eTag of the search result.
	Etag string `json:"etag,omitempty"`

	// Id: The id of the resource.
	Id *ResourceId `json:"id,omitempty"`

	// Kind: The type of this API resource.
	Kind string `json:"kind,omitempty"`

	// Snippet: Basic details about the search result: title, description,
	// author.
	Snippet *SearchResultSnippet `json:"snippet,omitempty"`
}

type SearchResultSnippet struct {
	// ChannelId: Author of the found resource.
	ChannelId string `json:"channelId,omitempty"`

	// Description: Description of the search result.
	Description string `json:"description,omitempty"`

	// PublishedAt: The date and time the found resource was created.
	PublishedAt string `json:"publishedAt,omitempty"`

	// Title: Title of the search result.
	Title string `json:"title,omitempty"`
}

type Thumbnail struct {
	// Url: The URL for the thumbnail.
	Url string `json:"url,omitempty"`
}

type TokenPagination struct {
	// NextPageToken: Token to the next page.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// PreviousPageToken: Token to the previous page.
	PreviousPageToken string `json:"previousPageToken,omitempty"`
}

type Video struct {
	// ContentDetails: Information about the video content, media file.
	ContentDetails *VideoContentDetails `json:"contentDetails,omitempty"`

	// Etag: The eTag of the video.
	Etag string `json:"etag,omitempty"`

	// Id: The unique id of the video.
	Id string `json:"id,omitempty"`

	// Kind: The type of this API resource.
	Kind string `json:"kind,omitempty"`

	// Player: Information used to play the video.
	Player *VideoPlayer `json:"player,omitempty"`

	// Snippet: Basic details about the video: title, description,
	// thumbnails.
	Snippet *VideoSnippet `json:"snippet,omitempty"`

	// Statistics: Statistics about the video: number of views, ratings.
	Statistics *VideoStatistics `json:"statistics,omitempty"`

	// Status: Status of the video upload, privacy status.
	Status *VideoStatus `json:"status,omitempty"`
}

type VideoContentDetails struct {
	// AspectRatio: The aspect ratio of the video.
	AspectRatio string `json:"aspectRatio,omitempty"`

	// Duration: Duration of the video.
	Duration string `json:"duration,omitempty"`
}

type VideoListResponse struct {
	// Etag: The eTag of the response.
	Etag string `json:"etag,omitempty"`

	// Kind: The type of this API response.
	Kind string `json:"kind,omitempty"`

	// Videos: Map of videos matching the request criteria, keyed by video
	// id.
	Videos *VideoListResponseVideos `json:"videos,omitempty"`
}

type VideoListResponseVideos struct {
}

type VideoPlayer struct {
	// EmbedHtml: Iframe embed for the video.
	EmbedHtml string `json:"embedHtml,omitempty"`
}

type VideoSnippet struct {
	// CategoryId: Video category the video belongs to.
	CategoryId string `json:"categoryId,omitempty"`

	// ChannelId: Channel the video was uploaded into.
	ChannelId string `json:"channelId,omitempty"`

	// Description: Description of the video.
	Description string `json:"description,omitempty"`

	// PublishedAt: Date time the video was uploaded.
	PublishedAt string `json:"publishedAt,omitempty"`

	// Tags: Textual tags associated with the video.
	Tags []string `json:"tags,omitempty"`

	// Thumbnails: Video thumbnails.
	Thumbnails *VideoSnippetThumbnails `json:"thumbnails,omitempty"`

	// Title: Title of the video.
	Title string `json:"title,omitempty"`
}

type VideoSnippetThumbnails struct {
}

type VideoStatistics struct {
	// CommentCount: Number of comments for this video.
	CommentCount uint64 `json:"commentCount,omitempty,string"`

	// DislikeCount: Number of times the video was disliked.
	DislikeCount uint64 `json:"dislikeCount,omitempty,string"`

	// FavoriteCount: Number of times the video was added to a user's
	// favorites list.
	FavoriteCount uint64 `json:"favoriteCount,omitempty,string"`

	// LikeCount: Number of times the video was liked.
	LikeCount uint64 `json:"likeCount,omitempty,string"`

	// ViewCount: Number of times the video was viewed.
	ViewCount uint64 `json:"viewCount,omitempty,string"`
}

type VideoStatus struct {
	// FailureReason: Present only if the uploadStatus indicates a failed
	// upload.
	FailureReason string `json:"failureReason,omitempty"`

	// PrivacyStatus: Privacy of the video.
	PrivacyStatus string `json:"privacyStatus,omitempty"`

	// RejectionReason: Present only if the uploadStatus indicates a
	// rejected upload.
	RejectionReason string `json:"rejectionReason,omitempty"`

	// UploadStatus: Status of the video upload.
	UploadStatus string `json:"uploadStatus,omitempty"`
}

// method id "youtube.activities.list":

type ActivitiesListCall struct {
	s    *Service
	home string
	opt_ map[string]interface{}
}

// List: Browse the YouTube channel activity collection.
func (r *ActivitiesService) List(home string) *ActivitiesListCall {
	c := &ActivitiesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.home = home
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *ActivitiesListCall) MaxResults(maxResults int64) *ActivitiesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first element to return (starts at 0)
func (c *ActivitiesListCall) StartIndex(startIndex int64) *ActivitiesListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *ActivitiesListCall) Do() (*ActivityListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("home", fmt.Sprintf("%v", c.home))
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/youtube/v3alpha/", "activities")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(ActivityListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Browse the YouTube channel activity collection.",
	//   "httpMethod": "GET",
	//   "id": "youtube.activities.list",
	//   "parameterOrder": [
	//     "home"
	//   ],
	//   "parameters": {
	//     "home": {
	//       "description": "Flag indicating to return user's homepage feed.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first element to return (starts at 0)",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "activities",
	//   "response": {
	//     "$ref": "ActivityListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly"
	//   ]
	// }

}

// method id "youtube.channels.list":

type ChannelsListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Browse the YouTube channel collection. Either the 'id' or
// 'mine' parameter must be set.
func (r *ChannelsService) List(part string) *ChannelsListCall {
	c := &ChannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// CategoryId sets the optional parameter "categoryId": Fiter to
// retrieve the channels within the given category ID.
func (c *ChannelsListCall) CategoryId(categoryId string) *ChannelsListCall {
	c.opt_["categoryId"] = categoryId
	return c
}

// ContentOwnerId sets the optional parameter "contentOwnerId": The
// authenticated user acts on behalf of this content owner.
func (c *ChannelsListCall) ContentOwnerId(contentOwnerId string) *ChannelsListCall {
	c.opt_["contentOwnerId"] = contentOwnerId
	return c
}

// Id sets the optional parameter "id": YouTube IDs of the channels to
// be returned.
func (c *ChannelsListCall) Id(id string) *ChannelsListCall {
	c.opt_["id"] = id
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *ChannelsListCall) MaxResults(maxResults int64) *ChannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Mine sets the optional parameter "mine": Filter to only channels
// owned by authenticated user.
func (c *ChannelsListCall) Mine(mine string) *ChannelsListCall {
	c.opt_["mine"] = mine
	return c
}

// MySubscribers sets the optional parameter "mySubscribers": Filter to
// channels that subscribed to the channel of the authenticated user.
func (c *ChannelsListCall) MySubscribers(mySubscribers string) *ChannelsListCall {
	c.opt_["mySubscribers"] = mySubscribers
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first element to return (starts at 0)
func (c *ChannelsListCall) StartIndex(startIndex int64) *ChannelsListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *ChannelsListCall) Do() (*ChannelListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["categoryId"]; ok {
		params.Set("categoryId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["contentOwnerId"]; ok {
		params.Set("contentOwnerId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["id"]; ok {
		params.Set("id", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["mine"]; ok {
		params.Set("mine", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["mySubscribers"]; ok {
		params.Set("mySubscribers", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/youtube/v3alpha/", "channels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(ChannelListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Browse the YouTube channel collection. Either the 'id' or 'mine' parameter must be set.",
	//   "httpMethod": "GET",
	//   "id": "youtube.channels.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "categoryId": {
	//       "description": "Fiter to retrieve the channels within the given category ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "contentOwnerId": {
	//       "description": "The authenticated user acts on behalf of this content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "YouTube IDs of the channels to be returned.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "mine": {
	//       "description": "Filter to only channels owned by authenticated user.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "mySubscribers": {
	//       "description": "Filter to channels that subscribed to the channel of the authenticated user.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "Parts of the channel resource to be returned.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first element to return (starts at 0)",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "channels",
	//   "response": {
	//     "$ref": "ChannelListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.playlistItems.delete":

type PlaylistItemsDeleteCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Delete: Deletes playlist items by IDs.
func (r *PlaylistItemsService) Delete(id string) *PlaylistItemsDeleteCall {
	c := &PlaylistItemsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *PlaylistItemsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	urls := googleapi.ResolveRelative("https://www.googleapis.com/youtube/v3alpha/", "playlistItems")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes playlist items by IDs.",
	//   "httpMethod": "DELETE",
	//   "id": "youtube.playlistItems.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "YouTube IDs of the playlist items to be deleted.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "playlistItems",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.playlistItems.insert":

type PlaylistItemsInsertCall struct {
	s            *Service
	playlistitem *PlaylistItem
	opt_         map[string]interface{}
}

// Insert: Insert a resource into a playlist.
func (r *PlaylistItemsService) Insert(playlistitem *PlaylistItem) *PlaylistItemsInsertCall {
	c := &PlaylistItemsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.playlistitem = playlistitem
	return c
}

func (c *PlaylistItemsInsertCall) Do() (*PlaylistItem, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.playlistitem)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/youtube/v3alpha/", "playlistItems")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PlaylistItem)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Insert a resource into a playlist.",
	//   "httpMethod": "POST",
	//   "id": "youtube.playlistItems.insert",
	//   "path": "playlistItems",
	//   "request": {
	//     "$ref": "PlaylistItem"
	//   },
	//   "response": {
	//     "$ref": "PlaylistItem"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.playlistItems.list":

type PlaylistItemsListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Browse the YouTube playlist collection.
func (r *PlaylistItemsService) List(part string) *PlaylistItemsListCall {
	c := &PlaylistItemsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// ContentOwnerId sets the optional parameter "contentOwnerId": The
// authenticated user acts on behalf of this content owner.
func (c *PlaylistItemsListCall) ContentOwnerId(contentOwnerId string) *PlaylistItemsListCall {
	c.opt_["contentOwnerId"] = contentOwnerId
	return c
}

// Id sets the optional parameter "id": YouTube IDs of the playlist
// items to be returned.
func (c *PlaylistItemsListCall) Id(id string) *PlaylistItemsListCall {
	c.opt_["id"] = id
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *PlaylistItemsListCall) MaxResults(maxResults int64) *PlaylistItemsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PlaylistId sets the optional parameter "playlistId": Retrieves
// playlist items from the given playlist id.
func (c *PlaylistItemsListCall) PlaylistId(playlistId string) *PlaylistItemsListCall {
	c.opt_["playlistId"] = playlistId
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first element to return (starts at 0)
func (c *PlaylistItemsListCall) StartIndex(startIndex int64) *PlaylistItemsListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *PlaylistItemsListCall) Do() (*PlaylistItemListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["contentOwnerId"]; ok {
		params.Set("contentOwnerId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["id"]; ok {
		params.Set("id", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["playlistId"]; ok {
		params.Set("playlistId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/youtube/v3alpha/", "playlistItems")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PlaylistItemListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Browse the YouTube playlist collection.",
	//   "httpMethod": "GET",
	//   "id": "youtube.playlistItems.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "contentOwnerId": {
	//       "description": "The authenticated user acts on behalf of this content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "YouTube IDs of the playlist items to be returned.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "part": {
	//       "description": "Parts of the playlist resource to be returned.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "playlistId": {
	//       "description": "Retrieves playlist items from the given playlist id.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first element to return (starts at 0)",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "playlistItems",
	//   "response": {
	//     "$ref": "PlaylistItemListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.playlistItems.update":

type PlaylistItemsUpdateCall struct {
	s            *Service
	playlistitem *PlaylistItem
	opt_         map[string]interface{}
}

// Update: Update a playlist item.
func (r *PlaylistItemsService) Update(playlistitem *PlaylistItem) *PlaylistItemsUpdateCall {
	c := &PlaylistItemsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.playlistitem = playlistitem
	return c
}

func (c *PlaylistItemsUpdateCall) Do() (*PlaylistItem, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.playlistitem)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/youtube/v3alpha/", "playlistItems")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PlaylistItem)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update a playlist item.",
	//   "httpMethod": "PUT",
	//   "id": "youtube.playlistItems.update",
	//   "path": "playlistItems",
	//   "request": {
	//     "$ref": "PlaylistItem"
	//   },
	//   "response": {
	//     "$ref": "PlaylistItem"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.playlists.delete":

type PlaylistsDeleteCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Delete: Deletes playlists by IDs.
func (r *PlaylistsService) Delete(id string) *PlaylistsDeleteCall {
	c := &PlaylistsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *PlaylistsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	urls := googleapi.ResolveRelative("https://www.googleapis.com/youtube/v3alpha/", "playlists")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Deletes playlists by IDs.",
	//   "httpMethod": "DELETE",
	//   "id": "youtube.playlists.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "YouTube IDs of the playlists to be deleted.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "playlists",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.playlists.list":

type PlaylistsListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Browse the YouTube playlist collection.
func (r *PlaylistsService) List(part string) *PlaylistsListCall {
	c := &PlaylistsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// ContentOwnerId sets the optional parameter "contentOwnerId": The
// authenticated user acts on behalf of this content owner.
func (c *PlaylistsListCall) ContentOwnerId(contentOwnerId string) *PlaylistsListCall {
	c.opt_["contentOwnerId"] = contentOwnerId
	return c
}

// Id sets the optional parameter "id": Comma-separated YouTube IDs of
// the playlists to be returned.
func (c *PlaylistsListCall) Id(id string) *PlaylistsListCall {
	c.opt_["id"] = id
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of results to return
func (c *PlaylistsListCall) MaxResults(maxResults int64) *PlaylistsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Mine sets the optional parameter "mine": Flag indicating only return
// the playlists of the authenticated user.
func (c *PlaylistsListCall) Mine(mine string) *PlaylistsListCall {
	c.opt_["mine"] = mine
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first element to return (starts at 0)
func (c *PlaylistsListCall) StartIndex(startIndex int64) *PlaylistsListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

func (c *PlaylistsListCall) Do() (*PlaylistListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["contentOwnerId"]; ok {
		params.Set("contentOwnerId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["id"]; ok {
		params.Set("id", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["mine"]; ok {
		params.Set("mine", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/youtube/v3alpha/", "playlists")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(PlaylistListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Browse the YouTube playlist collection.",
	//   "httpMethod": "GET",
	//   "id": "youtube.playlists.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "contentOwnerId": {
	//       "description": "The authenticated user acts on behalf of this content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "Comma-separated YouTube IDs of the playlists to be returned.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "Maximum number of results to return",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "mine": {
	//       "description": "Flag indicating only return the playlists of the authenticated user.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "Parts of the playlist resource to be returned.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "description": "Index of the first element to return (starts at 0)",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "0",
	//       "type": "integer"
	//     }
	//   },
	//   "path": "playlists",
	//   "response": {
	//     "$ref": "PlaylistListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.search.list":

type SearchListCall struct {
	s    *Service
	q    string
	opt_ map[string]interface{}
}

// List: Universal search for youtube.
func (r *SearchService) List(q string) *SearchListCall {
	c := &SearchListCall{s: r.s, opt_: make(map[string]interface{})}
	c.q = q
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of search results to return per page.
func (c *SearchListCall) MaxResults(maxResults int64) *SearchListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Order sets the optional parameter "order": Sort order.
func (c *SearchListCall) Order(order string) *SearchListCall {
	c.opt_["order"] = order
	return c
}

// Published sets the optional parameter "published": Only search for
// resources uploaded at a specific pediod
func (c *SearchListCall) Published(published string) *SearchListCall {
	c.opt_["published"] = published
	return c
}

// StartIndex sets the optional parameter "startIndex": Index of the
// first element to return (starts at 0)
func (c *SearchListCall) StartIndex(startIndex int64) *SearchListCall {
	c.opt_["startIndex"] = startIndex
	return c
}

// Type sets the optional parameter "type": Type of resource to search.
func (c *SearchListCall) Type(type_ string) *SearchListCall {
	c.opt_["type"] = type_
	return c
}

// VideoCaption sets the optional parameter "videoCaption": Add a filter
// on the the presence of captions on the videos.
func (c *SearchListCall) VideoCaption(videoCaption string) *SearchListCall {
	c.opt_["videoCaption"] = videoCaption
	return c
}

// VideoDefinition sets the optional parameter "videoDefinition": Add a
// filter for the definition of the videos.
func (c *SearchListCall) VideoDefinition(videoDefinition string) *SearchListCall {
	c.opt_["videoDefinition"] = videoDefinition
	return c
}

// VideoDimension sets the optional parameter "videoDimension": Add a
// filter for the number of dimensions in the videos.
func (c *SearchListCall) VideoDimension(videoDimension string) *SearchListCall {
	c.opt_["videoDimension"] = videoDimension
	return c
}

// VideoDuration sets the optional parameter "videoDuration": Add a
// filter on the duration of the videos.
func (c *SearchListCall) VideoDuration(videoDuration string) *SearchListCall {
	c.opt_["videoDuration"] = videoDuration
	return c
}

// VideoLicense sets the optional parameter "videoLicense": Add a filter
// on the licensing of the videos.
func (c *SearchListCall) VideoLicense(videoLicense string) *SearchListCall {
	c.opt_["videoLicense"] = videoLicense
	return c
}

func (c *SearchListCall) Do() (*SearchListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("q", fmt.Sprintf("%v", c.q))
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["order"]; ok {
		params.Set("order", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["published"]; ok {
		params.Set("published", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startIndex"]; ok {
		params.Set("startIndex", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["type"]; ok {
		params.Set("type", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["videoCaption"]; ok {
		params.Set("videoCaption", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["videoDefinition"]; ok {
		params.Set("videoDefinition", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["videoDimension"]; ok {
		params.Set("videoDimension", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["videoDuration"]; ok {
		params.Set("videoDuration", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["videoLicense"]; ok {
		params.Set("videoLicense", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/youtube/v3alpha/", "search")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(SearchListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Universal search for youtube.",
	//   "httpMethod": "GET",
	//   "id": "youtube.search.list",
	//   "parameterOrder": [
	//     "q"
	//   ],
	//   "parameters": {
	//     "maxResults": {
	//       "default": "5",
	//       "description": "Maximum number of search results to return per page.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "50",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "order": {
	//       "default": "relevance",
	//       "description": "Sort order.",
	//       "enum": [
	//         "date",
	//         "rating",
	//         "relevance",
	//         "view_count"
	//       ],
	//       "enumDescriptions": [
	//         "Sort according to the date.",
	//         "Sort according to the rating.",
	//         "Sort according to the relevance.",
	//         "Sort according to the view count."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "published": {
	//       "description": "Only search for resources uploaded at a specific pediod",
	//       "enum": [
	//         "any",
	//         "thisWeek",
	//         "today"
	//       ],
	//       "enumDescriptions": [
	//         "No filter on the release date",
	//         "Videos uploaded this month",
	//         "Videos uploaded today"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "q": {
	//       "description": "Query to search in Youtube.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startIndex": {
	//       "default": "0",
	//       "description": "Index of the first element to return (starts at 0)",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "1000",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "type": {
	//       "description": "Type of resource to search.",
	//       "enum": [
	//         "channel",
	//         "playlist",
	//         "video"
	//       ],
	//       "enumDescriptions": [
	//         "Search for channels.",
	//         "Search for playlists.",
	//         "Search for videos."
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "videoCaption": {
	//       "description": "Add a filter on the the presence of captions on the videos.",
	//       "enum": [
	//         "any",
	//         "closedCaption",
	//         "none"
	//       ],
	//       "enumDescriptions": [
	//         "No filter on the captions.",
	//         "Videos with closed captions.",
	//         "Videos without captions."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoDefinition": {
	//       "description": "Add a filter for the definition of the videos.",
	//       "enum": [
	//         "any",
	//         "high",
	//         "standard"
	//       ],
	//       "enumDescriptions": [
	//         "No filter on the definition.",
	//         "Videos in high definition.",
	//         "Videos in standard definition."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoDimension": {
	//       "description": "Add a filter for the number of dimensions in the videos.",
	//       "enum": [
	//         "2d",
	//         "3d",
	//         "any"
	//       ],
	//       "enumDescriptions": [
	//         "Videos in two dimensions.",
	//         "Videos in three dimensions.",
	//         "No filter on the dimension."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoDuration": {
	//       "description": "Add a filter on the duration of the videos.",
	//       "enum": [
	//         "any",
	//         "long",
	//         "medium",
	//         "short"
	//       ],
	//       "enumDescriptions": [
	//         "No filter on the duration.",
	//         "Videos with a duration longer than 20 minutes.",
	//         "Videos with a duration between 4 and 20 minutes.",
	//         "Videos with a duration under 4 minutes."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoLicense": {
	//       "description": "Add a filter on the licensing of the videos.",
	//       "enum": [
	//         "any",
	//         "creativeCommon",
	//         "youtube"
	//       ],
	//       "enumDescriptions": [
	//         "No filter on the license.",
	//         "Videos under the Creative Common license.",
	//         "Videos under the YouTube license."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "search",
	//   "response": {
	//     "$ref": "SearchListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.videos.delete":

type VideosDeleteCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Delete: Delete a YouTube video.
func (r *VideosService) Delete(id string) *VideosDeleteCall {
	c := &VideosDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

// ContentOwnerId sets the optional parameter "contentOwnerId": The
// authenticated user acts on behalf of this content owner.
func (c *VideosDeleteCall) ContentOwnerId(contentOwnerId string) *VideosDeleteCall {
	c.opt_["contentOwnerId"] = contentOwnerId
	return c
}

func (c *VideosDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	if v, ok := c.opt_["contentOwnerId"]; ok {
		params.Set("contentOwnerId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/youtube/v3alpha/", "videos")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Delete a YouTube video.",
	//   "httpMethod": "DELETE",
	//   "id": "youtube.videos.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "contentOwnerId": {
	//       "description": "The authenticated user acts on behalf of this content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "YouTube ID of the video to be deleted.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "videos",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.videos.insert":

type VideosInsertCall struct {
	s      *Service
	video  *Video
	opt_   map[string]interface{}
	media_ io.Reader
}

// Insert: Upload a video to YouTube.
func (r *VideosService) Insert(video *Video) *VideosInsertCall {
	c := &VideosInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.video = video
	return c
}
func (c *VideosInsertCall) Media(r io.Reader) *VideosInsertCall {
	c.media_ = r
	return c
}

func (c *VideosInsertCall) Do() (*Video, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.video)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative("https://www.googleapis.com/youtube/v3alpha/", "videos")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	contentLength_, hasMedia_ := googleapi.ConditionallyIncludeMedia(c.media_, &body, &ctype)
	req, _ := http.NewRequest("POST", urls, body)
	if hasMedia_ {
		req.ContentLength = contentLength_
	}
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Video)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Upload a video to YouTube.",
	//   "httpMethod": "POST",
	//   "id": "youtube.videos.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "application/octet-stream",
	//       "video/*"
	//     ],
	//     "maxSize": "64GB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/youtube/v3alpha/videos"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/youtube/v3alpha/videos"
	//       }
	//     }
	//   },
	//   "path": "videos",
	//   "request": {
	//     "$ref": "Video"
	//   },
	//   "response": {
	//     "$ref": "Video"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.upload"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "youtube.videos.list":

type VideosListCall struct {
	s    *Service
	id   string
	part string
	opt_ map[string]interface{}
}

// List: Browse the YouTube video collection.
func (r *VideosService) List(id string, part string) *VideosListCall {
	c := &VideosListCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.part = part
	return c
}

// ContentOwnerId sets the optional parameter "contentOwnerId": The
// authenticated user acts on behalf of this content owner.
func (c *VideosListCall) ContentOwnerId(contentOwnerId string) *VideosListCall {
	c.opt_["contentOwnerId"] = contentOwnerId
	return c
}

func (c *VideosListCall) Do() (*VideoListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["contentOwnerId"]; ok {
		params.Set("contentOwnerId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative("https://www.googleapis.com/youtube/v3alpha/", "videos")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header.Set("User-Agent", "google-api-go-client/0.5")
	res, err := c.s.client.Do(req)
	if err != nil {
		return nil, err
	}
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(VideoListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Browse the YouTube video collection.",
	//   "httpMethod": "GET",
	//   "id": "youtube.videos.list",
	//   "parameterOrder": [
	//     "id",
	//     "part"
	//   ],
	//   "parameters": {
	//     "contentOwnerId": {
	//       "description": "The authenticated user acts on behalf of this content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "YouTube IDs of the videos to be returned.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "Parts of the video resource to be returned.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "videos",
	//   "response": {
	//     "$ref": "VideoListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

func cleanPathString(s string) string {
	return strings.Map(func(r rune) rune {
		if r >= 0x2d && r <= 0x7a {
			return r
		}
		return -1
	}, s)
}
