// Package youtube provides access to the YouTube Data API.
//
// See https://developers.google.com/youtube/v3
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/youtube/v3"
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

const apiId = "youtube:v3"
const apiName = "youtube"
const apiVersion = "v3"
const basePath = "https://www.googleapis.com/youtube/v3/"

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

	// View private information of your YouTube channel relevant during the
	// audit process with a YouTube partner
	YoutubepartnerChannelAuditScope = "https://www.googleapis.com/auth/youtubepartner-channel-audit"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Activities = NewActivitiesService(s)
	s.ChannelBanners = NewChannelBannersService(s)
	s.ChannelSections = NewChannelSectionsService(s)
	s.Channels = NewChannelsService(s)
	s.GuideCategories = NewGuideCategoriesService(s)
	s.I18nLanguages = NewI18nLanguagesService(s)
	s.I18nRegions = NewI18nRegionsService(s)
	s.LiveBroadcasts = NewLiveBroadcastsService(s)
	s.LiveStreams = NewLiveStreamsService(s)
	s.PlaylistItems = NewPlaylistItemsService(s)
	s.Playlists = NewPlaylistsService(s)
	s.Search = NewSearchService(s)
	s.Subscriptions = NewSubscriptionsService(s)
	s.Thumbnails = NewThumbnailsService(s)
	s.VideoCategories = NewVideoCategoriesService(s)
	s.Videos = NewVideosService(s)
	s.Watermarks = NewWatermarksService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Activities *ActivitiesService

	ChannelBanners *ChannelBannersService

	ChannelSections *ChannelSectionsService

	Channels *ChannelsService

	GuideCategories *GuideCategoriesService

	I18nLanguages *I18nLanguagesService

	I18nRegions *I18nRegionsService

	LiveBroadcasts *LiveBroadcastsService

	LiveStreams *LiveStreamsService

	PlaylistItems *PlaylistItemsService

	Playlists *PlaylistsService

	Search *SearchService

	Subscriptions *SubscriptionsService

	Thumbnails *ThumbnailsService

	VideoCategories *VideoCategoriesService

	Videos *VideosService

	Watermarks *WatermarksService
}

func NewActivitiesService(s *Service) *ActivitiesService {
	rs := &ActivitiesService{s: s}
	return rs
}

type ActivitiesService struct {
	s *Service
}

func NewChannelBannersService(s *Service) *ChannelBannersService {
	rs := &ChannelBannersService{s: s}
	return rs
}

type ChannelBannersService struct {
	s *Service
}

func NewChannelSectionsService(s *Service) *ChannelSectionsService {
	rs := &ChannelSectionsService{s: s}
	return rs
}

type ChannelSectionsService struct {
	s *Service
}

func NewChannelsService(s *Service) *ChannelsService {
	rs := &ChannelsService{s: s}
	return rs
}

type ChannelsService struct {
	s *Service
}

func NewGuideCategoriesService(s *Service) *GuideCategoriesService {
	rs := &GuideCategoriesService{s: s}
	return rs
}

type GuideCategoriesService struct {
	s *Service
}

func NewI18nLanguagesService(s *Service) *I18nLanguagesService {
	rs := &I18nLanguagesService{s: s}
	return rs
}

type I18nLanguagesService struct {
	s *Service
}

func NewI18nRegionsService(s *Service) *I18nRegionsService {
	rs := &I18nRegionsService{s: s}
	return rs
}

type I18nRegionsService struct {
	s *Service
}

func NewLiveBroadcastsService(s *Service) *LiveBroadcastsService {
	rs := &LiveBroadcastsService{s: s}
	return rs
}

type LiveBroadcastsService struct {
	s *Service
}

func NewLiveStreamsService(s *Service) *LiveStreamsService {
	rs := &LiveStreamsService{s: s}
	return rs
}

type LiveStreamsService struct {
	s *Service
}

func NewPlaylistItemsService(s *Service) *PlaylistItemsService {
	rs := &PlaylistItemsService{s: s}
	return rs
}

type PlaylistItemsService struct {
	s *Service
}

func NewPlaylistsService(s *Service) *PlaylistsService {
	rs := &PlaylistsService{s: s}
	return rs
}

type PlaylistsService struct {
	s *Service
}

func NewSearchService(s *Service) *SearchService {
	rs := &SearchService{s: s}
	return rs
}

type SearchService struct {
	s *Service
}

func NewSubscriptionsService(s *Service) *SubscriptionsService {
	rs := &SubscriptionsService{s: s}
	return rs
}

type SubscriptionsService struct {
	s *Service
}

func NewThumbnailsService(s *Service) *ThumbnailsService {
	rs := &ThumbnailsService{s: s}
	return rs
}

type ThumbnailsService struct {
	s *Service
}

func NewVideoCategoriesService(s *Service) *VideoCategoriesService {
	rs := &VideoCategoriesService{s: s}
	return rs
}

type VideoCategoriesService struct {
	s *Service
}

func NewVideosService(s *Service) *VideosService {
	rs := &VideosService{s: s}
	return rs
}

type VideosService struct {
	s *Service
}

func NewWatermarksService(s *Service) *WatermarksService {
	rs := &WatermarksService{s: s}
	return rs
}

type WatermarksService struct {
	s *Service
}

type AccessPolicy struct {
	// Allowed: The value of allowed indicates whether the access to the
	// policy is allowed or denied by default.
	Allowed bool `json:"allowed,omitempty"`

	// Exception: A list of region codes that identify countries where the
	// default policy do not apply.
	Exception []string `json:"exception,omitempty"`
}

type Activity struct {
	// ContentDetails: The contentDetails object contains information about
	// the content associated with the activity. For example, if the
	// snippet.type value is videoRated, then the contentDetails object's
	// content identifies the rated video.
	ContentDetails *ActivityContentDetails `json:"contentDetails,omitempty"`

	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The ID that YouTube uses to uniquely identify the activity.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#activity".
	Kind string `json:"kind,omitempty"`

	// Snippet: The snippet object contains basic details about the
	// activity, including the activity's type and group ID.
	Snippet *ActivitySnippet `json:"snippet,omitempty"`
}

type ActivityContentDetails struct {
	// Bulletin: The bulletin object contains details about a channel
	// bulletin post. This object is only present if the snippet.type is
	// bulletin.
	Bulletin *ActivityContentDetailsBulletin `json:"bulletin,omitempty"`

	// ChannelItem: The channelItem object contains details about a resource
	// which was added to a channel. This property is only present if the
	// snippet.type is channelItem.
	ChannelItem *ActivityContentDetailsChannelItem `json:"channelItem,omitempty"`

	// Comment: The comment object contains information about a resource
	// that received a comment. This property is only present if the
	// snippet.type is comment.
	Comment *ActivityContentDetailsComment `json:"comment,omitempty"`

	// Favorite: The favorite object contains information about a video that
	// was marked as a favorite video. This property is only present if the
	// snippet.type is favorite.
	Favorite *ActivityContentDetailsFavorite `json:"favorite,omitempty"`

	// Like: The like object contains information about a resource that
	// received a positive (like) rating. This property is only present if
	// the snippet.type is like.
	Like *ActivityContentDetailsLike `json:"like,omitempty"`

	// PlaylistItem: The playlistItem object contains information about a
	// new playlist item. This property is only present if the snippet.type
	// is playlistItem.
	PlaylistItem *ActivityContentDetailsPlaylistItem `json:"playlistItem,omitempty"`

	// PromotedItem: The promotedItem object contains details about a
	// resource which is being promoted. This property is only present if
	// the snippet.type is promotedItem.
	PromotedItem *ActivityContentDetailsPromotedItem `json:"promotedItem,omitempty"`

	// Recommendation: The recommendation object contains information about
	// a recommended resource. This property is only present if the
	// snippet.type is recommendation.
	Recommendation *ActivityContentDetailsRecommendation `json:"recommendation,omitempty"`

	// Social: The social object contains details about a social network
	// post. This property is only present if the snippet.type is social.
	Social *ActivityContentDetailsSocial `json:"social,omitempty"`

	// Subscription: The subscription object contains information about a
	// channel that a user subscribed to. This property is only present if
	// the snippet.type is subscription.
	Subscription *ActivityContentDetailsSubscription `json:"subscription,omitempty"`

	// Upload: The upload object contains information about the uploaded
	// video. This property is only present if the snippet.type is upload.
	Upload *ActivityContentDetailsUpload `json:"upload,omitempty"`
}

type ActivityContentDetailsBulletin struct {
	// ResourceId: The resourceId object contains information that
	// identifies the resource associated with a bulletin post.
	ResourceId *ResourceId `json:"resourceId,omitempty"`
}

type ActivityContentDetailsChannelItem struct {
	// ResourceId: The resourceId object contains information that
	// identifies the resource that was added to the channel.
	ResourceId *ResourceId `json:"resourceId,omitempty"`
}

type ActivityContentDetailsComment struct {
	// ResourceId: The resourceId object contains information that
	// identifies the resource associated with the comment.
	ResourceId *ResourceId `json:"resourceId,omitempty"`
}

type ActivityContentDetailsFavorite struct {
	// ResourceId: The resourceId object contains information that
	// identifies the resource that was marked as a favorite.
	ResourceId *ResourceId `json:"resourceId,omitempty"`
}

type ActivityContentDetailsLike struct {
	// ResourceId: The resourceId object contains information that
	// identifies the rated resource.
	ResourceId *ResourceId `json:"resourceId,omitempty"`
}

type ActivityContentDetailsPlaylistItem struct {
	// PlaylistId: The value that YouTube uses to uniquely identify the
	// playlist.
	PlaylistId string `json:"playlistId,omitempty"`

	// PlaylistItemId: ID of the item within the playlist.
	PlaylistItemId string `json:"playlistItemId,omitempty"`

	// ResourceId: The resourceId object contains information about the
	// resource that was added to the playlist.
	ResourceId *ResourceId `json:"resourceId,omitempty"`
}

type ActivityContentDetailsPromotedItem struct {
	// AdTag: The URL the client should fetch to request a promoted item.
	AdTag string `json:"adTag,omitempty"`

	// ClickTrackingUrl: The URL the client should ping to indicate that the
	// user clicked through on this promoted item.
	ClickTrackingUrl string `json:"clickTrackingUrl,omitempty"`

	// CreativeViewUrl: The URL the client should ping to indicate that the
	// user was shown this promoted item.
	CreativeViewUrl string `json:"creativeViewUrl,omitempty"`

	// CtaType: The type of call-to-action, a message to the user indicating
	// action that can be taken.
	CtaType string `json:"ctaType,omitempty"`

	// CustomCtaButtonText: The custom call-to-action button text. If
	// specified, it will override the default button text for the cta_type.
	CustomCtaButtonText string `json:"customCtaButtonText,omitempty"`

	// DescriptionText: The text description to accompany the promoted item.
	DescriptionText string `json:"descriptionText,omitempty"`

	// DestinationUrl: The URL the client should direct the user to, if the
	// user chooses to visit the advertiser's website.
	DestinationUrl string `json:"destinationUrl,omitempty"`

	// ForecastingUrl: The list of forecasting URLs. The client should ping
	// all of these URLs when a promoted item is not available, to indicate
	// that a promoted item could have been shown.
	ForecastingUrl []string `json:"forecastingUrl,omitempty"`

	// ImpressionUrl: The list of impression URLs. The client should ping
	// all of these URLs to indicate that the user was shown this promoted
	// item.
	ImpressionUrl []string `json:"impressionUrl,omitempty"`

	// VideoId: The ID that YouTube uses to uniquely identify the promoted
	// video.
	VideoId string `json:"videoId,omitempty"`
}

type ActivityContentDetailsRecommendation struct {
	// Reason: The reason that the resource is recommended to the user.
	Reason string `json:"reason,omitempty"`

	// ResourceId: The resourceId object contains information that
	// identifies the recommended resource.
	ResourceId *ResourceId `json:"resourceId,omitempty"`

	// SeedResourceId: The seedResourceId object contains information about
	// the resource that caused the recommendation.
	SeedResourceId *ResourceId `json:"seedResourceId,omitempty"`
}

type ActivityContentDetailsSocial struct {
	// Author: The author of the social network post.
	Author string `json:"author,omitempty"`

	// ImageUrl: An image of the post's author.
	ImageUrl string `json:"imageUrl,omitempty"`

	// ReferenceUrl: The URL of the social network post.
	ReferenceUrl string `json:"referenceUrl,omitempty"`

	// ResourceId: The resourceId object encapsulates information that
	// identifies the resource associated with a social network post.
	ResourceId *ResourceId `json:"resourceId,omitempty"`

	// Type: The name of the social network.
	Type string `json:"type,omitempty"`
}

type ActivityContentDetailsSubscription struct {
	// ResourceId: The resourceId object contains information that
	// identifies the resource that the user subscribed to.
	ResourceId *ResourceId `json:"resourceId,omitempty"`
}

type ActivityContentDetailsUpload struct {
	// VideoId: The ID that YouTube uses to uniquely identify the uploaded
	// video.
	VideoId string `json:"videoId,omitempty"`
}

type ActivityListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of activities, or events, that match the request
	// criteria.
	Items []*Activity `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#activityListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the next page in the result set.
	NextPageToken string `json:"nextPageToken,omitempty"`

	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// PrevPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the previous page in the result set.
	PrevPageToken string `json:"prevPageToken,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type ActivitySnippet struct {
	// ChannelId: The ID that YouTube uses to uniquely identify the channel
	// associated with the activity.
	ChannelId string `json:"channelId,omitempty"`

	// ChannelTitle: Channel title for the channel responsible for this
	// activity
	ChannelTitle string `json:"channelTitle,omitempty"`

	// Description: The description of the resource primarily associated
	// with the activity.
	Description string `json:"description,omitempty"`

	// GroupId: The group ID associated with the activity. A group ID
	// identifies user events that are associated with the same user and
	// resource. For example, if a user rates a video and marks the same
	// video as a favorite, the entries for those events would have the same
	// group ID in the user's activity feed. In your user interface, you can
	// avoid repetition by grouping events with the same groupId value.
	GroupId string `json:"groupId,omitempty"`

	// PublishedAt: The date and time that the video was uploaded. The value
	// is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format.
	PublishedAt string `json:"publishedAt,omitempty"`

	// Thumbnails: A map of thumbnail images associated with the resource
	// that is primarily associated with the activity. For each object in
	// the map, the key is the name of the thumbnail image, and the value is
	// an object that contains other information about the thumbnail.
	Thumbnails *ThumbnailDetails `json:"thumbnails,omitempty"`

	// Title: The title of the resource primarily associated with the
	// activity.
	Title string `json:"title,omitempty"`

	// Type: The type of activity that the resource describes.
	Type string `json:"type,omitempty"`
}

type CdnSettings struct {
	// Format: The format of the video stream that you are sending to
	// Youtube.
	Format string `json:"format,omitempty"`

	// IngestionInfo: The ingestionInfo object contains information that
	// YouTube provides that you need to transmit your RTMP or HTTP stream
	// to YouTube.
	IngestionInfo *IngestionInfo `json:"ingestionInfo,omitempty"`

	// IngestionType: The method or protocol used to transmit the video
	// stream.
	IngestionType string `json:"ingestionType,omitempty"`
}

type Channel struct {
	// AuditDetails: The auditionDetails object encapsulates channel data
	// that is relevant for YouTube Partners during the audition process.
	AuditDetails *ChannelAuditDetails `json:"auditDetails,omitempty"`

	// BrandingSettings: The brandingSettings object encapsulates
	// information about the branding of the channel.
	BrandingSettings *ChannelBrandingSettings `json:"brandingSettings,omitempty"`

	// ContentDetails: The contentDetails object encapsulates information
	// about the channel's content.
	ContentDetails *ChannelContentDetails `json:"contentDetails,omitempty"`

	// ContentOwnerDetails: The contentOwnerDetails object encapsulates
	// channel data that is relevant for YouTube Partners linked with the
	// channel.
	ContentOwnerDetails *ChannelContentOwnerDetails `json:"contentOwnerDetails,omitempty"`

	// ConversionPings: The conversionPings object encapsulates information
	// about conversion pings that need to be respected by the channel.
	ConversionPings *ChannelConversionPings `json:"conversionPings,omitempty"`

	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The ID that YouTube uses to uniquely identify the channel.
	Id string `json:"id,omitempty"`

	// InvideoPromotion: The invideoPromotion object encapsulates
	// information about promotion campaign associated with the channel.
	InvideoPromotion *InvideoPromotion `json:"invideoPromotion,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#channel".
	Kind string `json:"kind,omitempty"`

	// Snippet: The snippet object contains basic details about the channel,
	// such as its title, description, and thumbnail images.
	Snippet *ChannelSnippet `json:"snippet,omitempty"`

	// Statistics: The statistics object encapsulates statistics for the
	// channel.
	Statistics *ChannelStatistics `json:"statistics,omitempty"`

	// Status: The status object encapsulates information about the privacy
	// status of the channel.
	Status *ChannelStatus `json:"status,omitempty"`

	// TopicDetails: The topicDetails object encapsulates information about
	// Freebase topics associated with the channel.
	TopicDetails *ChannelTopicDetails `json:"topicDetails,omitempty"`
}

type ChannelAuditDetails struct {
	// CommunityGuidelinesGoodStanding: Whether or not the channel respects
	// the community guidelines.
	CommunityGuidelinesGoodStanding bool `json:"communityGuidelinesGoodStanding,omitempty"`

	// ContentIdClaimsGoodStanding: Whether or not the channel has any
	// unresolved claims.
	ContentIdClaimsGoodStanding bool `json:"contentIdClaimsGoodStanding,omitempty"`

	// CopyrightStrikesGoodStanding: Whether or not the channel has any
	// copyright strikes.
	CopyrightStrikesGoodStanding bool `json:"copyrightStrikesGoodStanding,omitempty"`

	// OverallGoodStanding: Describes the general state of the channel. This
	// field will always show if there are any issues whatsoever with the
	// channel. Currently this field represents the result of the logical
	// and operation over the community guidelines good standing, the
	// copyright strikes good standing and the content ID claims good
	// standing, but this may change in the future.
	OverallGoodStanding bool `json:"overallGoodStanding,omitempty"`
}

type ChannelBannerResource struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#channelBannerResource".
	Kind string `json:"kind,omitempty"`

	// Url: The URL of this banner image.
	Url string `json:"url,omitempty"`
}

type ChannelBrandingSettings struct {
	// Channel: Branding properties for the channel view.
	Channel *ChannelSettings `json:"channel,omitempty"`

	// Hints: Additional experimental branding properties.
	Hints []*PropertyValue `json:"hints,omitempty"`

	// Image: Branding properties for branding images.
	Image *ImageSettings `json:"image,omitempty"`

	// Watch: Branding properties for the watch page.
	Watch *WatchSettings `json:"watch,omitempty"`
}

type ChannelContentDetails struct {
	// GooglePlusUserId: The googlePlusUserId object identifies the Google+
	// profile ID associated with this channel.
	GooglePlusUserId string `json:"googlePlusUserId,omitempty"`

	RelatedPlaylists *ChannelContentDetailsRelatedPlaylists `json:"relatedPlaylists,omitempty"`
}

type ChannelContentDetailsRelatedPlaylists struct {
	// Favorites: The ID of the playlist that contains the channel"s
	// favorite videos. Use the  playlistItems.insert and
	// playlistItems.delete to add or remove items from that list.
	Favorites string `json:"favorites,omitempty"`

	// Likes: The ID of the playlist that contains the channel"s liked
	// videos. Use the   playlistItems.insert and  playlistItems.delete to
	// add or remove items from that list.
	Likes string `json:"likes,omitempty"`

	// Uploads: The ID of the playlist that contains the channel"s uploaded
	// videos. Use the  videos.insert method to upload new videos and the
	// videos.delete method to delete previously uploaded videos.
	Uploads string `json:"uploads,omitempty"`

	// WatchHistory: The ID of the playlist that contains the channel"s
	// watch history. Use the  playlistItems.insert and
	// playlistItems.delete to add or remove items from that list.
	WatchHistory string `json:"watchHistory,omitempty"`

	// WatchLater: The ID of the playlist that contains the channel"s watch
	// later playlist. Use the playlistItems.insert and
	// playlistItems.delete to add or remove items from that list.
	WatchLater string `json:"watchLater,omitempty"`
}

type ChannelContentOwnerDetails struct {
	// ContentOwner: The ID of the content owner linked to the channel.
	ContentOwner string `json:"contentOwner,omitempty"`

	// TimeLinked: The date and time of when the channel was linked to the
	// content owner. The value is specified in ISO 8601
	// (YYYY-MM-DDThh:mm:ss.sZ) format.
	TimeLinked string `json:"timeLinked,omitempty"`
}

type ChannelConversionPing struct {
	// Context: Defines the context of the ping.
	Context string `json:"context,omitempty"`

	// ConversionUrl: The url (without the schema) that the player shall
	// send the ping to. It's at caller's descretion to decide which schema
	// to use (http vs https) Example of a returned url:
	// //googleads.g.doubleclick.net/pagead/
	// viewthroughconversion/962985656/?data=path%3DtHe_path%3Btype%3D
	// cview%3Butuid%3DGISQtTNGYqaYl4sKxoVvKA&labe=default The caller must
	// append biscotti authentication (ms param in case of mobile, for
	// example) to this ping.
	ConversionUrl string `json:"conversionUrl,omitempty"`
}

type ChannelConversionPings struct {
	// Pings: Pings that the app shall fire (authenticated by biscotti
	// cookie). Each ping has a context, in which the app must fire the
	// ping, and a url identifying the ping.
	Pings []*ChannelConversionPing `json:"pings,omitempty"`
}

type ChannelListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of channels that match the request criteria.
	Items []*Channel `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#channelListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the next page in the result set.
	NextPageToken string `json:"nextPageToken,omitempty"`

	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// PrevPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the previous page in the result set.
	PrevPageToken string `json:"prevPageToken,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type ChannelSection struct {
	// ContentDetails: The contentDetails object contains details about the
	// ChannelSection content, such as playlists and channels.
	ContentDetails *ChannelSectionContentDetails `json:"contentDetails,omitempty"`

	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The ID that YouTube uses to uniquely identify the ChannelSection.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#channelSection".
	Kind string `json:"kind,omitempty"`

	// Snippet: The snippet object contains basic details about the
	// ChannelSection, such as its type, style and title.
	Snippet *ChannelSectionSnippet `json:"snippet,omitempty"`
}

type ChannelSectionContentDetails struct {
	// Channels: The channel ids for type multiple_channels.
	Channels []string `json:"channels,omitempty"`

	// Playlists: The playlist ids for type single_playlist and
	// multiple_playlists. For singlePlaylist, only one playlistId is
	// allowed.
	Playlists []string `json:"playlists,omitempty"`
}

type ChannelSectionListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of ChannelSections that match the request criteria.
	Items []*ChannelSection `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#channelSectionListResponse".
	Kind string `json:"kind,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type ChannelSectionSnippet struct {
	// ChannelId: The ID that YouTube uses to uniquely identify the channel
	// that published the channelSection.
	ChannelId string `json:"channelId,omitempty"`

	// Position: The position of the channelSection in the channel.
	Position int64 `json:"position,omitempty"`

	// Style: The style of the channelSection.
	Style string `json:"style,omitempty"`

	// Title: The channelSection's title for multiple_playlists and
	// multiple_channels.
	Title string `json:"title,omitempty"`

	// Type: The type of the channelSection.
	Type string `json:"type,omitempty"`
}

type ChannelSettings struct {
	// DefaultTab: Which content tab users should see when viewing the
	// channel.
	DefaultTab string `json:"defaultTab,omitempty"`

	// Description: Specifies the channel description.
	Description string `json:"description,omitempty"`

	// FeaturedChannelsTitle: Title for the featured channels tab.
	FeaturedChannelsTitle string `json:"featuredChannelsTitle,omitempty"`

	// FeaturedChannelsUrls: The list of featured channels.
	FeaturedChannelsUrls []string `json:"featuredChannelsUrls,omitempty"`

	// Keywords: Lists keywords associated with the channel,
	// comma-separated.
	Keywords string `json:"keywords,omitempty"`

	// ModerateComments: Whether user-submitted comments left on the channel
	// page need to be approved by the channel owner to be publicly visible.
	ModerateComments bool `json:"moderateComments,omitempty"`

	// ProfileColor: A prominent color that can be rendered on this channel
	// page.
	ProfileColor string `json:"profileColor,omitempty"`

	// ShowBrowseView: Whether the tab to browse the videos should be
	// displayed.
	ShowBrowseView bool `json:"showBrowseView,omitempty"`

	// ShowRelatedChannels: Whether related channels should be proposed.
	ShowRelatedChannels bool `json:"showRelatedChannels,omitempty"`

	// Title: Specifies the channel title.
	Title string `json:"title,omitempty"`

	// TrackingAnalyticsAccountId: The ID for a Google Analytics account to
	// track and measure traffic to the channels.
	TrackingAnalyticsAccountId string `json:"trackingAnalyticsAccountId,omitempty"`

	// UnsubscribedTrailer: The trailer of the channel, for users that are
	// not subscribers.
	UnsubscribedTrailer string `json:"unsubscribedTrailer,omitempty"`
}

type ChannelSnippet struct {
	// Description: The description of the channel.
	Description string `json:"description,omitempty"`

	// PublishedAt: The date and time that the channel was created. The
	// value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format.
	PublishedAt string `json:"publishedAt,omitempty"`

	// Thumbnails: A map of thumbnail images associated with the channel.
	// For each object in the map, the key is the name of the thumbnail
	// image, and the value is an object that contains other information
	// about the thumbnail.
	Thumbnails *ThumbnailDetails `json:"thumbnails,omitempty"`

	// Title: The channel's title.
	Title string `json:"title,omitempty"`
}

type ChannelStatistics struct {
	// CommentCount: The number of comments for the channel.
	CommentCount uint64 `json:"commentCount,omitempty,string"`

	// HiddenSubscriberCount: Whether or not the number of subscribers is
	// shown for this user.
	HiddenSubscriberCount bool `json:"hiddenSubscriberCount,omitempty"`

	// SubscriberCount: The number of subscribers that the channel has.
	SubscriberCount uint64 `json:"subscriberCount,omitempty,string"`

	// VideoCount: The number of videos uploaded to the channel.
	VideoCount uint64 `json:"videoCount,omitempty,string"`

	// ViewCount: The number of times the channel has been viewed.
	ViewCount uint64 `json:"viewCount,omitempty,string"`
}

type ChannelStatus struct {
	// IsLinked: If true, then the user is linked to either a YouTube
	// username or G+ account. Otherwise, the user doesn't have a public
	// YouTube identity.
	IsLinked bool `json:"isLinked,omitempty"`

	// PrivacyStatus: Privacy status of the channel.
	PrivacyStatus string `json:"privacyStatus,omitempty"`
}

type ChannelTopicDetails struct {
	// TopicIds: A list of Freebase topic IDs associated with the channel.
	// You can retrieve information about each topic using the Freebase
	// Topic API.
	TopicIds []string `json:"topicIds,omitempty"`
}

type ContentRating struct {
	// AcbRating: Rating system in Australia - Australian Classification
	// Board
	AcbRating string `json:"acbRating,omitempty"`

	// BbfcRating: British Board of Film Classification
	BbfcRating string `json:"bbfcRating,omitempty"`

	// CatvRating: Rating system for Canadian TV - Canadian TV
	// Classification System
	CatvRating string `json:"catvRating,omitempty"`

	// CatvfrRating: Rating system for French Canadian TV - Regie du cinema
	CatvfrRating string `json:"catvfrRating,omitempty"`

	// CbfcRating: Rating system in India - Central Board of Film
	// Certification
	CbfcRating string `json:"cbfcRating,omitempty"`

	// ChvrsRating: Canadian Home Video Rating System
	ChvrsRating string `json:"chvrsRating,omitempty"`

	// DjctqRating: Rating system in Brazil - Department of Justice, Rating,
	// Titles and Qualification
	DjctqRating string `json:"djctqRating,omitempty"`

	// EirinRating: Rating system in Japan - Eiga Rinri Kanri Iinkai
	EirinRating string `json:"eirinRating,omitempty"`

	// FmocRating: Rating system in France - French Minister of Culture
	FmocRating string `json:"fmocRating,omitempty"`

	// FskRating: Rating system in Germany - Voluntary Self Regulation of
	// the Movie Industry
	FskRating string `json:"fskRating,omitempty"`

	// IcaaRating: Rating system in Spain - Instituto de Cinematografia y de
	// las Artes Audiovisuales
	IcaaRating string `json:"icaaRating,omitempty"`

	// KmrbRating: Rating system in South Korea - Korea Media Rating Board
	KmrbRating string `json:"kmrbRating,omitempty"`

	// MibacRating: Rating system in Italy - Ministero dei Beni e delle
	// Attivita Culturali e del Turismo
	MibacRating string `json:"mibacRating,omitempty"`

	// MpaaRating: Motion Picture Association of America rating for the
	// content.
	MpaaRating string `json:"mpaaRating,omitempty"`

	// OflcRating: Rating system in New Zealand - Office of Film and
	// Literature Classification
	OflcRating string `json:"oflcRating,omitempty"`

	// RtcRating: Rating system in Mexico - General Directorate of Radio,
	// Television and Cinematography
	RtcRating string `json:"rtcRating,omitempty"`

	// RussiaRating: Rating system in Russia
	RussiaRating string `json:"russiaRating,omitempty"`

	// TvpgRating: TV Parental Guidelines rating of the content.
	TvpgRating string `json:"tvpgRating,omitempty"`

	// YtRating: Internal YouTube rating.
	YtRating string `json:"ytRating,omitempty"`
}

type GeoPoint struct {
	// Altitude: Altitude above the reference ellipsoid, in meters.
	Altitude float64 `json:"altitude,omitempty"`

	// Latitude: Latitude in degrees.
	Latitude float64 `json:"latitude,omitempty"`

	// Longitude: Longitude in degrees.
	Longitude float64 `json:"longitude,omitempty"`
}

type GuideCategory struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The ID that YouTube uses to uniquely identify the guide category.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#guideCategory".
	Kind string `json:"kind,omitempty"`

	// Snippet: The snippet object contains basic details about the
	// category, such as its title.
	Snippet *GuideCategorySnippet `json:"snippet,omitempty"`
}

type GuideCategoryListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of categories that can be associated with YouTube
	// channels. In this map, the category ID is the map key, and its value
	// is the corresponding guideCategory resource.
	Items []*GuideCategory `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#guideCategoryListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the next page in the result set.
	NextPageToken string `json:"nextPageToken,omitempty"`

	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// PrevPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the previous page in the result set.
	PrevPageToken string `json:"prevPageToken,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type GuideCategorySnippet struct {
	ChannelId string `json:"channelId,omitempty"`

	// Title: Description of the guide category.
	Title string `json:"title,omitempty"`
}

type I18nLanguage struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The ID that YouTube uses to uniquely identify the i18n language.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#i18nLanguage".
	Kind string `json:"kind,omitempty"`

	// Snippet: The snippet object contains basic details about the i18n
	// language, such as language code and human-readable name.
	Snippet *I18nLanguageSnippet `json:"snippet,omitempty"`
}

type I18nLanguageListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of supported i18n languages. In this map, the i18n
	// language ID is the map key, and its value is the corresponding
	// i18nLanguage resource.
	Items []*I18nLanguage `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#i18nLanguageListResponse".
	Kind string `json:"kind,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type I18nLanguageSnippet struct {
	// Hl: A short BCP-47 code that uniquely identifies a language.
	Hl string `json:"hl,omitempty"`

	// Name: The human-readable name of the language in the language itself.
	Name string `json:"name,omitempty"`
}

type I18nRegion struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The ID that YouTube uses to uniquely identify the i18n region.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#i18nRegion".
	Kind string `json:"kind,omitempty"`

	// Snippet: The snippet object contains basic details about the i18n
	// region, such as region code and human-readable name.
	Snippet *I18nRegionSnippet `json:"snippet,omitempty"`
}

type I18nRegionListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of regions where YouTube is available. In this map, the
	// i18n region ID is the map key, and its value is the corresponding
	// i18nRegion resource.
	Items []*I18nRegion `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#i18nRegionListResponse".
	Kind string `json:"kind,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type I18nRegionSnippet struct {
	// Gl: The region code as a 2-letter ISO country code.
	Gl string `json:"gl,omitempty"`

	// Name: The human-readable name of the region.
	Name string `json:"name,omitempty"`
}

type ImageSettings struct {
	// BackgroundImageUrl: The URL for the background image shown on the
	// video watch page. The image should be 1200px by 615px, with a maximum
	// file size of 128k.
	BackgroundImageUrl *LocalizedProperty `json:"backgroundImageUrl,omitempty"`

	// BannerExternalUrl: This is used only in update requests; if it's set,
	// we use this URL to generate all of the above banner URLs.
	BannerExternalUrl string `json:"bannerExternalUrl,omitempty"`

	// BannerImageUrl: Banner image. Desktop size (1060x175).
	BannerImageUrl string `json:"bannerImageUrl,omitempty"`

	// BannerMobileExtraHdImageUrl: Banner image. Mobile size high
	// resolution (1440x395).
	BannerMobileExtraHdImageUrl string `json:"bannerMobileExtraHdImageUrl,omitempty"`

	// BannerMobileHdImageUrl: Banner image. Mobile size high resolution
	// (1280x360).
	BannerMobileHdImageUrl string `json:"bannerMobileHdImageUrl,omitempty"`

	// BannerMobileImageUrl: Banner image. Mobile size (640x175).
	BannerMobileImageUrl string `json:"bannerMobileImageUrl,omitempty"`

	// BannerMobileLowImageUrl: Banner image. Mobile size low resolution
	// (320x88).
	BannerMobileLowImageUrl string `json:"bannerMobileLowImageUrl,omitempty"`

	// BannerMobileMediumHdImageUrl: Banner image. Mobile size medium/high
	// resolution (960x263).
	BannerMobileMediumHdImageUrl string `json:"bannerMobileMediumHdImageUrl,omitempty"`

	// BannerTabletExtraHdImageUrl: Banner image. Tablet size extra high
	// resolution (2560x424).
	BannerTabletExtraHdImageUrl string `json:"bannerTabletExtraHdImageUrl,omitempty"`

	// BannerTabletHdImageUrl: Banner image. Tablet size high resolution
	// (2276x377).
	BannerTabletHdImageUrl string `json:"bannerTabletHdImageUrl,omitempty"`

	// BannerTabletImageUrl: Banner image. Tablet size (1707x283).
	BannerTabletImageUrl string `json:"bannerTabletImageUrl,omitempty"`

	// BannerTabletLowImageUrl: Banner image. Tablet size low resolution
	// (1138x188).
	BannerTabletLowImageUrl string `json:"bannerTabletLowImageUrl,omitempty"`

	// BannerTvHighImageUrl: Banner image. TV size high resolution
	// (1920x1080).
	BannerTvHighImageUrl string `json:"bannerTvHighImageUrl,omitempty"`

	// BannerTvImageUrl: Banner image. TV size extra high resolution
	// (2120x1192).
	BannerTvImageUrl string `json:"bannerTvImageUrl,omitempty"`

	// BannerTvLowImageUrl: Banner image. TV size low resolution (854x480).
	BannerTvLowImageUrl string `json:"bannerTvLowImageUrl,omitempty"`

	// BannerTvMediumImageUrl: Banner image. TV size medium resolution
	// (1280x720).
	BannerTvMediumImageUrl string `json:"bannerTvMediumImageUrl,omitempty"`

	// LargeBrandedBannerImageImapScript: The image map script for the large
	// banner image.
	LargeBrandedBannerImageImapScript *LocalizedProperty `json:"largeBrandedBannerImageImapScript,omitempty"`

	// LargeBrandedBannerImageUrl: The URL for the 854px by 70px image that
	// appears below the video player in the expanded video view of the
	// video watch page.
	LargeBrandedBannerImageUrl *LocalizedProperty `json:"largeBrandedBannerImageUrl,omitempty"`

	// SmallBrandedBannerImageImapScript: The image map script for the small
	// banner image.
	SmallBrandedBannerImageImapScript *LocalizedProperty `json:"smallBrandedBannerImageImapScript,omitempty"`

	// SmallBrandedBannerImageUrl: The URL for the 640px by 70px banner
	// image that appears below the video player in the default view of the
	// video watch page.
	SmallBrandedBannerImageUrl *LocalizedProperty `json:"smallBrandedBannerImageUrl,omitempty"`

	// TrackingImageUrl: The URL for a 1px by 1px tracking pixel that can be
	// used to collect statistics for views of the channel or video pages.
	TrackingImageUrl string `json:"trackingImageUrl,omitempty"`

	// WatchIconImageUrl: The URL for the image that appears above the
	// top-left corner of the video player. This is a 25-pixel-high image
	// with a flexible width that cannot exceed 170 pixels.
	WatchIconImageUrl string `json:"watchIconImageUrl,omitempty"`
}

type IngestionInfo struct {
	// BackupIngestionAddress: The backup ingestion URL that you should use
	// to stream video to YouTube. You have the option of simultaneously
	// streaming the content that you are sending to the ingestionAddress to
	// this URL.
	BackupIngestionAddress string `json:"backupIngestionAddress,omitempty"`

	// IngestionAddress: The primary ingestion URL that you should use to
	// stream video to YouTube. You must stream video to this
	// URL.
	//
	// Depending on which application or tool you use to encode your
	// video stream, you may need to enter the stream URL and stream name
	// separately or you may need to concatenate them in the following
	// format:
	//
	// STREAM_URL/STREAM_NAME
	IngestionAddress string `json:"ingestionAddress,omitempty"`

	// StreamName: The HTTP or RTMP stream name that YouTube assigns to the
	// video stream.
	StreamName string `json:"streamName,omitempty"`
}

type InvideoBranding struct {
	ImageBytes string `json:"imageBytes,omitempty"`

	ImageUrl string `json:"imageUrl,omitempty"`

	Position *InvideoPosition `json:"position,omitempty"`

	TargetChannelId string `json:"targetChannelId,omitempty"`

	Timing *InvideoTiming `json:"timing,omitempty"`
}

type InvideoPosition struct {
	// CornerPosition: Describes in which corner of the video the visual
	// widget will appear.
	CornerPosition string `json:"cornerPosition,omitempty"`

	// Type: Defines the position type.
	Type string `json:"type,omitempty"`
}

type InvideoPromotion struct {
	// DefaultTiming: The default temporal position within the video where
	// the promoted item will be displayed. Can be overriden by more
	// specific timing in the item.
	DefaultTiming *InvideoTiming `json:"defaultTiming,omitempty"`

	// Items: List of promoted items in decreasing priority.
	Items []*PromotedItem `json:"items,omitempty"`

	// Position: The spatial position within the video where the promoted
	// item will be displayed.
	Position *InvideoPosition `json:"position,omitempty"`
}

type InvideoTiming struct {
	// DurationMs: Defines the duration in milliseconds for which the
	// promotion should be displayed. If missing, the client should use the
	// default.
	DurationMs uint64 `json:"durationMs,omitempty,string"`

	// OffsetMs: Defines the time at which the promotion will appear.
	// Depending on the value of type the value of the offsetMs field will
	// represent a time offset from the start or from the end of the video,
	// expressed in milliseconds.
	OffsetMs uint64 `json:"offsetMs,omitempty,string"`

	// Type: Describes a timing type. If the value is offsetFromStart, then
	// the offsetMs field represents an offset from the start of the video.
	// If the value is offsetFromEnd, then the offsetMs field represents an
	// offset from the end of the video.
	Type string `json:"type,omitempty"`
}

type LiveBroadcast struct {
	// ContentDetails: The contentDetails object contains information about
	// the event's video content, such as whether the content can be shown
	// in an embedded video player or if it will be archived and therefore
	// available for viewing after the event has concluded.
	ContentDetails *LiveBroadcastContentDetails `json:"contentDetails,omitempty"`

	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The ID that YouTube assigns to uniquely identify the broadcast.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#liveBroadcast".
	Kind string `json:"kind,omitempty"`

	// Snippet: The snippet object contains basic details about the event,
	// including its title, description, start time, and end time.
	Snippet *LiveBroadcastSnippet `json:"snippet,omitempty"`

	// Status: The status object contains information about the event's
	// status.
	Status *LiveBroadcastStatus `json:"status,omitempty"`
}

type LiveBroadcastContentDetails struct {
	// BoundStreamId: This value uniquely identifies the live stream bound
	// to the broadcast.
	BoundStreamId string `json:"boundStreamId,omitempty"`

	// EnableClosedCaptions: This setting indicates whether closed
	// captioning is enabled for this broadcast. The ingestion URL of the
	// closed captions is returned through the liveStreams API.
	EnableClosedCaptions bool `json:"enableClosedCaptions,omitempty"`

	// EnableContentEncryption: This setting indicates whether YouTube
	// should enable content encryption for the broadcast.
	EnableContentEncryption bool `json:"enableContentEncryption,omitempty"`

	// EnableDvr: This setting determines whether viewers can access DVR
	// controls while watching the video. DVR controls enable the viewer to
	// control the video playback experience by pausing, rewinding, or fast
	// forwarding content. The default value for this property is
	// true.
	//
	//
	//
	// Important: You must set the value to true and also set the
	// enableArchive property's value to true if you want to make playback
	// available immediately after the broadcast ends.
	EnableDvr bool `json:"enableDvr,omitempty"`

	// EnableEmbed: This setting indicates whether the broadcast video can
	// be played in an embedded player. If you choose to archive the video
	// (using the enableArchive property), this setting will also apply to
	// the archived video.
	EnableEmbed bool `json:"enableEmbed,omitempty"`

	// MonitorStream: The monitorStream object contains information about
	// the monitor stream, which the broadcaster can use to review the event
	// content before the broadcast stream is shown publicly.
	MonitorStream *MonitorStreamInfo `json:"monitorStream,omitempty"`

	// RecordFromStart: Automatically start recording after the event goes
	// live. The default value for this property is true.
	//
	//
	//
	// Important: You
	// must also set the enableDvr property's value to true if you want the
	// playback to be available immediately after the broadcast ends. If you
	// set this property's value to true but do not also set the enableDvr
	// property to true, there may be a delay of around one day before the
	// archived video will be available for playback.
	RecordFromStart bool `json:"recordFromStart,omitempty"`

	// StartWithSlate: This setting indicates whether the broadcast should
	// automatically begin with an in-stream slate when you update the
	// broadcast's status to live. After updating the status, you then need
	// to send a liveCuepoints.insert request that sets the cuepoint's
	// eventState to end to remove the in-stream slate and make your
	// broadcast stream visible to viewers.
	StartWithSlate bool `json:"startWithSlate,omitempty"`
}

type LiveBroadcastListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of broadcasts that match the request criteria.
	Items []*LiveBroadcast `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#liveBroadcastListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the next page in the result set.
	NextPageToken string `json:"nextPageToken,omitempty"`

	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// PrevPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the previous page in the result set.
	PrevPageToken string `json:"prevPageToken,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type LiveBroadcastSnippet struct {
	// ActualEndTime: The date and time that the broadcast actually ended.
	// This information is only available once the broadcast's state is
	// complete. The value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ)
	// format.
	ActualEndTime string `json:"actualEndTime,omitempty"`

	// ActualStartTime: The date and time that the broadcast actually
	// started. This information is only available once the broadcast's
	// state is live. The value is specified in ISO 8601
	// (YYYY-MM-DDThh:mm:ss.sZ) format.
	ActualStartTime string `json:"actualStartTime,omitempty"`

	// ChannelId: The ID that YouTube uses to uniquely identify the channel
	// that is publishing the broadcast.
	ChannelId string `json:"channelId,omitempty"`

	// Description: The broadcast's description. As with the title, you can
	// set this field by modifying the broadcast resource or by setting the
	// description field of the corresponding video resource.
	Description string `json:"description,omitempty"`

	// PublishedAt: The date and time that the broadcast was added to
	// YouTube's live broadcast schedule. The value is specified in ISO 8601
	// (YYYY-MM-DDThh:mm:ss.sZ) format.
	PublishedAt string `json:"publishedAt,omitempty"`

	// ScheduledEndTime: The date and time that the broadcast is scheduled
	// to end. The value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ)
	// format.
	ScheduledEndTime string `json:"scheduledEndTime,omitempty"`

	// ScheduledStartTime: The date and time that the broadcast is scheduled
	// to start. The value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ)
	// format.
	ScheduledStartTime string `json:"scheduledStartTime,omitempty"`

	// Thumbnails: A map of thumbnail images associated with the broadcast.
	// For each nested object in this object, the key is the name of the
	// thumbnail image, and the value is an object that contains other
	// information about the thumbnail.
	Thumbnails *ThumbnailDetails `json:"thumbnails,omitempty"`

	// Title: The broadcast's title. Note that the broadcast represents
	// exactly one YouTube video. You can set this field by modifying the
	// broadcast resource or by setting the title field of the corresponding
	// video resource.
	Title string `json:"title,omitempty"`
}

type LiveBroadcastStatus struct {
	// LifeCycleStatus: The broadcast's status. The status can be updated
	// using the API's liveBroadcasts.transition method.
	LifeCycleStatus string `json:"lifeCycleStatus,omitempty"`

	// PrivacyStatus: The broadcast's privacy status. Note that the
	// broadcast represents exactly one YouTube video, so the privacy
	// settings are identical to those supported for videos. In addition,
	// you can set this field by modifying the broadcast resource or by
	// setting the privacyStatus field of the corresponding video resource.
	PrivacyStatus string `json:"privacyStatus,omitempty"`

	// RecordingStatus: The broadcast's recording status.
	RecordingStatus string `json:"recordingStatus,omitempty"`
}

type LiveStream struct {
	// Cdn: The cdn object defines the live stream's content delivery
	// network (CDN) settings. These settings provide details about the
	// manner in which you stream your content to YouTube.
	Cdn *CdnSettings `json:"cdn,omitempty"`

	// ContentDetails: The content_details object contains information about
	// the stream, including the closed captions ingestion URL.
	ContentDetails *LiveStreamContentDetails `json:"contentDetails,omitempty"`

	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The ID that YouTube assigns to uniquely identify the stream.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#liveStream".
	Kind string `json:"kind,omitempty"`

	// Snippet: The snippet object contains basic details about the stream,
	// including its channel, title, and description.
	Snippet *LiveStreamSnippet `json:"snippet,omitempty"`

	// Status: The status object contains information about live stream's
	// status.
	Status *LiveStreamStatus `json:"status,omitempty"`
}

type LiveStreamContentDetails struct {
	// ClosedCaptionsIngestionUrl: The ingestion URL where the closed
	// captions of this stream are sent.
	ClosedCaptionsIngestionUrl string `json:"closedCaptionsIngestionUrl,omitempty"`
}

type LiveStreamListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of live streams that match the request criteria.
	Items []*LiveStream `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#liveStreamListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the next page in the result set.
	NextPageToken string `json:"nextPageToken,omitempty"`

	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// PrevPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the previous page in the result set.
	PrevPageToken string `json:"prevPageToken,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type LiveStreamSnippet struct {
	// ChannelId: The ID that YouTube uses to uniquely identify the channel
	// that is transmitting the stream.
	ChannelId string `json:"channelId,omitempty"`

	// Description: The stream's description. The value cannot be longer
	// than 10000 characters.
	Description string `json:"description,omitempty"`

	// PublishedAt: The date and time that the stream was created. The value
	// is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format.
	PublishedAt string `json:"publishedAt,omitempty"`

	// Title: The stream's title. The value must be between 1 and 128
	// characters long.
	Title string `json:"title,omitempty"`
}

type LiveStreamStatus struct {
	StreamStatus string `json:"streamStatus,omitempty"`
}

type LocalizedProperty struct {
	// Default: Default value for the localized property.
	Default string `json:"default,omitempty"`

	// Localized: The localized values.
	Localized []*LocalizedString `json:"localized,omitempty"`
}

type LocalizedString struct {
	// Language: Language associated to this value.
	Language string `json:"language,omitempty"`

	// Value: Value of the property.
	Value string `json:"value,omitempty"`
}

type MonitorStreamInfo struct {
	// BroadcastStreamDelayMs: If you have set the enableMonitorStream
	// property to true, then this property determines the length of the
	// live broadcast delay.
	BroadcastStreamDelayMs int64 `json:"broadcastStreamDelayMs,omitempty"`

	// EmbedHtml: HTML code that embeds a player that plays the monitor
	// stream.
	EmbedHtml string `json:"embedHtml,omitempty"`

	// EnableMonitorStream: This value determines whether the monitor stream
	// is enabled for the broadcast. If the monitor stream is enabled, then
	// YouTube will broadcast the event content on a special stream intended
	// only for the broadcaster's consumption. The broadcaster can use the
	// stream to review the event content and also to identify the optimal
	// times to insert cuepoints.
	//
	// You need to set this value to true if you
	// intend to have a broadcast delay for your event.
	//
	// Note: This property
	// cannot be updated once the broadcast is in the testing or live state.
	EnableMonitorStream bool `json:"enableMonitorStream,omitempty"`
}

type PageInfo struct {
	// ResultsPerPage: The number of results included in the API response.
	ResultsPerPage int64 `json:"resultsPerPage,omitempty"`

	// TotalResults: The total number of results in the result set.
	TotalResults int64 `json:"totalResults,omitempty"`
}

type Playlist struct {
	// ContentDetails: The contentDetails object contains information like
	// video count.
	ContentDetails *PlaylistContentDetails `json:"contentDetails,omitempty"`

	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The ID that YouTube uses to uniquely identify the playlist.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#playlist".
	Kind string `json:"kind,omitempty"`

	// Player: The player object contains information that you would use to
	// play the playlist in an embedded player.
	Player *PlaylistPlayer `json:"player,omitempty"`

	// Snippet: The snippet object contains basic details about the
	// playlist, such as its title and description.
	Snippet *PlaylistSnippet `json:"snippet,omitempty"`

	// Status: The status object contains status information for the
	// playlist.
	Status *PlaylistStatus `json:"status,omitempty"`
}

type PlaylistContentDetails struct {
	// ItemCount: The number of videos in the playlist.
	ItemCount int64 `json:"itemCount,omitempty"`
}

type PlaylistItem struct {
	// ContentDetails: The contentDetails object is included in the resource
	// if the included item is a YouTube video. The object contains
	// additional information about the video.
	ContentDetails *PlaylistItemContentDetails `json:"contentDetails,omitempty"`

	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The ID that YouTube uses to uniquely identify the playlist item.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#playlistItem".
	Kind string `json:"kind,omitempty"`

	// Snippet: The snippet object contains basic details about the playlist
	// item, such as its title and position in the playlist.
	Snippet *PlaylistItemSnippet `json:"snippet,omitempty"`

	// Status: The status object contains information about the playlist
	// item's privacy status.
	Status *PlaylistItemStatus `json:"status,omitempty"`
}

type PlaylistItemContentDetails struct {
	// EndAt: The time, measured in seconds from the start of the video,
	// when the video should stop playing. (The playlist owner can specify
	// the times when the video should start and stop playing when the video
	// is played in the context of the playlist.) By default, assume that
	// the video.endTime is the end of the video.
	EndAt string `json:"endAt,omitempty"`

	// Note: A user-generated note for this item.
	Note string `json:"note,omitempty"`

	// StartAt: The time, measured in seconds from the start of the video,
	// when the video should start playing. (The playlist owner can specify
	// the times when the video should start and stop playing when the video
	// is played in the context of the playlist.) The default value is 0.
	StartAt string `json:"startAt,omitempty"`

	// VideoId: The ID that YouTube uses to uniquely identify a video. To
	// retrieve the video resource, set the id query parameter to this value
	// in your API request.
	VideoId string `json:"videoId,omitempty"`
}

type PlaylistItemListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of playlist items that match the request criteria.
	Items []*PlaylistItem `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#playlistItemListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the next page in the result set.
	NextPageToken string `json:"nextPageToken,omitempty"`

	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// PrevPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the previous page in the result set.
	PrevPageToken string `json:"prevPageToken,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type PlaylistItemSnippet struct {
	// ChannelId: The ID that YouTube uses to uniquely identify the user
	// that added the item to the playlist.
	ChannelId string `json:"channelId,omitempty"`

	// ChannelTitle: Channel title for the channel that the playlist item
	// belongs to.
	ChannelTitle string `json:"channelTitle,omitempty"`

	// Description: The item's description.
	Description string `json:"description,omitempty"`

	// PlaylistId: The ID that YouTube uses to uniquely identify the
	// playlist that the playlist item is in.
	PlaylistId string `json:"playlistId,omitempty"`

	// Position: The order in which the item appears in the playlist. The
	// value uses a zero-based index, so the first item has a position of 0,
	// the second item has a position of 1, and so forth.
	Position int64 `json:"position,omitempty"`

	// PublishedAt: The date and time that the item was added to the
	// playlist. The value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ)
	// format.
	PublishedAt string `json:"publishedAt,omitempty"`

	// ResourceId: The id object contains information that can be used to
	// uniquely identify the resource that is included in the playlist as
	// the playlist item.
	ResourceId *ResourceId `json:"resourceId,omitempty"`

	// Thumbnails: A map of thumbnail images associated with the playlist
	// item. For each object in the map, the key is the name of the
	// thumbnail image, and the value is an object that contains other
	// information about the thumbnail.
	Thumbnails *ThumbnailDetails `json:"thumbnails,omitempty"`

	// Title: The item's title.
	Title string `json:"title,omitempty"`
}

type PlaylistItemStatus struct {
	// PrivacyStatus: This resource's privacy status.
	PrivacyStatus string `json:"privacyStatus,omitempty"`
}

type PlaylistListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of playlists that match the request criteria.
	Items []*Playlist `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#playlistListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the next page in the result set.
	NextPageToken string `json:"nextPageToken,omitempty"`

	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// PrevPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the previous page in the result set.
	PrevPageToken string `json:"prevPageToken,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type PlaylistPlayer struct {
	// EmbedHtml: An <iframe> tag that embeds a player that will play the
	// playlist.
	EmbedHtml string `json:"embedHtml,omitempty"`
}

type PlaylistSnippet struct {
	// ChannelId: The ID that YouTube uses to uniquely identify the channel
	// that published the playlist.
	ChannelId string `json:"channelId,omitempty"`

	// ChannelTitle: The channel title of the channel that the video belongs
	// to.
	ChannelTitle string `json:"channelTitle,omitempty"`

	// Description: The playlist's description.
	Description string `json:"description,omitempty"`

	// PublishedAt: The date and time that the playlist was created. The
	// value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format.
	PublishedAt string `json:"publishedAt,omitempty"`

	// Tags: Keyword tags associated with the playlist.
	Tags []string `json:"tags,omitempty"`

	// Thumbnails: A map of thumbnail images associated with the playlist.
	// For each object in the map, the key is the name of the thumbnail
	// image, and the value is an object that contains other information
	// about the thumbnail.
	Thumbnails *ThumbnailDetails `json:"thumbnails,omitempty"`

	// Title: The playlist's title.
	Title string `json:"title,omitempty"`
}

type PlaylistStatus struct {
	// PrivacyStatus: The playlist's privacy status.
	PrivacyStatus string `json:"privacyStatus,omitempty"`
}

type PromotedItem struct {
	// CustomMessage: A custom message to display for this promotion. This
	// field is currently ignored unless the promoted item is a website.
	CustomMessage string `json:"customMessage,omitempty"`

	// Id: Identifies the promoted item.
	Id *PromotedItemId `json:"id,omitempty"`

	// PromotedByContentOwner: If true, the content owner's name will be
	// used when displaying the promotion. This field can only be set when
	// the update is made on behalf of the content owner.
	PromotedByContentOwner bool `json:"promotedByContentOwner,omitempty"`

	// Timing: The temporal position within the video where the promoted
	// item will be displayed. If present, it overrides the default timing.
	Timing *InvideoTiming `json:"timing,omitempty"`
}

type PromotedItemId struct {
	// RecentlyUploadedBy: If type is recentUpload, this field identifies
	// the channel from which to take the recent upload. If missing, the
	// channel is assumed to be the same channel for which the
	// invideoPromotion is set.
	RecentlyUploadedBy string `json:"recentlyUploadedBy,omitempty"`

	// Type: Describes the type of the promoted item.
	Type string `json:"type,omitempty"`

	// VideoId: If the promoted item represents a video, this field
	// represents the unique YouTube ID identifying it. This field will be
	// present only if type has the value video.
	VideoId string `json:"videoId,omitempty"`

	// WebsiteUrl: If the promoted item represents a website, this field
	// represents the url pointing to the website. This field will be
	// present only if type has the value website.
	WebsiteUrl string `json:"websiteUrl,omitempty"`
}

type PropertyValue struct {
	// Property: A property.
	Property string `json:"property,omitempty"`

	// Value: The property's value.
	Value string `json:"value,omitempty"`
}

type ResourceId struct {
	// ChannelId: The ID that YouTube uses to uniquely identify the referred
	// resource, if that resource is a channel. This property is only
	// present if the resourceId.kind value is youtube#channel.
	ChannelId string `json:"channelId,omitempty"`

	// Kind: The type of the API resource.
	Kind string `json:"kind,omitempty"`

	// PlaylistId: The ID that YouTube uses to uniquely identify the
	// referred resource, if that resource is a playlist. This property is
	// only present if the resourceId.kind value is youtube#playlist.
	PlaylistId string `json:"playlistId,omitempty"`

	// VideoId: The ID that YouTube uses to uniquely identify the referred
	// resource, if that resource is a video. This property is only present
	// if the resourceId.kind value is youtube#video.
	VideoId string `json:"videoId,omitempty"`
}

type SearchListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of results that match the search criteria.
	Items []*SearchResult `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#searchListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the next page in the result set.
	NextPageToken string `json:"nextPageToken,omitempty"`

	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// PrevPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the previous page in the result set.
	PrevPageToken string `json:"prevPageToken,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type SearchResult struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The id object contains information that can be used to uniquely
	// identify the resource that matches the search request.
	Id *ResourceId `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#searchResult".
	Kind string `json:"kind,omitempty"`

	// Snippet: The snippet object contains basic details about a search
	// result, such as its title or description. For example, if the search
	// result is a video, then the title will be the video's title and the
	// description will be the video's description.
	Snippet *SearchResultSnippet `json:"snippet,omitempty"`
}

type SearchResultSnippet struct {
	// ChannelId: The value that YouTube uses to uniquely identify the
	// channel that published the resource that the search result
	// identifies.
	ChannelId string `json:"channelId,omitempty"`

	// ChannelTitle: The title of the channel that published the resource
	// that the search result identifies.
	ChannelTitle string `json:"channelTitle,omitempty"`

	// Description: A description of the search result.
	Description string `json:"description,omitempty"`

	// LiveBroadcastContent: It indicates if the resource (video or channel)
	// has upcoming/active live broadcast content. Or it's "none" if there
	// is not any upcoming/active live broadcasts.
	LiveBroadcastContent string `json:"liveBroadcastContent,omitempty"`

	// PublishedAt: The creation date and time of the resource that the
	// search result identifies. The value is specified in ISO 8601
	// (YYYY-MM-DDThh:mm:ss.sZ) format.
	PublishedAt string `json:"publishedAt,omitempty"`

	// Thumbnails: A map of thumbnail images associated with the search
	// result. For each object in the map, the key is the name of the
	// thumbnail image, and the value is an object that contains other
	// information about the thumbnail.
	Thumbnails *ThumbnailDetails `json:"thumbnails,omitempty"`

	// Title: The title of the search result.
	Title string `json:"title,omitempty"`
}

type Subscription struct {
	// ContentDetails: The contentDetails object contains basic statistics
	// about the subscription.
	ContentDetails *SubscriptionContentDetails `json:"contentDetails,omitempty"`

	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The ID that YouTube uses to uniquely identify the subscription.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#subscription".
	Kind string `json:"kind,omitempty"`

	// Snippet: The snippet object contains basic details about the
	// subscription, including its title and the channel that the user
	// subscribed to.
	Snippet *SubscriptionSnippet `json:"snippet,omitempty"`

	// SubscriberSnippet: The subscriberSnippet object contains basic
	// details about the sbuscriber.
	SubscriberSnippet *SubscriptionSubscriberSnippet `json:"subscriberSnippet,omitempty"`
}

type SubscriptionContentDetails struct {
	// ActivityType: The type of activity this subscription is for (only
	// uploads, everything).
	ActivityType string `json:"activityType,omitempty"`

	// NewItemCount: The number of new items in the subscription since its
	// content was last read.
	NewItemCount int64 `json:"newItemCount,omitempty"`

	// TotalItemCount: The approximate number of items that the subscription
	// points to.
	TotalItemCount int64 `json:"totalItemCount,omitempty"`
}

type SubscriptionListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of subscriptions that match the request criteria.
	Items []*Subscription `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#subscriptionListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the next page in the result set.
	NextPageToken string `json:"nextPageToken,omitempty"`

	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// PrevPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the previous page in the result set.
	PrevPageToken string `json:"prevPageToken,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type SubscriptionSnippet struct {
	// ChannelId: The ID that YouTube uses to uniquely identify the
	// subscriber's channel.
	ChannelId string `json:"channelId,omitempty"`

	// ChannelTitle: Channel title for the channel that the subscription
	// belongs to.
	ChannelTitle string `json:"channelTitle,omitempty"`

	// Description: The subscription's details.
	Description string `json:"description,omitempty"`

	// PublishedAt: The date and time that the subscription was created. The
	// value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format.
	PublishedAt string `json:"publishedAt,omitempty"`

	// ResourceId: The id object contains information about the channel that
	// the user subscribed to.
	ResourceId *ResourceId `json:"resourceId,omitempty"`

	// Thumbnails: A map of thumbnail images associated with the video. For
	// each object in the map, the key is the name of the thumbnail image,
	// and the value is an object that contains other information about the
	// thumbnail.
	Thumbnails *ThumbnailDetails `json:"thumbnails,omitempty"`

	// Title: The subscription's title.
	Title string `json:"title,omitempty"`
}

type SubscriptionSubscriberSnippet struct {
	// ChannelId: The channel ID of the subscriber.
	ChannelId string `json:"channelId,omitempty"`

	// Description: The description of the subscriber.
	Description string `json:"description,omitempty"`

	// Thumbnails: Thumbnails for this subscriber.
	Thumbnails *ThumbnailDetails `json:"thumbnails,omitempty"`

	// Title: The title of the subscriber.
	Title string `json:"title,omitempty"`
}

type Thumbnail struct {
	// Height: (Optional) Height of the thumbnail image.
	Height int64 `json:"height,omitempty"`

	// Url: The thumbnail image's URL.
	Url string `json:"url,omitempty"`

	// Width: (Optional) Width of the thumbnail image.
	Width int64 `json:"width,omitempty"`
}

type ThumbnailDetails struct {
	// Default: The default image for this resource.
	Default *Thumbnail `json:"default,omitempty"`

	// High: The high quality image for this resource.
	High *Thumbnail `json:"high,omitempty"`

	// Maxres: The maximum resolution quality image for this resource.
	Maxres *Thumbnail `json:"maxres,omitempty"`

	// Medium: The medium quality image for this resource.
	Medium *Thumbnail `json:"medium,omitempty"`

	// Standard: The standard quality image for this resource.
	Standard *Thumbnail `json:"standard,omitempty"`
}

type ThumbnailSetResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of thumbnails.
	Items []*ThumbnailDetails `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#thumbnailSetResponse".
	Kind string `json:"kind,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type TokenPagination struct {
}

type Video struct {
	// AgeGating: Age restriction details related to a video.
	AgeGating *VideoAgeGating `json:"ageGating,omitempty"`

	// ContentDetails: The contentDetails object contains information about
	// the video content, including the length of the video and its aspect
	// ratio.
	ContentDetails *VideoContentDetails `json:"contentDetails,omitempty"`

	// ConversionPings: The conversionPings object encapsulates information
	// about url pings that need to be respected by the App in different
	// video contexts.
	ConversionPings *VideoConversionPings `json:"conversionPings,omitempty"`

	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// FileDetails: The fileDetails object encapsulates information about
	// the video file that was uploaded to YouTube, including the file's
	// resolution, duration, audio and video codecs, stream bitrates, and
	// more. This data can only be retrieved by the video owner.
	FileDetails *VideoFileDetails `json:"fileDetails,omitempty"`

	// Id: The ID that YouTube uses to uniquely identify the video.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#video".
	Kind string `json:"kind,omitempty"`

	// LiveStreamingDetails: The liveStreamingDetails object contains
	// metadata about a live video broadcast. The object will only be
	// present in a video resource if the video is an upcoming, live, or
	// completed live broadcast.
	LiveStreamingDetails *VideoLiveStreamingDetails `json:"liveStreamingDetails,omitempty"`

	// MonetizationDetails: The monetizationDetails object encapsulates
	// information about the monetization status of the video.
	MonetizationDetails *VideoMonetizationDetails `json:"monetizationDetails,omitempty"`

	// Player: The player object contains information that you would use to
	// play the video in an embedded player.
	Player *VideoPlayer `json:"player,omitempty"`

	// ProcessingDetails: The processingProgress object encapsulates
	// information about YouTube's progress in processing the uploaded video
	// file. The properties in the object identify the current processing
	// status and an estimate of the time remaining until YouTube finishes
	// processing the video. This part also indicates whether different
	// types of data or content, such as file details or thumbnail images,
	// are available for the video.
	//
	// The processingProgress object is
	// designed to be polled so that the video uploaded can track the
	// progress that YouTube has made in processing the uploaded video file.
	// This data can only be retrieved by the video owner.
	ProcessingDetails *VideoProcessingDetails `json:"processingDetails,omitempty"`

	// ProjectDetails: The projectDetails object contains information about
	// the project specific video metadata.
	ProjectDetails *VideoProjectDetails `json:"projectDetails,omitempty"`

	// RecordingDetails: The recordingDetails object encapsulates
	// information about the location, date and address where the video was
	// recorded.
	RecordingDetails *VideoRecordingDetails `json:"recordingDetails,omitempty"`

	// Snippet: The snippet object contains basic details about the video,
	// such as its title, description, and category.
	Snippet *VideoSnippet `json:"snippet,omitempty"`

	// Statistics: The statistics object contains statistics about the
	// video.
	Statistics *VideoStatistics `json:"statistics,omitempty"`

	// Status: The status object contains information about the video's
	// uploading, processing, and privacy statuses.
	Status *VideoStatus `json:"status,omitempty"`

	// Suggestions: The suggestions object encapsulates suggestions that
	// identify opportunities to improve the video quality or the metadata
	// for the uploaded video. This data can only be retrieved by the video
	// owner.
	Suggestions *VideoSuggestions `json:"suggestions,omitempty"`

	// TopicDetails: The topicDetails object encapsulates information about
	// Freebase topics associated with the video.
	TopicDetails *VideoTopicDetails `json:"topicDetails,omitempty"`
}

type VideoAgeGating struct {
	// AlcoholContent: Indicates whether or not the video has alcoholic
	// beverage content. Only users of legal purchasing age in a particular
	// country, as identified by ICAP, can view the content.
	AlcoholContent bool `json:"alcoholContent,omitempty"`

	// Restricted: Age-restricted trailers. For redband trailers and
	// adult-rated video-games. Only users aged 18+ can view the content.
	// The the field is true the content is restricted to viewers aged 18+.
	// Otherwise The field won't be present.
	Restricted bool `json:"restricted,omitempty"`

	// VideoGameRating: Video game rating, if any.
	VideoGameRating string `json:"videoGameRating,omitempty"`
}

type VideoCategory struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Id: The ID that YouTube uses to uniquely identify the video category.
	Id string `json:"id,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#videoCategory".
	Kind string `json:"kind,omitempty"`

	// Snippet: The snippet object contains basic details about the video
	// category, including its title.
	Snippet *VideoCategorySnippet `json:"snippet,omitempty"`
}

type VideoCategoryListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of video categories that can be associated with YouTube
	// videos. In this map, the video category ID is the map key, and its
	// value is the corresponding videoCategory resource.
	Items []*VideoCategory `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#videoCategoryListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the next page in the result set.
	NextPageToken string `json:"nextPageToken,omitempty"`

	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// PrevPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the previous page in the result set.
	PrevPageToken string `json:"prevPageToken,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type VideoCategorySnippet struct {
	Assignable bool `json:"assignable,omitempty"`

	// ChannelId: The YouTube channel that created the video category.
	ChannelId string `json:"channelId,omitempty"`

	// Title: The video category's title.
	Title string `json:"title,omitempty"`
}

type VideoContentDetails struct {
	// Caption: The value of captions indicates whether the video has
	// captions or not.
	Caption string `json:"caption,omitempty"`

	// ContentRating: Specifies the ratings that the video received under
	// various rating schemes.
	ContentRating *ContentRating `json:"contentRating,omitempty"`

	// CountryRestriction: The countryRestriction object contains
	// information about the countries where a video is (or is not)
	// viewable.
	CountryRestriction *AccessPolicy `json:"countryRestriction,omitempty"`

	// Definition: The value of definition indicates whether the video is
	// available in high definition or only in standard definition.
	Definition string `json:"definition,omitempty"`

	// Dimension: The value of dimension indicates whether the video is
	// available in 3D or in 2D.
	Dimension string `json:"dimension,omitempty"`

	// Duration: The length of the video. The tag value is an ISO 8601
	// duration in the format PT#M#S, in which the letters PT indicate that
	// the value specifies a period of time, and the letters M and S refer
	// to length in minutes and seconds, respectively. The # characters
	// preceding the M and S letters are both integers that specify the
	// number of minutes (or seconds) of the video. For example, a value of
	// PT15M51S indicates that the video is 15 minutes and 51 seconds long.
	Duration string `json:"duration,omitempty"`

	// LicensedContent: The value of is_license_content indicates whether
	// the video is licensed content.
	LicensedContent bool `json:"licensedContent,omitempty"`

	// RegionRestriction: The regionRestriction object contains information
	// about the countries where a video is (or is not) viewable. The object
	// will contain either the contentDetails.regionRestriction.allowed
	// property or the contentDetails.regionRestriction.blocked property.
	RegionRestriction *VideoContentDetailsRegionRestriction `json:"regionRestriction,omitempty"`
}

type VideoContentDetailsRegionRestriction struct {
	// Allowed: A list of region codes that identify countries where the
	// video is viewable. If this property is present and a country is not
	// listed in its value, then the video is blocked from appearing in that
	// country. If this property is present and contains an empty list, the
	// video is blocked in all countries.
	Allowed []string `json:"allowed,omitempty"`

	// Blocked: A list of region codes that identify countries where the
	// video is blocked. If this property is present and a country is not
	// listed in its value, then the video is viewable in that country. If
	// this property is present and contains an empty list, the video is
	// viewable in all countries.
	Blocked []string `json:"blocked,omitempty"`
}

type VideoConversionPing struct {
	// Context: Defines the context of the ping.
	Context string `json:"context,omitempty"`

	// ConversionUrl: The url (without the schema) that the app shall send
	// the ping to. It's at caller's descretion to decide which schema to
	// use (http vs https) Example of a returned url:
	// //googleads.g.doubleclick.net/pagead/
	// viewthroughconversion/962985656/?data=path%3DtHe_path%3Btype%3D
	// like%3Butuid%3DGISQtTNGYqaYl4sKxoVvKA%3Bytvid%3DUrIaJUvIQDg&labe=defau
	// lt The caller must append biscotti authentication (ms param in case
	// of mobile, for example) to this ping.
	ConversionUrl string `json:"conversionUrl,omitempty"`
}

type VideoConversionPings struct {
	// Pings: Pings that the app shall fire for a video (authenticated by
	// biscotti cookie). Each ping has a context, in which the app must fire
	// the ping, and a url identifying the ping.
	Pings []*VideoConversionPing `json:"pings,omitempty"`
}

type VideoFileDetails struct {
	// AudioStreams: A list of audio streams contained in the uploaded video
	// file. Each item in the list contains detailed metadata about an audio
	// stream.
	AudioStreams []*VideoFileDetailsAudioStream `json:"audioStreams,omitempty"`

	// BitrateBps: The uploaded video file's combined (video and audio)
	// bitrate in bits per second.
	BitrateBps uint64 `json:"bitrateBps,omitempty,string"`

	// Container: The uploaded video file's container format.
	Container string `json:"container,omitempty"`

	// CreationTime: The date and time when the uploaded video file was
	// created. The value is specified in ISO 8601 format. Currently, the
	// following ISO 8601 formats are supported:
	// - Date only: YYYY-MM-DD
	//
	// - Naive time: YYYY-MM-DDTHH:MM:SS
	// - Time with timezone:
	// YYYY-MM-DDTHH:MM:SS+HH:MM
	CreationTime string `json:"creationTime,omitempty"`

	// DurationMs: The length of the uploaded video in milliseconds.
	DurationMs uint64 `json:"durationMs,omitempty,string"`

	// FileName: The uploaded file's name. This field is present whether a
	// video file or another type of file was uploaded.
	FileName string `json:"fileName,omitempty"`

	// FileSize: The uploaded file's size in bytes. This field is present
	// whether a video file or another type of file was uploaded.
	FileSize uint64 `json:"fileSize,omitempty,string"`

	// FileType: The uploaded file's type as detected by YouTube's video
	// processing engine. Currently, YouTube only processes video files, but
	// this field is present whether a video file or another type of file
	// was uploaded.
	FileType string `json:"fileType,omitempty"`

	// RecordingLocation: Geographic coordinates that identify the place
	// where the uploaded video was recorded. Coordinates are defined using
	// WGS 84.
	RecordingLocation *GeoPoint `json:"recordingLocation,omitempty"`

	// VideoStreams: A list of video streams contained in the uploaded video
	// file. Each item in the list contains detailed metadata about a video
	// stream.
	VideoStreams []*VideoFileDetailsVideoStream `json:"videoStreams,omitempty"`
}

type VideoFileDetailsAudioStream struct {
	// BitrateBps: The audio stream's bitrate, in bits per second.
	BitrateBps uint64 `json:"bitrateBps,omitempty,string"`

	// ChannelCount: The number of audio channels that the stream contains.
	ChannelCount int64 `json:"channelCount,omitempty"`

	// Codec: The audio codec that the stream uses.
	Codec string `json:"codec,omitempty"`

	// Vendor: A value that uniquely identifies a video vendor. Typically,
	// the value is a four-letter vendor code.
	Vendor string `json:"vendor,omitempty"`
}

type VideoFileDetailsVideoStream struct {
	// AspectRatio: The video content's display aspect ratio, which
	// specifies the aspect ratio in which the video should be displayed.
	AspectRatio float64 `json:"aspectRatio,omitempty"`

	// BitrateBps: The video stream's bitrate, in bits per second.
	BitrateBps uint64 `json:"bitrateBps,omitempty,string"`

	// Codec: The video codec that the stream uses.
	Codec string `json:"codec,omitempty"`

	// FrameRateFps: The video stream's frame rate, in frames per second.
	FrameRateFps float64 `json:"frameRateFps,omitempty"`

	// HeightPixels: The encoded video content's height in pixels.
	HeightPixels int64 `json:"heightPixels,omitempty"`

	// Rotation: The amount that YouTube needs to rotate the original source
	// content to properly display the video.
	Rotation string `json:"rotation,omitempty"`

	// Vendor: A value that uniquely identifies a video vendor. Typically,
	// the value is a four-letter vendor code.
	Vendor string `json:"vendor,omitempty"`

	// WidthPixels: The encoded video content's width in pixels. You can
	// calculate the video's encoding aspect ratio as
	// width_pixels / height_pixels.
	WidthPixels int64 `json:"widthPixels,omitempty"`
}

type VideoGetRatingResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// Items: A list of ratings that match the request criteria.
	Items []*VideoRating `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#videoGetRatingResponse".
	Kind string `json:"kind,omitempty"`
}

type VideoListResponse struct {
	// Etag: Etag of this resource.
	Etag string `json:"etag,omitempty"`

	// EventId: Serialized EventId of the request which produced this
	// response.
	EventId string `json:"eventId,omitempty"`

	// Items: A list of videos that match the request criteria.
	Items []*Video `json:"items,omitempty"`

	// Kind: Identifies what kind of resource this is. Value: the fixed
	// string "youtube#videoListResponse".
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the next page in the result set.
	NextPageToken string `json:"nextPageToken,omitempty"`

	PageInfo *PageInfo `json:"pageInfo,omitempty"`

	// PrevPageToken: The token that can be used as the value of the
	// pageToken parameter to retrieve the previous page in the result set.
	PrevPageToken string `json:"prevPageToken,omitempty"`

	TokenPagination *TokenPagination `json:"tokenPagination,omitempty"`

	// VisitorId: The visitorId identifies the visitor.
	VisitorId string `json:"visitorId,omitempty"`
}

type VideoLiveStreamingDetails struct {
	// ActualEndTime: The time that the broadcast actually ended. The value
	// is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format. This value
	// will not be available until the broadcast is over.
	ActualEndTime string `json:"actualEndTime,omitempty"`

	// ActualStartTime: The time that the broadcast actually started. The
	// value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format. This
	// value will not be available until the broadcast begins.
	ActualStartTime string `json:"actualStartTime,omitempty"`

	// ConcurrentViewers: The number of viewers currently watching the
	// broadcast. The property and its value will be present if the
	// broadcast has current viewers and the broadcast owner has not hidden
	// the viewcount for the video. Note that YouTube stops tracking the
	// number of concurrent viewers for a broadcast when the broadcast ends.
	// So, this property would not identify the number of viewers watching
	// an archived video of a live broadcast that already ended.
	ConcurrentViewers uint64 `json:"concurrentViewers,omitempty,string"`

	// ScheduledEndTime: The time that the broadcast is scheduled to end.
	// The value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format.
	// If the value is empty or the property is not present, then the
	// broadcast is scheduled to continue indefinitely.
	ScheduledEndTime string `json:"scheduledEndTime,omitempty"`

	// ScheduledStartTime: The time that the broadcast is scheduled to
	// begin. The value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ)
	// format.
	ScheduledStartTime string `json:"scheduledStartTime,omitempty"`
}

type VideoMonetizationDetails struct {
	// Access: The value of access indicates whether the video can be
	// monetized or not.
	Access *AccessPolicy `json:"access,omitempty"`
}

type VideoPlayer struct {
	// EmbedHtml: An <iframe> tag that embeds a player that will play the
	// video.
	EmbedHtml string `json:"embedHtml,omitempty"`
}

type VideoProcessingDetails struct {
	// EditorSuggestionsAvailability: This value indicates whether video
	// editing suggestions, which might improve video quality or the
	// playback experience, are available for the video. You can retrieve
	// these suggestions by requesting the suggestions part in your
	// videos.list() request.
	EditorSuggestionsAvailability string `json:"editorSuggestionsAvailability,omitempty"`

	// FileDetailsAvailability: This value indicates whether file details
	// are available for the uploaded video. You can retrieve a video's file
	// details by requesting the fileDetails part in your videos.list()
	// request.
	FileDetailsAvailability string `json:"fileDetailsAvailability,omitempty"`

	// ProcessingFailureReason: The reason that YouTube failed to process
	// the video. This property will only have a value if the
	// processingStatus property's value is failed.
	ProcessingFailureReason string `json:"processingFailureReason,omitempty"`

	// ProcessingIssuesAvailability: This value indicates whether the video
	// processing engine has generated suggestions that might improve
	// YouTube's ability to process the the video, warnings that explain
	// video processing problems, or errors that cause video processing
	// problems. You can retrieve these suggestions by requesting the
	// suggestions part in your videos.list() request.
	ProcessingIssuesAvailability string `json:"processingIssuesAvailability,omitempty"`

	// ProcessingProgress: The processingProgress object contains
	// information about the progress YouTube has made in processing the
	// video. The values are really only relevant if the video's processing
	// status is processing.
	ProcessingProgress *VideoProcessingDetailsProcessingProgress `json:"processingProgress,omitempty"`

	// ProcessingStatus: The video's processing status. This value indicates
	// whether YouTube was able to process the video or if the video is
	// still being processed.
	ProcessingStatus string `json:"processingStatus,omitempty"`

	// TagSuggestionsAvailability: This value indicates whether keyword
	// (tag) suggestions are available for the video. Tags can be added to a
	// video's metadata to make it easier for other users to find the video.
	// You can retrieve these suggestions by requesting the suggestions part
	// in your videos.list() request.
	TagSuggestionsAvailability string `json:"tagSuggestionsAvailability,omitempty"`

	// ThumbnailsAvailability: This value indicates whether thumbnail images
	// have been generated for the video.
	ThumbnailsAvailability string `json:"thumbnailsAvailability,omitempty"`
}

type VideoProcessingDetailsProcessingProgress struct {
	// PartsProcessed: The number of parts of the video that YouTube has
	// already processed. You can estimate the percentage of the video that
	// YouTube has already processed by calculating:
	// 100 * parts_processed /
	// parts_total
	//
	// Note that since the estimated number of parts could
	// increase without a corresponding increase in the number of parts that
	// have already been processed, it is possible that the calculated
	// progress could periodically decrease while YouTube processes a video.
	PartsProcessed uint64 `json:"partsProcessed,omitempty,string"`

	// PartsTotal: An estimate of the total number of parts that need to be
	// processed for the video. The number may be updated with more precise
	// estimates while YouTube processes the video.
	PartsTotal uint64 `json:"partsTotal,omitempty,string"`

	// TimeLeftMs: An estimate of the amount of time, in millseconds, that
	// YouTube needs to finish processing the video.
	TimeLeftMs uint64 `json:"timeLeftMs,omitempty,string"`
}

type VideoProjectDetails struct {
	// Tags: A list of project tags associated with the video during the
	// upload.
	Tags []string `json:"tags,omitempty"`
}

type VideoRating struct {
	Rating string `json:"rating,omitempty"`

	VideoId string `json:"videoId,omitempty"`
}

type VideoRecordingDetails struct {
	// Location: The geolocation information associated with the video.
	Location *GeoPoint `json:"location,omitempty"`

	// LocationDescription: The text description of the location where the
	// video was recorded.
	LocationDescription string `json:"locationDescription,omitempty"`

	// RecordingDate: The date and time when the video was recorded. The
	// value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format.
	RecordingDate string `json:"recordingDate,omitempty"`
}

type VideoSnippet struct {
	// CategoryId: The YouTube video category associated with the video.
	CategoryId string `json:"categoryId,omitempty"`

	// ChannelId: The ID that YouTube uses to uniquely identify the channel
	// that the video was uploaded to.
	ChannelId string `json:"channelId,omitempty"`

	// ChannelTitle: Channel title for the channel that the video belongs
	// to.
	ChannelTitle string `json:"channelTitle,omitempty"`

	// Description: The video's description.
	Description string `json:"description,omitempty"`

	// LiveBroadcastContent: Indicates if the video is an upcoming/active
	// live broadcast. Or it's "none" if the video is not an upcoming/active
	// live broadcast.
	LiveBroadcastContent string `json:"liveBroadcastContent,omitempty"`

	// PublishedAt: The date and time that the video was uploaded. The value
	// is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format.
	PublishedAt string `json:"publishedAt,omitempty"`

	// Tags: A list of keyword tags associated with the video. Tags may
	// contain spaces. This field is only visible to the video's uploader.
	Tags []string `json:"tags,omitempty"`

	// Thumbnails: A map of thumbnail images associated with the video. For
	// each object in the map, the key is the name of the thumbnail image,
	// and the value is an object that contains other information about the
	// thumbnail.
	Thumbnails *ThumbnailDetails `json:"thumbnails,omitempty"`

	// Title: The video's title.
	Title string `json:"title,omitempty"`
}

type VideoStatistics struct {
	// CommentCount: The number of comments for the video.
	CommentCount uint64 `json:"commentCount,omitempty,string"`

	// DislikeCount: The number of users who have indicated that they
	// disliked the video by giving it a negative rating.
	DislikeCount uint64 `json:"dislikeCount,omitempty,string"`

	// FavoriteCount: The number of users who currently have the video
	// marked as a favorite video.
	FavoriteCount uint64 `json:"favoriteCount,omitempty,string"`

	// LikeCount: The number of users who have indicated that they liked the
	// video by giving it a positive rating.
	LikeCount uint64 `json:"likeCount,omitempty,string"`

	// ViewCount: The number of times the video has been viewed.
	ViewCount uint64 `json:"viewCount,omitempty,string"`
}

type VideoStatus struct {
	// Embeddable: This value indicates if the video can be embedded on
	// another website.
	Embeddable bool `json:"embeddable,omitempty"`

	// FailureReason: This value explains why a video failed to upload. This
	// property is only present if the uploadStatus property indicates that
	// the upload failed.
	FailureReason string `json:"failureReason,omitempty"`

	// License: The video's license.
	License string `json:"license,omitempty"`

	// PrivacyStatus: The video's privacy status.
	PrivacyStatus string `json:"privacyStatus,omitempty"`

	// PublicStatsViewable: This value indicates if the extended video
	// statistics on the watch page can be viewed by everyone. Note that the
	// view count, likes, etc will still be visible if this is disabled.
	PublicStatsViewable bool `json:"publicStatsViewable,omitempty"`

	// PublishAt: The date and time when the video is scheduled to publish.
	// It can be set only if the privacy status of the video is private. The
	// value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format.
	PublishAt string `json:"publishAt,omitempty"`

	// RejectionReason: This value explains why YouTube rejected an uploaded
	// video. This property is only present if the uploadStatus property
	// indicates that the upload was rejected.
	RejectionReason string `json:"rejectionReason,omitempty"`

	// UploadStatus: The status of the uploaded video.
	UploadStatus string `json:"uploadStatus,omitempty"`
}

type VideoSuggestions struct {
	// EditorSuggestions: A list of video editing operations that might
	// improve the video quality or playback experience of the uploaded
	// video.
	EditorSuggestions []string `json:"editorSuggestions,omitempty"`

	// ProcessingErrors: A list of errors that will prevent YouTube from
	// successfully processing the uploaded video video. These errors
	// indicate that, regardless of the video's current processing status,
	// eventually, that status will almost certainly be failed.
	ProcessingErrors []string `json:"processingErrors,omitempty"`

	// ProcessingHints: A list of suggestions that may improve YouTube's
	// ability to process the video.
	ProcessingHints []string `json:"processingHints,omitempty"`

	// ProcessingWarnings: A list of reasons why YouTube may have difficulty
	// transcoding the uploaded video or that might result in an erroneous
	// transcoding. These warnings are generated before YouTube actually
	// processes the uploaded video file. In addition, they identify issues
	// that are unlikely to cause the video processing to fail but that
	// might cause problems such as sync issues, video artifacts, or a
	// missing audio track.
	ProcessingWarnings []string `json:"processingWarnings,omitempty"`

	// TagSuggestions: A list of keyword tags that could be added to the
	// video's metadata to increase the likelihood that users will locate
	// your video when searching or browsing on YouTube.
	TagSuggestions []*VideoSuggestionsTagSuggestion `json:"tagSuggestions,omitempty"`
}

type VideoSuggestionsTagSuggestion struct {
	// CategoryRestricts: A set of video categories for which the tag is
	// relevant. You can use this information to display appropriate tag
	// suggestions based on the video category that the video uploader
	// associates with the video. By default, tag suggestions are relevant
	// for all categories if there are no restricts defined for the keyword.
	CategoryRestricts []string `json:"categoryRestricts,omitempty"`

	// Tag: The keyword tag suggested for the video.
	Tag string `json:"tag,omitempty"`
}

type VideoTopicDetails struct {
	// RelevantTopicIds: Similar to topic_id, except that these topics are
	// merely relevant to the video. These are topics that may be mentioned
	// in, or appear in the video. You can retrieve information about each
	// topic using Freebase Topic API.
	RelevantTopicIds []string `json:"relevantTopicIds,omitempty"`

	// TopicIds: A list of Freebase topic IDs that are centrally associated
	// with the video. These are topics that are centrally featured in the
	// video, and it can be said that the video is mainly about each of
	// these. You can retrieve information about each topic using the
	// Freebase Topic API.
	TopicIds []string `json:"topicIds,omitempty"`
}

type WatchSettings struct {
	// BackgroundColor: The text color for the video watch page's branded
	// area.
	BackgroundColor string `json:"backgroundColor,omitempty"`

	// FeaturedPlaylistId: An ID that uniquely identifies a playlist that
	// displays next to the video player.
	FeaturedPlaylistId string `json:"featuredPlaylistId,omitempty"`

	// TextColor: The background color for the video watch page's branded
	// area.
	TextColor string `json:"textColor,omitempty"`
}

// method id "youtube.activities.insert":

type ActivitiesInsertCall struct {
	s        *Service
	part     string
	activity *Activity
	opt_     map[string]interface{}
}

// Insert: Posts a bulletin for a specific channel. (The user submitting
// the request must be authorized to act on the channel's
// behalf.)
//
// Note: Even though an activity resource can contain
// information about actions like a user rating a video or marking a
// video as a favorite, you need to use other API methods to generate
// those activity resources. For example, you would use the API's
// videos.rate() method to rate a video and the playlistItems.insert()
// method to mark a video as a favorite.
func (r *ActivitiesService) Insert(part string, activity *Activity) *ActivitiesInsertCall {
	c := &ActivitiesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.activity = activity
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
	params.Set("part", fmt.Sprintf("%v", c.part))
	urls := googleapi.ResolveRelative(c.s.BasePath, "activities")
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
	ret := new(Activity)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Posts a bulletin for a specific channel. (The user submitting the request must be authorized to act on the channel's behalf.)\n\nNote: Even though an activity resource can contain information about actions like a user rating a video or marking a video as a favorite, you need to use other API methods to generate those activity resources. For example, you would use the API's videos.rate() method to rate a video and the playlistItems.insert() method to mark a video as a favorite.",
	//   "httpMethod": "POST",
	//   "id": "youtube.activities.insert",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part names that you can include in the parameter value are snippet and contentDetails.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "activities",
	//   "request": {
	//     "$ref": "Activity"
	//   },
	//   "response": {
	//     "$ref": "Activity"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube"
	//   ]
	// }

}

// method id "youtube.activities.list":

type ActivitiesListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Returns a list of channel activity events that match the
// request criteria. For example, you can retrieve events associated
// with a particular channel, events associated with the user's
// subscriptions and Google+ friends, or the YouTube home page feed,
// which is customized for each user.
func (r *ActivitiesService) List(part string) *ActivitiesListCall {
	c := &ActivitiesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// ChannelId sets the optional parameter "channelId": The channelId
// parameter specifies a unique YouTube channel ID. The API will then
// return a list of that channel's activities.
func (c *ActivitiesListCall) ChannelId(channelId string) *ActivitiesListCall {
	c.opt_["channelId"] = channelId
	return c
}

// Home sets the optional parameter "home": Set this parameter's value
// to true to retrieve the activity feed that displays on the YouTube
// home page for the currently authenticated user.
func (c *ActivitiesListCall) Home(home bool) *ActivitiesListCall {
	c.opt_["home"] = home
	return c
}

// MaxResults sets the optional parameter "maxResults": The maxResults
// parameter specifies the maximum number of items that should be
// returned in the result set.
func (c *ActivitiesListCall) MaxResults(maxResults int64) *ActivitiesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Mine sets the optional parameter "mine": Set this parameter's value
// to true to retrieve a feed of the authenticated user's activities.
func (c *ActivitiesListCall) Mine(mine bool) *ActivitiesListCall {
	c.opt_["mine"] = mine
	return c
}

// PageToken sets the optional parameter "pageToken": The pageToken
// parameter identifies a specific page in the result set that should be
// returned. In an API response, the nextPageToken and prevPageToken
// properties identify other pages that could be retrieved.
func (c *ActivitiesListCall) PageToken(pageToken string) *ActivitiesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// PublishedAfter sets the optional parameter "publishedAfter": The
// publishedAfter parameter specifies the earliest date and time that an
// activity could have occurred for that activity to be included in the
// API response. If the parameter value specifies a day, but not a time,
// then any activities that occurred that day will be included in the
// result set. The value is specified in ISO 8601
// (YYYY-MM-DDThh:mm:ss.sZ) format.
func (c *ActivitiesListCall) PublishedAfter(publishedAfter string) *ActivitiesListCall {
	c.opt_["publishedAfter"] = publishedAfter
	return c
}

// PublishedBefore sets the optional parameter "publishedBefore": The
// publishedBefore parameter specifies the date and time before which an
// activity must have occurred for that activity to be included in the
// API response. If the parameter value specifies a day, but not a time,
// then any activities that occurred that day will be excluded from the
// result set. The value is specified in ISO 8601
// (YYYY-MM-DDThh:mm:ss.sZ) format.
func (c *ActivitiesListCall) PublishedBefore(publishedBefore string) *ActivitiesListCall {
	c.opt_["publishedBefore"] = publishedBefore
	return c
}

// RegionCode sets the optional parameter "regionCode": The regionCode
// parameter instructs the API to return results for the specified
// country. The parameter value is an ISO 3166-1 alpha-2 country code.
// YouTube uses this value when the authorized user's previous activity
// on YouTube does not provide enough information to generate the
// activity feed.
func (c *ActivitiesListCall) RegionCode(regionCode string) *ActivitiesListCall {
	c.opt_["regionCode"] = regionCode
	return c
}

func (c *ActivitiesListCall) Do() (*ActivityListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["channelId"]; ok {
		params.Set("channelId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["home"]; ok {
		params.Set("home", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["mine"]; ok {
		params.Set("mine", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["publishedAfter"]; ok {
		params.Set("publishedAfter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["publishedBefore"]; ok {
		params.Set("publishedBefore", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["regionCode"]; ok {
		params.Set("regionCode", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "activities")
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
	ret := new(ActivityListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of channel activity events that match the request criteria. For example, you can retrieve events associated with a particular channel, events associated with the user's subscriptions and Google+ friends, or the YouTube home page feed, which is customized for each user.",
	//   "httpMethod": "GET",
	//   "id": "youtube.activities.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "channelId": {
	//       "description": "The channelId parameter specifies a unique YouTube channel ID. The API will then return a list of that channel's activities.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "home": {
	//       "description": "Set this parameter's value to true to retrieve the activity feed that displays on the YouTube home page for the currently authenticated user.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "The maxResults parameter specifies the maximum number of items that should be returned in the result set.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "50",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "mine": {
	//       "description": "Set this parameter's value to true to retrieve a feed of the authenticated user's activities.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "pageToken": {
	//       "description": "The pageToken parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more activity resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, and contentDetails.\n\nIf the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a activity resource, the snippet property contains other properties that identify the type of activity, a display title for the activity, and so forth. If you set part=snippet, the API response will also contain all of those nested properties.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "publishedAfter": {
	//       "description": "The publishedAfter parameter specifies the earliest date and time that an activity could have occurred for that activity to be included in the API response. If the parameter value specifies a day, but not a time, then any activities that occurred that day will be included in the result set. The value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format.",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "publishedBefore": {
	//       "description": "The publishedBefore parameter specifies the date and time before which an activity must have occurred for that activity to be included in the API response. If the parameter value specifies a day, but not a time, then any activities that occurred that day will be excluded from the result set. The value is specified in ISO 8601 (YYYY-MM-DDThh:mm:ss.sZ) format.",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "regionCode": {
	//       "description": "The regionCode parameter instructs the API to return results for the specified country. The parameter value is an ISO 3166-1 alpha-2 country code. YouTube uses this value when the authorized user's previous activity on YouTube does not provide enough information to generate the activity feed.",
	//       "location": "query",
	//       "type": "string"
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

// method id "youtube.channelBanners.insert":

type ChannelBannersInsertCall struct {
	s                     *Service
	channelbannerresource *ChannelBannerResource
	opt_                  map[string]interface{}
	media_                io.Reader
}

// Insert: Uploads a channel banner image to YouTube. This method
// represents the first two steps in a three-step process to update the
// banner image for a channel:
//
// - Call the channelBanners.insert method
// to upload the binary image data to YouTube. The image must have a
// 16:9 aspect ratio and be at least 2120x1192 pixels.
// - Extract the url
// property's value from the response that the API returns for step 1.
// -
// Call the channels.update method to update the channel's branding
// settings. Set the brandingSettings.image.bannerExternalUrl property's
// value to the URL obtained in step 2.
func (r *ChannelBannersService) Insert(channelbannerresource *ChannelBannerResource) *ChannelBannersInsertCall {
	c := &ChannelBannersInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.channelbannerresource = channelbannerresource
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *ChannelBannersInsertCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *ChannelBannersInsertCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}
func (c *ChannelBannersInsertCall) Media(r io.Reader) *ChannelBannersInsertCall {
	c.media_ = r
	return c
}

func (c *ChannelBannersInsertCall) Do() (*ChannelBannerResource, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.channelbannerresource)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "channelBanners/insert")
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
	ret := new(ChannelBannerResource)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Uploads a channel banner image to YouTube. This method represents the first two steps in a three-step process to update the banner image for a channel:\n\n- Call the channelBanners.insert method to upload the binary image data to YouTube. The image must have a 16:9 aspect ratio and be at least 2120x1192 pixels.\n- Extract the url property's value from the response that the API returns for step 1.\n- Call the channels.update method to update the channel's branding settings. Set the brandingSettings.image.bannerExternalUrl property's value to the URL obtained in step 2.",
	//   "httpMethod": "POST",
	//   "id": "youtube.channelBanners.insert",
	//   "mediaUpload": {
	//     "accept": [
	//       "application/octet-stream",
	//       "image/jpeg",
	//       "image/png"
	//     ],
	//     "maxSize": "6MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/youtube/v3/channelBanners/insert"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/youtube/v3/channelBanners/insert"
	//       }
	//     }
	//   },
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "channelBanners/insert",
	//   "request": {
	//     "$ref": "ChannelBannerResource"
	//   },
	//   "response": {
	//     "$ref": "ChannelBannerResource"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.upload"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "youtube.channelSections.delete":

type ChannelSectionsDeleteCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Delete: Deletes a channelSection.
func (r *ChannelSectionsService) Delete(id string) *ChannelSectionsDeleteCall {
	c := &ChannelSectionsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

func (c *ChannelSectionsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	urls := googleapi.ResolveRelative(c.s.BasePath, "channelSections")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
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
	//   "description": "Deletes a channelSection.",
	//   "httpMethod": "DELETE",
	//   "id": "youtube.channelSections.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies the YouTube channelSection ID for the resource that is being deleted. In a channelSection resource, the id property specifies the YouTube channelSection ID.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "channelSections",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.channelSections.insert":

type ChannelSectionsInsertCall struct {
	s              *Service
	part           string
	channelsection *ChannelSection
	opt_           map[string]interface{}
}

// Insert: Adds a channelSection for the authenticated user's channel.
func (r *ChannelSectionsService) Insert(part string, channelsection *ChannelSection) *ChannelSectionsInsertCall {
	c := &ChannelSectionsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.channelsection = channelsection
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *ChannelSectionsInsertCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *ChannelSectionsInsertCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

func (c *ChannelSectionsInsertCall) Do() (*ChannelSection, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.channelsection)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "channelSections")
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
	ret := new(ChannelSection)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds a channelSection for the authenticated user's channel.",
	//   "httpMethod": "POST",
	//   "id": "youtube.channelSections.insert",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part names that you can include in the parameter value are snippet and contentDetails.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "channelSections",
	//   "request": {
	//     "$ref": "ChannelSection"
	//   },
	//   "response": {
	//     "$ref": "ChannelSection"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.channelSections.list":

type ChannelSectionsListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Returns channelSection resources that match the API request
// criteria.
func (r *ChannelSectionsService) List(part string) *ChannelSectionsListCall {
	c := &ChannelSectionsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// ChannelId sets the optional parameter "channelId": The channelId
// parameter specifies a YouTube channel ID. The API will only return
// that channel's channelSections.
func (c *ChannelSectionsListCall) ChannelId(channelId string) *ChannelSectionsListCall {
	c.opt_["channelId"] = channelId
	return c
}

// Id sets the optional parameter "id": The id parameter specifies a
// comma-separated list of the YouTube channelSection ID(s) for the
// resource(s) that are being retrieved. In a channelSection resource,
// the id property specifies the YouTube channelSection ID.
func (c *ChannelSectionsListCall) Id(id string) *ChannelSectionsListCall {
	c.opt_["id"] = id
	return c
}

// Mine sets the optional parameter "mine": Set this parameter's value
// to true to retrieve a feed of the authenticated user's
// channelSections.
func (c *ChannelSectionsListCall) Mine(mine bool) *ChannelSectionsListCall {
	c.opt_["mine"] = mine
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *ChannelSectionsListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *ChannelSectionsListCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

func (c *ChannelSectionsListCall) Do() (*ChannelSectionListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["channelId"]; ok {
		params.Set("channelId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["id"]; ok {
		params.Set("id", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["mine"]; ok {
		params.Set("mine", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "channelSections")
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
	ret := new(ChannelSectionListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns channelSection resources that match the API request criteria.",
	//   "httpMethod": "GET",
	//   "id": "youtube.channelSections.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "channelId": {
	//       "description": "The channelId parameter specifies a YouTube channel ID. The API will only return that channel's channelSections.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "The id parameter specifies a comma-separated list of the YouTube channelSection ID(s) for the resource(s) that are being retrieved. In a channelSection resource, the id property specifies the YouTube channelSection ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "mine": {
	//       "description": "Set this parameter's value to true to retrieve a feed of the authenticated user's channelSections.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more channelSection resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, and contentDetails.\n\nIf the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a channelSection resource, the snippet property contains other properties, such as a display title for the channelSection. If you set part=snippet, the API response will also contain all of those nested properties.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "channelSections",
	//   "response": {
	//     "$ref": "ChannelSectionListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.channelSections.update":

type ChannelSectionsUpdateCall struct {
	s              *Service
	part           string
	channelsection *ChannelSection
	opt_           map[string]interface{}
}

// Update: Update a channelSection.
func (r *ChannelSectionsService) Update(part string, channelsection *ChannelSection) *ChannelSectionsUpdateCall {
	c := &ChannelSectionsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.channelsection = channelsection
	return c
}

func (c *ChannelSectionsUpdateCall) Do() (*ChannelSection, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.channelsection)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	urls := googleapi.ResolveRelative(c.s.BasePath, "channelSections")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
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
	ret := new(ChannelSection)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update a channelSection.",
	//   "httpMethod": "PUT",
	//   "id": "youtube.channelSections.update",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part names that you can include in the parameter value are snippet and contentDetails.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "channelSections",
	//   "request": {
	//     "$ref": "ChannelSection"
	//   },
	//   "response": {
	//     "$ref": "ChannelSection"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.channels.list":

type ChannelsListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Returns a collection of zero or more channel resources that
// match the request criteria.
func (r *ChannelsService) List(part string) *ChannelsListCall {
	c := &ChannelsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// CategoryId sets the optional parameter "categoryId": The categoryId
// parameter specifies a YouTube guide category, thereby requesting
// YouTube channels associated with that category.
func (c *ChannelsListCall) CategoryId(categoryId string) *ChannelsListCall {
	c.opt_["categoryId"] = categoryId
	return c
}

// ForUsername sets the optional parameter "forUsername": The
// forUsername parameter specifies a YouTube username, thereby
// requesting the channel associated with that username.
func (c *ChannelsListCall) ForUsername(forUsername string) *ChannelsListCall {
	c.opt_["forUsername"] = forUsername
	return c
}

// Id sets the optional parameter "id": The id parameter specifies a
// comma-separated list of the YouTube channel ID(s) for the resource(s)
// that are being retrieved. In a channel resource, the id property
// specifies the channel's YouTube channel ID.
func (c *ChannelsListCall) Id(id string) *ChannelsListCall {
	c.opt_["id"] = id
	return c
}

// ManagedByMe sets the optional parameter "managedByMe": Set this
// parameter's value to true to instruct the API to only return channels
// managed by the content owner that the onBehalfOfContentOwner
// parameter specifies. The user must be authenticated as a CMS account
// linked to the specified content owner and onBehalfOfContentOwner must
// be provided.
func (c *ChannelsListCall) ManagedByMe(managedByMe bool) *ChannelsListCall {
	c.opt_["managedByMe"] = managedByMe
	return c
}

// MaxResults sets the optional parameter "maxResults": The maxResults
// parameter specifies the maximum number of items that should be
// returned in the result set.
func (c *ChannelsListCall) MaxResults(maxResults int64) *ChannelsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Mine sets the optional parameter "mine": Set this parameter's value
// to true to instruct the API to only return channels owned by the
// authenticated user.
func (c *ChannelsListCall) Mine(mine bool) *ChannelsListCall {
	c.opt_["mine"] = mine
	return c
}

// MySubscribers sets the optional parameter "mySubscribers": Set this
// parameter's value to true to retrieve a list of channels that
// subscribed to the authenticated user's channel.
func (c *ChannelsListCall) MySubscribers(mySubscribers bool) *ChannelsListCall {
	c.opt_["mySubscribers"] = mySubscribers
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": The onBehalfOfContentOwner parameter
// indicates that the authenticated user is acting on behalf of the
// content owner specified in the parameter value. This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and get access to all their video and channel data, without
// having to provide authentication credentials for each individual
// channel. The actual CMS account that the user authenticates with
// needs to be linked to the specified YouTube content owner.
func (c *ChannelsListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *ChannelsListCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// PageToken sets the optional parameter "pageToken": The pageToken
// parameter identifies a specific page in the result set that should be
// returned. In an API response, the nextPageToken and prevPageToken
// properties identify other pages that could be retrieved.
func (c *ChannelsListCall) PageToken(pageToken string) *ChannelsListCall {
	c.opt_["pageToken"] = pageToken
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
	if v, ok := c.opt_["forUsername"]; ok {
		params.Set("forUsername", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["id"]; ok {
		params.Set("id", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["managedByMe"]; ok {
		params.Set("managedByMe", fmt.Sprintf("%v", v))
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
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "channels")
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
	ret := new(ChannelListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a collection of zero or more channel resources that match the request criteria.",
	//   "httpMethod": "GET",
	//   "id": "youtube.channels.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "categoryId": {
	//       "description": "The categoryId parameter specifies a YouTube guide category, thereby requesting YouTube channels associated with that category.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "forUsername": {
	//       "description": "The forUsername parameter specifies a YouTube username, thereby requesting the channel associated with that username.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "The id parameter specifies a comma-separated list of the YouTube channel ID(s) for the resource(s) that are being retrieved. In a channel resource, the id property specifies the channel's YouTube channel ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "managedByMe": {
	//       "description": "Set this parameter's value to true to instruct the API to only return channels managed by the content owner that the onBehalfOfContentOwner parameter specifies. The user must be authenticated as a CMS account linked to the specified content owner and onBehalfOfContentOwner must be provided.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "The maxResults parameter specifies the maximum number of items that should be returned in the result set.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "50",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "mine": {
	//       "description": "Set this parameter's value to true to instruct the API to only return channels owned by the authenticated user.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "mySubscribers": {
	//       "description": "Set this parameter's value to true to retrieve a list of channels that subscribed to the authenticated user's channel.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "The onBehalfOfContentOwner parameter indicates that the authenticated user is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The actual CMS account that the user authenticates with needs to be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The pageToken parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more channel resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, contentDetails, statistics, topicDetails, and invideoPromotion.\n\nIf the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a channel resource, the contentDetails property contains other properties, such as the uploads properties. As such, if you set part=contentDetails, the API response will also contain all of those nested properties.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "channels",
	//   "response": {
	//     "$ref": "ChannelListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner",
	//     "https://www.googleapis.com/auth/youtubepartner-channel-audit"
	//   ]
	// }

}

// method id "youtube.channels.update":

type ChannelsUpdateCall struct {
	s       *Service
	part    string
	channel *Channel
	opt_    map[string]interface{}
}

// Update: Updates a channel's metadata.
func (r *ChannelsService) Update(part string, channel *Channel) *ChannelsUpdateCall {
	c := &ChannelsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.channel = channel
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": The onBehalfOfContentOwner parameter
// indicates that the authenticated user is acting on behalf of the
// content owner specified in the parameter value. This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and get access to all their video and channel data, without
// having to provide authentication credentials for each individual
// channel. The actual CMS account that the user authenticates with
// needs to be linked to the specified YouTube content owner.
func (c *ChannelsUpdateCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *ChannelsUpdateCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

func (c *ChannelsUpdateCall) Do() (*Channel, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.channel)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "channels")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
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
	ret := new(Channel)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a channel's metadata.",
	//   "httpMethod": "PUT",
	//   "id": "youtube.channels.update",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "The onBehalfOfContentOwner parameter indicates that the authenticated user is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The actual CMS account that the user authenticates with needs to be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part names that you can include in the parameter value are id and invideoPromotion.\n\nNote that this method will override the existing values for all of the mutable properties that are contained in any parts that the parameter value specifies.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "channels",
	//   "request": {
	//     "$ref": "Channel"
	//   },
	//   "response": {
	//     "$ref": "Channel"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.guideCategories.list":

type GuideCategoriesListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Returns a list of categories that can be associated with
// YouTube channels.
func (r *GuideCategoriesService) List(part string) *GuideCategoriesListCall {
	c := &GuideCategoriesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// Hl sets the optional parameter "hl": The hl parameter specifies the
// language that will be used for text values in the API response.
func (c *GuideCategoriesListCall) Hl(hl string) *GuideCategoriesListCall {
	c.opt_["hl"] = hl
	return c
}

// Id sets the optional parameter "id": The id parameter specifies a
// comma-separated list of the YouTube channel category ID(s) for the
// resource(s) that are being retrieved. In a guideCategory resource,
// the id property specifies the YouTube channel category ID.
func (c *GuideCategoriesListCall) Id(id string) *GuideCategoriesListCall {
	c.opt_["id"] = id
	return c
}

// RegionCode sets the optional parameter "regionCode": The regionCode
// parameter instructs the API to return the list of guide categories
// available in the specified country. The parameter value is an ISO
// 3166-1 alpha-2 country code.
func (c *GuideCategoriesListCall) RegionCode(regionCode string) *GuideCategoriesListCall {
	c.opt_["regionCode"] = regionCode
	return c
}

func (c *GuideCategoriesListCall) Do() (*GuideCategoryListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["id"]; ok {
		params.Set("id", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["regionCode"]; ok {
		params.Set("regionCode", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "guideCategories")
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
	ret := new(GuideCategoryListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of categories that can be associated with YouTube channels.",
	//   "httpMethod": "GET",
	//   "id": "youtube.guideCategories.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "hl": {
	//       "default": "en-US",
	//       "description": "The hl parameter specifies the language that will be used for text values in the API response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "The id parameter specifies a comma-separated list of the YouTube channel category ID(s) for the resource(s) that are being retrieved. In a guideCategory resource, the id property specifies the YouTube channel category ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more guideCategory resource properties that the API response will include. The part names that you can include in the parameter value are id and snippet.\n\nIf the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a guideCategory resource, the snippet property contains other properties, such as the category's title. If you set part=snippet, the API response will also contain all of those nested properties.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "regionCode": {
	//       "description": "The regionCode parameter instructs the API to return the list of guide categories available in the specified country. The parameter value is an ISO 3166-1 alpha-2 country code.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "guideCategories",
	//   "response": {
	//     "$ref": "GuideCategoryListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.i18nLanguages.list":

type I18nLanguagesListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Returns a list of supported languages.
func (r *I18nLanguagesService) List(part string) *I18nLanguagesListCall {
	c := &I18nLanguagesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// Hl sets the optional parameter "hl": The hl parameter specifies the
// language that should be used for text values in the API response.
func (c *I18nLanguagesListCall) Hl(hl string) *I18nLanguagesListCall {
	c.opt_["hl"] = hl
	return c
}

func (c *I18nLanguagesListCall) Do() (*I18nLanguageListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "i18nLanguages")
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
	ret := new(I18nLanguageListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of supported languages.",
	//   "httpMethod": "GET",
	//   "id": "youtube.i18nLanguages.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "hl": {
	//       "default": "en_US",
	//       "description": "The hl parameter specifies the language that should be used for text values in the API response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more i18nLanguage resource properties that the API response will include. The part names that you can include in the parameter value are id and snippet.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "i18nLanguages",
	//   "response": {
	//     "$ref": "I18nLanguageListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.i18nRegions.list":

type I18nRegionsListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Returns a list of supported regions.
func (r *I18nRegionsService) List(part string) *I18nRegionsListCall {
	c := &I18nRegionsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// Hl sets the optional parameter "hl": The hl parameter specifies the
// language that should be used for text values in the API response.
func (c *I18nRegionsListCall) Hl(hl string) *I18nRegionsListCall {
	c.opt_["hl"] = hl
	return c
}

func (c *I18nRegionsListCall) Do() (*I18nRegionListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "i18nRegions")
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
	ret := new(I18nRegionListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of supported regions.",
	//   "httpMethod": "GET",
	//   "id": "youtube.i18nRegions.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "hl": {
	//       "default": "en_US",
	//       "description": "The hl parameter specifies the language that should be used for text values in the API response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more i18nRegion resource properties that the API response will include. The part names that you can include in the parameter value are id and snippet.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "i18nRegions",
	//   "response": {
	//     "$ref": "I18nRegionListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.liveBroadcasts.bind":

type LiveBroadcastsBindCall struct {
	s    *Service
	id   string
	part string
	opt_ map[string]interface{}
}

// Bind: Binds a YouTube broadcast to a stream or removes an existing
// binding between a broadcast and a stream. A broadcast can only be
// bound to one video stream.
func (r *LiveBroadcastsService) Bind(id string, part string) *LiveBroadcastsBindCall {
	c := &LiveBroadcastsBindCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.part = part
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *LiveBroadcastsBindCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *LiveBroadcastsBindCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *LiveBroadcastsBindCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *LiveBroadcastsBindCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

// StreamId sets the optional parameter "streamId": The streamId
// parameter specifies the unique ID of the video stream that is being
// bound to a broadcast. If this parameter is omitted, the API will
// remove any existing binding between the broadcast and a video stream.
func (c *LiveBroadcastsBindCall) StreamId(streamId string) *LiveBroadcastsBindCall {
	c.opt_["streamId"] = streamId
	return c
}

func (c *LiveBroadcastsBindCall) Do() (*LiveBroadcast, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["streamId"]; ok {
		params.Set("streamId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "liveBroadcasts/bind")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(LiveBroadcast)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Binds a YouTube broadcast to a stream or removes an existing binding between a broadcast and a stream. A broadcast can only be bound to one video stream.",
	//   "httpMethod": "POST",
	//   "id": "youtube.liveBroadcasts.bind",
	//   "parameterOrder": [
	//     "id",
	//     "part"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies the unique ID of the broadcast that is being bound to a video stream.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more liveBroadcast resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, contentDetails, and status.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "streamId": {
	//       "description": "The streamId parameter specifies the unique ID of the video stream that is being bound to a broadcast. If this parameter is omitted, the API will remove any existing binding between the broadcast and a video stream.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "liveBroadcasts/bind",
	//   "response": {
	//     "$ref": "LiveBroadcast"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube"
	//   ]
	// }

}

// method id "youtube.liveBroadcasts.control":

type LiveBroadcastsControlCall struct {
	s    *Service
	id   string
	part string
	opt_ map[string]interface{}
}

// Control: Controls the settings for a slate that can be displayed in
// the broadcast stream.
func (r *LiveBroadcastsService) Control(id string, part string) *LiveBroadcastsControlCall {
	c := &LiveBroadcastsControlCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.part = part
	return c
}

// DisplaySlate sets the optional parameter "displaySlate": The
// displaySlate parameter specifies whether the slate is being enabled
// or disabled.
func (c *LiveBroadcastsControlCall) DisplaySlate(displaySlate bool) *LiveBroadcastsControlCall {
	c.opt_["displaySlate"] = displaySlate
	return c
}

// OffsetTimeMs sets the optional parameter "offsetTimeMs": The
// offsetTimeMs parameter specifies a positive time offset when the
// specified slate change will occur. The value is measured in
// milliseconds from the beginning of the broadcast's monitor stream,
// which is the time that the testing phase for the broadcast began.
// Even though it is specified in milliseconds, the value is actually an
// approximation, and YouTube completes the requested action as closely
// as possible to that time.
//
// If you do not specify a value for this
// parameter, then YouTube performs the action as soon as possible. See
// the Getting started guide for more details.
//
// Important: You should
// only specify a value for this parameter if your broadcast stream is
// delayed.
func (c *LiveBroadcastsControlCall) OffsetTimeMs(offsetTimeMs uint64) *LiveBroadcastsControlCall {
	c.opt_["offsetTimeMs"] = offsetTimeMs
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *LiveBroadcastsControlCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *LiveBroadcastsControlCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *LiveBroadcastsControlCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *LiveBroadcastsControlCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

// Walltime sets the optional parameter "walltime": The walltime
// parameter specifies the wall clock time at which the specified slate
// change will occur.
func (c *LiveBroadcastsControlCall) Walltime(walltime string) *LiveBroadcastsControlCall {
	c.opt_["walltime"] = walltime
	return c
}

func (c *LiveBroadcastsControlCall) Do() (*LiveBroadcast, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["displaySlate"]; ok {
		params.Set("displaySlate", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["offsetTimeMs"]; ok {
		params.Set("offsetTimeMs", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["walltime"]; ok {
		params.Set("walltime", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "liveBroadcasts/control")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(LiveBroadcast)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Controls the settings for a slate that can be displayed in the broadcast stream.",
	//   "httpMethod": "POST",
	//   "id": "youtube.liveBroadcasts.control",
	//   "parameterOrder": [
	//     "id",
	//     "part"
	//   ],
	//   "parameters": {
	//     "displaySlate": {
	//       "description": "The displaySlate parameter specifies whether the slate is being enabled or disabled.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "id": {
	//       "description": "The id parameter specifies the YouTube live broadcast ID that uniquely identifies the broadcast in which the slate is being updated.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "offsetTimeMs": {
	//       "description": "The offsetTimeMs parameter specifies a positive time offset when the specified slate change will occur. The value is measured in milliseconds from the beginning of the broadcast's monitor stream, which is the time that the testing phase for the broadcast began. Even though it is specified in milliseconds, the value is actually an approximation, and YouTube completes the requested action as closely as possible to that time.\n\nIf you do not specify a value for this parameter, then YouTube performs the action as soon as possible. See the Getting started guide for more details.\n\nImportant: You should only specify a value for this parameter if your broadcast stream is delayed.",
	//       "format": "uint64",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more liveBroadcast resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, contentDetails, and status.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "walltime": {
	//       "description": "The walltime parameter specifies the wall clock time at which the specified slate change will occur.",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "liveBroadcasts/control",
	//   "response": {
	//     "$ref": "LiveBroadcast"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube"
	//   ]
	// }

}

// method id "youtube.liveBroadcasts.delete":

type LiveBroadcastsDeleteCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Delete: Deletes a broadcast.
func (r *LiveBroadcastsService) Delete(id string) *LiveBroadcastsDeleteCall {
	c := &LiveBroadcastsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *LiveBroadcastsDeleteCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *LiveBroadcastsDeleteCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *LiveBroadcastsDeleteCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *LiveBroadcastsDeleteCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

func (c *LiveBroadcastsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "liveBroadcasts")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
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
	//   "description": "Deletes a broadcast.",
	//   "httpMethod": "DELETE",
	//   "id": "youtube.liveBroadcasts.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies the YouTube live broadcast ID for the resource that is being deleted.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "liveBroadcasts",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube"
	//   ]
	// }

}

// method id "youtube.liveBroadcasts.insert":

type LiveBroadcastsInsertCall struct {
	s             *Service
	part          string
	livebroadcast *LiveBroadcast
	opt_          map[string]interface{}
}

// Insert: Creates a broadcast.
func (r *LiveBroadcastsService) Insert(part string, livebroadcast *LiveBroadcast) *LiveBroadcastsInsertCall {
	c := &LiveBroadcastsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.livebroadcast = livebroadcast
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *LiveBroadcastsInsertCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *LiveBroadcastsInsertCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *LiveBroadcastsInsertCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *LiveBroadcastsInsertCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

func (c *LiveBroadcastsInsertCall) Do() (*LiveBroadcast, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.livebroadcast)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "liveBroadcasts")
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
	ret := new(LiveBroadcast)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a broadcast.",
	//   "httpMethod": "POST",
	//   "id": "youtube.liveBroadcasts.insert",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part properties that you can include in the parameter value are id, snippet, contentDetails, and status.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "liveBroadcasts",
	//   "request": {
	//     "$ref": "LiveBroadcast"
	//   },
	//   "response": {
	//     "$ref": "LiveBroadcast"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube"
	//   ]
	// }

}

// method id "youtube.liveBroadcasts.list":

type LiveBroadcastsListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Returns a list of YouTube broadcasts that match the API request
// parameters.
func (r *LiveBroadcastsService) List(part string) *LiveBroadcastsListCall {
	c := &LiveBroadcastsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// BroadcastStatus sets the optional parameter "broadcastStatus": The
// broadcastStatus parameter filters the API response to only include
// broadcasts with the specified status.
func (c *LiveBroadcastsListCall) BroadcastStatus(broadcastStatus string) *LiveBroadcastsListCall {
	c.opt_["broadcastStatus"] = broadcastStatus
	return c
}

// Id sets the optional parameter "id": The id parameter specifies a
// comma-separated list of YouTube broadcast IDs that identify the
// broadcasts being retrieved. In a liveBroadcast resource, the id
// property specifies the broadcast's ID.
func (c *LiveBroadcastsListCall) Id(id string) *LiveBroadcastsListCall {
	c.opt_["id"] = id
	return c
}

// MaxResults sets the optional parameter "maxResults": The maxResults
// parameter specifies the maximum number of items that should be
// returned in the result set.
func (c *LiveBroadcastsListCall) MaxResults(maxResults int64) *LiveBroadcastsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Mine sets the optional parameter "mine": The mine parameter can be
// used to instruct the API to only return broadcasts owned by the
// authenticated user. Set the parameter value to true to only retrieve
// your own broadcasts.
func (c *LiveBroadcastsListCall) Mine(mine bool) *LiveBroadcastsListCall {
	c.opt_["mine"] = mine
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *LiveBroadcastsListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *LiveBroadcastsListCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *LiveBroadcastsListCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *LiveBroadcastsListCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

// PageToken sets the optional parameter "pageToken": The pageToken
// parameter identifies a specific page in the result set that should be
// returned. In an API response, the nextPageToken and prevPageToken
// properties identify other pages that could be retrieved.
func (c *LiveBroadcastsListCall) PageToken(pageToken string) *LiveBroadcastsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *LiveBroadcastsListCall) Do() (*LiveBroadcastListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["broadcastStatus"]; ok {
		params.Set("broadcastStatus", fmt.Sprintf("%v", v))
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
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "liveBroadcasts")
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
	ret := new(LiveBroadcastListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of YouTube broadcasts that match the API request parameters.",
	//   "httpMethod": "GET",
	//   "id": "youtube.liveBroadcasts.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "broadcastStatus": {
	//       "description": "The broadcastStatus parameter filters the API response to only include broadcasts with the specified status.",
	//       "enum": [
	//         "active",
	//         "all",
	//         "completed",
	//         "upcoming"
	//       ],
	//       "enumDescriptions": [
	//         "Return current live broadcasts.",
	//         "Return all broadcasts.",
	//         "Return broadcasts that have already ended.",
	//         "Return broadcasts that have not yet started."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "The id parameter specifies a comma-separated list of YouTube broadcast IDs that identify the broadcasts being retrieved. In a liveBroadcast resource, the id property specifies the broadcast's ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "The maxResults parameter specifies the maximum number of items that should be returned in the result set.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "50",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "mine": {
	//       "description": "The mine parameter can be used to instruct the API to only return broadcasts owned by the authenticated user. Set the parameter value to true to only retrieve your own broadcasts.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The pageToken parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more liveBroadcast resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, contentDetails, and status.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "liveBroadcasts",
	//   "response": {
	//     "$ref": "LiveBroadcastListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly"
	//   ]
	// }

}

// method id "youtube.liveBroadcasts.transition":

type LiveBroadcastsTransitionCall struct {
	s               *Service
	broadcastStatus string
	id              string
	part            string
	opt_            map[string]interface{}
}

// Transition: Changes the status of a YouTube live broadcast and
// initiates any processes associated with the new status. For example,
// when you transition a broadcast's status to testing, YouTube starts
// to transmit video to that broadcast's monitor stream. Before calling
// this method, you should confirm that the value of the
// status.streamStatus property for the stream bound to your broadcast
// is active.
func (r *LiveBroadcastsService) Transition(broadcastStatus string, id string, part string) *LiveBroadcastsTransitionCall {
	c := &LiveBroadcastsTransitionCall{s: r.s, opt_: make(map[string]interface{})}
	c.broadcastStatus = broadcastStatus
	c.id = id
	c.part = part
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *LiveBroadcastsTransitionCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *LiveBroadcastsTransitionCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *LiveBroadcastsTransitionCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *LiveBroadcastsTransitionCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

func (c *LiveBroadcastsTransitionCall) Do() (*LiveBroadcast, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("broadcastStatus", fmt.Sprintf("%v", c.broadcastStatus))
	params.Set("id", fmt.Sprintf("%v", c.id))
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "liveBroadcasts/transition")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	ret := new(LiveBroadcast)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Changes the status of a YouTube live broadcast and initiates any processes associated with the new status. For example, when you transition a broadcast's status to testing, YouTube starts to transmit video to that broadcast's monitor stream. Before calling this method, you should confirm that the value of the status.streamStatus property for the stream bound to your broadcast is active.",
	//   "httpMethod": "POST",
	//   "id": "youtube.liveBroadcasts.transition",
	//   "parameterOrder": [
	//     "broadcastStatus",
	//     "id",
	//     "part"
	//   ],
	//   "parameters": {
	//     "broadcastStatus": {
	//       "description": "The broadcastStatus parameter identifies the state to which the broadcast is changing. Note that to transition a broadcast to either the testing or live state, the status.streamStatus must be active for the stream that the broadcast is bound to.",
	//       "enum": [
	//         "complete",
	//         "live",
	//         "testing"
	//       ],
	//       "enumDescriptions": [
	//         "The broadcast is over. YouTube stops transmitting video.",
	//         "The broadcast is visible to its audience. YouTube transmits video to the broadcast's monitor stream and its broadcast stream.",
	//         "Start testing the broadcast. YouTube transmits video to the broadcast's monitor stream. Note that you can only transition a broadcast to the testing state if its contentDetails.monitorStream.enableMonitorStream property is set to true."
	//       ],
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "The id parameter specifies the unique ID of the broadcast that is transitioning to another status.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more liveBroadcast resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, contentDetails, and status.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "liveBroadcasts/transition",
	//   "response": {
	//     "$ref": "LiveBroadcast"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube"
	//   ]
	// }

}

// method id "youtube.liveBroadcasts.update":

type LiveBroadcastsUpdateCall struct {
	s             *Service
	part          string
	livebroadcast *LiveBroadcast
	opt_          map[string]interface{}
}

// Update: Updates a broadcast. For example, you could modify the
// broadcast settings defined in the liveBroadcast resource's
// contentDetails object.
func (r *LiveBroadcastsService) Update(part string, livebroadcast *LiveBroadcast) *LiveBroadcastsUpdateCall {
	c := &LiveBroadcastsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.livebroadcast = livebroadcast
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *LiveBroadcastsUpdateCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *LiveBroadcastsUpdateCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *LiveBroadcastsUpdateCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *LiveBroadcastsUpdateCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

func (c *LiveBroadcastsUpdateCall) Do() (*LiveBroadcast, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.livebroadcast)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "liveBroadcasts")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
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
	ret := new(LiveBroadcast)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a broadcast. For example, you could modify the broadcast settings defined in the liveBroadcast resource's contentDetails object.",
	//   "httpMethod": "PUT",
	//   "id": "youtube.liveBroadcasts.update",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part properties that you can include in the parameter value are id, snippet, contentDetails, and status.\n\nNote that this method will override the existing values for all of the mutable properties that are contained in any parts that the parameter value specifies. For example, a broadcast's privacy status is defined in the status part. As such, if your request is updating a private or unlisted broadcast, and the request's part parameter value includes the status part, the broadcast's privacy setting will be updated to whatever value the request body specifies. If the request body does not specify a value, the existing privacy setting will be removed and the broadcast will revert to the default privacy setting.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "liveBroadcasts",
	//   "request": {
	//     "$ref": "LiveBroadcast"
	//   },
	//   "response": {
	//     "$ref": "LiveBroadcast"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube"
	//   ]
	// }

}

// method id "youtube.liveStreams.delete":

type LiveStreamsDeleteCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Delete: Deletes a video stream.
func (r *LiveStreamsService) Delete(id string) *LiveStreamsDeleteCall {
	c := &LiveStreamsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *LiveStreamsDeleteCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *LiveStreamsDeleteCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *LiveStreamsDeleteCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *LiveStreamsDeleteCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

func (c *LiveStreamsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "liveStreams")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
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
	//   "description": "Deletes a video stream.",
	//   "httpMethod": "DELETE",
	//   "id": "youtube.liveStreams.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies the YouTube live stream ID for the resource that is being deleted.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "liveStreams",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube"
	//   ]
	// }

}

// method id "youtube.liveStreams.insert":

type LiveStreamsInsertCall struct {
	s          *Service
	part       string
	livestream *LiveStream
	opt_       map[string]interface{}
}

// Insert: Creates a video stream. The stream enables you to send your
// video to YouTube, which can then broadcast the video to your
// audience.
func (r *LiveStreamsService) Insert(part string, livestream *LiveStream) *LiveStreamsInsertCall {
	c := &LiveStreamsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.livestream = livestream
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *LiveStreamsInsertCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *LiveStreamsInsertCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *LiveStreamsInsertCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *LiveStreamsInsertCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

func (c *LiveStreamsInsertCall) Do() (*LiveStream, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.livestream)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "liveStreams")
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
	ret := new(LiveStream)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a video stream. The stream enables you to send your video to YouTube, which can then broadcast the video to your audience.",
	//   "httpMethod": "POST",
	//   "id": "youtube.liveStreams.insert",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part properties that you can include in the parameter value are id, snippet, cdn, and status.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "liveStreams",
	//   "request": {
	//     "$ref": "LiveStream"
	//   },
	//   "response": {
	//     "$ref": "LiveStream"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube"
	//   ]
	// }

}

// method id "youtube.liveStreams.list":

type LiveStreamsListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Returns a list of video streams that match the API request
// parameters.
func (r *LiveStreamsService) List(part string) *LiveStreamsListCall {
	c := &LiveStreamsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// Id sets the optional parameter "id": The id parameter specifies a
// comma-separated list of YouTube stream IDs that identify the streams
// being retrieved. In a liveStream resource, the id property specifies
// the stream's ID.
func (c *LiveStreamsListCall) Id(id string) *LiveStreamsListCall {
	c.opt_["id"] = id
	return c
}

// MaxResults sets the optional parameter "maxResults": The maxResults
// parameter specifies the maximum number of items that should be
// returned in the result set. Acceptable values are 0 to 50, inclusive.
// The default value is 5.
func (c *LiveStreamsListCall) MaxResults(maxResults int64) *LiveStreamsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Mine sets the optional parameter "mine": The mine parameter can be
// used to instruct the API to only return streams owned by the
// authenticated user. Set the parameter value to true to only retrieve
// your own streams.
func (c *LiveStreamsListCall) Mine(mine bool) *LiveStreamsListCall {
	c.opt_["mine"] = mine
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *LiveStreamsListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *LiveStreamsListCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *LiveStreamsListCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *LiveStreamsListCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

// PageToken sets the optional parameter "pageToken": The pageToken
// parameter identifies a specific page in the result set that should be
// returned. In an API response, the nextPageToken and prevPageToken
// properties identify other pages that could be retrieved.
func (c *LiveStreamsListCall) PageToken(pageToken string) *LiveStreamsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *LiveStreamsListCall) Do() (*LiveStreamListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["id"]; ok {
		params.Set("id", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["mine"]; ok {
		params.Set("mine", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "liveStreams")
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
	ret := new(LiveStreamListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of video streams that match the API request parameters.",
	//   "httpMethod": "GET",
	//   "id": "youtube.liveStreams.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies a comma-separated list of YouTube stream IDs that identify the streams being retrieved. In a liveStream resource, the id property specifies the stream's ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "The maxResults parameter specifies the maximum number of items that should be returned in the result set. Acceptable values are 0 to 50, inclusive. The default value is 5.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "50",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "mine": {
	//       "description": "The mine parameter can be used to instruct the API to only return streams owned by the authenticated user. Set the parameter value to true to only retrieve your own streams.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The pageToken parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more liveStream resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, cdn, and status.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "liveStreams",
	//   "response": {
	//     "$ref": "LiveStreamListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly"
	//   ]
	// }

}

// method id "youtube.liveStreams.update":

type LiveStreamsUpdateCall struct {
	s          *Service
	part       string
	livestream *LiveStream
	opt_       map[string]interface{}
}

// Update: Updates a video stream. If the properties that you want to
// change cannot be updated, then you need to create a new stream with
// the proper settings.
func (r *LiveStreamsService) Update(part string, livestream *LiveStream) *LiveStreamsUpdateCall {
	c := &LiveStreamsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.livestream = livestream
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *LiveStreamsUpdateCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *LiveStreamsUpdateCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *LiveStreamsUpdateCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *LiveStreamsUpdateCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

func (c *LiveStreamsUpdateCall) Do() (*LiveStream, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.livestream)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "liveStreams")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
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
	ret := new(LiveStream)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a video stream. If the properties that you want to change cannot be updated, then you need to create a new stream with the proper settings.",
	//   "httpMethod": "PUT",
	//   "id": "youtube.liveStreams.update",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part properties that you can include in the parameter value are id, snippet, cdn, and status.\n\nNote that this method will override the existing values for all of the mutable properties that are contained in any parts that the parameter value specifies. If the request body does not specify a value for a mutable property, the existing value for that property will be removed.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "liveStreams",
	//   "request": {
	//     "$ref": "LiveStream"
	//   },
	//   "response": {
	//     "$ref": "LiveStream"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube"
	//   ]
	// }

}

// method id "youtube.playlistItems.delete":

type PlaylistItemsDeleteCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// Delete: Deletes a playlist item.
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "playlistItems")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
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
	//   "description": "Deletes a playlist item.",
	//   "httpMethod": "DELETE",
	//   "id": "youtube.playlistItems.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies the YouTube playlist item ID for the playlist item that is being deleted. In a playlistItem resource, the id property specifies the playlist item's ID.",
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
	part         string
	playlistitem *PlaylistItem
	opt_         map[string]interface{}
}

// Insert: Adds a resource to a playlist.
func (r *PlaylistItemsService) Insert(part string, playlistitem *PlaylistItem) *PlaylistItemsInsertCall {
	c := &PlaylistItemsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.playlistitem = playlistitem
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *PlaylistItemsInsertCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *PlaylistItemsInsertCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
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
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "playlistItems")
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
	ret := new(PlaylistItem)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds a resource to a playlist.",
	//   "httpMethod": "POST",
	//   "id": "youtube.playlistItems.insert",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part names that you can include in the parameter value are snippet, contentDetails, and status.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
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

// List: Returns a collection of playlist items that match the API
// request parameters. You can retrieve all of the playlist items in a
// specified playlist or retrieve one or more playlist items by their
// unique IDs.
func (r *PlaylistItemsService) List(part string) *PlaylistItemsListCall {
	c := &PlaylistItemsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// Id sets the optional parameter "id": The id parameter specifies a
// comma-separated list of one or more unique playlist item IDs.
func (c *PlaylistItemsListCall) Id(id string) *PlaylistItemsListCall {
	c.opt_["id"] = id
	return c
}

// MaxResults sets the optional parameter "maxResults": The maxResults
// parameter specifies the maximum number of items that should be
// returned in the result set.
func (c *PlaylistItemsListCall) MaxResults(maxResults int64) *PlaylistItemsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *PlaylistItemsListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *PlaylistItemsListCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// PageToken sets the optional parameter "pageToken": The pageToken
// parameter identifies a specific page in the result set that should be
// returned. In an API response, the nextPageToken and prevPageToken
// properties identify other pages that could be retrieved.
func (c *PlaylistItemsListCall) PageToken(pageToken string) *PlaylistItemsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// PlaylistId sets the optional parameter "playlistId": The playlistId
// parameter specifies the unique ID of the playlist for which you want
// to retrieve playlist items. Note that even though this is an optional
// parameter, every request to retrieve playlist items must specify a
// value for either the id parameter or the playlistId parameter.
func (c *PlaylistItemsListCall) PlaylistId(playlistId string) *PlaylistItemsListCall {
	c.opt_["playlistId"] = playlistId
	return c
}

// VideoId sets the optional parameter "videoId": The videoId parameter
// specifies that the request should return only the playlist items that
// contain the specified video.
func (c *PlaylistItemsListCall) VideoId(videoId string) *PlaylistItemsListCall {
	c.opt_["videoId"] = videoId
	return c
}

func (c *PlaylistItemsListCall) Do() (*PlaylistItemListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["id"]; ok {
		params.Set("id", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["playlistId"]; ok {
		params.Set("playlistId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["videoId"]; ok {
		params.Set("videoId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "playlistItems")
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
	ret := new(PlaylistItemListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a collection of playlist items that match the API request parameters. You can retrieve all of the playlist items in a specified playlist or retrieve one or more playlist items by their unique IDs.",
	//   "httpMethod": "GET",
	//   "id": "youtube.playlistItems.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies a comma-separated list of one or more unique playlist item IDs.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "The maxResults parameter specifies the maximum number of items that should be returned in the result set.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "50",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The pageToken parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more playlistItem resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, contentDetails, and status.\n\nIf the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a playlistItem resource, the snippet property contains numerous fields, including the title, description, position, and resourceId properties. As such, if you set part=snippet, the API response will contain all of those properties.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "playlistId": {
	//       "description": "The playlistId parameter specifies the unique ID of the playlist for which you want to retrieve playlist items. Note that even though this is an optional parameter, every request to retrieve playlist items must specify a value for either the id parameter or the playlistId parameter.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoId": {
	//       "description": "The videoId parameter specifies that the request should return only the playlist items that contain the specified video.",
	//       "location": "query",
	//       "type": "string"
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
	//   ],
	//   "supportsSubscription": true
	// }

}

// method id "youtube.playlistItems.update":

type PlaylistItemsUpdateCall struct {
	s            *Service
	part         string
	playlistitem *PlaylistItem
	opt_         map[string]interface{}
}

// Update: Modifies a playlist item. For example, you could update the
// item's position in the playlist.
func (r *PlaylistItemsService) Update(part string, playlistitem *PlaylistItem) *PlaylistItemsUpdateCall {
	c := &PlaylistItemsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
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
	params.Set("part", fmt.Sprintf("%v", c.part))
	urls := googleapi.ResolveRelative(c.s.BasePath, "playlistItems")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
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
	ret := new(PlaylistItem)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Modifies a playlist item. For example, you could update the item's position in the playlist.",
	//   "httpMethod": "PUT",
	//   "id": "youtube.playlistItems.update",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part names that you can include in the parameter value are snippet, contentDetails, and status.\n\nNote that this method will override the existing values for all of the mutable properties that are contained in any parts that the parameter value specifies. For example, a playlist item can specify a start time and end time, which identify the times portion of the video that should play when users watch the video in the playlist. If your request is updating a playlist item that sets these values, and the request's part parameter value includes the contentDetails part, the playlist item's start and end times will be updated to whatever value the request body specifies. If the request body does not specify values, the existing start and end times will be removed and replaced with the default settings.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
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

// Delete: Deletes a playlist.
func (r *PlaylistsService) Delete(id string) *PlaylistsDeleteCall {
	c := &PlaylistsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *PlaylistsDeleteCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *PlaylistsDeleteCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

func (c *PlaylistsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "playlists")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
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
	//   "description": "Deletes a playlist.",
	//   "httpMethod": "DELETE",
	//   "id": "youtube.playlists.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies the YouTube playlist ID for the playlist that is being deleted. In a playlist resource, the id property specifies the playlist's ID.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
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

// method id "youtube.playlists.insert":

type PlaylistsInsertCall struct {
	s        *Service
	part     string
	playlist *Playlist
	opt_     map[string]interface{}
}

// Insert: Creates a playlist.
func (r *PlaylistsService) Insert(part string, playlist *Playlist) *PlaylistsInsertCall {
	c := &PlaylistsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.playlist = playlist
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *PlaylistsInsertCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *PlaylistsInsertCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *PlaylistsInsertCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *PlaylistsInsertCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

func (c *PlaylistsInsertCall) Do() (*Playlist, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.playlist)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "playlists")
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
	ret := new(Playlist)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Creates a playlist.",
	//   "httpMethod": "POST",
	//   "id": "youtube.playlists.insert",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part names that you can include in the parameter value are snippet and status.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "playlists",
	//   "request": {
	//     "$ref": "Playlist"
	//   },
	//   "response": {
	//     "$ref": "Playlist"
	//   },
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

// List: Returns a collection of playlists that match the API request
// parameters. For example, you can retrieve all playlists that the
// authenticated user owns, or you can retrieve one or more playlists by
// their unique IDs.
func (r *PlaylistsService) List(part string) *PlaylistsListCall {
	c := &PlaylistsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// ChannelId sets the optional parameter "channelId": This value
// indicates that the API should only return the specified channel's
// playlists.
func (c *PlaylistsListCall) ChannelId(channelId string) *PlaylistsListCall {
	c.opt_["channelId"] = channelId
	return c
}

// Id sets the optional parameter "id": The id parameter specifies a
// comma-separated list of the YouTube playlist ID(s) for the
// resource(s) that are being retrieved. In a playlist resource, the id
// property specifies the playlist's YouTube playlist ID.
func (c *PlaylistsListCall) Id(id string) *PlaylistsListCall {
	c.opt_["id"] = id
	return c
}

// MaxResults sets the optional parameter "maxResults": The maxResults
// parameter specifies the maximum number of items that should be
// returned in the result set.
func (c *PlaylistsListCall) MaxResults(maxResults int64) *PlaylistsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Mine sets the optional parameter "mine": Set this parameter's value
// to true to instruct the API to only return playlists owned by the
// authenticated user.
func (c *PlaylistsListCall) Mine(mine bool) *PlaylistsListCall {
	c.opt_["mine"] = mine
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *PlaylistsListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *PlaylistsListCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *PlaylistsListCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *PlaylistsListCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

// PageToken sets the optional parameter "pageToken": The pageToken
// parameter identifies a specific page in the result set that should be
// returned. In an API response, the nextPageToken and prevPageToken
// properties identify other pages that could be retrieved.
func (c *PlaylistsListCall) PageToken(pageToken string) *PlaylistsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *PlaylistsListCall) Do() (*PlaylistListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["channelId"]; ok {
		params.Set("channelId", fmt.Sprintf("%v", v))
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
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "playlists")
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
	ret := new(PlaylistListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a collection of playlists that match the API request parameters. For example, you can retrieve all playlists that the authenticated user owns, or you can retrieve one or more playlists by their unique IDs.",
	//   "httpMethod": "GET",
	//   "id": "youtube.playlists.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "channelId": {
	//       "description": "This value indicates that the API should only return the specified channel's playlists.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "The id parameter specifies a comma-separated list of the YouTube playlist ID(s) for the resource(s) that are being retrieved. In a playlist resource, the id property specifies the playlist's YouTube playlist ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "The maxResults parameter specifies the maximum number of items that should be returned in the result set.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "50",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "mine": {
	//       "description": "Set this parameter's value to true to instruct the API to only return playlists owned by the authenticated user.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The pageToken parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more playlist resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, and status.\n\nIf the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a playlist resource, the snippet property contains properties like author, title, description, tags, and timeCreated. As such, if you set part=snippet, the API response will contain all of those properties.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
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

// method id "youtube.playlists.update":

type PlaylistsUpdateCall struct {
	s        *Service
	part     string
	playlist *Playlist
	opt_     map[string]interface{}
}

// Update: Modifies a playlist. For example, you could change a
// playlist's title, description, or privacy status.
func (r *PlaylistsService) Update(part string, playlist *Playlist) *PlaylistsUpdateCall {
	c := &PlaylistsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.playlist = playlist
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *PlaylistsUpdateCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *PlaylistsUpdateCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

func (c *PlaylistsUpdateCall) Do() (*Playlist, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.playlist)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "playlists")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
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
	ret := new(Playlist)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Modifies a playlist. For example, you could change a playlist's title, description, or privacy status.",
	//   "httpMethod": "PUT",
	//   "id": "youtube.playlists.update",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part names that you can include in the parameter value are snippet and status.\n\nNote that this method will override the existing values for all of the mutable properties that are contained in any parts that the parameter value specifies. For example, a playlist's privacy setting is contained in the status part. As such, if your request is updating a private playlist, and the request's part parameter value includes the status part, the playlist's privacy setting will be updated to whatever value the request body specifies. If the request body does not specify a value, the existing privacy setting will be removed and the playlist will revert to the default privacy setting.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "playlists",
	//   "request": {
	//     "$ref": "Playlist"
	//   },
	//   "response": {
	//     "$ref": "Playlist"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.search.list":

type SearchListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Returns a collection of search results that match the query
// parameters specified in the API request. By default, a search result
// set identifies matching video, channel, and playlist resources, but
// you can also configure queries to only retrieve a specific type of
// resource.
func (r *SearchService) List(part string) *SearchListCall {
	c := &SearchListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// ChannelId sets the optional parameter "channelId": The channelId
// parameter indicates that the API response should only contain
// resources created by the channel
func (c *SearchListCall) ChannelId(channelId string) *SearchListCall {
	c.opt_["channelId"] = channelId
	return c
}

// ChannelType sets the optional parameter "channelType": The
// channelType parameter lets you restrict a search to a particular type
// of channel.
func (c *SearchListCall) ChannelType(channelType string) *SearchListCall {
	c.opt_["channelType"] = channelType
	return c
}

// EventType sets the optional parameter "eventType": The eventType
// parameter restricts a search to broadcast events.
func (c *SearchListCall) EventType(eventType string) *SearchListCall {
	c.opt_["eventType"] = eventType
	return c
}

// ForContentOwner sets the optional parameter "forContentOwner": Note:
// This parameter is intended exclusively for YouTube content
// partners.
//
// The forContentOwner parameter restricts the search to only
// retrieve resources owned by the content owner specified by the
// onBehalfOfContentOwner parameter. The user must be authenticated
// using a CMS account linked to the specified content owner and
// onBehalfOfContentOwner must be provided.
func (c *SearchListCall) ForContentOwner(forContentOwner bool) *SearchListCall {
	c.opt_["forContentOwner"] = forContentOwner
	return c
}

// ForMine sets the optional parameter "forMine": The forMine parameter
// restricts the search to only retrieve videos owned by the
// authenticated user. If you set this parameter to true, then the type
// parameter's value must also be set to video.
func (c *SearchListCall) ForMine(forMine bool) *SearchListCall {
	c.opt_["forMine"] = forMine
	return c
}

// MaxResults sets the optional parameter "maxResults": The maxResults
// parameter specifies the maximum number of items that should be
// returned in the result set.
func (c *SearchListCall) MaxResults(maxResults int64) *SearchListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *SearchListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *SearchListCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// Order sets the optional parameter "order": The order parameter
// specifies the method that will be used to order resources in the API
// response.
func (c *SearchListCall) Order(order string) *SearchListCall {
	c.opt_["order"] = order
	return c
}

// PageToken sets the optional parameter "pageToken": The pageToken
// parameter identifies a specific page in the result set that should be
// returned. In an API response, the nextPageToken and prevPageToken
// properties identify other pages that could be retrieved.
func (c *SearchListCall) PageToken(pageToken string) *SearchListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// PublishedAfter sets the optional parameter "publishedAfter": The
// publishedAfter parameter indicates that the API response should only
// contain resources created after the specified time. The value is an
// RFC 3339 formatted date-time value (1970-01-01T00:00:00Z).
func (c *SearchListCall) PublishedAfter(publishedAfter string) *SearchListCall {
	c.opt_["publishedAfter"] = publishedAfter
	return c
}

// PublishedBefore sets the optional parameter "publishedBefore": The
// publishedBefore parameter indicates that the API response should only
// contain resources created before the specified time. The value is an
// RFC 3339 formatted date-time value (1970-01-01T00:00:00Z).
func (c *SearchListCall) PublishedBefore(publishedBefore string) *SearchListCall {
	c.opt_["publishedBefore"] = publishedBefore
	return c
}

// Q sets the optional parameter "q": The q parameter specifies the
// query term to search for.
func (c *SearchListCall) Q(q string) *SearchListCall {
	c.opt_["q"] = q
	return c
}

// RegionCode sets the optional parameter "regionCode": The regionCode
// parameter instructs the API to return search results for the
// specified country. The parameter value is an ISO 3166-1 alpha-2
// country code.
func (c *SearchListCall) RegionCode(regionCode string) *SearchListCall {
	c.opt_["regionCode"] = regionCode
	return c
}

// RelatedToVideoId sets the optional parameter "relatedToVideoId": The
// relatedToVideoId parameter retrieves a list of videos that are
// related to the video that the parameter value identifies. The
// parameter value must be set to a YouTube video ID and, if you are
// using this parameter, the type parameter must be set to video.
func (c *SearchListCall) RelatedToVideoId(relatedToVideoId string) *SearchListCall {
	c.opt_["relatedToVideoId"] = relatedToVideoId
	return c
}

// SafeSearch sets the optional parameter "safeSearch": The safeSearch
// parameter indicates whether the search results should include
// restricted content as well as standard content.
func (c *SearchListCall) SafeSearch(safeSearch string) *SearchListCall {
	c.opt_["safeSearch"] = safeSearch
	return c
}

// TopicId sets the optional parameter "topicId": The topicId parameter
// indicates that the API response should only contain resources
// associated with the specified topic. The value identifies a Freebase
// topic ID.
func (c *SearchListCall) TopicId(topicId string) *SearchListCall {
	c.opt_["topicId"] = topicId
	return c
}

// Type sets the optional parameter "type": The type parameter restricts
// a search query to only retrieve a particular type of resource. The
// value is a comma-separated list of resource types.
func (c *SearchListCall) Type(type_ string) *SearchListCall {
	c.opt_["type"] = type_
	return c
}

// VideoCaption sets the optional parameter "videoCaption": The
// videoCaption parameter indicates whether the API should filter video
// search results based on whether they have captions.
func (c *SearchListCall) VideoCaption(videoCaption string) *SearchListCall {
	c.opt_["videoCaption"] = videoCaption
	return c
}

// VideoCategoryId sets the optional parameter "videoCategoryId": The
// videoCategoryId parameter filters video search results based on their
// category.
func (c *SearchListCall) VideoCategoryId(videoCategoryId string) *SearchListCall {
	c.opt_["videoCategoryId"] = videoCategoryId
	return c
}

// VideoDefinition sets the optional parameter "videoDefinition": The
// videoDefinition parameter lets you restrict a search to only include
// either high definition (HD) or standard definition (SD) videos. HD
// videos are available for playback in at least 720p, though higher
// resolutions, like 1080p, might also be available.
func (c *SearchListCall) VideoDefinition(videoDefinition string) *SearchListCall {
	c.opt_["videoDefinition"] = videoDefinition
	return c
}

// VideoDimension sets the optional parameter "videoDimension": The
// videoDimension parameter lets you restrict a search to only retrieve
// 2D or 3D videos.
func (c *SearchListCall) VideoDimension(videoDimension string) *SearchListCall {
	c.opt_["videoDimension"] = videoDimension
	return c
}

// VideoDuration sets the optional parameter "videoDuration": The
// videoDuration parameter filters video search results based on their
// duration.
func (c *SearchListCall) VideoDuration(videoDuration string) *SearchListCall {
	c.opt_["videoDuration"] = videoDuration
	return c
}

// VideoEmbeddable sets the optional parameter "videoEmbeddable": The
// videoEmbeddable parameter lets you to restrict a search to only
// videos that can be embedded into a webpage.
func (c *SearchListCall) VideoEmbeddable(videoEmbeddable string) *SearchListCall {
	c.opt_["videoEmbeddable"] = videoEmbeddable
	return c
}

// VideoLicense sets the optional parameter "videoLicense": The
// videoLicense parameter filters search results to only include videos
// with a particular license. YouTube lets video uploaders choose to
// attach either the Creative Commons license or the standard YouTube
// license to each of their videos.
func (c *SearchListCall) VideoLicense(videoLicense string) *SearchListCall {
	c.opt_["videoLicense"] = videoLicense
	return c
}

// VideoSyndicated sets the optional parameter "videoSyndicated": The
// videoSyndicated parameter lets you to restrict a search to only
// videos that can be played outside youtube.com.
func (c *SearchListCall) VideoSyndicated(videoSyndicated string) *SearchListCall {
	c.opt_["videoSyndicated"] = videoSyndicated
	return c
}

// VideoType sets the optional parameter "videoType": The videoType
// parameter lets you restrict a search to a particular type of videos.
func (c *SearchListCall) VideoType(videoType string) *SearchListCall {
	c.opt_["videoType"] = videoType
	return c
}

func (c *SearchListCall) Do() (*SearchListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["channelId"]; ok {
		params.Set("channelId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["channelType"]; ok {
		params.Set("channelType", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["eventType"]; ok {
		params.Set("eventType", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["forContentOwner"]; ok {
		params.Set("forContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["forMine"]; ok {
		params.Set("forMine", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["order"]; ok {
		params.Set("order", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["publishedAfter"]; ok {
		params.Set("publishedAfter", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["publishedBefore"]; ok {
		params.Set("publishedBefore", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["q"]; ok {
		params.Set("q", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["regionCode"]; ok {
		params.Set("regionCode", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["relatedToVideoId"]; ok {
		params.Set("relatedToVideoId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["safeSearch"]; ok {
		params.Set("safeSearch", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["topicId"]; ok {
		params.Set("topicId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["type"]; ok {
		params.Set("type", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["videoCaption"]; ok {
		params.Set("videoCaption", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["videoCategoryId"]; ok {
		params.Set("videoCategoryId", fmt.Sprintf("%v", v))
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
	if v, ok := c.opt_["videoEmbeddable"]; ok {
		params.Set("videoEmbeddable", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["videoLicense"]; ok {
		params.Set("videoLicense", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["videoSyndicated"]; ok {
		params.Set("videoSyndicated", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["videoType"]; ok {
		params.Set("videoType", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "search")
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
	ret := new(SearchListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a collection of search results that match the query parameters specified in the API request. By default, a search result set identifies matching video, channel, and playlist resources, but you can also configure queries to only retrieve a specific type of resource.",
	//   "httpMethod": "GET",
	//   "id": "youtube.search.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "channelId": {
	//       "description": "The channelId parameter indicates that the API response should only contain resources created by the channel",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "channelType": {
	//       "description": "The channelType parameter lets you restrict a search to a particular type of channel.",
	//       "enum": [
	//         "any",
	//         "show"
	//       ],
	//       "enumDescriptions": [
	//         "Return all channels.",
	//         "Only retrieve shows."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "eventType": {
	//       "description": "The eventType parameter restricts a search to broadcast events.",
	//       "enum": [
	//         "completed",
	//         "live",
	//         "upcoming"
	//       ],
	//       "enumDescriptions": [
	//         "Only include completed broadcasts.",
	//         "Only include active broadcasts.",
	//         "Only include upcoming broadcasts."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "forContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe forContentOwner parameter restricts the search to only retrieve resources owned by the content owner specified by the onBehalfOfContentOwner parameter. The user must be authenticated using a CMS account linked to the specified content owner and onBehalfOfContentOwner must be provided.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "forMine": {
	//       "description": "The forMine parameter restricts the search to only retrieve videos owned by the authenticated user. If you set this parameter to true, then the type parameter's value must also be set to video.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "The maxResults parameter specifies the maximum number of items that should be returned in the result set.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "50",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "order": {
	//       "default": "SEARCH_SORT_RELEVANCE",
	//       "description": "The order parameter specifies the method that will be used to order resources in the API response.",
	//       "enum": [
	//         "date",
	//         "rating",
	//         "relevance",
	//         "title",
	//         "videoCount",
	//         "viewCount"
	//       ],
	//       "enumDescriptions": [
	//         "Resources are sorted in reverse chronological order based on the date they were created.",
	//         "Resources are sorted from highest to lowest rating.",
	//         "Resources are sorted based on their relevance to the search query. This is the default value for this parameter.",
	//         "Resources are sorted alphabetically by title.",
	//         "Channels are sorted in descending order of their number of uploaded videos.",
	//         "Resources are sorted from highest to lowest number of views."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The pageToken parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more search resource properties that the API response will include. The part names that you can include in the parameter value are id and snippet.\n\nIf the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a search result, the snippet property contains other properties that identify the result's title, description, and so forth. If you set part=snippet, the API response will also contain all of those nested properties.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "publishedAfter": {
	//       "description": "The publishedAfter parameter indicates that the API response should only contain resources created after the specified time. The value is an RFC 3339 formatted date-time value (1970-01-01T00:00:00Z).",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "publishedBefore": {
	//       "description": "The publishedBefore parameter indicates that the API response should only contain resources created before the specified time. The value is an RFC 3339 formatted date-time value (1970-01-01T00:00:00Z).",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "q": {
	//       "description": "The q parameter specifies the query term to search for.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "regionCode": {
	//       "description": "The regionCode parameter instructs the API to return search results for the specified country. The parameter value is an ISO 3166-1 alpha-2 country code.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "relatedToVideoId": {
	//       "description": "The relatedToVideoId parameter retrieves a list of videos that are related to the video that the parameter value identifies. The parameter value must be set to a YouTube video ID and, if you are using this parameter, the type parameter must be set to video.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "safeSearch": {
	//       "description": "The safeSearch parameter indicates whether the search results should include restricted content as well as standard content.",
	//       "enum": [
	//         "moderate",
	//         "none",
	//         "strict"
	//       ],
	//       "enumDescriptions": [
	//         "YouTube will filter some content from search results and, at the least, will filter content that is restricted in your locale. Based on their content, search results could be removed from search results or demoted in search results. This is the default parameter value.",
	//         "YouTube will not filter the search result set.",
	//         "YouTube will try to exclude all restricted content from the search result set. Based on their content, search results could be removed from search results or demoted in search results."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "topicId": {
	//       "description": "The topicId parameter indicates that the API response should only contain resources associated with the specified topic. The value identifies a Freebase topic ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "type": {
	//       "default": "video,channel,playlist",
	//       "description": "The type parameter restricts a search query to only retrieve a particular type of resource. The value is a comma-separated list of resource types.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoCaption": {
	//       "description": "The videoCaption parameter indicates whether the API should filter video search results based on whether they have captions.",
	//       "enum": [
	//         "any",
	//         "closedCaption",
	//         "none"
	//       ],
	//       "enumDescriptions": [
	//         "Do not filter results based on caption availability.",
	//         "Only include videos that have captions.",
	//         "Only include videos that do not have captions."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoCategoryId": {
	//       "description": "The videoCategoryId parameter filters video search results based on their category.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoDefinition": {
	//       "description": "The videoDefinition parameter lets you restrict a search to only include either high definition (HD) or standard definition (SD) videos. HD videos are available for playback in at least 720p, though higher resolutions, like 1080p, might also be available.",
	//       "enum": [
	//         "any",
	//         "high",
	//         "standard"
	//       ],
	//       "enumDescriptions": [
	//         "Return all videos, regardless of their resolution.",
	//         "Only retrieve HD videos.",
	//         "Only retrieve videos in standard definition."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoDimension": {
	//       "description": "The videoDimension parameter lets you restrict a search to only retrieve 2D or 3D videos.",
	//       "enum": [
	//         "2d",
	//         "3d",
	//         "any"
	//       ],
	//       "enumDescriptions": [
	//         "Restrict search results to exclude 3D videos.",
	//         "Restrict search results to only include 3D videos.",
	//         "Include both 3D and non-3D videos in returned results. This is the default value."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoDuration": {
	//       "description": "The videoDuration parameter filters video search results based on their duration.",
	//       "enum": [
	//         "any",
	//         "long",
	//         "medium",
	//         "short"
	//       ],
	//       "enumDescriptions": [
	//         "Do not filter video search results based on their duration. This is the default value.",
	//         "Only include videos longer than 20 minutes.",
	//         "Only include videos that are between four and 20 minutes long (inclusive).",
	//         "Only include videos that are less than four minutes long."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoEmbeddable": {
	//       "description": "The videoEmbeddable parameter lets you to restrict a search to only videos that can be embedded into a webpage.",
	//       "enum": [
	//         "any",
	//         "true"
	//       ],
	//       "enumDescriptions": [
	//         "Return all videos, embeddable or not.",
	//         "Only retrieve embeddable videos."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoLicense": {
	//       "description": "The videoLicense parameter filters search results to only include videos with a particular license. YouTube lets video uploaders choose to attach either the Creative Commons license or the standard YouTube license to each of their videos.",
	//       "enum": [
	//         "any",
	//         "creativeCommon",
	//         "youtube"
	//       ],
	//       "enumDescriptions": [
	//         "Return all videos, regardless of which license they have, that match the query parameters.",
	//         "Only return videos that have a Creative Commons license. Users can reuse videos with this license in other videos that they create. Learn more.",
	//         "Only return videos that have the standard YouTube license."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoSyndicated": {
	//       "description": "The videoSyndicated parameter lets you to restrict a search to only videos that can be played outside youtube.com.",
	//       "enum": [
	//         "any",
	//         "true"
	//       ],
	//       "enumDescriptions": [
	//         "Return all videos, syndicated or not.",
	//         "Only retrieve syndicated videos."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoType": {
	//       "description": "The videoType parameter lets you restrict a search to a particular type of videos.",
	//       "enum": [
	//         "any",
	//         "episode",
	//         "movie"
	//       ],
	//       "enumDescriptions": [
	//         "Return all videos.",
	//         "Only retrieve episodes of shows.",
	//         "Only retrieve movies."
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

// method id "youtube.subscriptions.delete":

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
	params.Set("id", fmt.Sprintf("%v", c.id))
	urls := googleapi.ResolveRelative(c.s.BasePath, "subscriptions")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
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
	//   "id": "youtube.subscriptions.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies the YouTube subscription ID for the resource that is being deleted. In a subscription resource, the id property specifies the YouTube subscription ID.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "subscriptions",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.subscriptions.insert":

type SubscriptionsInsertCall struct {
	s            *Service
	part         string
	subscription *Subscription
	opt_         map[string]interface{}
}

// Insert: Adds a subscription for the authenticated user's channel.
func (r *SubscriptionsService) Insert(part string, subscription *Subscription) *SubscriptionsInsertCall {
	c := &SubscriptionsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
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
	params.Set("part", fmt.Sprintf("%v", c.part))
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
	//   "description": "Adds a subscription for the authenticated user's channel.",
	//   "httpMethod": "POST",
	//   "id": "youtube.subscriptions.insert",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part names that you can include in the parameter value are snippet and contentDetails.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "subscriptions",
	//   "request": {
	//     "$ref": "Subscription"
	//   },
	//   "response": {
	//     "$ref": "Subscription"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.subscriptions.list":

type SubscriptionsListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Returns subscription resources that match the API request
// criteria.
func (r *SubscriptionsService) List(part string) *SubscriptionsListCall {
	c := &SubscriptionsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// ChannelId sets the optional parameter "channelId": The channelId
// parameter specifies a YouTube channel ID. The API will only return
// that channel's subscriptions.
func (c *SubscriptionsListCall) ChannelId(channelId string) *SubscriptionsListCall {
	c.opt_["channelId"] = channelId
	return c
}

// ForChannelId sets the optional parameter "forChannelId": The
// forChannelId parameter specifies a comma-separated list of channel
// IDs. The API response will then only contain subscriptions matching
// those channels.
func (c *SubscriptionsListCall) ForChannelId(forChannelId string) *SubscriptionsListCall {
	c.opt_["forChannelId"] = forChannelId
	return c
}

// Id sets the optional parameter "id": The id parameter specifies a
// comma-separated list of the YouTube subscription ID(s) for the
// resource(s) that are being retrieved. In a subscription resource, the
// id property specifies the YouTube subscription ID.
func (c *SubscriptionsListCall) Id(id string) *SubscriptionsListCall {
	c.opt_["id"] = id
	return c
}

// MaxResults sets the optional parameter "maxResults": The maxResults
// parameter specifies the maximum number of items that should be
// returned in the result set.
func (c *SubscriptionsListCall) MaxResults(maxResults int64) *SubscriptionsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// Mine sets the optional parameter "mine": Set this parameter's value
// to true to retrieve a feed of the authenticated user's subscriptions.
func (c *SubscriptionsListCall) Mine(mine bool) *SubscriptionsListCall {
	c.opt_["mine"] = mine
	return c
}

// MySubscribers sets the optional parameter "mySubscribers": Set this
// parameter's value to true to retrieve a feed of the subscribers of
// the authenticated user.
func (c *SubscriptionsListCall) MySubscribers(mySubscribers bool) *SubscriptionsListCall {
	c.opt_["mySubscribers"] = mySubscribers
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *SubscriptionsListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *SubscriptionsListCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *SubscriptionsListCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *SubscriptionsListCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

// Order sets the optional parameter "order": The order parameter
// specifies the method that will be used to sort resources in the API
// response.
func (c *SubscriptionsListCall) Order(order string) *SubscriptionsListCall {
	c.opt_["order"] = order
	return c
}

// PageToken sets the optional parameter "pageToken": The pageToken
// parameter identifies a specific page in the result set that should be
// returned. In an API response, the nextPageToken and prevPageToken
// properties identify other pages that could be retrieved.
func (c *SubscriptionsListCall) PageToken(pageToken string) *SubscriptionsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *SubscriptionsListCall) Do() (*SubscriptionListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["channelId"]; ok {
		params.Set("channelId", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["forChannelId"]; ok {
		params.Set("forChannelId", fmt.Sprintf("%v", v))
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
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["order"]; ok {
		params.Set("order", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
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
	ret := new(SubscriptionListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns subscription resources that match the API request criteria.",
	//   "httpMethod": "GET",
	//   "id": "youtube.subscriptions.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "channelId": {
	//       "description": "The channelId parameter specifies a YouTube channel ID. The API will only return that channel's subscriptions.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "forChannelId": {
	//       "description": "The forChannelId parameter specifies a comma-separated list of channel IDs. The API response will then only contain subscriptions matching those channels.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "The id parameter specifies a comma-separated list of the YouTube subscription ID(s) for the resource(s) that are being retrieved. In a subscription resource, the id property specifies the YouTube subscription ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "The maxResults parameter specifies the maximum number of items that should be returned in the result set.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "50",
	//       "minimum": "0",
	//       "type": "integer"
	//     },
	//     "mine": {
	//       "description": "Set this parameter's value to true to retrieve a feed of the authenticated user's subscriptions.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "mySubscribers": {
	//       "description": "Set this parameter's value to true to retrieve a feed of the subscribers of the authenticated user.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "order": {
	//       "default": "SUBSCRIPTION_ORDER_RELEVANCE",
	//       "description": "The order parameter specifies the method that will be used to sort resources in the API response.",
	//       "enum": [
	//         "alphabetical",
	//         "relevance",
	//         "unread"
	//       ],
	//       "enumDescriptions": [
	//         "Sort alphabetically.",
	//         "Sort by relevance.",
	//         "Sort by order of activity."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The pageToken parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more subscription resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, and contentDetails.\n\nIf the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a subscription resource, the snippet property contains other properties, such as a display title for the subscription. If you set part=snippet, the API response will also contain all of those nested properties.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "subscriptions",
	//   "response": {
	//     "$ref": "SubscriptionListResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.readonly",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.thumbnails.set":

type ThumbnailsSetCall struct {
	s       *Service
	videoId string
	opt_    map[string]interface{}
	media_  io.Reader
}

// Set: Uploads a custom video thumbnail to YouTube and sets it for a
// video.
func (r *ThumbnailsService) Set(videoId string) *ThumbnailsSetCall {
	c := &ThumbnailsSetCall{s: r.s, opt_: make(map[string]interface{})}
	c.videoId = videoId
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": The onBehalfOfContentOwner parameter
// indicates that the authenticated user is acting on behalf of the
// content owner specified in the parameter value. This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and get access to all their video and channel data, without
// having to provide authentication credentials for each individual
// channel. The actual CMS account that the user authenticates with
// needs to be linked to the specified YouTube content owner.
func (c *ThumbnailsSetCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *ThumbnailsSetCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}
func (c *ThumbnailsSetCall) Media(r io.Reader) *ThumbnailsSetCall {
	c.media_ = r
	return c
}

func (c *ThumbnailsSetCall) Do() (*ThumbnailSetResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("videoId", fmt.Sprintf("%v", c.videoId))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "thumbnails/set")
	if c.media_ != nil {
		urls = strings.Replace(urls, "https://www.googleapis.com/", "https://www.googleapis.com/upload/", 1)
		params.Set("uploadType", "multipart")
	}
	urls += "?" + params.Encode()
	body = new(bytes.Buffer)
	ctype := "application/json"
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
	ret := new(ThumbnailSetResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Uploads a custom video thumbnail to YouTube and sets it for a video.",
	//   "httpMethod": "POST",
	//   "id": "youtube.thumbnails.set",
	//   "mediaUpload": {
	//     "accept": [
	//       "application/octet-stream",
	//       "image/jpeg",
	//       "image/png"
	//     ],
	//     "maxSize": "2MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/youtube/v3/thumbnails/set"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/youtube/v3/thumbnails/set"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "videoId"
	//   ],
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "The onBehalfOfContentOwner parameter indicates that the authenticated user is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The actual CMS account that the user authenticates with needs to be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoId": {
	//       "description": "The videoId parameter specifies a YouTube video ID for which the custom video thumbnail is being provided.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "thumbnails/set",
	//   "response": {
	//     "$ref": "ThumbnailSetResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.upload",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "youtube.videoCategories.list":

type VideoCategoriesListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Returns a list of categories that can be associated with
// YouTube videos.
func (r *VideoCategoriesService) List(part string) *VideoCategoriesListCall {
	c := &VideoCategoriesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// Hl sets the optional parameter "hl": The hl parameter specifies the
// language that should be used for text values in the API response.
func (c *VideoCategoriesListCall) Hl(hl string) *VideoCategoriesListCall {
	c.opt_["hl"] = hl
	return c
}

// Id sets the optional parameter "id": The id parameter specifies a
// comma-separated list of video category IDs for the resources that you
// are retrieving.
func (c *VideoCategoriesListCall) Id(id string) *VideoCategoriesListCall {
	c.opt_["id"] = id
	return c
}

// RegionCode sets the optional parameter "regionCode": The regionCode
// parameter instructs the API to return the list of video categories
// available in the specified country. The parameter value is an ISO
// 3166-1 alpha-2 country code.
func (c *VideoCategoriesListCall) RegionCode(regionCode string) *VideoCategoriesListCall {
	c.opt_["regionCode"] = regionCode
	return c
}

func (c *VideoCategoriesListCall) Do() (*VideoCategoryListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["id"]; ok {
		params.Set("id", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["regionCode"]; ok {
		params.Set("regionCode", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "videoCategories")
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
	ret := new(VideoCategoryListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of categories that can be associated with YouTube videos.",
	//   "httpMethod": "GET",
	//   "id": "youtube.videoCategories.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "hl": {
	//       "default": "en_US",
	//       "description": "The hl parameter specifies the language that should be used for text values in the API response.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "The id parameter specifies a comma-separated list of video category IDs for the resources that you are retrieving.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies the videoCategory resource parts that the API response will include. Supported values are id and snippet.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "regionCode": {
	//       "description": "The regionCode parameter instructs the API to return the list of video categories available in the specified country. The parameter value is an ISO 3166-1 alpha-2 country code.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "videoCategories",
	//   "response": {
	//     "$ref": "VideoCategoryListResponse"
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

// Delete: Deletes a YouTube video.
func (r *VideosService) Delete(id string) *VideosDeleteCall {
	c := &VideosDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// actual CMS account that the user authenticates with must be linked to
// the specified YouTube content owner.
func (c *VideosDeleteCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *VideosDeleteCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

func (c *VideosDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "videos")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
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
	//   "description": "Deletes a YouTube video.",
	//   "httpMethod": "DELETE",
	//   "id": "youtube.videos.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies the YouTube video ID for the resource that is being deleted. In a video resource, the id property specifies the video's ID.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The actual CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
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

// method id "youtube.videos.getRating":

type VideosGetRatingCall struct {
	s    *Service
	id   string
	opt_ map[string]interface{}
}

// GetRating: Retrieves the ratings that the authorized user gave to a
// list of specified videos.
func (r *VideosService) GetRating(id string) *VideosGetRatingCall {
	c := &VideosGetRatingCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *VideosGetRatingCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *VideosGetRatingCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

func (c *VideosGetRatingCall) Do() (*VideoGetRatingResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "videos/getRating")
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
	ret := new(VideoGetRatingResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the ratings that the authorized user gave to a list of specified videos.",
	//   "httpMethod": "GET",
	//   "id": "youtube.videos.getRating",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies a comma-separated list of the YouTube video ID(s) for the resource(s) for which you are retrieving rating data. In a video resource, the id property specifies the video's ID.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "videos/getRating",
	//   "response": {
	//     "$ref": "VideoGetRatingResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.videos.insert":

type VideosInsertCall struct {
	s      *Service
	part   string
	video  *Video
	opt_   map[string]interface{}
	media_ io.Reader
}

// Insert: Uploads a video to YouTube and optionally sets the video's
// metadata.
func (r *VideosService) Insert(part string, video *Video) *VideosInsertCall {
	c := &VideosInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.video = video
	return c
}

// AutoLevels sets the optional parameter "autoLevels": The autoLevels
// parameter indicates whether YouTube should automatically enhance the
// video's lighting and color.
func (c *VideosInsertCall) AutoLevels(autoLevels bool) *VideosInsertCall {
	c.opt_["autoLevels"] = autoLevels
	return c
}

// NotifySubscribers sets the optional parameter "notifySubscribers":
// The notifySubscribers parameter indicates whether YouTube should send
// notification to subscribers about the inserted video.
func (c *VideosInsertCall) NotifySubscribers(notifySubscribers bool) *VideosInsertCall {
	c.opt_["notifySubscribers"] = notifySubscribers
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *VideosInsertCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *VideosInsertCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// OnBehalfOfContentOwnerChannel sets the optional parameter
// "onBehalfOfContentOwnerChannel": This parameter can only be used in a
// properly authorized request. Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The
// onBehalfOfContentOwnerChannel parameter specifies the YouTube channel
// ID of the channel to which a video is being added. This parameter is
// required when a request specifies a value for the
// onBehalfOfContentOwner parameter, and it can only be used in
// conjunction with that parameter. In addition, the request must be
// authorized using a CMS account that is linked to the content owner
// that the onBehalfOfContentOwner parameter specifies. Finally, the
// channel that the onBehalfOfContentOwnerChannel parameter value
// specifies must be linked to the content owner that the
// onBehalfOfContentOwner parameter specifies.
//
// This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and perform actions on behalf of the channel specified in the
// parameter value, without having to provide authentication credentials
// for each separate channel.
func (c *VideosInsertCall) OnBehalfOfContentOwnerChannel(onBehalfOfContentOwnerChannel string) *VideosInsertCall {
	c.opt_["onBehalfOfContentOwnerChannel"] = onBehalfOfContentOwnerChannel
	return c
}

// Stabilize sets the optional parameter "stabilize": The stabilize
// parameter indicates whether YouTube should adjust the video to remove
// shaky camera motions.
func (c *VideosInsertCall) Stabilize(stabilize bool) *VideosInsertCall {
	c.opt_["stabilize"] = stabilize
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
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["autoLevels"]; ok {
		params.Set("autoLevels", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["notifySubscribers"]; ok {
		params.Set("notifySubscribers", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwnerChannel"]; ok {
		params.Set("onBehalfOfContentOwnerChannel", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["stabilize"]; ok {
		params.Set("stabilize", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "videos")
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
	ret := new(Video)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Uploads a video to YouTube and optionally sets the video's metadata.",
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
	//         "path": "/resumable/upload/youtube/v3/videos"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/youtube/v3/videos"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "autoLevels": {
	//       "description": "The autoLevels parameter indicates whether YouTube should automatically enhance the video's lighting and color.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "notifySubscribers": {
	//       "default": "true",
	//       "description": "The notifySubscribers parameter indicates whether YouTube should send notification to subscribers about the inserted video.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwnerChannel": {
	//       "description": "This parameter can only be used in a properly authorized request. Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwnerChannel parameter specifies the YouTube channel ID of the channel to which a video is being added. This parameter is required when a request specifies a value for the onBehalfOfContentOwner parameter, and it can only be used in conjunction with that parameter. In addition, the request must be authorized using a CMS account that is linked to the content owner that the onBehalfOfContentOwner parameter specifies. Finally, the channel that the onBehalfOfContentOwnerChannel parameter value specifies must be linked to the content owner that the onBehalfOfContentOwner parameter specifies.\n\nThis parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and perform actions on behalf of the channel specified in the parameter value, without having to provide authentication credentials for each separate channel.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part names that you can include in the parameter value are snippet, contentDetails, fileDetails, liveStreamingDetails, player, processingDetails, recordingDetails, statistics, status, suggestions, and topicDetails. However, not all of those parts contain properties that can be set when setting or updating a video's metadata. For example, the statistics object encapsulates statistics that YouTube calculates for a video and does not contain values that you can set or modify. If the parameter value specifies a part that does not contain mutable values, that part will still be included in the API response.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "stabilize": {
	//       "description": "The stabilize parameter indicates whether YouTube should adjust the video to remove shaky camera motions.",
	//       "location": "query",
	//       "type": "boolean"
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
	//     "https://www.googleapis.com/auth/youtube.upload",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "youtube.videos.list":

type VideosListCall struct {
	s    *Service
	part string
	opt_ map[string]interface{}
}

// List: Returns a list of videos that match the API request parameters.
func (r *VideosService) List(part string) *VideosListCall {
	c := &VideosListCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	return c
}

// Chart sets the optional parameter "chart": The chart parameter
// identifies the chart that you want to retrieve.
func (c *VideosListCall) Chart(chart string) *VideosListCall {
	c.opt_["chart"] = chart
	return c
}

// Id sets the optional parameter "id": The id parameter specifies a
// comma-separated list of the YouTube video ID(s) for the resource(s)
// that are being retrieved. In a video resource, the id property
// specifies the video's ID.
func (c *VideosListCall) Id(id string) *VideosListCall {
	c.opt_["id"] = id
	return c
}

// Locale sets the optional parameter "locale": DEPRECATED
func (c *VideosListCall) Locale(locale string) *VideosListCall {
	c.opt_["locale"] = locale
	return c
}

// MaxResults sets the optional parameter "maxResults": The maxResults
// parameter specifies the maximum number of items that should be
// returned in the result set.
//
// Note: This parameter is supported for
// use in conjunction with the myRating parameter, but it is not
// supported for use in conjunction with the id parameter.
func (c *VideosListCall) MaxResults(maxResults int64) *VideosListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// MyRating sets the optional parameter "myRating": Set this parameter's
// value to like or dislike to instruct the API to only return videos
// liked or disliked by the authenticated user.
func (c *VideosListCall) MyRating(myRating string) *VideosListCall {
	c.opt_["myRating"] = myRating
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *VideosListCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *VideosListCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

// PageToken sets the optional parameter "pageToken": The pageToken
// parameter identifies a specific page in the result set that should be
// returned. In an API response, the nextPageToken and prevPageToken
// properties identify other pages that could be retrieved.
//
// Note: This
// parameter is supported for use in conjunction with the myRating
// parameter, but it is not supported for use in conjunction with the id
// parameter.
func (c *VideosListCall) PageToken(pageToken string) *VideosListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// RegionCode sets the optional parameter "regionCode": The regionCode
// parameter instructs the API to select a video chart available in the
// specified region. This parameter can only be used in conjunction with
// the chart parameter. The parameter value is an ISO 3166-1 alpha-2
// country code.
func (c *VideosListCall) RegionCode(regionCode string) *VideosListCall {
	c.opt_["regionCode"] = regionCode
	return c
}

// VideoCategoryId sets the optional parameter "videoCategoryId": The
// videoCategoryId parameter identifies the video category for which the
// chart should be retrieved. This parameter can only be used in
// conjunction with the chart parameter. By default, charts are not
// restricted to a particular category.
func (c *VideosListCall) VideoCategoryId(videoCategoryId string) *VideosListCall {
	c.opt_["videoCategoryId"] = videoCategoryId
	return c
}

func (c *VideosListCall) Do() (*VideoListResponse, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["chart"]; ok {
		params.Set("chart", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["id"]; ok {
		params.Set("id", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["locale"]; ok {
		params.Set("locale", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["myRating"]; ok {
		params.Set("myRating", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["regionCode"]; ok {
		params.Set("regionCode", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["videoCategoryId"]; ok {
		params.Set("videoCategoryId", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "videos")
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
	ret := new(VideoListResponse)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Returns a list of videos that match the API request parameters.",
	//   "httpMethod": "GET",
	//   "id": "youtube.videos.list",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "chart": {
	//       "description": "The chart parameter identifies the chart that you want to retrieve.",
	//       "enum": [
	//         "mostPopular"
	//       ],
	//       "enumDescriptions": [
	//         "Return the most popular videos for the specified content region and video category."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "id": {
	//       "description": "The id parameter specifies a comma-separated list of the YouTube video ID(s) for the resource(s) that are being retrieved. In a video resource, the id property specifies the video's ID.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "locale": {
	//       "description": "DEPRECATED",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "default": "5",
	//       "description": "The maxResults parameter specifies the maximum number of items that should be returned in the result set.\n\nNote: This parameter is supported for use in conjunction with the myRating parameter, but it is not supported for use in conjunction with the id parameter.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "50",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "myRating": {
	//       "description": "Set this parameter's value to like or dislike to instruct the API to only return videos liked or disliked by the authenticated user.",
	//       "enum": [
	//         "dislike",
	//         "like"
	//       ],
	//       "enumDescriptions": [
	//         "Returns only videos disliked by the authenticated user.",
	//         "Returns only video liked by the authenticated user."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "The pageToken parameter identifies a specific page in the result set that should be returned. In an API response, the nextPageToken and prevPageToken properties identify other pages that could be retrieved.\n\nNote: This parameter is supported for use in conjunction with the myRating parameter, but it is not supported for use in conjunction with the id parameter.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter specifies a comma-separated list of one or more video resource properties that the API response will include. The part names that you can include in the parameter value are id, snippet, contentDetails, fileDetails, liveStreamingDetails, player, processingDetails, recordingDetails, statistics, status, suggestions, and topicDetails.\n\nIf the parameter identifies a property that contains child properties, the child properties will be included in the response. For example, in a video resource, the snippet property contains the channelId, title, description, tags, and categoryId properties. As such, if you set part=snippet, the API response will contain all of those properties.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "regionCode": {
	//       "description": "The regionCode parameter instructs the API to select a video chart available in the specified region. This parameter can only be used in conjunction with the chart parameter. The parameter value is an ISO 3166-1 alpha-2 country code.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "videoCategoryId": {
	//       "default": "0",
	//       "description": "The videoCategoryId parameter identifies the video category for which the chart should be retrieved. This parameter can only be used in conjunction with the chart parameter. By default, charts are not restricted to a particular category.",
	//       "location": "query",
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

// method id "youtube.videos.rate":

type VideosRateCall struct {
	s      *Service
	id     string
	rating string
	opt_   map[string]interface{}
}

// Rate: Add a like or dislike rating to a video or remove a rating from
// a video.
func (r *VideosService) Rate(id string, rating string) *VideosRateCall {
	c := &VideosRateCall{s: r.s, opt_: make(map[string]interface{})}
	c.id = id
	c.rating = rating
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// CMS account that the user authenticates with must be linked to the
// specified YouTube content owner.
func (c *VideosRateCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *VideosRateCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

func (c *VideosRateCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("id", fmt.Sprintf("%v", c.id))
	params.Set("rating", fmt.Sprintf("%v", c.rating))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "videos/rate")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	//   "description": "Add a like or dislike rating to a video or remove a rating from a video.",
	//   "httpMethod": "POST",
	//   "id": "youtube.videos.rate",
	//   "parameterOrder": [
	//     "id",
	//     "rating"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "The id parameter specifies the YouTube video ID of the video that is being rated or having its rating removed.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "rating": {
	//       "description": "Specifies the rating to record.",
	//       "enum": [
	//         "dislike",
	//         "like",
	//         "none"
	//       ],
	//       "enumDescriptions": [
	//         "Records that the authenticated user disliked the video.",
	//         "Records that the authenticated user liked the video.",
	//         "Removes any rating that the authenticated user had previously set for the video."
	//       ],
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "videos/rate",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.videos.update":

type VideosUpdateCall struct {
	s     *Service
	part  string
	video *Video
	opt_  map[string]interface{}
}

// Update: Updates a video's metadata.
func (r *VideosService) Update(part string, video *Video) *VideosUpdateCall {
	c := &VideosUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.part = part
	c.video = video
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": Note: This parameter is intended
// exclusively for YouTube content partners.
//
// The onBehalfOfContentOwner
// parameter indicates that the request's authorization credentials
// identify a YouTube CMS user who is acting on behalf of the content
// owner specified in the parameter value. This parameter is intended
// for YouTube content partners that own and manage many different
// YouTube channels. It allows content owners to authenticate once and
// get access to all their video and channel data, without having to
// provide authentication credentials for each individual channel. The
// actual CMS account that the user authenticates with must be linked to
// the specified YouTube content owner.
func (c *VideosUpdateCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *VideosUpdateCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

func (c *VideosUpdateCall) Do() (*Video, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.video)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("part", fmt.Sprintf("%v", c.part))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "videos")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
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
	ret := new(Video)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates a video's metadata.",
	//   "httpMethod": "PUT",
	//   "id": "youtube.videos.update",
	//   "parameterOrder": [
	//     "part"
	//   ],
	//   "parameters": {
	//     "onBehalfOfContentOwner": {
	//       "description": "Note: This parameter is intended exclusively for YouTube content partners.\n\nThe onBehalfOfContentOwner parameter indicates that the request's authorization credentials identify a YouTube CMS user who is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The actual CMS account that the user authenticates with must be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "part": {
	//       "description": "The part parameter serves two purposes in this operation. It identifies the properties that the write operation will set as well as the properties that the API response will include.\n\nThe part names that you can include in the parameter value are snippet, contentDetails, fileDetails, liveStreamingDetails, player, processingDetails, recordingDetails, statistics, status, suggestions, and topicDetails.\n\nNote that this method will override the existing values for all of the mutable properties that are contained in any parts that the parameter value specifies. For example, a video's privacy setting is contained in the status part. As such, if your request is updating a private video, and the request's part parameter value includes the status part, the video's privacy setting will be updated to whatever value the request body specifies. If the request body does not specify a value, the existing privacy setting will be removed and the video will revert to the default privacy setting.\n\nIn addition, not all of those parts contain properties that can be set when setting or updating a video's metadata. For example, the statistics object encapsulates statistics that YouTube calculates for a video and does not contain values that you can set or modify. If the parameter value specifies a part that does not contain mutable values, that part will still be included in the API response.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
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
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}

// method id "youtube.watermarks.set":

type WatermarksSetCall struct {
	s               *Service
	channelId       string
	invideobranding *InvideoBranding
	opt_            map[string]interface{}
	media_          io.Reader
}

// Set: Uploads a watermark image to YouTube and sets it for a channel.
func (r *WatermarksService) Set(channelId string, invideobranding *InvideoBranding) *WatermarksSetCall {
	c := &WatermarksSetCall{s: r.s, opt_: make(map[string]interface{})}
	c.channelId = channelId
	c.invideobranding = invideobranding
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": The onBehalfOfContentOwner parameter
// indicates that the authenticated user is acting on behalf of the
// content owner specified in the parameter value. This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and get access to all their video and channel data, without
// having to provide authentication credentials for each individual
// channel. The actual CMS account that the user authenticates with
// needs to be linked to the specified YouTube content owner.
func (c *WatermarksSetCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *WatermarksSetCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}
func (c *WatermarksSetCall) Media(r io.Reader) *WatermarksSetCall {
	c.media_ = r
	return c
}

func (c *WatermarksSetCall) Do() error {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.invideobranding)
	if err != nil {
		return err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("channelId", fmt.Sprintf("%v", c.channelId))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "watermarks/set")
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
		return err
	}
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return err
	}
	return nil
	// {
	//   "description": "Uploads a watermark image to YouTube and sets it for a channel.",
	//   "httpMethod": "POST",
	//   "id": "youtube.watermarks.set",
	//   "mediaUpload": {
	//     "accept": [
	//       "application/octet-stream",
	//       "image/jpeg",
	//       "image/png"
	//     ],
	//     "maxSize": "10MB",
	//     "protocols": {
	//       "resumable": {
	//         "multipart": true,
	//         "path": "/resumable/upload/youtube/v3/watermarks/set"
	//       },
	//       "simple": {
	//         "multipart": true,
	//         "path": "/upload/youtube/v3/watermarks/set"
	//       }
	//     }
	//   },
	//   "parameterOrder": [
	//     "channelId"
	//   ],
	//   "parameters": {
	//     "channelId": {
	//       "description": "The channelId parameter specifies a YouTube channel ID for which the watermark is being provided.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "The onBehalfOfContentOwner parameter indicates that the authenticated user is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The actual CMS account that the user authenticates with needs to be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "watermarks/set",
	//   "request": {
	//     "$ref": "InvideoBranding"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtube.upload",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ],
	//   "supportsMediaUpload": true
	// }

}

// method id "youtube.watermarks.unset":

type WatermarksUnsetCall struct {
	s         *Service
	channelId string
	opt_      map[string]interface{}
}

// Unset: Deletes a watermark.
func (r *WatermarksService) Unset(channelId string) *WatermarksUnsetCall {
	c := &WatermarksUnsetCall{s: r.s, opt_: make(map[string]interface{})}
	c.channelId = channelId
	return c
}

// OnBehalfOfContentOwner sets the optional parameter
// "onBehalfOfContentOwner": The onBehalfOfContentOwner parameter
// indicates that the authenticated user is acting on behalf of the
// content owner specified in the parameter value. This parameter is
// intended for YouTube content partners that own and manage many
// different YouTube channels. It allows content owners to authenticate
// once and get access to all their video and channel data, without
// having to provide authentication credentials for each individual
// channel. The actual CMS account that the user authenticates with
// needs to be linked to the specified YouTube content owner.
func (c *WatermarksUnsetCall) OnBehalfOfContentOwner(onBehalfOfContentOwner string) *WatermarksUnsetCall {
	c.opt_["onBehalfOfContentOwner"] = onBehalfOfContentOwner
	return c
}

func (c *WatermarksUnsetCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("channelId", fmt.Sprintf("%v", c.channelId))
	if v, ok := c.opt_["onBehalfOfContentOwner"]; ok {
		params.Set("onBehalfOfContentOwner", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "watermarks/unset")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
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
	//   "description": "Deletes a watermark.",
	//   "httpMethod": "POST",
	//   "id": "youtube.watermarks.unset",
	//   "parameterOrder": [
	//     "channelId"
	//   ],
	//   "parameters": {
	//     "channelId": {
	//       "description": "The channelId parameter specifies a YouTube channel ID for which the watermark is being unset.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "onBehalfOfContentOwner": {
	//       "description": "The onBehalfOfContentOwner parameter indicates that the authenticated user is acting on behalf of the content owner specified in the parameter value. This parameter is intended for YouTube content partners that own and manage many different YouTube channels. It allows content owners to authenticate once and get access to all their video and channel data, without having to provide authentication credentials for each individual channel. The actual CMS account that the user authenticates with needs to be linked to the specified YouTube content owner.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "watermarks/unset",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/youtube",
	//     "https://www.googleapis.com/auth/youtubepartner"
	//   ]
	// }

}
