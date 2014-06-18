// Package orkut provides access to the Orkut API.
//
// See http://code.google.com/apis/orkut/v2/reference.html
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/orkut/v2"
//   ...
//   orkutService, err := orkut.New(oauthHttpClient)
package orkut

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

const apiId = "orkut:v2"
const apiName = "orkut"
const apiVersion = "v2"
const basePath = "https://www.googleapis.com/orkut/v2/"

// OAuth2 scopes used by this API.
const (
	// Manage your Orkut activity
	OrkutScope = "https://www.googleapis.com/auth/orkut"

	// View your Orkut data
	OrkutReadonlyScope = "https://www.googleapis.com/auth/orkut.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Acl = NewAclService(s)
	s.Activities = NewActivitiesService(s)
	s.ActivityVisibility = NewActivityVisibilityService(s)
	s.Badges = NewBadgesService(s)
	s.Comments = NewCommentsService(s)
	s.Communities = NewCommunitiesService(s)
	s.CommunityFollow = NewCommunityFollowService(s)
	s.CommunityMembers = NewCommunityMembersService(s)
	s.CommunityMessages = NewCommunityMessagesService(s)
	s.CommunityPollComments = NewCommunityPollCommentsService(s)
	s.CommunityPollVotes = NewCommunityPollVotesService(s)
	s.CommunityPolls = NewCommunityPollsService(s)
	s.CommunityRelated = NewCommunityRelatedService(s)
	s.CommunityTopics = NewCommunityTopicsService(s)
	s.Counters = NewCountersService(s)
	s.Scraps = NewScrapsService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Acl *AclService

	Activities *ActivitiesService

	ActivityVisibility *ActivityVisibilityService

	Badges *BadgesService

	Comments *CommentsService

	Communities *CommunitiesService

	CommunityFollow *CommunityFollowService

	CommunityMembers *CommunityMembersService

	CommunityMessages *CommunityMessagesService

	CommunityPollComments *CommunityPollCommentsService

	CommunityPollVotes *CommunityPollVotesService

	CommunityPolls *CommunityPollsService

	CommunityRelated *CommunityRelatedService

	CommunityTopics *CommunityTopicsService

	Counters *CountersService

	Scraps *ScrapsService
}

func NewAclService(s *Service) *AclService {
	rs := &AclService{s: s}
	return rs
}

type AclService struct {
	s *Service
}

func NewActivitiesService(s *Service) *ActivitiesService {
	rs := &ActivitiesService{s: s}
	return rs
}

type ActivitiesService struct {
	s *Service
}

func NewActivityVisibilityService(s *Service) *ActivityVisibilityService {
	rs := &ActivityVisibilityService{s: s}
	return rs
}

type ActivityVisibilityService struct {
	s *Service
}

func NewBadgesService(s *Service) *BadgesService {
	rs := &BadgesService{s: s}
	return rs
}

type BadgesService struct {
	s *Service
}

func NewCommentsService(s *Service) *CommentsService {
	rs := &CommentsService{s: s}
	return rs
}

type CommentsService struct {
	s *Service
}

func NewCommunitiesService(s *Service) *CommunitiesService {
	rs := &CommunitiesService{s: s}
	return rs
}

type CommunitiesService struct {
	s *Service
}

func NewCommunityFollowService(s *Service) *CommunityFollowService {
	rs := &CommunityFollowService{s: s}
	return rs
}

type CommunityFollowService struct {
	s *Service
}

func NewCommunityMembersService(s *Service) *CommunityMembersService {
	rs := &CommunityMembersService{s: s}
	return rs
}

type CommunityMembersService struct {
	s *Service
}

func NewCommunityMessagesService(s *Service) *CommunityMessagesService {
	rs := &CommunityMessagesService{s: s}
	return rs
}

type CommunityMessagesService struct {
	s *Service
}

func NewCommunityPollCommentsService(s *Service) *CommunityPollCommentsService {
	rs := &CommunityPollCommentsService{s: s}
	return rs
}

type CommunityPollCommentsService struct {
	s *Service
}

func NewCommunityPollVotesService(s *Service) *CommunityPollVotesService {
	rs := &CommunityPollVotesService{s: s}
	return rs
}

type CommunityPollVotesService struct {
	s *Service
}

func NewCommunityPollsService(s *Service) *CommunityPollsService {
	rs := &CommunityPollsService{s: s}
	return rs
}

type CommunityPollsService struct {
	s *Service
}

func NewCommunityRelatedService(s *Service) *CommunityRelatedService {
	rs := &CommunityRelatedService{s: s}
	return rs
}

type CommunityRelatedService struct {
	s *Service
}

func NewCommunityTopicsService(s *Service) *CommunityTopicsService {
	rs := &CommunityTopicsService{s: s}
	return rs
}

type CommunityTopicsService struct {
	s *Service
}

func NewCountersService(s *Service) *CountersService {
	rs := &CountersService{s: s}
	return rs
}

type CountersService struct {
	s *Service
}

func NewScrapsService(s *Service) *ScrapsService {
	rs := &ScrapsService{s: s}
	return rs
}

type ScrapsService struct {
	s *Service
}

type Acl struct {
	// Description: Human readable description of the access granted.
	Description string `json:"description,omitempty"`

	// Items: The list of ACL entries.
	Items []*AclItems `json:"items,omitempty"`

	// Kind: Identifies this resource as an access control list. Value:
	// "orkut#acl"
	Kind string `json:"kind,omitempty"`

	// TotalParticipants: The total count of participants of the parent
	// resource.
	TotalParticipants int64 `json:"totalParticipants,omitempty"`
}

type AclItems struct {
	// Id: The ID of the entity. For entities of type "person" or "circle",
	// this is the ID of the resource. For other types, this will be unset.
	Id string `json:"id,omitempty"`

	// Type: The type of entity to whom access is granted.
	Type string `json:"type,omitempty"`
}

type Activity struct {
	// Access: Identifies who has access to see this activity.
	Access *Acl `json:"access,omitempty"`

	// Actor: The person who performed the activity.
	Actor *OrkutAuthorResource `json:"actor,omitempty"`

	// Id: The ID for the activity.
	Id string `json:"id,omitempty"`

	// Kind: The kind of activity. Always orkut#activity.
	Kind string `json:"kind,omitempty"`

	// Links: Links to resources related to this activity.
	Links []*OrkutLinkResource `json:"links,omitempty"`

	// Object: The activity's object.
	Object *ActivityObject `json:"object,omitempty"`

	// Published: The time at which the activity was initially published.
	Published string `json:"published,omitempty"`

	// Title: Title of the activity.
	Title string `json:"title,omitempty"`

	// Updated: The time at which the activity was last updated.
	Updated string `json:"updated,omitempty"`

	// Verb: This activity's verb, indicating what action was performed.
	// Possible values are:
	// - add - User added new content to profile or
	// album, e.g. video, photo.
	// - post - User publish content to the
	// stream, e.g. status, scrap.
	// - update - User commented on an
	// activity.
	// - make-friend - User added a new friend.
	// - birthday -
	// User has a birthday.
	Verb string `json:"verb,omitempty"`
}

type ActivityObject struct {
	// Content: The HTML-formatted content, suitable for display. When
	// updating an activity's content, post the changes to this property,
	// using the value of originalContent as a starting point. If the update
	// is successful, the server adds HTML formatting and responds with this
	// formatted content.
	Content string `json:"content,omitempty"`

	// Items: The list of additional items.
	Items []*OrkutActivityobjectsResource `json:"items,omitempty"`

	// ObjectType: The type of the object affected by the activity. Clients
	// can use this information to style the rendered activity object
	// differently depending on the content.
	ObjectType string `json:"objectType,omitempty"`

	// Replies: Comments in reply to this activity.
	Replies *ActivityObjectReplies `json:"replies,omitempty"`
}

type ActivityObjectReplies struct {
	// Items: The list of comments.
	Items []*Comment `json:"items,omitempty"`

	// TotalItems: Total number of comments.
	TotalItems uint64 `json:"totalItems,omitempty,string"`

	// Url: URL for the collection of comments in reply to this activity.
	Url string `json:"url,omitempty"`
}

type ActivityList struct {
	// Items: List of activities retrieved.
	Items []*Activity `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of activities. Value:
	// "orkut#activityList"
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The value of pageToken query parameter in
	// activities.list request to get the next page, if there are more to
	// retrieve.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type Badge struct {
	// BadgeLargeLogo: The URL for the 64x64 badge logo.
	BadgeLargeLogo string `json:"badgeLargeLogo,omitempty"`

	// BadgeSmallLogo: The URL for the 24x24 badge logo.
	BadgeSmallLogo string `json:"badgeSmallLogo,omitempty"`

	// Caption: The name of the badge, suitable for display.
	Caption string `json:"caption,omitempty"`

	// Description: The description for the badge, suitable for display.
	Description string `json:"description,omitempty"`

	// Id: The unique ID for the badge.
	Id int64 `json:"id,omitempty,string"`

	// Kind: Identifies this resource as a badge. Value: "orkut#badge"
	Kind string `json:"kind,omitempty"`

	// SponsorLogo: The URL for the 32x32 badge sponsor logo.
	SponsorLogo string `json:"sponsorLogo,omitempty"`

	// SponsorName: The name of the badge sponsor, suitable for display.
	SponsorName string `json:"sponsorName,omitempty"`

	// SponsorUrl: The URL for the badge sponsor.
	SponsorUrl string `json:"sponsorUrl,omitempty"`
}

type BadgeList struct {
	// Items: List of badges retrieved.
	Items []*Badge `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of badges. Value:
	// "orkut#badgeList"
	Kind string `json:"kind,omitempty"`
}

type Comment struct {
	// Actor: The person who posted the comment.
	Actor *OrkutAuthorResource `json:"actor,omitempty"`

	// Content: The content of the comment in text/html
	Content string `json:"content,omitempty"`

	// Id: The unique ID for the comment.
	Id string `json:"id,omitempty"`

	// InReplyTo: Link to the original activity where this comment was
	// posted.
	InReplyTo *CommentInReplyTo `json:"inReplyTo,omitempty"`

	// Kind: Identifies this resource as a comment. Value: "orkut#comment"
	Kind string `json:"kind,omitempty"`

	// Links: List of resources for the comment.
	Links []*OrkutLinkResource `json:"links,omitempty"`

	// Published: The time the comment was initially published, in RFC 3339
	// format.
	Published string `json:"published,omitempty"`
}

type CommentInReplyTo struct {
	// Href: Link to the post on activity stream being commented.
	Href string `json:"href,omitempty"`

	// Ref: Unique identifier of the post on activity stream being
	// commented.
	Ref string `json:"ref,omitempty"`

	// Rel: Relationship between the comment and the post on activity stream
	// being commented. Always inReplyTo.
	Rel string `json:"rel,omitempty"`

	// Type: Type of the post on activity stream being commented. Always
	// text/html.
	Type string `json:"type,omitempty"`
}

type CommentList struct {
	// Items: List of comments retrieved.
	Items []*Comment `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of comments. Value:
	// "orkut#commentList"
	Kind string `json:"kind,omitempty"`

	// NextPageToken: The value of pageToken query parameter in
	// comments.list request to get the next page, if there are more to
	// retrieve.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// PreviousPageToken: The value of pageToken query parameter in
	// comments.list request to get the previous page, if there are more to
	// retrieve.
	PreviousPageToken string `json:"previousPageToken,omitempty"`
}

type Community struct {
	// Category: The category of the community.
	Category string `json:"category,omitempty"`

	// Co_owners: The co-owners of the community.
	Co_owners []*OrkutAuthorResource `json:"co_owners,omitempty"`

	// Creation_date: The time the community was created, in RFC 3339
	// format.
	Creation_date string `json:"creation_date,omitempty"`

	// Description: The description of the community.
	Description string `json:"description,omitempty"`

	// Id: The id of the community.
	Id int64 `json:"id,omitempty"`

	// Kind: Identifies this resource as a community. Value:
	// "orkut#community"
	Kind string `json:"kind,omitempty"`

	// Language: The official language of the community.
	Language string `json:"language,omitempty"`

	// Links: List of resources for the community.
	Links []*OrkutLinkResource `json:"links,omitempty"`

	// Location: The location of the community.
	Location string `json:"location,omitempty"`

	// Member_count: The number of users who are part of the community. This
	// number may be approximate, so do not rely on it for iteration.
	Member_count int64 `json:"member_count,omitempty"`

	// Moderators: The list of moderators of the community.
	Moderators []*OrkutAuthorResource `json:"moderators,omitempty"`

	// Name: The name of the community.
	Name string `json:"name,omitempty"`

	// Owner: The person who owns the community.
	Owner *OrkutAuthorResource `json:"owner,omitempty"`

	// Photo_url: The photo of the community.
	Photo_url string `json:"photo_url,omitempty"`
}

type CommunityList struct {
	// Items: List of communities retrieved.
	Items []*Community `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of communities. Value:
	// "orkut#communityList"
	Kind string `json:"kind,omitempty"`
}

type CommunityMembers struct {
	// CommunityMembershipStatus: Status and permissions of the user related
	// to the community.
	CommunityMembershipStatus *CommunityMembershipStatus `json:"communityMembershipStatus,omitempty"`

	// Kind: Kind of this item. Always orkut#communityMembers.
	Kind string `json:"kind,omitempty"`

	// Person: Description of the community member.
	Person *OrkutActivitypersonResource `json:"person,omitempty"`
}

type CommunityMembersList struct {
	// FirstPageToken: The value of pageToken query parameter in
	// community_members.list request to get the first page.
	FirstPageToken string `json:"firstPageToken,omitempty"`

	// Items: List of community members retrieved.
	Items []*CommunityMembers `json:"items,omitempty"`

	// Kind: Kind of this item. Always orkut#communityMembersList.
	Kind string `json:"kind,omitempty"`

	// LastPageToken: The value of pageToken query parameter in
	// community_members.list request to get the last page.
	LastPageToken string `json:"lastPageToken,omitempty"`

	// NextPageToken: The value of pageToken query parameter in
	// community_members.list request to get the next page, if there are
	// more to retrieve.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// PrevPageToken: The value of pageToken query parameter in
	// community_members.list request to get the previous page, if there are
	// more to retrieve.
	PrevPageToken string `json:"prevPageToken,omitempty"`
}

type CommunityMembershipStatus struct {
	// CanCreatePoll: Whether the user can create a poll in this community.
	CanCreatePoll bool `json:"canCreatePoll,omitempty"`

	// CanCreateTopic: Whether the user can create a topic in this
	// community.
	CanCreateTopic bool `json:"canCreateTopic,omitempty"`

	// CanShout: Whether the user can perform a shout operation in this
	// community.
	CanShout bool `json:"canShout,omitempty"`

	// IsCoOwner: Whether the session user is a community co-owner.
	IsCoOwner bool `json:"isCoOwner,omitempty"`

	// IsFollowing: Whether the user is following this community.
	IsFollowing bool `json:"isFollowing,omitempty"`

	// IsModerator: Whether the session user is a community moderator.
	IsModerator bool `json:"isModerator,omitempty"`

	// IsOwner: Whether the session user is the community owner.
	IsOwner bool `json:"isOwner,omitempty"`

	// IsRestoreAvailable: Whether the restore operation is available for
	// the community.
	IsRestoreAvailable bool `json:"isRestoreAvailable,omitempty"`

	// IsTakebackAvailable: Whether the take-back operation is available for
	// the community.
	IsTakebackAvailable bool `json:"isTakebackAvailable,omitempty"`

	// Kind: Kind of this item. Always orkut#communityMembershipStatus.
	Kind string `json:"kind,omitempty"`

	// Status: The status of the current link between the community and the
	// user.
	Status string `json:"status,omitempty"`
}

type CommunityMessage struct {
	// AddedDate: The timestamp of the date when the message was added, in
	// RFC 3339 format.
	AddedDate string `json:"addedDate,omitempty"`

	// Author: The creator of the message. If ommited, the message is
	// annonimous.
	Author *OrkutAuthorResource `json:"author,omitempty"`

	// Body: The body of the message.
	Body string `json:"body,omitempty"`

	// Id: The ID of the message.
	Id int64 `json:"id,omitempty,string"`

	// IsSpam: Whether this post was marked as spam by the viewer, when
	// he/she is not the community owner or one of its moderators.
	IsSpam bool `json:"isSpam,omitempty"`

	// Kind: Identifies this resource as a community message. Value:
	// "orkut#communityMessage"
	Kind string `json:"kind,omitempty"`

	// Links: List of resources for the community message.
	Links []*OrkutLinkResource `json:"links,omitempty"`

	// Subject: The subject of the message.
	Subject string `json:"subject,omitempty"`
}

type CommunityMessageList struct {
	// FirstPageToken: The value of pageToken query parameter in
	// community_messages.list request to get the first page.
	FirstPageToken string `json:"firstPageToken,omitempty"`

	// Items: List of messages retrieved.
	Items []*CommunityMessage `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of community messages.
	// Value: "orkut#communityMessageList"
	Kind string `json:"kind,omitempty"`

	// LastPageToken: The value of pageToken query parameter in
	// community_messages.list request to get the last page.
	LastPageToken string `json:"lastPageToken,omitempty"`

	// NextPageToken: The value of pageToken query parameter in
	// community_messages.list request to get the next page, if there are
	// more to retrieve.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// PrevPageToken: The value of pageToken query parameter in
	// community_messages.list request to get the previous page, if there
	// are more to retrieve.
	PrevPageToken string `json:"prevPageToken,omitempty"`
}

type CommunityPoll struct {
	// Author: The person who created the poll.
	Author *OrkutAuthorResource `json:"author,omitempty"`

	// CommunityId: The ID of the community.
	CommunityId int64 `json:"communityId,omitempty"`

	// CreationTime: The date of creation of this poll
	CreationTime string `json:"creationTime,omitempty"`

	// Description: The poll description.
	Description string `json:"description,omitempty"`

	// EndingTime: The ending date of this poll or empty if the poll doesn't
	// have one.
	EndingTime string `json:"endingTime,omitempty"`

	// HasVoted: Whether the user has voted on this poll.
	HasVoted bool `json:"hasVoted,omitempty"`

	// Id: The poll ID.
	Id string `json:"id,omitempty"`

	// Image: The image representing the poll. Field is omitted if no image
	// exists.
	Image *CommunityPollImage `json:"image,omitempty"`

	// IsClosed: Whether the poll is not expired if there is an expiration
	// date. A poll is open (that is, not closed for voting) if it either is
	// not expired or doesn't have an expiration date at all. Note that just
	// because a poll is open, it doesn't mean that the requester can vote
	// on it.
	IsClosed bool `json:"isClosed,omitempty"`

	// IsMultipleAnswers: Whether this poll allows voting for more than one
	// option.
	IsMultipleAnswers bool `json:"isMultipleAnswers,omitempty"`

	// IsOpenForVoting: Whether this poll is still opened for voting. A poll
	// is open for voting if it is not closed, the user has not yet voted on
	// it and the user has the permission to do so, which happens if he/she
	// is either a community member or the poll is open for everybody.
	IsOpenForVoting bool `json:"isOpenForVoting,omitempty"`

	// IsRestricted: Whether this poll is restricted for members only. If a
	// poll is open but the user can't vote on it, it's been restricted to
	// members only. This information is important to tell this case apart
	// from the one where the user can't vote simply because the poll is
	// already closed.
	IsRestricted bool `json:"isRestricted,omitempty"`

	// IsSpam: Whether the user has marked this poll as spam. This only
	// affects the poll for this user, not globally.
	IsSpam bool `json:"isSpam,omitempty"`

	// IsUsersVotePublic: If user has already voted, whether his vote is
	// publicly visible.
	IsUsersVotePublic bool `json:"isUsersVotePublic,omitempty"`

	// IsVotingAllowedForNonMembers: Whether non-members of the community
	// can vote on the poll.
	IsVotingAllowedForNonMembers bool `json:"isVotingAllowedForNonMembers,omitempty"`

	// Kind: Identifies this resource as a community poll. Value:
	// "orkut#communityPoll"
	Kind string `json:"kind,omitempty"`

	// LastUpdate: The date of the last update of this poll.
	LastUpdate string `json:"lastUpdate,omitempty"`

	// Links: List of resources for the community poll.
	Links []*OrkutLinkResource `json:"links,omitempty"`

	// Options: List of options of this poll.
	Options []*OrkutCommunitypolloptionResource `json:"options,omitempty"`

	// Question: The poll question.
	Question string `json:"question,omitempty"`

	// TotalNumberOfVotes: The total number of votes this poll has received.
	TotalNumberOfVotes int64 `json:"totalNumberOfVotes,omitempty"`

	// VotedOptions: List of options the user has voted on, if there are
	// any.
	VotedOptions []int64 `json:"votedOptions,omitempty"`
}

type CommunityPollImage struct {
	// Url: A URL that points to an image of the poll.
	Url string `json:"url,omitempty"`
}

type CommunityPollComment struct {
	// AddedDate: The date when the message was added, in RFC 3339 format.
	AddedDate string `json:"addedDate,omitempty"`

	// Author: The creator of the comment.
	Author *OrkutAuthorResource `json:"author,omitempty"`

	// Body: The body of the message.
	Body string `json:"body,omitempty"`

	// Id: The ID of the comment.
	Id int64 `json:"id,omitempty"`

	// Kind: Identifies this resource as a community poll comment. Value:
	// "orkut#communityPollComment"
	Kind string `json:"kind,omitempty"`
}

type CommunityPollCommentList struct {
	// FirstPageToken: The value of pageToken query parameter in
	// community_poll_comments.list request to get the first page.
	FirstPageToken string `json:"firstPageToken,omitempty"`

	// Items: List of community poll comments retrieved.
	Items []*CommunityPollComment `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of community poll
	// comments. Value: "orkut#CommunityPollCommentList"
	Kind string `json:"kind,omitempty"`

	// LastPageToken: The value of pageToken query parameter in
	// community_poll_comments.list request to get the last page.
	LastPageToken string `json:"lastPageToken,omitempty"`

	// NextPageToken: The value of pageToken query parameter in
	// community_poll_comments.list request to get the next page, if there
	// are more to retrieve.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// PrevPageToken: The value of pageToken query parameter in
	// community_poll_comments.list request to get the previous page, if
	// there are more to retrieve.
	PrevPageToken string `json:"prevPageToken,omitempty"`
}

type CommunityPollList struct {
	// FirstPageToken: The value of pageToken query parameter in
	// community_polls.list request to get the first page.
	FirstPageToken string `json:"firstPageToken,omitempty"`

	// Items: List of community polls retrieved.
	Items []*CommunityPoll `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of community polls.
	// Value: "orkut#communityPollList"
	Kind string `json:"kind,omitempty"`

	// LastPageToken: The value of pageToken query parameter in
	// community_polls.list request to get the last page.
	LastPageToken string `json:"lastPageToken,omitempty"`

	// NextPageToken: The value of pageToken query parameter in
	// community_polls.list request to get the next page, if there are more
	// to retrieve.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// PrevPageToken: The value of pageToken query parameter in
	// community_polls.list request to get the previous page, if there are
	// more to retrieve.
	PrevPageToken string `json:"prevPageToken,omitempty"`
}

type CommunityPollVote struct {
	// IsVotevisible: Whether this vote is visible to other users or not.
	IsVotevisible bool `json:"isVotevisible,omitempty"`

	// Kind: Identifies this resource as a community poll vote. Value:
	// "orkut#communityPollVote"
	Kind string `json:"kind,omitempty"`

	// OptionIds: The ids of the voted options.
	OptionIds []int64 `json:"optionIds,omitempty"`
}

type CommunityTopic struct {
	// Author: The creator of the topic.
	Author *OrkutAuthorResource `json:"author,omitempty"`

	// Body: The body of the topic.
	Body string `json:"body,omitempty"`

	// Id: The ID of the topic.
	Id int64 `json:"id,omitempty,string"`

	// IsClosed: Whether the topic is closed for new messages.
	IsClosed bool `json:"isClosed,omitempty"`

	// Kind: Identifies this resource as a community topic. Value:
	// "orkut#communityTopic"
	Kind string `json:"kind,omitempty"`

	// LastUpdate: The timestamp of the last update, in RFC 3339 format.
	LastUpdate string `json:"lastUpdate,omitempty"`

	// LatestMessageSnippet: Snippet of the last message posted on this
	// topic.
	LatestMessageSnippet string `json:"latestMessageSnippet,omitempty"`

	// Links: List of resources for the community.
	Links []*OrkutLinkResource `json:"links,omitempty"`

	// Messages: Most recent messages.
	Messages []*CommunityMessage `json:"messages,omitempty"`

	// NumberOfReplies: The total number of replies this topic has received.
	NumberOfReplies int64 `json:"numberOfReplies,omitempty"`

	// Title: The title of the topic.
	Title string `json:"title,omitempty"`
}

type CommunityTopicList struct {
	// FirstPageToken: The value of pageToken query parameter in
	// community_topic.list request to get the first page.
	FirstPageToken string `json:"firstPageToken,omitempty"`

	// Items: List of topics retrieved.
	Items []*CommunityTopic `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of community topics.
	// Value: "orkut#communityTopicList"
	Kind string `json:"kind,omitempty"`

	// LastPageToken: The value of pageToken query parameter in
	// community_topic.list request to get the last page.
	LastPageToken string `json:"lastPageToken,omitempty"`

	// NextPageToken: The value of pageToken query parameter in
	// community_topic.list request to get the next page, if there are more
	// to retrieve.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// PrevPageToken: The value of pageToken query parameter in
	// community_topic.list request to get the previous page, if there are
	// more to retrieve.
	PrevPageToken string `json:"prevPageToken,omitempty"`
}

type Counters struct {
	// Items: List of counters retrieved.
	Items []*OrkutCounterResource `json:"items,omitempty"`

	// Kind: Identifies this resource as a collection of counters. Value:
	// "orkut#counters"
	Kind string `json:"kind,omitempty"`
}

type OrkutActivityobjectsResource struct {
	// Community: The community which is related with this activity, e.g. a
	// joined community.
	Community *Community `json:"community,omitempty"`

	// Content: The HTML-formatted content, suitable for display. When
	// updating an activity's content, post the changes to this property,
	// using the value of originalContent as a starting point. If the update
	// is successful, the server adds HTML formatting and responds with this
	// formatted content.
	Content string `json:"content,omitempty"`

	// DisplayName: The title of the object.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The ID for the object.
	Id string `json:"id,omitempty"`

	// Links: Links to other resources related to this object.
	Links []*OrkutLinkResource `json:"links,omitempty"`

	// ObjectType: The object type.
	ObjectType string `json:"objectType,omitempty"`

	// Person: The person who is related with this activity, e.g. an Added
	// User.
	Person *OrkutActivitypersonResource `json:"person,omitempty"`
}

type OrkutActivitypersonResource struct {
	// Birthday: The person's date of birth, represented as YYYY-MM-DD.
	Birthday string `json:"birthday,omitempty"`

	// Gender: The person's gender. Values include "male", "female", and
	// "other".
	Gender string `json:"gender,omitempty"`

	// Id: The person's opensocial ID.
	Id string `json:"id,omitempty"`

	// Image: The person's profile photo. This is adapted from Google+ and
	// was originaly introduced as extra OpenSocial convenience fields.
	Image *OrkutActivitypersonResourceImage `json:"image,omitempty"`

	// Name: An object that encapsulates the individual components of a
	// person's name.
	Name *OrkutActivitypersonResourceName `json:"name,omitempty"`

	// Url: The person's profile url. This is adapted from Google+ and was
	// originaly introduced as extra OpenSocial convenience fields.
	Url string `json:"url,omitempty"`
}

type OrkutActivitypersonResourceImage struct {
	// Url: The URL of the person's profile photo.
	Url string `json:"url,omitempty"`
}

type OrkutActivitypersonResourceName struct {
	// FamilyName: The family name (last name) of this person.
	FamilyName string `json:"familyName,omitempty"`

	// GivenName: The given name (first name) of this person.
	GivenName string `json:"givenName,omitempty"`
}

type OrkutAuthorResource struct {
	// DisplayName: The name of the author, suitable for display.
	DisplayName string `json:"displayName,omitempty"`

	// Id: Unique identifier of the person who posted the comment. This is
	// the person's OpenSocial ID.
	Id string `json:"id,omitempty"`

	// Image: Image data about the author.
	Image *OrkutAuthorResourceImage `json:"image,omitempty"`

	// Url: The URL of the author who posted the comment [not yet
	// implemented]
	Url string `json:"url,omitempty"`
}

type OrkutAuthorResourceImage struct {
	// Url: A URL that points to a thumbnail photo of the author.
	Url string `json:"url,omitempty"`
}

type OrkutCommunitypolloptionResource struct {
	// Description: The option description.
	Description string `json:"description,omitempty"`

	// Image: Image data about the poll option. Field is omitted if no image
	// exists.
	Image *OrkutCommunitypolloptionResourceImage `json:"image,omitempty"`

	// NumberOfVotes: The total number of votes that this option received.
	NumberOfVotes int64 `json:"numberOfVotes,omitempty"`

	// OptionId: The poll option ID
	OptionId int64 `json:"optionId,omitempty"`
}

type OrkutCommunitypolloptionResourceImage struct {
	// Url: A URL that points to an image of the poll question.
	Url string `json:"url,omitempty"`
}

type OrkutCounterResource struct {
	// Link: Link to the collection being counted.
	Link *OrkutLinkResource `json:"link,omitempty"`

	// Name: The name of the counted collection. Currently supported
	// collections are:
	// - scraps - The scraps of the user.
	// - photos - The
	// photos of the user.
	// - videos - The videos of the user.
	// -
	// pendingTestimonials - The pending testimonials of the user.
	Name string `json:"name,omitempty"`

	// Total: The number of resources on the counted collection.
	Total int64 `json:"total,omitempty"`
}

type OrkutLinkResource struct {
	// Href: URL of the link.
	Href string `json:"href,omitempty"`

	// Rel: Relation between the resource and the parent object.
	Rel string `json:"rel,omitempty"`

	// Title: Title of the link.
	Title string `json:"title,omitempty"`

	// Type: Media type of the link.
	Type string `json:"type,omitempty"`
}

type Visibility struct {
	// Kind: Identifies this resource as a visibility item. Value:
	// "orkut#visibility"
	Kind string `json:"kind,omitempty"`

	// Links: List of resources for the visibility item.
	Links []*OrkutLinkResource `json:"links,omitempty"`

	// Visibility: The visibility of the resource. Possible values are:
	// -
	// default: not hidden by the user
	// - hidden: hidden
	Visibility string `json:"visibility,omitempty"`
}

// method id "orkut.acl.delete":

type AclDeleteCall struct {
	s          *Service
	activityId string
	userId     string
	opt_       map[string]interface{}
}

// Delete: Excludes an element from the ACL of the activity.
func (r *AclService) Delete(activityId string, userId string) *AclDeleteCall {
	c := &AclDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.activityId = activityId
	c.userId = userId
	return c
}

func (c *AclDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "activities/{activityId}/acl/{userId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{activityId}", url.QueryEscape(c.activityId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	//   "description": "Excludes an element from the ACL of the activity.",
	//   "httpMethod": "DELETE",
	//   "id": "orkut.acl.delete",
	//   "parameterOrder": [
	//     "activityId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "activityId": {
	//       "description": "ID of the activity.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "ID of the user to be removed from the activity.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "activities/{activityId}/acl/{userId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.activities.delete":

type ActivitiesDeleteCall struct {
	s          *Service
	activityId string
	opt_       map[string]interface{}
}

// Delete: Deletes an existing activity, if the access controls allow
// it.
func (r *ActivitiesService) Delete(activityId string) *ActivitiesDeleteCall {
	c := &ActivitiesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.activityId = activityId
	return c
}

func (c *ActivitiesDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "activities/{activityId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{activityId}", url.QueryEscape(c.activityId), 1)
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
	//   "description": "Deletes an existing activity, if the access controls allow it.",
	//   "httpMethod": "DELETE",
	//   "id": "orkut.activities.delete",
	//   "parameterOrder": [
	//     "activityId"
	//   ],
	//   "parameters": {
	//     "activityId": {
	//       "description": "ID of the activity to remove.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "activities/{activityId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.activities.list":

type ActivitiesListCall struct {
	s          *Service
	userId     string
	collection string
	opt_       map[string]interface{}
}

// List: Retrieves a list of activities.
func (r *ActivitiesService) List(userId string, collection string) *ActivitiesListCall {
	c := &ActivitiesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	c.collection = collection
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *ActivitiesListCall) Hl(hl string) *ActivitiesListCall {
	c.opt_["hl"] = hl
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of activities to include in the response.
func (c *ActivitiesListCall) MaxResults(maxResults int64) *ActivitiesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token that allows pagination.
func (c *ActivitiesListCall) PageToken(pageToken string) *ActivitiesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *ActivitiesListCall) Do() (*ActivityList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "people/{userId}/activities/{collection}")
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
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(ActivityList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of activities.",
	//   "httpMethod": "GET",
	//   "id": "orkut.activities.list",
	//   "parameterOrder": [
	//     "userId",
	//     "collection"
	//   ],
	//   "parameters": {
	//     "collection": {
	//       "description": "The collection of activities to list.",
	//       "enum": [
	//         "all",
	//         "scraps",
	//         "stream"
	//       ],
	//       "enumDescriptions": [
	//         "All activities created by the specified user that the authenticated user is authorized to view.",
	//         "The specified user's scrapbook.",
	//         "The specified user's stream feed, intended for consumption. This includes activities posted by people that the user is following, and activities in which the user has been mentioned."
	//       ],
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of activities to include in the response.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token that allows pagination.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user whose activities will be listed. Can be me to refer to the viewer (i.e. the authenticated user).",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}/activities/{collection}",
	//   "response": {
	//     "$ref": "ActivityList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.activityVisibility.get":

type ActivityVisibilityGetCall struct {
	s          *Service
	activityId string
	opt_       map[string]interface{}
}

// Get: Gets the visibility of an existing activity.
func (r *ActivityVisibilityService) Get(activityId string) *ActivityVisibilityGetCall {
	c := &ActivityVisibilityGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.activityId = activityId
	return c
}

func (c *ActivityVisibilityGetCall) Do() (*Visibility, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "activities/{activityId}/visibility")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{activityId}", url.QueryEscape(c.activityId), 1)
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
	ret := new(Visibility)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets the visibility of an existing activity.",
	//   "httpMethod": "GET",
	//   "id": "orkut.activityVisibility.get",
	//   "parameterOrder": [
	//     "activityId"
	//   ],
	//   "parameters": {
	//     "activityId": {
	//       "description": "ID of the activity to get the visibility.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "activities/{activityId}/visibility",
	//   "response": {
	//     "$ref": "Visibility"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.activityVisibility.patch":

type ActivityVisibilityPatchCall struct {
	s          *Service
	activityId string
	visibility *Visibility
	opt_       map[string]interface{}
}

// Patch: Updates the visibility of an existing activity. This method
// supports patch semantics.
func (r *ActivityVisibilityService) Patch(activityId string, visibility *Visibility) *ActivityVisibilityPatchCall {
	c := &ActivityVisibilityPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.activityId = activityId
	c.visibility = visibility
	return c
}

func (c *ActivityVisibilityPatchCall) Do() (*Visibility, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.visibility)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "activities/{activityId}/visibility")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{activityId}", url.QueryEscape(c.activityId), 1)
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
	ret := new(Visibility)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the visibility of an existing activity. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "orkut.activityVisibility.patch",
	//   "parameterOrder": [
	//     "activityId"
	//   ],
	//   "parameters": {
	//     "activityId": {
	//       "description": "ID of the activity.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "activities/{activityId}/visibility",
	//   "request": {
	//     "$ref": "Visibility"
	//   },
	//   "response": {
	//     "$ref": "Visibility"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.activityVisibility.update":

type ActivityVisibilityUpdateCall struct {
	s          *Service
	activityId string
	visibility *Visibility
	opt_       map[string]interface{}
}

// Update: Updates the visibility of an existing activity.
func (r *ActivityVisibilityService) Update(activityId string, visibility *Visibility) *ActivityVisibilityUpdateCall {
	c := &ActivityVisibilityUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.activityId = activityId
	c.visibility = visibility
	return c
}

func (c *ActivityVisibilityUpdateCall) Do() (*Visibility, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.visibility)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "activities/{activityId}/visibility")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{activityId}", url.QueryEscape(c.activityId), 1)
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
	ret := new(Visibility)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Updates the visibility of an existing activity.",
	//   "httpMethod": "PUT",
	//   "id": "orkut.activityVisibility.update",
	//   "parameterOrder": [
	//     "activityId"
	//   ],
	//   "parameters": {
	//     "activityId": {
	//       "description": "ID of the activity.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "activities/{activityId}/visibility",
	//   "request": {
	//     "$ref": "Visibility"
	//   },
	//   "response": {
	//     "$ref": "Visibility"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.badges.get":

type BadgesGetCall struct {
	s       *Service
	userId  string
	badgeId int64
	opt_    map[string]interface{}
}

// Get: Retrieves a badge from a user.
func (r *BadgesService) Get(userId string, badgeId int64) *BadgesGetCall {
	c := &BadgesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	c.badgeId = badgeId
	return c
}

func (c *BadgesGetCall) Do() (*Badge, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "people/{userId}/badges/{badgeId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{badgeId}", strconv.FormatInt(c.badgeId, 10), 1)
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
	ret := new(Badge)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a badge from a user.",
	//   "httpMethod": "GET",
	//   "id": "orkut.badges.get",
	//   "parameterOrder": [
	//     "userId",
	//     "badgeId"
	//   ],
	//   "parameters": {
	//     "badgeId": {
	//       "description": "The ID of the badge that will be retrieved.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user whose badges will be listed. Can be me to refer to caller.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}/badges/{badgeId}",
	//   "response": {
	//     "$ref": "Badge"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.badges.list":

type BadgesListCall struct {
	s      *Service
	userId string
	opt_   map[string]interface{}
}

// List: Retrieves the list of visible badges of a user.
func (r *BadgesService) List(userId string) *BadgesListCall {
	c := &BadgesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	return c
}

func (c *BadgesListCall) Do() (*BadgeList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "people/{userId}/badges")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	ret := new(BadgeList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of visible badges of a user.",
	//   "httpMethod": "GET",
	//   "id": "orkut.badges.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "description": "The id of the user whose badges will be listed. Can be me to refer to caller.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}/badges",
	//   "response": {
	//     "$ref": "BadgeList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.comments.delete":

type CommentsDeleteCall struct {
	s         *Service
	commentId string
	opt_      map[string]interface{}
}

// Delete: Deletes an existing comment.
func (r *CommentsService) Delete(commentId string) *CommentsDeleteCall {
	c := &CommentsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.commentId = commentId
	return c
}

func (c *CommentsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "comments/{commentId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{commentId}", url.QueryEscape(c.commentId), 1)
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
	//   "description": "Deletes an existing comment.",
	//   "httpMethod": "DELETE",
	//   "id": "orkut.comments.delete",
	//   "parameterOrder": [
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "commentId": {
	//       "description": "ID of the comment to remove.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "comments/{commentId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.comments.get":

type CommentsGetCall struct {
	s         *Service
	commentId string
	opt_      map[string]interface{}
}

// Get: Retrieves an existing comment.
func (r *CommentsService) Get(commentId string) *CommentsGetCall {
	c := &CommentsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.commentId = commentId
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommentsGetCall) Hl(hl string) *CommentsGetCall {
	c.opt_["hl"] = hl
	return c
}

func (c *CommentsGetCall) Do() (*Comment, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "comments/{commentId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{commentId}", url.QueryEscape(c.commentId), 1)
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
	ret := new(Comment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves an existing comment.",
	//   "httpMethod": "GET",
	//   "id": "orkut.comments.get",
	//   "parameterOrder": [
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "commentId": {
	//       "description": "ID of the comment to get.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "comments/{commentId}",
	//   "response": {
	//     "$ref": "Comment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.comments.insert":

type CommentsInsertCall struct {
	s          *Service
	activityId string
	comment    *Comment
	opt_       map[string]interface{}
}

// Insert: Inserts a new comment to an activity.
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "activities/{activityId}/comments")
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
	defer googleapi.CloseBody(res)
	if err := googleapi.CheckResponse(res); err != nil {
		return nil, err
	}
	ret := new(Comment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Inserts a new comment to an activity.",
	//   "httpMethod": "POST",
	//   "id": "orkut.comments.insert",
	//   "parameterOrder": [
	//     "activityId"
	//   ],
	//   "parameters": {
	//     "activityId": {
	//       "description": "The ID of the activity to contain the new comment.",
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
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.comments.list":

type CommentsListCall struct {
	s          *Service
	activityId string
	opt_       map[string]interface{}
}

// List: Retrieves a list of comments, possibly filtered.
func (r *CommentsService) List(activityId string) *CommentsListCall {
	c := &CommentsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.activityId = activityId
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommentsListCall) Hl(hl string) *CommentsListCall {
	c.opt_["hl"] = hl
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of activities to include in the response.
func (c *CommentsListCall) MaxResults(maxResults int64) *CommentsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// OrderBy sets the optional parameter "orderBy": Sort search results.
func (c *CommentsListCall) OrderBy(orderBy string) *CommentsListCall {
	c.opt_["orderBy"] = orderBy
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token that allows pagination.
func (c *CommentsListCall) PageToken(pageToken string) *CommentsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *CommentsListCall) Do() (*CommentList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
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
	urls := googleapi.ResolveRelative(c.s.BasePath, "activities/{activityId}/comments")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{activityId}", url.QueryEscape(c.activityId), 1)
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
	ret := new(CommentList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of comments, possibly filtered.",
	//   "httpMethod": "GET",
	//   "id": "orkut.comments.list",
	//   "parameterOrder": [
	//     "activityId"
	//   ],
	//   "parameters": {
	//     "activityId": {
	//       "description": "The ID of the activity containing the comments.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of activities to include in the response.",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "orderBy": {
	//       "default": "DESCENDING_SORT",
	//       "description": "Sort search results.",
	//       "enum": [
	//         "ascending",
	//         "descending"
	//       ],
	//       "enumDescriptions": [
	//         "Use ascending sort order.",
	//         "Use descending sort order."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token that allows pagination.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "activities/{activityId}/comments",
	//   "response": {
	//     "$ref": "CommentList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.communities.get":

type CommunitiesGetCall struct {
	s           *Service
	communityId int64
	opt_        map[string]interface{}
}

// Get: Retrieves the basic information (aka. profile) of a community.
func (r *CommunitiesService) Get(communityId int64) *CommunitiesGetCall {
	c := &CommunitiesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommunitiesGetCall) Hl(hl string) *CommunitiesGetCall {
	c.opt_["hl"] = hl
	return c
}

func (c *CommunitiesGetCall) Do() (*Community, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
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
	ret := new(Community)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the basic information (aka. profile) of a community.",
	//   "httpMethod": "GET",
	//   "id": "orkut.communities.get",
	//   "parameterOrder": [
	//     "communityId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community to get.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}",
	//   "response": {
	//     "$ref": "Community"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.communities.list":

type CommunitiesListCall struct {
	s      *Service
	userId string
	opt_   map[string]interface{}
}

// List: Retrieves the list of communities the current user is a member
// of.
func (r *CommunitiesService) List(userId string) *CommunitiesListCall {
	c := &CommunitiesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommunitiesListCall) Hl(hl string) *CommunitiesListCall {
	c.opt_["hl"] = hl
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of communities to include in the response.
func (c *CommunitiesListCall) MaxResults(maxResults int64) *CommunitiesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// OrderBy sets the optional parameter "orderBy": How to order the
// communities by.
func (c *CommunitiesListCall) OrderBy(orderBy string) *CommunitiesListCall {
	c.opt_["orderBy"] = orderBy
	return c
}

func (c *CommunitiesListCall) Do() (*CommunityList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["orderBy"]; ok {
		params.Set("orderBy", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "people/{userId}/communities")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	ret := new(CommunityList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the list of communities the current user is a member of.",
	//   "httpMethod": "GET",
	//   "id": "orkut.communities.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of communities to include in the response.",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "orderBy": {
	//       "description": "How to order the communities by.",
	//       "enum": [
	//         "id",
	//         "ranked"
	//       ],
	//       "enumDescriptions": [
	//         "Returns the communities sorted by a fixed, natural order.",
	//         "Returns the communities ranked accordingly to how they are displayed on the orkut web application."
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "The ID of the user whose communities will be listed. Can be me to refer to caller.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}/communities",
	//   "response": {
	//     "$ref": "CommunityList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.communityFollow.delete":

type CommunityFollowDeleteCall struct {
	s           *Service
	communityId int64
	userId      string
	opt_        map[string]interface{}
}

// Delete: Removes a user from the followers of a community.
func (r *CommunityFollowService) Delete(communityId int64, userId string) *CommunityFollowDeleteCall {
	c := &CommunityFollowDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.userId = userId
	return c
}

func (c *CommunityFollowDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/followers/{userId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	//   "description": "Removes a user from the followers of a community.",
	//   "httpMethod": "DELETE",
	//   "id": "orkut.communityFollow.delete",
	//   "parameterOrder": [
	//     "communityId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "ID of the community.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "userId": {
	//       "description": "ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/followers/{userId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.communityFollow.insert":

type CommunityFollowInsertCall struct {
	s           *Service
	communityId int64
	userId      string
	opt_        map[string]interface{}
}

// Insert: Adds a user as a follower of a community.
func (r *CommunityFollowService) Insert(communityId int64, userId string) *CommunityFollowInsertCall {
	c := &CommunityFollowInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.userId = userId
	return c
}

func (c *CommunityFollowInsertCall) Do() (*CommunityMembers, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/followers/{userId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	ret := new(CommunityMembers)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds a user as a follower of a community.",
	//   "httpMethod": "POST",
	//   "id": "orkut.communityFollow.insert",
	//   "parameterOrder": [
	//     "communityId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "ID of the community.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "userId": {
	//       "description": "ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/followers/{userId}",
	//   "response": {
	//     "$ref": "CommunityMembers"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.communityMembers.delete":

type CommunityMembersDeleteCall struct {
	s           *Service
	communityId int64
	userId      string
	opt_        map[string]interface{}
}

// Delete: Makes the user leave a community.
func (r *CommunityMembersService) Delete(communityId int64, userId string) *CommunityMembersDeleteCall {
	c := &CommunityMembersDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.userId = userId
	return c
}

func (c *CommunityMembersDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/members/{userId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	//   "description": "Makes the user leave a community.",
	//   "httpMethod": "DELETE",
	//   "id": "orkut.communityMembers.delete",
	//   "parameterOrder": [
	//     "communityId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "ID of the community.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "userId": {
	//       "description": "ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/members/{userId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.communityMembers.get":

type CommunityMembersGetCall struct {
	s           *Service
	communityId int64
	userId      string
	opt_        map[string]interface{}
}

// Get: Retrieves the relationship between a user and a community.
func (r *CommunityMembersService) Get(communityId int64, userId string) *CommunityMembersGetCall {
	c := &CommunityMembersGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.userId = userId
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommunityMembersGetCall) Hl(hl string) *CommunityMembersGetCall {
	c.opt_["hl"] = hl
	return c
}

func (c *CommunityMembersGetCall) Do() (*CommunityMembers, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/members/{userId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	ret := new(CommunityMembers)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the relationship between a user and a community.",
	//   "httpMethod": "GET",
	//   "id": "orkut.communityMembers.get",
	//   "parameterOrder": [
	//     "communityId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "ID of the community.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/members/{userId}",
	//   "response": {
	//     "$ref": "CommunityMembers"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.communityMembers.insert":

type CommunityMembersInsertCall struct {
	s           *Service
	communityId int64
	userId      string
	opt_        map[string]interface{}
}

// Insert: Makes the user join a community.
func (r *CommunityMembersService) Insert(communityId int64, userId string) *CommunityMembersInsertCall {
	c := &CommunityMembersInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.userId = userId
	return c
}

func (c *CommunityMembersInsertCall) Do() (*CommunityMembers, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/members/{userId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	ret := new(CommunityMembers)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Makes the user join a community.",
	//   "httpMethod": "POST",
	//   "id": "orkut.communityMembers.insert",
	//   "parameterOrder": [
	//     "communityId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "ID of the community.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "userId": {
	//       "description": "ID of the user.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/members/{userId}",
	//   "response": {
	//     "$ref": "CommunityMembers"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.communityMembers.list":

type CommunityMembersListCall struct {
	s           *Service
	communityId int64
	opt_        map[string]interface{}
}

// List: Lists members of a community. Use the pagination tokens to
// retrieve the full list; do not rely on the member count available in
// the community profile information to know when to stop iterating, as
// that count may be approximate.
func (r *CommunityMembersService) List(communityId int64) *CommunityMembersListCall {
	c := &CommunityMembersListCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	return c
}

// FriendsOnly sets the optional parameter "friendsOnly": Whether to
// list only community members who are friends of the user.
func (c *CommunityMembersListCall) FriendsOnly(friendsOnly bool) *CommunityMembersListCall {
	c.opt_["friendsOnly"] = friendsOnly
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommunityMembersListCall) Hl(hl string) *CommunityMembersListCall {
	c.opt_["hl"] = hl
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of members to include in the response.
func (c *CommunityMembersListCall) MaxResults(maxResults int64) *CommunityMembersListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token that allows pagination.
func (c *CommunityMembersListCall) PageToken(pageToken string) *CommunityMembersListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *CommunityMembersListCall) Do() (*CommunityMembersList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["friendsOnly"]; ok {
		params.Set("friendsOnly", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/members")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
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
	ret := new(CommunityMembersList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Lists members of a community. Use the pagination tokens to retrieve the full list; do not rely on the member count available in the community profile information to know when to stop iterating, as that count may be approximate.",
	//   "httpMethod": "GET",
	//   "id": "orkut.communityMembers.list",
	//   "parameterOrder": [
	//     "communityId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community whose members will be listed.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "friendsOnly": {
	//       "description": "Whether to list only community members who are friends of the user.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of members to include in the response.",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token that allows pagination.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/members",
	//   "response": {
	//     "$ref": "CommunityMembersList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.communityMessages.delete":

type CommunityMessagesDeleteCall struct {
	s           *Service
	communityId int64
	topicId     int64
	messageId   int64
	opt_        map[string]interface{}
}

// Delete: Moves a message of the community to the trash folder.
func (r *CommunityMessagesService) Delete(communityId int64, topicId int64, messageId int64) *CommunityMessagesDeleteCall {
	c := &CommunityMessagesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.topicId = topicId
	c.messageId = messageId
	return c
}

func (c *CommunityMessagesDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/topics/{topicId}/messages/{messageId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{topicId}", strconv.FormatInt(c.topicId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{messageId}", strconv.FormatInt(c.messageId, 10), 1)
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
	//   "description": "Moves a message of the community to the trash folder.",
	//   "httpMethod": "DELETE",
	//   "id": "orkut.communityMessages.delete",
	//   "parameterOrder": [
	//     "communityId",
	//     "topicId",
	//     "messageId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community whose message will be moved to the trash folder.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "messageId": {
	//       "description": "The ID of the message to be moved to the trash folder.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "topicId": {
	//       "description": "The ID of the topic whose message will be moved to the trash folder.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/topics/{topicId}/messages/{messageId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.communityMessages.insert":

type CommunityMessagesInsertCall struct {
	s                *Service
	communityId      int64
	topicId          int64
	communitymessage *CommunityMessage
	opt_             map[string]interface{}
}

// Insert: Adds a message to a given community topic.
func (r *CommunityMessagesService) Insert(communityId int64, topicId int64, communitymessage *CommunityMessage) *CommunityMessagesInsertCall {
	c := &CommunityMessagesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.topicId = topicId
	c.communitymessage = communitymessage
	return c
}

func (c *CommunityMessagesInsertCall) Do() (*CommunityMessage, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.communitymessage)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/topics/{topicId}/messages")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{topicId}", strconv.FormatInt(c.topicId, 10), 1)
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
	ret := new(CommunityMessage)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds a message to a given community topic.",
	//   "httpMethod": "POST",
	//   "id": "orkut.communityMessages.insert",
	//   "parameterOrder": [
	//     "communityId",
	//     "topicId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community the message should be added to.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "topicId": {
	//       "description": "The ID of the topic the message should be added to.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/topics/{topicId}/messages",
	//   "request": {
	//     "$ref": "CommunityMessage"
	//   },
	//   "response": {
	//     "$ref": "CommunityMessage"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.communityMessages.list":

type CommunityMessagesListCall struct {
	s           *Service
	communityId int64
	topicId     int64
	opt_        map[string]interface{}
}

// List: Retrieves the messages of a topic of a community.
func (r *CommunityMessagesService) List(communityId int64, topicId int64) *CommunityMessagesListCall {
	c := &CommunityMessagesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.topicId = topicId
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommunityMessagesListCall) Hl(hl string) *CommunityMessagesListCall {
	c.opt_["hl"] = hl
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of messages to include in the response.
func (c *CommunityMessagesListCall) MaxResults(maxResults int64) *CommunityMessagesListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token that allows pagination.
func (c *CommunityMessagesListCall) PageToken(pageToken string) *CommunityMessagesListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *CommunityMessagesListCall) Do() (*CommunityMessageList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/topics/{topicId}/messages")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{topicId}", strconv.FormatInt(c.topicId, 10), 1)
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
	ret := new(CommunityMessageList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the messages of a topic of a community.",
	//   "httpMethod": "GET",
	//   "id": "orkut.communityMessages.list",
	//   "parameterOrder": [
	//     "communityId",
	//     "topicId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community which messages will be listed.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of messages to include in the response.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token that allows pagination.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "topicId": {
	//       "description": "The ID of the topic which messages will be listed.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/topics/{topicId}/messages",
	//   "response": {
	//     "$ref": "CommunityMessageList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.communityPollComments.insert":

type CommunityPollCommentsInsertCall struct {
	s                    *Service
	communityId          int64
	pollId               string
	communitypollcomment *CommunityPollComment
	opt_                 map[string]interface{}
}

// Insert: Adds a comment on a community poll.
func (r *CommunityPollCommentsService) Insert(communityId int64, pollId string, communitypollcomment *CommunityPollComment) *CommunityPollCommentsInsertCall {
	c := &CommunityPollCommentsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.pollId = pollId
	c.communitypollcomment = communitypollcomment
	return c
}

func (c *CommunityPollCommentsInsertCall) Do() (*CommunityPollComment, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.communitypollcomment)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/polls/{pollId}/comments")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{pollId}", url.QueryEscape(c.pollId), 1)
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
	ret := new(CommunityPollComment)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds a comment on a community poll.",
	//   "httpMethod": "POST",
	//   "id": "orkut.communityPollComments.insert",
	//   "parameterOrder": [
	//     "communityId",
	//     "pollId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community whose poll is being commented.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "pollId": {
	//       "description": "The ID of the poll being commented.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/polls/{pollId}/comments",
	//   "request": {
	//     "$ref": "CommunityPollComment"
	//   },
	//   "response": {
	//     "$ref": "CommunityPollComment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.communityPollComments.list":

type CommunityPollCommentsListCall struct {
	s           *Service
	communityId int64
	pollId      string
	opt_        map[string]interface{}
}

// List: Retrieves the comments of a community poll.
func (r *CommunityPollCommentsService) List(communityId int64, pollId string) *CommunityPollCommentsListCall {
	c := &CommunityPollCommentsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.pollId = pollId
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommunityPollCommentsListCall) Hl(hl string) *CommunityPollCommentsListCall {
	c.opt_["hl"] = hl
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of comments to include in the response.
func (c *CommunityPollCommentsListCall) MaxResults(maxResults int64) *CommunityPollCommentsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token that allows pagination.
func (c *CommunityPollCommentsListCall) PageToken(pageToken string) *CommunityPollCommentsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *CommunityPollCommentsListCall) Do() (*CommunityPollCommentList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/polls/{pollId}/comments")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{pollId}", url.QueryEscape(c.pollId), 1)
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
	ret := new(CommunityPollCommentList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the comments of a community poll.",
	//   "httpMethod": "GET",
	//   "id": "orkut.communityPollComments.list",
	//   "parameterOrder": [
	//     "communityId",
	//     "pollId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community whose poll is having its comments listed.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of comments to include in the response.",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token that allows pagination.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pollId": {
	//       "description": "The ID of the community whose polls will be listed.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/polls/{pollId}/comments",
	//   "response": {
	//     "$ref": "CommunityPollCommentList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.communityPollVotes.insert":

type CommunityPollVotesInsertCall struct {
	s                 *Service
	communityId       int64
	pollId            string
	communitypollvote *CommunityPollVote
	opt_              map[string]interface{}
}

// Insert: Votes on a community poll.
func (r *CommunityPollVotesService) Insert(communityId int64, pollId string, communitypollvote *CommunityPollVote) *CommunityPollVotesInsertCall {
	c := &CommunityPollVotesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.pollId = pollId
	c.communitypollvote = communitypollvote
	return c
}

func (c *CommunityPollVotesInsertCall) Do() (*CommunityPollVote, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.communitypollvote)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/polls/{pollId}/votes")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{pollId}", url.QueryEscape(c.pollId), 1)
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
	ret := new(CommunityPollVote)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Votes on a community poll.",
	//   "httpMethod": "POST",
	//   "id": "orkut.communityPollVotes.insert",
	//   "parameterOrder": [
	//     "communityId",
	//     "pollId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community whose poll is being voted.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "pollId": {
	//       "description": "The ID of the poll being voted.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/polls/{pollId}/votes",
	//   "request": {
	//     "$ref": "CommunityPollVote"
	//   },
	//   "response": {
	//     "$ref": "CommunityPollVote"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.communityPolls.get":

type CommunityPollsGetCall struct {
	s           *Service
	communityId int64
	pollId      string
	opt_        map[string]interface{}
}

// Get: Retrieves one specific poll of a community.
func (r *CommunityPollsService) Get(communityId int64, pollId string) *CommunityPollsGetCall {
	c := &CommunityPollsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.pollId = pollId
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommunityPollsGetCall) Hl(hl string) *CommunityPollsGetCall {
	c.opt_["hl"] = hl
	return c
}

func (c *CommunityPollsGetCall) Do() (*CommunityPoll, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/polls/{pollId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{pollId}", url.QueryEscape(c.pollId), 1)
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
	ret := new(CommunityPoll)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves one specific poll of a community.",
	//   "httpMethod": "GET",
	//   "id": "orkut.communityPolls.get",
	//   "parameterOrder": [
	//     "communityId",
	//     "pollId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community for whose poll will be retrieved.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pollId": {
	//       "description": "The ID of the poll to get.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/polls/{pollId}",
	//   "response": {
	//     "$ref": "CommunityPoll"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.communityPolls.list":

type CommunityPollsListCall struct {
	s           *Service
	communityId int64
	opt_        map[string]interface{}
}

// List: Retrieves the polls of a community.
func (r *CommunityPollsService) List(communityId int64) *CommunityPollsListCall {
	c := &CommunityPollsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommunityPollsListCall) Hl(hl string) *CommunityPollsListCall {
	c.opt_["hl"] = hl
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of polls to include in the response.
func (c *CommunityPollsListCall) MaxResults(maxResults int64) *CommunityPollsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token that allows pagination.
func (c *CommunityPollsListCall) PageToken(pageToken string) *CommunityPollsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *CommunityPollsListCall) Do() (*CommunityPollList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/polls")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
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
	ret := new(CommunityPollList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the polls of a community.",
	//   "httpMethod": "GET",
	//   "id": "orkut.communityPolls.list",
	//   "parameterOrder": [
	//     "communityId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community which polls will be listed.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of polls to include in the response.",
	//       "format": "uint32",
	//       "location": "query",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token that allows pagination.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/polls",
	//   "response": {
	//     "$ref": "CommunityPollList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.communityRelated.list":

type CommunityRelatedListCall struct {
	s           *Service
	communityId int64
	opt_        map[string]interface{}
}

// List: Retrieves the communities related to another one.
func (r *CommunityRelatedService) List(communityId int64) *CommunityRelatedListCall {
	c := &CommunityRelatedListCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommunityRelatedListCall) Hl(hl string) *CommunityRelatedListCall {
	c.opt_["hl"] = hl
	return c
}

func (c *CommunityRelatedListCall) Do() (*CommunityList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/related")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
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
	ret := new(CommunityList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the communities related to another one.",
	//   "httpMethod": "GET",
	//   "id": "orkut.communityRelated.list",
	//   "parameterOrder": [
	//     "communityId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community whose related communities will be listed.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/related",
	//   "response": {
	//     "$ref": "CommunityList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.communityTopics.delete":

type CommunityTopicsDeleteCall struct {
	s           *Service
	communityId int64
	topicId     int64
	opt_        map[string]interface{}
}

// Delete: Moves a topic of the community to the trash folder.
func (r *CommunityTopicsService) Delete(communityId int64, topicId int64) *CommunityTopicsDeleteCall {
	c := &CommunityTopicsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.topicId = topicId
	return c
}

func (c *CommunityTopicsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/topics/{topicId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{topicId}", strconv.FormatInt(c.topicId, 10), 1)
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
	//   "description": "Moves a topic of the community to the trash folder.",
	//   "httpMethod": "DELETE",
	//   "id": "orkut.communityTopics.delete",
	//   "parameterOrder": [
	//     "communityId",
	//     "topicId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community whose topic will be moved to the trash folder.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "topicId": {
	//       "description": "The ID of the topic to be moved to the trash folder.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/topics/{topicId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.communityTopics.get":

type CommunityTopicsGetCall struct {
	s           *Service
	communityId int64
	topicId     int64
	opt_        map[string]interface{}
}

// Get: Retrieves a topic of a community.
func (r *CommunityTopicsService) Get(communityId int64, topicId int64) *CommunityTopicsGetCall {
	c := &CommunityTopicsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.topicId = topicId
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommunityTopicsGetCall) Hl(hl string) *CommunityTopicsGetCall {
	c.opt_["hl"] = hl
	return c
}

func (c *CommunityTopicsGetCall) Do() (*CommunityTopic, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/topics/{topicId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{topicId}", strconv.FormatInt(c.topicId, 10), 1)
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
	ret := new(CommunityTopic)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a topic of a community.",
	//   "httpMethod": "GET",
	//   "id": "orkut.communityTopics.get",
	//   "parameterOrder": [
	//     "communityId",
	//     "topicId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community whose topic will be retrieved.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "topicId": {
	//       "description": "The ID of the topic to get.",
	//       "format": "int64",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/topics/{topicId}",
	//   "response": {
	//     "$ref": "CommunityTopic"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.communityTopics.insert":

type CommunityTopicsInsertCall struct {
	s              *Service
	communityId    int64
	communitytopic *CommunityTopic
	opt_           map[string]interface{}
}

// Insert: Adds a topic to a given community.
func (r *CommunityTopicsService) Insert(communityId int64, communitytopic *CommunityTopic) *CommunityTopicsInsertCall {
	c := &CommunityTopicsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	c.communitytopic = communitytopic
	return c
}

// IsShout sets the optional parameter "isShout": Whether this topic is
// a shout.
func (c *CommunityTopicsInsertCall) IsShout(isShout bool) *CommunityTopicsInsertCall {
	c.opt_["isShout"] = isShout
	return c
}

func (c *CommunityTopicsInsertCall) Do() (*CommunityTopic, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.communitytopic)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["isShout"]; ok {
		params.Set("isShout", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/topics")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
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
	ret := new(CommunityTopic)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Adds a topic to a given community.",
	//   "httpMethod": "POST",
	//   "id": "orkut.communityTopics.insert",
	//   "parameterOrder": [
	//     "communityId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community the topic should be added to.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "isShout": {
	//       "description": "Whether this topic is a shout.",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "communities/{communityId}/topics",
	//   "request": {
	//     "$ref": "CommunityTopic"
	//   },
	//   "response": {
	//     "$ref": "CommunityTopic"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}

// method id "orkut.communityTopics.list":

type CommunityTopicsListCall struct {
	s           *Service
	communityId int64
	opt_        map[string]interface{}
}

// List: Retrieves the topics of a community.
func (r *CommunityTopicsService) List(communityId int64) *CommunityTopicsListCall {
	c := &CommunityTopicsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.communityId = communityId
	return c
}

// Hl sets the optional parameter "hl": Specifies the interface language
// (host language) of your user interface.
func (c *CommunityTopicsListCall) Hl(hl string) *CommunityTopicsListCall {
	c.opt_["hl"] = hl
	return c
}

// MaxResults sets the optional parameter "maxResults": The maximum
// number of topics to include in the response.
func (c *CommunityTopicsListCall) MaxResults(maxResults int64) *CommunityTopicsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": A continuation
// token that allows pagination.
func (c *CommunityTopicsListCall) PageToken(pageToken string) *CommunityTopicsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

func (c *CommunityTopicsListCall) Do() (*CommunityTopicList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["hl"]; ok {
		params.Set("hl", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "communities/{communityId}/topics")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{communityId}", strconv.FormatInt(c.communityId, 10), 1)
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
	ret := new(CommunityTopicList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the topics of a community.",
	//   "httpMethod": "GET",
	//   "id": "orkut.communityTopics.list",
	//   "parameterOrder": [
	//     "communityId"
	//   ],
	//   "parameters": {
	//     "communityId": {
	//       "description": "The ID of the community which topics will be listed.",
	//       "format": "int32",
	//       "location": "path",
	//       "required": true,
	//       "type": "integer"
	//     },
	//     "hl": {
	//       "description": "Specifies the interface language (host language) of your user interface.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "The maximum number of topics to include in the response.",
	//       "format": "uint32",
	//       "location": "query",
	//       "maximum": "100",
	//       "minimum": "1",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "A continuation token that allows pagination.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "communities/{communityId}/topics",
	//   "response": {
	//     "$ref": "CommunityTopicList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.counters.list":

type CountersListCall struct {
	s      *Service
	userId string
	opt_   map[string]interface{}
}

// List: Retrieves the counters of a user.
func (r *CountersService) List(userId string) *CountersListCall {
	c := &CountersListCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	return c
}

func (c *CountersListCall) Do() (*Counters, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "people/{userId}/counters")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
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
	ret := new(Counters)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the counters of a user.",
	//   "httpMethod": "GET",
	//   "id": "orkut.counters.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "description": "The ID of the user whose counters will be listed. Can be me to refer to caller.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "people/{userId}/counters",
	//   "response": {
	//     "$ref": "Counters"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut",
	//     "https://www.googleapis.com/auth/orkut.readonly"
	//   ]
	// }

}

// method id "orkut.scraps.insert":

type ScrapsInsertCall struct {
	s        *Service
	activity *Activity
	opt_     map[string]interface{}
}

// Insert: Creates a new scrap.
func (r *ScrapsService) Insert(activity *Activity) *ScrapsInsertCall {
	c := &ScrapsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.activity = activity
	return c
}

func (c *ScrapsInsertCall) Do() (*Activity, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.activity)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "activities/scraps")
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
	//   "description": "Creates a new scrap.",
	//   "httpMethod": "POST",
	//   "id": "orkut.scraps.insert",
	//   "path": "activities/scraps",
	//   "request": {
	//     "$ref": "Activity"
	//   },
	//   "response": {
	//     "$ref": "Activity"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/orkut"
	//   ]
	// }

}
