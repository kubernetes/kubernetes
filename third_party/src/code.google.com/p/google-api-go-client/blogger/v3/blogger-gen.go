// Package blogger provides access to the Blogger API.
//
// See https://developers.google.com/blogger/docs/3.0/getting_started
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/blogger/v3"
//   ...
//   bloggerService, err := blogger.New(oauthHttpClient)
package blogger

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

const apiId = "blogger:v3"
const apiName = "blogger"
const apiVersion = "v3"
const basePath = "https://www.googleapis.com/blogger/v3/"

// OAuth2 scopes used by this API.
const (
	// Manage your Blogger account
	BloggerScope = "https://www.googleapis.com/auth/blogger"

	// View your Blogger account
	BloggerReadonlyScope = "https://www.googleapis.com/auth/blogger.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.BlogUserInfos = NewBlogUserInfosService(s)
	s.Blogs = NewBlogsService(s)
	s.Comments = NewCommentsService(s)
	s.PageViews = NewPageViewsService(s)
	s.Pages = NewPagesService(s)
	s.PostUserInfos = NewPostUserInfosService(s)
	s.Posts = NewPostsService(s)
	s.Users = NewUsersService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	BlogUserInfos *BlogUserInfosService

	Blogs *BlogsService

	Comments *CommentsService

	PageViews *PageViewsService

	Pages *PagesService

	PostUserInfos *PostUserInfosService

	Posts *PostsService

	Users *UsersService
}

func NewBlogUserInfosService(s *Service) *BlogUserInfosService {
	rs := &BlogUserInfosService{s: s}
	return rs
}

type BlogUserInfosService struct {
	s *Service
}

func NewBlogsService(s *Service) *BlogsService {
	rs := &BlogsService{s: s}
	return rs
}

type BlogsService struct {
	s *Service
}

func NewCommentsService(s *Service) *CommentsService {
	rs := &CommentsService{s: s}
	return rs
}

type CommentsService struct {
	s *Service
}

func NewPageViewsService(s *Service) *PageViewsService {
	rs := &PageViewsService{s: s}
	return rs
}

type PageViewsService struct {
	s *Service
}

func NewPagesService(s *Service) *PagesService {
	rs := &PagesService{s: s}
	return rs
}

type PagesService struct {
	s *Service
}

func NewPostUserInfosService(s *Service) *PostUserInfosService {
	rs := &PostUserInfosService{s: s}
	return rs
}

type PostUserInfosService struct {
	s *Service
}

func NewPostsService(s *Service) *PostsService {
	rs := &PostsService{s: s}
	return rs
}

type PostsService struct {
	s *Service
}

func NewUsersService(s *Service) *UsersService {
	rs := &UsersService{s: s}
	return rs
}

type UsersService struct {
	s *Service
}

type Blog struct {
	// CustomMetaData: The JSON custom meta-data for the Blog
	CustomMetaData string `json:"customMetaData,omitempty"`

	// Description: The description of this blog. This is displayed
	// underneath the title.
	Description string `json:"description,omitempty"`

	// Id: The identifier for this resource.
	Id string `json:"id,omitempty"`

	// Kind: The kind of this entry. Always blogger#blog
	Kind string `json:"kind,omitempty"`

	// Locale: The locale this Blog is set to.
	Locale *BlogLocale `json:"locale,omitempty"`

	// Name: The name of this blog. This is displayed as the title.
	Name string `json:"name,omitempty"`

	// Pages: The container of pages in this blog.
	Pages *BlogPages `json:"pages,omitempty"`

	// Posts: The container of posts in this blog.
	Posts *BlogPosts `json:"posts,omitempty"`

	// Published: RFC 3339 date-time when this blog was published.
	Published string `json:"published,omitempty"`

	// SelfLink: The API REST URL to fetch this resource from.
	SelfLink string `json:"selfLink,omitempty"`

	// Updated: RFC 3339 date-time when this blog was last updated.
	Updated string `json:"updated,omitempty"`

	// Url: The URL where this blog is published.
	Url string `json:"url,omitempty"`
}

type BlogLocale struct {
	// Country: The country this blog's locale is set to.
	Country string `json:"country,omitempty"`

	// Language: The language this blog is authored in.
	Language string `json:"language,omitempty"`

	// Variant: The language variant this blog is authored in.
	Variant string `json:"variant,omitempty"`
}

type BlogPages struct {
	// SelfLink: The URL of the container for pages in this blog.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalItems: The count of pages in this blog.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type BlogPosts struct {
	// Items: The List of Posts for this Blog.
	Items []*Post `json:"items,omitempty"`

	// SelfLink: The URL of the container for posts in this blog.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalItems: The count of posts in this blog.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type BlogList struct {
	// BlogUserInfos: Admin level list of blog per-user information
	BlogUserInfos []*BlogUserInfo `json:"blogUserInfos,omitempty"`

	// Items: The list of Blogs this user has Authorship or Admin rights
	// over.
	Items []*Blog `json:"items,omitempty"`

	// Kind: The kind of this entity. Always blogger#blogList
	Kind string `json:"kind,omitempty"`
}

type BlogPerUserInfo struct {
	// BlogId: ID of the Blog resource
	BlogId string `json:"blogId,omitempty"`

	// HasAdminAccess: True if the user has Admin level access to the blog.
	HasAdminAccess bool `json:"hasAdminAccess,omitempty"`

	// Kind: The kind of this entity. Always blogger#blogPerUserInfo
	Kind string `json:"kind,omitempty"`

	// PhotosAlbumKey: The Photo Album Key for the user when adding photos
	// to the blog
	PhotosAlbumKey string `json:"photosAlbumKey,omitempty"`

	// Role: Access permissions that the user has for the blog (ADMIN,
	// AUTHOR, or READER).
	Role string `json:"role,omitempty"`

	// UserId: ID of the User
	UserId string `json:"userId,omitempty"`
}

type BlogUserInfo struct {
	// Blog: The Blog resource.
	Blog *Blog `json:"blog,omitempty"`

	// Blog_user_info: Information about a User for the Blog.
	Blog_user_info *BlogPerUserInfo `json:"blog_user_info,omitempty"`

	// Kind: The kind of this entity. Always blogger#blogUserInfo
	Kind string `json:"kind,omitempty"`
}

type Comment struct {
	// Author: The author of this Comment.
	Author *CommentAuthor `json:"author,omitempty"`

	// Blog: Data about the blog containing this comment.
	Blog *CommentBlog `json:"blog,omitempty"`

	// Content: The actual content of the comment. May include HTML markup.
	Content string `json:"content,omitempty"`

	// Id: The identifier for this resource.
	Id string `json:"id,omitempty"`

	// InReplyTo: Data about the comment this is in reply to.
	InReplyTo *CommentInReplyTo `json:"inReplyTo,omitempty"`

	// Kind: The kind of this entry. Always blogger#comment
	Kind string `json:"kind,omitempty"`

	// Post: Data about the post containing this comment.
	Post *CommentPost `json:"post,omitempty"`

	// Published: RFC 3339 date-time when this comment was published.
	Published string `json:"published,omitempty"`

	// SelfLink: The API REST URL to fetch this resource from.
	SelfLink string `json:"selfLink,omitempty"`

	// Status: The status of the comment (only populated for admin users)
	Status string `json:"status,omitempty"`

	// Updated: RFC 3339 date-time when this comment was last updated.
	Updated string `json:"updated,omitempty"`
}

type CommentAuthor struct {
	// DisplayName: The display name.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The identifier of the Comment creator.
	Id string `json:"id,omitempty"`

	// Image: The comment creator's avatar.
	Image *CommentAuthorImage `json:"image,omitempty"`

	// Url: The URL of the Comment creator's Profile page.
	Url string `json:"url,omitempty"`
}

type CommentAuthorImage struct {
	// Url: The comment creator's avatar URL.
	Url string `json:"url,omitempty"`
}

type CommentBlog struct {
	// Id: The identifier of the blog containing this comment.
	Id string `json:"id,omitempty"`
}

type CommentInReplyTo struct {
	// Id: The identified of the parent of this comment.
	Id string `json:"id,omitempty"`
}

type CommentPost struct {
	// Id: The identifier of the post containing this comment.
	Id string `json:"id,omitempty"`
}

type CommentList struct {
	// Items: The List of Comments for a Post.
	Items []*Comment `json:"items,omitempty"`

	// Kind: The kind of this entry. Always blogger#commentList
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Pagination token to fetch the next page, if one
	// exists.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// PrevPageToken: Pagination token to fetch the previous page, if one
	// exists.
	PrevPageToken string `json:"prevPageToken,omitempty"`
}

type Page struct {
	// Author: The author of this Page.
	Author *PageAuthor `json:"author,omitempty"`

	// Blog: Data about the blog containing this Page.
	Blog *PageBlog `json:"blog,omitempty"`

	// Content: The body content of this Page, in HTML.
	Content string `json:"content,omitempty"`

	// Id: The identifier for this resource.
	Id string `json:"id,omitempty"`

	// Kind: The kind of this entity. Always blogger#page
	Kind string `json:"kind,omitempty"`

	// Published: RFC 3339 date-time when this Page was published.
	Published string `json:"published,omitempty"`

	// SelfLink: The API REST URL to fetch this resource from.
	SelfLink string `json:"selfLink,omitempty"`

	// Status: The status of the page for admin resources (either LIVE or
	// DRAFT).
	Status string `json:"status,omitempty"`

	// Title: The title of this entity. This is the name displayed in the
	// Admin user interface.
	Title string `json:"title,omitempty"`

	// Updated: RFC 3339 date-time when this Page was last updated.
	Updated string `json:"updated,omitempty"`

	// Url: The URL that this Page is displayed at.
	Url string `json:"url,omitempty"`
}

type PageAuthor struct {
	// DisplayName: The display name.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The identifier of the Page creator.
	Id string `json:"id,omitempty"`

	// Image: The page author's avatar.
	Image *PageAuthorImage `json:"image,omitempty"`

	// Url: The URL of the Page creator's Profile page.
	Url string `json:"url,omitempty"`
}

type PageAuthorImage struct {
	// Url: The page author's avatar URL.
	Url string `json:"url,omitempty"`
}

type PageBlog struct {
	// Id: The identifier of the blog containing this page.
	Id string `json:"id,omitempty"`
}

type PageList struct {
	// Items: The list of Pages for a Blog.
	Items []*Page `json:"items,omitempty"`

	// Kind: The kind of this entity. Always blogger#pageList
	Kind string `json:"kind,omitempty"`
}

type Pageviews struct {
	// BlogId: Blog Id
	BlogId int64 `json:"blogId,omitempty,string"`

	// Counts: The container of posts in this blog.
	Counts []*PageviewsCounts `json:"counts,omitempty"`

	// Kind: The kind of this entry. Always blogger#page_views
	Kind string `json:"kind,omitempty"`
}

type PageviewsCounts struct {
	// Count: Count of page views for the given time range
	Count int64 `json:"count,omitempty,string"`

	// TimeRange: Time range the given count applies to
	TimeRange string `json:"timeRange,omitempty"`
}

type Post struct {
	// Author: The author of this Post.
	Author *PostAuthor `json:"author,omitempty"`

	// Blog: Data about the blog containing this Post.
	Blog *PostBlog `json:"blog,omitempty"`

	// Content: The content of the Post. May contain HTML markup.
	Content string `json:"content,omitempty"`

	// CustomMetaData: The JSON meta-data for the Post.
	CustomMetaData string `json:"customMetaData,omitempty"`

	// Id: The identifier of this Post.
	Id string `json:"id,omitempty"`

	// Images: Display image for the Post.
	Images []*PostImages `json:"images,omitempty"`

	// Kind: The kind of this entity. Always blogger#post
	Kind string `json:"kind,omitempty"`

	// Labels: The list of labels this Post was tagged with.
	Labels []string `json:"labels,omitempty"`

	// Location: The location for geotagged posts.
	Location *PostLocation `json:"location,omitempty"`

	// Published: RFC 3339 date-time when this Post was published.
	Published string `json:"published,omitempty"`

	// Replies: The container of comments on this Post.
	Replies *PostReplies `json:"replies,omitempty"`

	// SelfLink: The API REST URL to fetch this resource from.
	SelfLink string `json:"selfLink,omitempty"`

	// Status: Status of the post. Only set for admin-level requests
	Status string `json:"status,omitempty"`

	// Title: The title of the Post.
	Title string `json:"title,omitempty"`

	// TitleLink: The title link URL, similar to atom's related link.
	TitleLink string `json:"titleLink,omitempty"`

	// Updated: RFC 3339 date-time when this Post was last updated.
	Updated string `json:"updated,omitempty"`

	// Url: The URL where this Post is displayed.
	Url string `json:"url,omitempty"`
}

type PostAuthor struct {
	// DisplayName: The display name.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The identifier of the Post creator.
	Id string `json:"id,omitempty"`

	// Image: The Post author's avatar.
	Image *PostAuthorImage `json:"image,omitempty"`

	// Url: The URL of the Post creator's Profile page.
	Url string `json:"url,omitempty"`
}

type PostAuthorImage struct {
	// Url: The Post author's avatar URL.
	Url string `json:"url,omitempty"`
}

type PostBlog struct {
	// Id: The identifier of the Blog that contains this Post.
	Id string `json:"id,omitempty"`
}

type PostImages struct {
	Url string `json:"url,omitempty"`
}

type PostLocation struct {
	// Lat: Location's latitude.
	Lat float64 `json:"lat,omitempty"`

	// Lng: Location's longitude.
	Lng float64 `json:"lng,omitempty"`

	// Name: Location name.
	Name string `json:"name,omitempty"`

	// Span: Location's viewport span. Can be used when rendering a map
	// preview.
	Span string `json:"span,omitempty"`
}

type PostReplies struct {
	// Items: The List of Comments for this Post.
	Items []*Comment `json:"items,omitempty"`

	// SelfLink: The URL of the comments on this post.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalItems: The count of comments on this post.
	TotalItems int64 `json:"totalItems,omitempty,string"`
}

type PostList struct {
	// Items: The list of Posts for this Blog.
	Items []*Post `json:"items,omitempty"`

	// Kind: The kind of this entity. Always blogger#postList
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Pagination token to fetch the next page, if one
	// exists.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type PostPerUserInfo struct {
	// BlogId: ID of the Blog that the post resource belongs to.
	BlogId string `json:"blogId,omitempty"`

	// HasEditAccess: True if the user has Author level access to the post.
	HasEditAccess bool `json:"hasEditAccess,omitempty"`

	// Kind: The kind of this entity. Always blogger#postPerUserInfo
	Kind string `json:"kind,omitempty"`

	// PostId: ID of the Post resource.
	PostId string `json:"postId,omitempty"`

	// UserId: ID of the User.
	UserId string `json:"userId,omitempty"`
}

type PostUserInfo struct {
	// Kind: The kind of this entity. Always blogger#postUserInfo
	Kind string `json:"kind,omitempty"`

	// Post: The Post resource.
	Post *Post `json:"post,omitempty"`

	// Post_user_info: Information about a User for the Post.
	Post_user_info *PostPerUserInfo `json:"post_user_info,omitempty"`
}

type PostUserInfosList struct {
	// Items: The list of Posts with User information for the post, for this
	// Blog.
	Items []*PostUserInfo `json:"items,omitempty"`

	// Kind: The kind of this entity. Always blogger#postList
	Kind string `json:"kind,omitempty"`

	// NextPageToken: Pagination token to fetch the next page, if one
	// exists.
	NextPageToken string `json:"nextPageToken,omitempty"`
}

type User struct {
	// About: Profile summary information.
	About string `json:"about,omitempty"`

	// Blogs: The container of blogs for this user.
	Blogs *UserBlogs `json:"blogs,omitempty"`

	// Created: The timestamp of when this profile was created, in seconds
	// since epoch.
	Created string `json:"created,omitempty"`

	// DisplayName: The display name.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The identifier for this User.
	Id string `json:"id,omitempty"`

	// Kind: The kind of this entity. Always blogger#user
	Kind string `json:"kind,omitempty"`

	// Locale: This user's locale
	Locale *UserLocale `json:"locale,omitempty"`

	// SelfLink: The API REST URL to fetch this resource from.
	SelfLink string `json:"selfLink,omitempty"`

	// Url: The user's profile page.
	Url string `json:"url,omitempty"`
}

type UserBlogs struct {
	// SelfLink: The URL of the Blogs for this user.
	SelfLink string `json:"selfLink,omitempty"`
}

type UserLocale struct {
	// Country: The user's country setting.
	Country string `json:"country,omitempty"`

	// Language: The user's language setting.
	Language string `json:"language,omitempty"`

	// Variant: The user's language variant setting.
	Variant string `json:"variant,omitempty"`
}

// method id "blogger.blogUserInfos.get":

type BlogUserInfosGetCall struct {
	s      *Service
	userId string
	blogId string
	opt_   map[string]interface{}
}

// Get: Gets one blog and user info pair by blogId and userId.
func (r *BlogUserInfosService) Get(userId string, blogId string) *BlogUserInfosGetCall {
	c := &BlogUserInfosGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	c.blogId = blogId
	return c
}

// MaxPosts sets the optional parameter "maxPosts": Maximum number of
// posts to pull back with the blog.
func (c *BlogUserInfosGetCall) MaxPosts(maxPosts int64) *BlogUserInfosGetCall {
	c.opt_["maxPosts"] = maxPosts
	return c
}

func (c *BlogUserInfosGetCall) Do() (*BlogUserInfo, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxPosts"]; ok {
		params.Set("maxPosts", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/{userId}/blogs/{blogId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
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
	ret := new(BlogUserInfo)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets one blog and user info pair by blogId and userId.",
	//   "httpMethod": "GET",
	//   "id": "blogger.blogUserInfos.get",
	//   "parameterOrder": [
	//     "userId",
	//     "blogId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The ID of the blog to get.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxPosts": {
	//       "description": "Maximum number of posts to pull back with the blog.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "userId": {
	//       "description": "ID of the user whose blogs are to be fetched. Either the word 'self' (sans quote marks) or the user's profile identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/{userId}/blogs/{blogId}",
	//   "response": {
	//     "$ref": "BlogUserInfo"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.blogs.get":

type BlogsGetCall struct {
	s      *Service
	blogId string
	opt_   map[string]interface{}
}

// Get: Gets one blog by id.
func (r *BlogsService) Get(blogId string) *BlogsGetCall {
	c := &BlogsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	return c
}

// MaxPosts sets the optional parameter "maxPosts": Maximum number of
// posts to pull back with the blog.
func (c *BlogsGetCall) MaxPosts(maxPosts int64) *BlogsGetCall {
	c.opt_["maxPosts"] = maxPosts
	return c
}

// View sets the optional parameter "view": Access level with which to
// view the blogs. Note that some fields require elevated access.
func (c *BlogsGetCall) View(view string) *BlogsGetCall {
	c.opt_["view"] = view
	return c
}

func (c *BlogsGetCall) Do() (*Blog, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxPosts"]; ok {
		params.Set("maxPosts", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["view"]; ok {
		params.Set("view", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
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
	ret := new(Blog)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets one blog by id.",
	//   "httpMethod": "GET",
	//   "id": "blogger.blogs.get",
	//   "parameterOrder": [
	//     "blogId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The ID of the blog to get.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxPosts": {
	//       "description": "Maximum number of posts to pull back with the blog.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "view": {
	//       "description": "Access level with which to view the blogs. Note that some fields require elevated access.",
	//       "enum": [
	//         "ADMIN",
	//         "AUTHOR",
	//         "READER"
	//       ],
	//       "enumDescriptions": [
	//         "Admin level detail",
	//         "Author level detail",
	//         "Reader level detail"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}",
	//   "response": {
	//     "$ref": "Blog"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.blogs.getByUrl":

type BlogsGetByUrlCall struct {
	s    *Service
	url  string
	opt_ map[string]interface{}
}

// GetByUrl: Retrieve a Blog by URL.
func (r *BlogsService) GetByUrl(url string) *BlogsGetByUrlCall {
	c := &BlogsGetByUrlCall{s: r.s, opt_: make(map[string]interface{})}
	c.url = url
	return c
}

// View sets the optional parameter "view": Access level with which to
// view the blogs. Note that some fields require elevated access.
func (c *BlogsGetByUrlCall) View(view string) *BlogsGetByUrlCall {
	c.opt_["view"] = view
	return c
}

func (c *BlogsGetByUrlCall) Do() (*Blog, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("url", fmt.Sprintf("%v", c.url))
	if v, ok := c.opt_["view"]; ok {
		params.Set("view", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/byurl")
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
	ret := new(Blog)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieve a Blog by URL.",
	//   "httpMethod": "GET",
	//   "id": "blogger.blogs.getByUrl",
	//   "parameterOrder": [
	//     "url"
	//   ],
	//   "parameters": {
	//     "url": {
	//       "description": "The URL of the blog to retrieve.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Access level with which to view the blogs. Note that some fields require elevated access.",
	//       "enum": [
	//         "ADMIN",
	//         "AUTHOR",
	//         "READER"
	//       ],
	//       "enumDescriptions": [
	//         "Admin level detail",
	//         "Author level detail",
	//         "Reader level detail"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/byurl",
	//   "response": {
	//     "$ref": "Blog"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.blogs.listByUser":

type BlogsListByUserCall struct {
	s      *Service
	userId string
	opt_   map[string]interface{}
}

// ListByUser: Retrieves a list of blogs, possibly filtered.
func (r *BlogsService) ListByUser(userId string) *BlogsListByUserCall {
	c := &BlogsListByUserCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	return c
}

// FetchUserInfo sets the optional parameter "fetchUserInfo": Whether
// the response is a list of blogs with per-user information instead of
// just blogs.
func (c *BlogsListByUserCall) FetchUserInfo(fetchUserInfo bool) *BlogsListByUserCall {
	c.opt_["fetchUserInfo"] = fetchUserInfo
	return c
}

// Role sets the optional parameter "role": User access types for blogs
// to include in the results, e.g. AUTHOR will return blogs where the
// user has author level access. If no roles are specified, defaults to
// ADMIN and AUTHOR roles.
func (c *BlogsListByUserCall) Role(role string) *BlogsListByUserCall {
	c.opt_["role"] = role
	return c
}

// View sets the optional parameter "view": Access level with which to
// view the blogs. Note that some fields require elevated access.
func (c *BlogsListByUserCall) View(view string) *BlogsListByUserCall {
	c.opt_["view"] = view
	return c
}

func (c *BlogsListByUserCall) Do() (*BlogList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fetchUserInfo"]; ok {
		params.Set("fetchUserInfo", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["role"]; ok {
		params.Set("role", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["view"]; ok {
		params.Set("view", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/{userId}/blogs")
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
	ret := new(BlogList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of blogs, possibly filtered.",
	//   "httpMethod": "GET",
	//   "id": "blogger.blogs.listByUser",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "fetchUserInfo": {
	//       "description": "Whether the response is a list of blogs with per-user information instead of just blogs.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "role": {
	//       "description": "User access types for blogs to include in the results, e.g. AUTHOR will return blogs where the user has author level access. If no roles are specified, defaults to ADMIN and AUTHOR roles.",
	//       "enum": [
	//         "ADMIN",
	//         "AUTHOR",
	//         "READER"
	//       ],
	//       "enumDescriptions": [
	//         "Admin role - User has Admin level access.",
	//         "Author role - User has Author level access.",
	//         "Reader role - User has Reader level access."
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "ID of the user whose blogs are to be fetched. Either the word 'self' (sans quote marks) or the user's profile identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Access level with which to view the blogs. Note that some fields require elevated access.",
	//       "enum": [
	//         "ADMIN",
	//         "AUTHOR",
	//         "READER"
	//       ],
	//       "enumDescriptions": [
	//         "Admin level detail",
	//         "Author level detail",
	//         "Reader level detail"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/{userId}/blogs",
	//   "response": {
	//     "$ref": "BlogList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.comments.approve":

type CommentsApproveCall struct {
	s         *Service
	blogId    string
	postId    string
	commentId string
	opt_      map[string]interface{}
}

// Approve: Marks a comment as not spam.
func (r *CommentsService) Approve(blogId string, postId string, commentId string) *CommentsApproveCall {
	c := &CommentsApproveCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
	c.commentId = commentId
	return c
}

func (c *CommentsApproveCall) Do() (*Comment, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}/comments/{commentId}/approve")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	//   "description": "Marks a comment as not spam.",
	//   "httpMethod": "POST",
	//   "id": "blogger.comments.approve",
	//   "parameterOrder": [
	//     "blogId",
	//     "postId",
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The Id of the Blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "commentId": {
	//       "description": "The ID of the comment to mark as not spam.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "postId": {
	//       "description": "The ID of the Post.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}/comments/{commentId}/approve",
	//   "response": {
	//     "$ref": "Comment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.comments.delete":

type CommentsDeleteCall struct {
	s         *Service
	blogId    string
	postId    string
	commentId string
	opt_      map[string]interface{}
}

// Delete: Delete a comment by id.
func (r *CommentsService) Delete(blogId string, postId string, commentId string) *CommentsDeleteCall {
	c := &CommentsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
	c.commentId = commentId
	return c
}

func (c *CommentsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}/comments/{commentId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	//   "description": "Delete a comment by id.",
	//   "httpMethod": "DELETE",
	//   "id": "blogger.comments.delete",
	//   "parameterOrder": [
	//     "blogId",
	//     "postId",
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The Id of the Blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "commentId": {
	//       "description": "The ID of the comment to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "postId": {
	//       "description": "The ID of the Post.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}/comments/{commentId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.comments.get":

type CommentsGetCall struct {
	s         *Service
	blogId    string
	postId    string
	commentId string
	opt_      map[string]interface{}
}

// Get: Gets one comment by id.
func (r *CommentsService) Get(blogId string, postId string, commentId string) *CommentsGetCall {
	c := &CommentsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
	c.commentId = commentId
	return c
}

// View sets the optional parameter "view": Access level for the
// requested comment (default: READER). Note that some comments will
// require elevated permissions, for example comments where the parent
// posts which is in a draft state, or comments that are pending
// moderation.
func (c *CommentsGetCall) View(view string) *CommentsGetCall {
	c.opt_["view"] = view
	return c
}

func (c *CommentsGetCall) Do() (*Comment, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["view"]; ok {
		params.Set("view", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}/comments/{commentId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	//   "description": "Gets one comment by id.",
	//   "httpMethod": "GET",
	//   "id": "blogger.comments.get",
	//   "parameterOrder": [
	//     "blogId",
	//     "postId",
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "ID of the blog to containing the comment.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "commentId": {
	//       "description": "The ID of the comment to get.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "postId": {
	//       "description": "ID of the post to fetch posts from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Access level for the requested comment (default: READER). Note that some comments will require elevated permissions, for example comments where the parent posts which is in a draft state, or comments that are pending moderation.",
	//       "enum": [
	//         "ADMIN",
	//         "AUTHOR",
	//         "READER"
	//       ],
	//       "enumDescriptions": [
	//         "Admin level detail",
	//         "Author level detail",
	//         "Admin level detail"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}/comments/{commentId}",
	//   "response": {
	//     "$ref": "Comment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.comments.list":

type CommentsListCall struct {
	s      *Service
	blogId string
	postId string
	opt_   map[string]interface{}
}

// List: Retrieves the comments for a post, possibly filtered.
func (r *CommentsService) List(blogId string, postId string) *CommentsListCall {
	c := &CommentsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
	return c
}

// EndDate sets the optional parameter "endDate": Latest date of comment
// to fetch, a date-time with RFC 3339 formatting.
func (c *CommentsListCall) EndDate(endDate string) *CommentsListCall {
	c.opt_["endDate"] = endDate
	return c
}

// FetchBodies sets the optional parameter "fetchBodies": Whether the
// body content of the comments is included.
func (c *CommentsListCall) FetchBodies(fetchBodies bool) *CommentsListCall {
	c.opt_["fetchBodies"] = fetchBodies
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of comments to include in the result.
func (c *CommentsListCall) MaxResults(maxResults int64) *CommentsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// if request is paged.
func (c *CommentsListCall) PageToken(pageToken string) *CommentsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// StartDate sets the optional parameter "startDate": Earliest date of
// comment to fetch, a date-time with RFC 3339 formatting.
func (c *CommentsListCall) StartDate(startDate string) *CommentsListCall {
	c.opt_["startDate"] = startDate
	return c
}

// Status sets the optional parameter "status":
func (c *CommentsListCall) Status(status string) *CommentsListCall {
	c.opt_["status"] = status
	return c
}

// View sets the optional parameter "view": Access level with which to
// view the returned result. Note that some fields require elevated
// access.
func (c *CommentsListCall) View(view string) *CommentsListCall {
	c.opt_["view"] = view
	return c
}

func (c *CommentsListCall) Do() (*CommentList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["endDate"]; ok {
		params.Set("endDate", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fetchBodies"]; ok {
		params.Set("fetchBodies", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startDate"]; ok {
		params.Set("startDate", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["status"]; ok {
		params.Set("status", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["view"]; ok {
		params.Set("view", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}/comments")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	//   "description": "Retrieves the comments for a post, possibly filtered.",
	//   "httpMethod": "GET",
	//   "id": "blogger.comments.list",
	//   "parameterOrder": [
	//     "blogId",
	//     "postId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "ID of the blog to fetch comments from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "endDate": {
	//       "description": "Latest date of comment to fetch, a date-time with RFC 3339 formatting.",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "fetchBodies": {
	//       "description": "Whether the body content of the comments is included.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of comments to include in the result.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Continuation token if request is paged.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "postId": {
	//       "description": "ID of the post to fetch posts from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "startDate": {
	//       "description": "Earliest date of comment to fetch, a date-time with RFC 3339 formatting.",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "status": {
	//       "enum": [
	//         "emptied",
	//         "live",
	//         "pending",
	//         "spam"
	//       ],
	//       "enumDescriptions": [
	//         "Comments that have had their content removed",
	//         "Comments that are publicly visible",
	//         "Comments that are awaiting administrator approval",
	//         "Comments marked as spam by the administrator"
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Access level with which to view the returned result. Note that some fields require elevated access.",
	//       "enum": [
	//         "ADMIN",
	//         "AUTHOR",
	//         "READER"
	//       ],
	//       "enumDescriptions": [
	//         "Admin level detail",
	//         "Author level detail",
	//         "Reader level detail"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}/comments",
	//   "response": {
	//     "$ref": "CommentList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.comments.listByBlog":

type CommentsListByBlogCall struct {
	s      *Service
	blogId string
	opt_   map[string]interface{}
}

// ListByBlog: Retrieves the comments for a blog, across all posts,
// possibly filtered.
func (r *CommentsService) ListByBlog(blogId string) *CommentsListByBlogCall {
	c := &CommentsListByBlogCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	return c
}

// EndDate sets the optional parameter "endDate": Latest date of comment
// to fetch, a date-time with RFC 3339 formatting.
func (c *CommentsListByBlogCall) EndDate(endDate string) *CommentsListByBlogCall {
	c.opt_["endDate"] = endDate
	return c
}

// FetchBodies sets the optional parameter "fetchBodies": Whether the
// body content of the comments is included.
func (c *CommentsListByBlogCall) FetchBodies(fetchBodies bool) *CommentsListByBlogCall {
	c.opt_["fetchBodies"] = fetchBodies
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of comments to include in the result.
func (c *CommentsListByBlogCall) MaxResults(maxResults int64) *CommentsListByBlogCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// if request is paged.
func (c *CommentsListByBlogCall) PageToken(pageToken string) *CommentsListByBlogCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// StartDate sets the optional parameter "startDate": Earliest date of
// comment to fetch, a date-time with RFC 3339 formatting.
func (c *CommentsListByBlogCall) StartDate(startDate string) *CommentsListByBlogCall {
	c.opt_["startDate"] = startDate
	return c
}

func (c *CommentsListByBlogCall) Do() (*CommentList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["endDate"]; ok {
		params.Set("endDate", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fetchBodies"]; ok {
		params.Set("fetchBodies", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxResults"]; ok {
		params.Set("maxResults", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["pageToken"]; ok {
		params.Set("pageToken", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["startDate"]; ok {
		params.Set("startDate", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/comments")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
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
	//   "description": "Retrieves the comments for a blog, across all posts, possibly filtered.",
	//   "httpMethod": "GET",
	//   "id": "blogger.comments.listByBlog",
	//   "parameterOrder": [
	//     "blogId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "ID of the blog to fetch comments from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "endDate": {
	//       "description": "Latest date of comment to fetch, a date-time with RFC 3339 formatting.",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "fetchBodies": {
	//       "description": "Whether the body content of the comments is included.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of comments to include in the result.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "Continuation token if request is paged.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startDate": {
	//       "description": "Earliest date of comment to fetch, a date-time with RFC 3339 formatting.",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/comments",
	//   "response": {
	//     "$ref": "CommentList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.comments.markAsSpam":

type CommentsMarkAsSpamCall struct {
	s         *Service
	blogId    string
	postId    string
	commentId string
	opt_      map[string]interface{}
}

// MarkAsSpam: Marks a comment as spam.
func (r *CommentsService) MarkAsSpam(blogId string, postId string, commentId string) *CommentsMarkAsSpamCall {
	c := &CommentsMarkAsSpamCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
	c.commentId = commentId
	return c
}

func (c *CommentsMarkAsSpamCall) Do() (*Comment, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}/comments/{commentId}/spam")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	//   "description": "Marks a comment as spam.",
	//   "httpMethod": "POST",
	//   "id": "blogger.comments.markAsSpam",
	//   "parameterOrder": [
	//     "blogId",
	//     "postId",
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The Id of the Blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "commentId": {
	//       "description": "The ID of the comment to mark as spam.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "postId": {
	//       "description": "The ID of the Post.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}/comments/{commentId}/spam",
	//   "response": {
	//     "$ref": "Comment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.comments.removeContent":

type CommentsRemoveContentCall struct {
	s         *Service
	blogId    string
	postId    string
	commentId string
	opt_      map[string]interface{}
}

// RemoveContent: Removes the content of a comment.
func (r *CommentsService) RemoveContent(blogId string, postId string, commentId string) *CommentsRemoveContentCall {
	c := &CommentsRemoveContentCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
	c.commentId = commentId
	return c
}

func (c *CommentsRemoveContentCall) Do() (*Comment, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}/comments/{commentId}/removecontent")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	//   "description": "Removes the content of a comment.",
	//   "httpMethod": "POST",
	//   "id": "blogger.comments.removeContent",
	//   "parameterOrder": [
	//     "blogId",
	//     "postId",
	//     "commentId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The Id of the Blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "commentId": {
	//       "description": "The ID of the comment to delete content from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "postId": {
	//       "description": "The ID of the Post.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}/comments/{commentId}/removecontent",
	//   "response": {
	//     "$ref": "Comment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.pageViews.get":

type PageViewsGetCall struct {
	s      *Service
	blogId string
	opt_   map[string]interface{}
}

// Get: Retrieve pageview stats for a Blog.
func (r *PageViewsService) Get(blogId string) *PageViewsGetCall {
	c := &PageViewsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	return c
}

// Range sets the optional parameter "range":
func (c *PageViewsGetCall) Range(range_ string) *PageViewsGetCall {
	c.opt_["range"] = range_
	return c
}

func (c *PageViewsGetCall) Do() (*Pageviews, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["range"]; ok {
		params.Set("range", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/pageviews")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
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
	ret := new(Pageviews)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieve pageview stats for a Blog.",
	//   "httpMethod": "GET",
	//   "id": "blogger.pageViews.get",
	//   "parameterOrder": [
	//     "blogId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The ID of the blog to get.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "range": {
	//       "enum": [
	//         "30DAYS",
	//         "7DAYS",
	//         "all"
	//       ],
	//       "enumDescriptions": [
	//         "Page view counts from the last thirty days.",
	//         "Page view counts from the last seven days.",
	//         "Total page view counts from all time."
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/pageviews",
	//   "response": {
	//     "$ref": "Pageviews"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.pages.delete":

type PagesDeleteCall struct {
	s      *Service
	blogId string
	pageId string
	opt_   map[string]interface{}
}

// Delete: Delete a page by id.
func (r *PagesService) Delete(blogId string, pageId string) *PagesDeleteCall {
	c := &PagesDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.pageId = pageId
	return c
}

func (c *PagesDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/pages/{pageId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{pageId}", url.QueryEscape(c.pageId), 1)
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
	//   "description": "Delete a page by id.",
	//   "httpMethod": "DELETE",
	//   "id": "blogger.pages.delete",
	//   "parameterOrder": [
	//     "blogId",
	//     "pageId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The Id of the Blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the Page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/pages/{pageId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.pages.get":

type PagesGetCall struct {
	s      *Service
	blogId string
	pageId string
	opt_   map[string]interface{}
}

// Get: Gets one blog page by id.
func (r *PagesService) Get(blogId string, pageId string) *PagesGetCall {
	c := &PagesGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.pageId = pageId
	return c
}

// View sets the optional parameter "view":
func (c *PagesGetCall) View(view string) *PagesGetCall {
	c.opt_["view"] = view
	return c
}

func (c *PagesGetCall) Do() (*Page, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["view"]; ok {
		params.Set("view", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/pages/{pageId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{pageId}", url.QueryEscape(c.pageId), 1)
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
	ret := new(Page)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets one blog page by id.",
	//   "httpMethod": "GET",
	//   "id": "blogger.pages.get",
	//   "parameterOrder": [
	//     "blogId",
	//     "pageId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "ID of the blog containing the page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the page to get.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "enum": [
	//         "ADMIN",
	//         "AUTHOR",
	//         "READER"
	//       ],
	//       "enumDescriptions": [
	//         "Admin level detail",
	//         "Author level detail",
	//         "Reader level detail"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/pages/{pageId}",
	//   "response": {
	//     "$ref": "Page"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.pages.insert":

type PagesInsertCall struct {
	s      *Service
	blogId string
	page   *Page
	opt_   map[string]interface{}
}

// Insert: Add a page.
func (r *PagesService) Insert(blogId string, page *Page) *PagesInsertCall {
	c := &PagesInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.page = page
	return c
}

func (c *PagesInsertCall) Do() (*Page, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.page)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/pages")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
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
	ret := new(Page)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Add a page.",
	//   "httpMethod": "POST",
	//   "id": "blogger.pages.insert",
	//   "parameterOrder": [
	//     "blogId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "ID of the blog to add the page to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/pages",
	//   "request": {
	//     "$ref": "Page"
	//   },
	//   "response": {
	//     "$ref": "Page"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.pages.list":

type PagesListCall struct {
	s      *Service
	blogId string
	opt_   map[string]interface{}
}

// List: Retrieves the pages for a blog, optionally including non-LIVE
// statuses.
func (r *PagesService) List(blogId string) *PagesListCall {
	c := &PagesListCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	return c
}

// FetchBodies sets the optional parameter "fetchBodies": Whether to
// retrieve the Page bodies.
func (c *PagesListCall) FetchBodies(fetchBodies bool) *PagesListCall {
	c.opt_["fetchBodies"] = fetchBodies
	return c
}

// Status sets the optional parameter "status":
func (c *PagesListCall) Status(status string) *PagesListCall {
	c.opt_["status"] = status
	return c
}

// View sets the optional parameter "view": Access level with which to
// view the returned result. Note that some fields require elevated
// access.
func (c *PagesListCall) View(view string) *PagesListCall {
	c.opt_["view"] = view
	return c
}

func (c *PagesListCall) Do() (*PageList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fetchBodies"]; ok {
		params.Set("fetchBodies", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["status"]; ok {
		params.Set("status", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["view"]; ok {
		params.Set("view", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/pages")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
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
	ret := new(PageList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves the pages for a blog, optionally including non-LIVE statuses.",
	//   "httpMethod": "GET",
	//   "id": "blogger.pages.list",
	//   "parameterOrder": [
	//     "blogId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "ID of the blog to fetch pages from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fetchBodies": {
	//       "description": "Whether to retrieve the Page bodies.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "status": {
	//       "enum": [
	//         "draft",
	//         "live"
	//       ],
	//       "enumDescriptions": [
	//         "Draft (unpublished) Pages",
	//         "Pages that are publicly visible"
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Access level with which to view the returned result. Note that some fields require elevated access.",
	//       "enum": [
	//         "ADMIN",
	//         "AUTHOR",
	//         "READER"
	//       ],
	//       "enumDescriptions": [
	//         "Admin level detail",
	//         "Author level detail",
	//         "Reader level detail"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/pages",
	//   "response": {
	//     "$ref": "PageList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.pages.patch":

type PagesPatchCall struct {
	s      *Service
	blogId string
	pageId string
	page   *Page
	opt_   map[string]interface{}
}

// Patch: Update a page. This method supports patch semantics.
func (r *PagesService) Patch(blogId string, pageId string, page *Page) *PagesPatchCall {
	c := &PagesPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.pageId = pageId
	c.page = page
	return c
}

func (c *PagesPatchCall) Do() (*Page, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.page)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/pages/{pageId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{pageId}", url.QueryEscape(c.pageId), 1)
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
	ret := new(Page)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update a page. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "blogger.pages.patch",
	//   "parameterOrder": [
	//     "blogId",
	//     "pageId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The ID of the Blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the Page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/pages/{pageId}",
	//   "request": {
	//     "$ref": "Page"
	//   },
	//   "response": {
	//     "$ref": "Page"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.pages.update":

type PagesUpdateCall struct {
	s      *Service
	blogId string
	pageId string
	page   *Page
	opt_   map[string]interface{}
}

// Update: Update a page.
func (r *PagesService) Update(blogId string, pageId string, page *Page) *PagesUpdateCall {
	c := &PagesUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.pageId = pageId
	c.page = page
	return c
}

func (c *PagesUpdateCall) Do() (*Page, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.page)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/pages/{pageId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{pageId}", url.QueryEscape(c.pageId), 1)
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
	ret := new(Page)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update a page.",
	//   "httpMethod": "PUT",
	//   "id": "blogger.pages.update",
	//   "parameterOrder": [
	//     "blogId",
	//     "pageId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The ID of the Blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageId": {
	//       "description": "The ID of the Page.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/pages/{pageId}",
	//   "request": {
	//     "$ref": "Page"
	//   },
	//   "response": {
	//     "$ref": "Page"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.postUserInfos.get":

type PostUserInfosGetCall struct {
	s      *Service
	userId string
	blogId string
	postId string
	opt_   map[string]interface{}
}

// Get: Gets one post and user info pair, by post id and user id. The
// post user info contains per-user information about the post, such as
// access rights, specific to the user.
func (r *PostUserInfosService) Get(userId string, blogId string, postId string) *PostUserInfosGetCall {
	c := &PostUserInfosGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	c.blogId = blogId
	c.postId = postId
	return c
}

// MaxComments sets the optional parameter "maxComments": Maximum number
// of comments to pull back on a post.
func (c *PostUserInfosGetCall) MaxComments(maxComments int64) *PostUserInfosGetCall {
	c.opt_["maxComments"] = maxComments
	return c
}

func (c *PostUserInfosGetCall) Do() (*PostUserInfo, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["maxComments"]; ok {
		params.Set("maxComments", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/{userId}/blogs/{blogId}/posts/{postId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	ret := new(PostUserInfo)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets one post and user info pair, by post id and user id. The post user info contains per-user information about the post, such as access rights, specific to the user.",
	//   "httpMethod": "GET",
	//   "id": "blogger.postUserInfos.get",
	//   "parameterOrder": [
	//     "userId",
	//     "blogId",
	//     "postId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The ID of the blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxComments": {
	//       "description": "Maximum number of comments to pull back on a post.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "postId": {
	//       "description": "The ID of the post to get.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "ID of the user for the per-user information to be fetched. Either the word 'self' (sans quote marks) or the user's profile identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/{userId}/blogs/{blogId}/posts/{postId}",
	//   "response": {
	//     "$ref": "PostUserInfo"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.postUserInfos.list":

type PostUserInfosListCall struct {
	s      *Service
	userId string
	blogId string
	opt_   map[string]interface{}
}

// List: Retrieves a list of post and post user info pairs, possibly
// filtered. The post user info contains per-user information about the
// post, such as access rights, specific to the user.
func (r *PostUserInfosService) List(userId string, blogId string) *PostUserInfosListCall {
	c := &PostUserInfosListCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	c.blogId = blogId
	return c
}

// EndDate sets the optional parameter "endDate": Latest post date to
// fetch, a date-time with RFC 3339 formatting.
func (c *PostUserInfosListCall) EndDate(endDate string) *PostUserInfosListCall {
	c.opt_["endDate"] = endDate
	return c
}

// FetchBodies sets the optional parameter "fetchBodies": Whether the
// body content of posts is included. Default is false.
func (c *PostUserInfosListCall) FetchBodies(fetchBodies bool) *PostUserInfosListCall {
	c.opt_["fetchBodies"] = fetchBodies
	return c
}

// Labels sets the optional parameter "labels": Comma-separated list of
// labels to search for.
func (c *PostUserInfosListCall) Labels(labels string) *PostUserInfosListCall {
	c.opt_["labels"] = labels
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of posts to fetch.
func (c *PostUserInfosListCall) MaxResults(maxResults int64) *PostUserInfosListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// OrderBy sets the optional parameter "orderBy": Sort order applied to
// search results. Default is published.
func (c *PostUserInfosListCall) OrderBy(orderBy string) *PostUserInfosListCall {
	c.opt_["orderBy"] = orderBy
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// if the request is paged.
func (c *PostUserInfosListCall) PageToken(pageToken string) *PostUserInfosListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// StartDate sets the optional parameter "startDate": Earliest post date
// to fetch, a date-time with RFC 3339 formatting.
func (c *PostUserInfosListCall) StartDate(startDate string) *PostUserInfosListCall {
	c.opt_["startDate"] = startDate
	return c
}

// Status sets the optional parameter "status":
func (c *PostUserInfosListCall) Status(status string) *PostUserInfosListCall {
	c.opt_["status"] = status
	return c
}

// View sets the optional parameter "view": Access level with which to
// view the returned result. Note that some fields require elevated
// access.
func (c *PostUserInfosListCall) View(view string) *PostUserInfosListCall {
	c.opt_["view"] = view
	return c
}

func (c *PostUserInfosListCall) Do() (*PostUserInfosList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["endDate"]; ok {
		params.Set("endDate", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fetchBodies"]; ok {
		params.Set("fetchBodies", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["labels"]; ok {
		params.Set("labels", fmt.Sprintf("%v", v))
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
	if v, ok := c.opt_["startDate"]; ok {
		params.Set("startDate", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["status"]; ok {
		params.Set("status", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["view"]; ok {
		params.Set("view", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/{userId}/blogs/{blogId}/posts")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{userId}", url.QueryEscape(c.userId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
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
	ret := new(PostUserInfosList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of post and post user info pairs, possibly filtered. The post user info contains per-user information about the post, such as access rights, specific to the user.",
	//   "httpMethod": "GET",
	//   "id": "blogger.postUserInfos.list",
	//   "parameterOrder": [
	//     "userId",
	//     "blogId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "ID of the blog to fetch posts from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "endDate": {
	//       "description": "Latest post date to fetch, a date-time with RFC 3339 formatting.",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "fetchBodies": {
	//       "default": "false",
	//       "description": "Whether the body content of posts is included. Default is false.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "labels": {
	//       "description": "Comma-separated list of labels to search for.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of posts to fetch.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "orderBy": {
	//       "default": "PUBLISHED",
	//       "description": "Sort order applied to search results. Default is published.",
	//       "enum": [
	//         "published",
	//         "updated"
	//       ],
	//       "enumDescriptions": [
	//         "Order by the date the post was published",
	//         "Order by the date the post was last updated"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "Continuation token if the request is paged.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startDate": {
	//       "description": "Earliest post date to fetch, a date-time with RFC 3339 formatting.",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "status": {
	//       "enum": [
	//         "draft",
	//         "live",
	//         "scheduled"
	//       ],
	//       "enumDescriptions": [
	//         "Draft posts",
	//         "Published posts",
	//         "Posts that are scheduled to publish in future."
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "ID of the user for the per-user information to be fetched. Either the word 'self' (sans quote marks) or the user's profile identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Access level with which to view the returned result. Note that some fields require elevated access.",
	//       "enum": [
	//         "ADMIN",
	//         "AUTHOR",
	//         "READER"
	//       ],
	//       "enumDescriptions": [
	//         "Admin level detail",
	//         "Author level detail",
	//         "Reader level detail"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/{userId}/blogs/{blogId}/posts",
	//   "response": {
	//     "$ref": "PostUserInfosList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.posts.delete":

type PostsDeleteCall struct {
	s      *Service
	blogId string
	postId string
	opt_   map[string]interface{}
}

// Delete: Delete a post by id.
func (r *PostsService) Delete(blogId string, postId string) *PostsDeleteCall {
	c := &PostsDeleteCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
	return c
}

func (c *PostsDeleteCall) Do() error {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	//   "description": "Delete a post by id.",
	//   "httpMethod": "DELETE",
	//   "id": "blogger.posts.delete",
	//   "parameterOrder": [
	//     "blogId",
	//     "postId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The Id of the Blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "postId": {
	//       "description": "The ID of the Post.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}",
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.posts.get":

type PostsGetCall struct {
	s      *Service
	blogId string
	postId string
	opt_   map[string]interface{}
}

// Get: Get a post by id.
func (r *PostsService) Get(blogId string, postId string) *PostsGetCall {
	c := &PostsGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
	return c
}

// FetchBody sets the optional parameter "fetchBody": Whether the body
// content of the post is included (default: true). This should be set
// to false when the post bodies are not required, to help minimize
// traffic.
func (c *PostsGetCall) FetchBody(fetchBody bool) *PostsGetCall {
	c.opt_["fetchBody"] = fetchBody
	return c
}

// FetchImages sets the optional parameter "fetchImages": Whether image
// URL metadata for each post is included (default: false).
func (c *PostsGetCall) FetchImages(fetchImages bool) *PostsGetCall {
	c.opt_["fetchImages"] = fetchImages
	return c
}

// MaxComments sets the optional parameter "maxComments": Maximum number
// of comments to pull back on a post.
func (c *PostsGetCall) MaxComments(maxComments int64) *PostsGetCall {
	c.opt_["maxComments"] = maxComments
	return c
}

// View sets the optional parameter "view": Access level with which to
// view the returned result. Note that some fields require elevated
// access.
func (c *PostsGetCall) View(view string) *PostsGetCall {
	c.opt_["view"] = view
	return c
}

func (c *PostsGetCall) Do() (*Post, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fetchBody"]; ok {
		params.Set("fetchBody", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fetchImages"]; ok {
		params.Set("fetchImages", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxComments"]; ok {
		params.Set("maxComments", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["view"]; ok {
		params.Set("view", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	ret := new(Post)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Get a post by id.",
	//   "httpMethod": "GET",
	//   "id": "blogger.posts.get",
	//   "parameterOrder": [
	//     "blogId",
	//     "postId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "ID of the blog to fetch the post from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fetchBody": {
	//       "default": "true",
	//       "description": "Whether the body content of the post is included (default: true). This should be set to false when the post bodies are not required, to help minimize traffic.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "fetchImages": {
	//       "description": "Whether image URL metadata for each post is included (default: false).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxComments": {
	//       "description": "Maximum number of comments to pull back on a post.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "postId": {
	//       "description": "The ID of the post",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Access level with which to view the returned result. Note that some fields require elevated access.",
	//       "enum": [
	//         "ADMIN",
	//         "AUTHOR",
	//         "READER"
	//       ],
	//       "enumDescriptions": [
	//         "Admin level detail",
	//         "Author level detail",
	//         "Reader level detail"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}",
	//   "response": {
	//     "$ref": "Post"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.posts.getByPath":

type PostsGetByPathCall struct {
	s      *Service
	blogId string
	path   string
	opt_   map[string]interface{}
}

// GetByPath: Retrieve a Post by Path.
func (r *PostsService) GetByPath(blogId string, path string) *PostsGetByPathCall {
	c := &PostsGetByPathCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.path = path
	return c
}

// MaxComments sets the optional parameter "maxComments": Maximum number
// of comments to pull back on a post.
func (c *PostsGetByPathCall) MaxComments(maxComments int64) *PostsGetByPathCall {
	c.opt_["maxComments"] = maxComments
	return c
}

// View sets the optional parameter "view": Access level with which to
// view the returned result. Note that some fields require elevated
// access.
func (c *PostsGetByPathCall) View(view string) *PostsGetByPathCall {
	c.opt_["view"] = view
	return c
}

func (c *PostsGetByPathCall) Do() (*Post, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("path", fmt.Sprintf("%v", c.path))
	if v, ok := c.opt_["maxComments"]; ok {
		params.Set("maxComments", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["view"]; ok {
		params.Set("view", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/bypath")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
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
	ret := new(Post)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieve a Post by Path.",
	//   "httpMethod": "GET",
	//   "id": "blogger.posts.getByPath",
	//   "parameterOrder": [
	//     "blogId",
	//     "path"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "ID of the blog to fetch the post from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "maxComments": {
	//       "description": "Maximum number of comments to pull back on a post.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "path": {
	//       "description": "Path of the Post to retrieve.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Access level with which to view the returned result. Note that some fields require elevated access.",
	//       "enum": [
	//         "ADMIN",
	//         "AUTHOR",
	//         "READER"
	//       ],
	//       "enumDescriptions": [
	//         "Admin level detail",
	//         "Author level detail",
	//         "Reader level detail"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/bypath",
	//   "response": {
	//     "$ref": "Post"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.posts.insert":

type PostsInsertCall struct {
	s      *Service
	blogId string
	post   *Post
	opt_   map[string]interface{}
}

// Insert: Add a post.
func (r *PostsService) Insert(blogId string, post *Post) *PostsInsertCall {
	c := &PostsInsertCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.post = post
	return c
}

// FetchBody sets the optional parameter "fetchBody": Whether the body
// content of the post is included with the result (default: true).
func (c *PostsInsertCall) FetchBody(fetchBody bool) *PostsInsertCall {
	c.opt_["fetchBody"] = fetchBody
	return c
}

// FetchImages sets the optional parameter "fetchImages": Whether image
// URL metadata for each post is included in the returned result
// (default: false).
func (c *PostsInsertCall) FetchImages(fetchImages bool) *PostsInsertCall {
	c.opt_["fetchImages"] = fetchImages
	return c
}

// IsDraft sets the optional parameter "isDraft": Whether to create the
// post as a draft (default: false).
func (c *PostsInsertCall) IsDraft(isDraft bool) *PostsInsertCall {
	c.opt_["isDraft"] = isDraft
	return c
}

func (c *PostsInsertCall) Do() (*Post, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.post)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fetchBody"]; ok {
		params.Set("fetchBody", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fetchImages"]; ok {
		params.Set("fetchImages", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["isDraft"]; ok {
		params.Set("isDraft", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
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
	ret := new(Post)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Add a post.",
	//   "httpMethod": "POST",
	//   "id": "blogger.posts.insert",
	//   "parameterOrder": [
	//     "blogId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "ID of the blog to add the post to.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fetchBody": {
	//       "default": "true",
	//       "description": "Whether the body content of the post is included with the result (default: true).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "fetchImages": {
	//       "description": "Whether image URL metadata for each post is included in the returned result (default: false).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "isDraft": {
	//       "description": "Whether to create the post as a draft (default: false).",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts",
	//   "request": {
	//     "$ref": "Post"
	//   },
	//   "response": {
	//     "$ref": "Post"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.posts.list":

type PostsListCall struct {
	s      *Service
	blogId string
	opt_   map[string]interface{}
}

// List: Retrieves a list of posts, possibly filtered.
func (r *PostsService) List(blogId string) *PostsListCall {
	c := &PostsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	return c
}

// EndDate sets the optional parameter "endDate": Latest post date to
// fetch, a date-time with RFC 3339 formatting.
func (c *PostsListCall) EndDate(endDate string) *PostsListCall {
	c.opt_["endDate"] = endDate
	return c
}

// FetchBodies sets the optional parameter "fetchBodies": Whether the
// body content of posts is included (default: true). This should be set
// to false when the post bodies are not required, to help minimize
// traffic.
func (c *PostsListCall) FetchBodies(fetchBodies bool) *PostsListCall {
	c.opt_["fetchBodies"] = fetchBodies
	return c
}

// FetchImages sets the optional parameter "fetchImages": Whether image
// URL metadata for each post is included.
func (c *PostsListCall) FetchImages(fetchImages bool) *PostsListCall {
	c.opt_["fetchImages"] = fetchImages
	return c
}

// Labels sets the optional parameter "labels": Comma-separated list of
// labels to search for.
func (c *PostsListCall) Labels(labels string) *PostsListCall {
	c.opt_["labels"] = labels
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of posts to fetch.
func (c *PostsListCall) MaxResults(maxResults int64) *PostsListCall {
	c.opt_["maxResults"] = maxResults
	return c
}

// OrderBy sets the optional parameter "orderBy": Sort search results
func (c *PostsListCall) OrderBy(orderBy string) *PostsListCall {
	c.opt_["orderBy"] = orderBy
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// if the request is paged.
func (c *PostsListCall) PageToken(pageToken string) *PostsListCall {
	c.opt_["pageToken"] = pageToken
	return c
}

// StartDate sets the optional parameter "startDate": Earliest post date
// to fetch, a date-time with RFC 3339 formatting.
func (c *PostsListCall) StartDate(startDate string) *PostsListCall {
	c.opt_["startDate"] = startDate
	return c
}

// Status sets the optional parameter "status": Statuses to include in
// the results.
func (c *PostsListCall) Status(status string) *PostsListCall {
	c.opt_["status"] = status
	return c
}

// View sets the optional parameter "view": Access level with which to
// view the returned result. Note that some fields require escalated
// access.
func (c *PostsListCall) View(view string) *PostsListCall {
	c.opt_["view"] = view
	return c
}

func (c *PostsListCall) Do() (*PostList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["endDate"]; ok {
		params.Set("endDate", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fetchBodies"]; ok {
		params.Set("fetchBodies", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fetchImages"]; ok {
		params.Set("fetchImages", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["labels"]; ok {
		params.Set("labels", fmt.Sprintf("%v", v))
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
	if v, ok := c.opt_["startDate"]; ok {
		params.Set("startDate", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["status"]; ok {
		params.Set("status", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["view"]; ok {
		params.Set("view", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
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
	ret := new(PostList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Retrieves a list of posts, possibly filtered.",
	//   "httpMethod": "GET",
	//   "id": "blogger.posts.list",
	//   "parameterOrder": [
	//     "blogId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "ID of the blog to fetch posts from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "endDate": {
	//       "description": "Latest post date to fetch, a date-time with RFC 3339 formatting.",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "fetchBodies": {
	//       "default": "true",
	//       "description": "Whether the body content of posts is included (default: true). This should be set to false when the post bodies are not required, to help minimize traffic.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "fetchImages": {
	//       "description": "Whether image URL metadata for each post is included.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "labels": {
	//       "description": "Comma-separated list of labels to search for.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of posts to fetch.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "orderBy": {
	//       "default": "PUBLISHED",
	//       "description": "Sort search results",
	//       "enum": [
	//         "published",
	//         "updated"
	//       ],
	//       "enumDescriptions": [
	//         "Order by the date the post was published",
	//         "Order by the date the post was last updated"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageToken": {
	//       "description": "Continuation token if the request is paged.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "startDate": {
	//       "description": "Earliest post date to fetch, a date-time with RFC 3339 formatting.",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "status": {
	//       "description": "Statuses to include in the results.",
	//       "enum": [
	//         "draft",
	//         "live",
	//         "scheduled"
	//       ],
	//       "enumDescriptions": [
	//         "Draft (non-published) posts.",
	//         "Published posts",
	//         "Posts that are scheduled to publish in the future."
	//       ],
	//       "location": "query",
	//       "repeated": true,
	//       "type": "string"
	//     },
	//     "view": {
	//       "description": "Access level with which to view the returned result. Note that some fields require escalated access.",
	//       "enum": [
	//         "ADMIN",
	//         "AUTHOR",
	//         "READER"
	//       ],
	//       "enumDescriptions": [
	//         "Admin level detail",
	//         "Author level detail",
	//         "Reader level detail"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts",
	//   "response": {
	//     "$ref": "PostList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.posts.patch":

type PostsPatchCall struct {
	s      *Service
	blogId string
	postId string
	post   *Post
	opt_   map[string]interface{}
}

// Patch: Update a post. This method supports patch semantics.
func (r *PostsService) Patch(blogId string, postId string, post *Post) *PostsPatchCall {
	c := &PostsPatchCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
	c.post = post
	return c
}

// FetchBody sets the optional parameter "fetchBody": Whether the body
// content of the post is included with the result (default: true).
func (c *PostsPatchCall) FetchBody(fetchBody bool) *PostsPatchCall {
	c.opt_["fetchBody"] = fetchBody
	return c
}

// FetchImages sets the optional parameter "fetchImages": Whether image
// URL metadata for each post is included in the returned result
// (default: false).
func (c *PostsPatchCall) FetchImages(fetchImages bool) *PostsPatchCall {
	c.opt_["fetchImages"] = fetchImages
	return c
}

// MaxComments sets the optional parameter "maxComments": Maximum number
// of comments to retrieve with the returned post.
func (c *PostsPatchCall) MaxComments(maxComments int64) *PostsPatchCall {
	c.opt_["maxComments"] = maxComments
	return c
}

// Publish sets the optional parameter "publish": Whether a publish
// action should be performed when the post is updated (default: false).
func (c *PostsPatchCall) Publish(publish bool) *PostsPatchCall {
	c.opt_["publish"] = publish
	return c
}

// Revert sets the optional parameter "revert": Whether a revert action
// should be performed when the post is updated (default: false).
func (c *PostsPatchCall) Revert(revert bool) *PostsPatchCall {
	c.opt_["revert"] = revert
	return c
}

func (c *PostsPatchCall) Do() (*Post, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.post)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fetchBody"]; ok {
		params.Set("fetchBody", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fetchImages"]; ok {
		params.Set("fetchImages", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxComments"]; ok {
		params.Set("maxComments", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["publish"]; ok {
		params.Set("publish", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["revert"]; ok {
		params.Set("revert", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	ret := new(Post)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update a post. This method supports patch semantics.",
	//   "httpMethod": "PATCH",
	//   "id": "blogger.posts.patch",
	//   "parameterOrder": [
	//     "blogId",
	//     "postId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The ID of the Blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fetchBody": {
	//       "default": "true",
	//       "description": "Whether the body content of the post is included with the result (default: true).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "fetchImages": {
	//       "description": "Whether image URL metadata for each post is included in the returned result (default: false).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxComments": {
	//       "description": "Maximum number of comments to retrieve with the returned post.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "postId": {
	//       "description": "The ID of the Post.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "publish": {
	//       "description": "Whether a publish action should be performed when the post is updated (default: false).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "revert": {
	//       "description": "Whether a revert action should be performed when the post is updated (default: false).",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}",
	//   "request": {
	//     "$ref": "Post"
	//   },
	//   "response": {
	//     "$ref": "Post"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.posts.publish":

type PostsPublishCall struct {
	s      *Service
	blogId string
	postId string
	opt_   map[string]interface{}
}

// Publish: Publish a draft post.
func (r *PostsService) Publish(blogId string, postId string) *PostsPublishCall {
	c := &PostsPublishCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
	return c
}

// PublishDate sets the optional parameter "publishDate": The date and
// time to schedule the publishing of the Blog.
func (c *PostsPublishCall) PublishDate(publishDate string) *PostsPublishCall {
	c.opt_["publishDate"] = publishDate
	return c
}

func (c *PostsPublishCall) Do() (*Post, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["publishDate"]; ok {
		params.Set("publishDate", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}/publish")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	ret := new(Post)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Publish a draft post.",
	//   "httpMethod": "POST",
	//   "id": "blogger.posts.publish",
	//   "parameterOrder": [
	//     "blogId",
	//     "postId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The ID of the Blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "postId": {
	//       "description": "The ID of the Post.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "publishDate": {
	//       "description": "The date and time to schedule the publishing of the Blog.",
	//       "format": "date-time",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}/publish",
	//   "response": {
	//     "$ref": "Post"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.posts.revert":

type PostsRevertCall struct {
	s      *Service
	blogId string
	postId string
	opt_   map[string]interface{}
}

// Revert: Revert a published or scheduled post to draft state.
func (r *PostsService) Revert(blogId string, postId string) *PostsRevertCall {
	c := &PostsRevertCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
	return c
}

func (c *PostsRevertCall) Do() (*Post, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}/revert")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	ret := new(Post)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Revert a published or scheduled post to draft state.",
	//   "httpMethod": "POST",
	//   "id": "blogger.posts.revert",
	//   "parameterOrder": [
	//     "blogId",
	//     "postId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The ID of the Blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "postId": {
	//       "description": "The ID of the Post.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}/revert",
	//   "response": {
	//     "$ref": "Post"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.posts.search":

type PostsSearchCall struct {
	s      *Service
	blogId string
	q      string
	opt_   map[string]interface{}
}

// Search: Search for a post.
func (r *PostsService) Search(blogId string, q string) *PostsSearchCall {
	c := &PostsSearchCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.q = q
	return c
}

// FetchBodies sets the optional parameter "fetchBodies": Whether the
// body content of posts is included (default: true). This should be set
// to false when the post bodies are not required, to help minimize
// traffic.
func (c *PostsSearchCall) FetchBodies(fetchBodies bool) *PostsSearchCall {
	c.opt_["fetchBodies"] = fetchBodies
	return c
}

// OrderBy sets the optional parameter "orderBy": Sort search results
func (c *PostsSearchCall) OrderBy(orderBy string) *PostsSearchCall {
	c.opt_["orderBy"] = orderBy
	return c
}

func (c *PostsSearchCall) Do() (*PostList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	params.Set("q", fmt.Sprintf("%v", c.q))
	if v, ok := c.opt_["fetchBodies"]; ok {
		params.Set("fetchBodies", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["orderBy"]; ok {
		params.Set("orderBy", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/search")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
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
	ret := new(PostList)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Search for a post.",
	//   "httpMethod": "GET",
	//   "id": "blogger.posts.search",
	//   "parameterOrder": [
	//     "blogId",
	//     "q"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "ID of the blog to fetch the post from.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fetchBodies": {
	//       "default": "true",
	//       "description": "Whether the body content of posts is included (default: true). This should be set to false when the post bodies are not required, to help minimize traffic.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "orderBy": {
	//       "default": "PUBLISHED",
	//       "description": "Sort search results",
	//       "enum": [
	//         "published",
	//         "updated"
	//       ],
	//       "enumDescriptions": [
	//         "Order by the date the post was published",
	//         "Order by the date the post was last updated"
	//       ],
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "q": {
	//       "description": "Query terms to search this blog for matching posts.",
	//       "location": "query",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/search",
	//   "response": {
	//     "$ref": "PostList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}

// method id "blogger.posts.update":

type PostsUpdateCall struct {
	s      *Service
	blogId string
	postId string
	post   *Post
	opt_   map[string]interface{}
}

// Update: Update a post.
func (r *PostsService) Update(blogId string, postId string, post *Post) *PostsUpdateCall {
	c := &PostsUpdateCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
	c.post = post
	return c
}

// FetchBody sets the optional parameter "fetchBody": Whether the body
// content of the post is included with the result (default: true).
func (c *PostsUpdateCall) FetchBody(fetchBody bool) *PostsUpdateCall {
	c.opt_["fetchBody"] = fetchBody
	return c
}

// FetchImages sets the optional parameter "fetchImages": Whether image
// URL metadata for each post is included in the returned result
// (default: false).
func (c *PostsUpdateCall) FetchImages(fetchImages bool) *PostsUpdateCall {
	c.opt_["fetchImages"] = fetchImages
	return c
}

// MaxComments sets the optional parameter "maxComments": Maximum number
// of comments to retrieve with the returned post.
func (c *PostsUpdateCall) MaxComments(maxComments int64) *PostsUpdateCall {
	c.opt_["maxComments"] = maxComments
	return c
}

// Publish sets the optional parameter "publish": Whether a publish
// action should be performed when the post is updated (default: false).
func (c *PostsUpdateCall) Publish(publish bool) *PostsUpdateCall {
	c.opt_["publish"] = publish
	return c
}

// Revert sets the optional parameter "revert": Whether a revert action
// should be performed when the post is updated (default: false).
func (c *PostsUpdateCall) Revert(revert bool) *PostsUpdateCall {
	c.opt_["revert"] = revert
	return c
}

func (c *PostsUpdateCall) Do() (*Post, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.post)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fetchBody"]; ok {
		params.Set("fetchBody", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["fetchImages"]; ok {
		params.Set("fetchImages", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["maxComments"]; ok {
		params.Set("maxComments", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["publish"]; ok {
		params.Set("publish", fmt.Sprintf("%v", v))
	}
	if v, ok := c.opt_["revert"]; ok {
		params.Set("revert", fmt.Sprintf("%v", v))
	}
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}")
	urls += "?" + params.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	req.URL.Path = strings.Replace(req.URL.Path, "{blogId}", url.QueryEscape(c.blogId), 1)
	req.URL.Path = strings.Replace(req.URL.Path, "{postId}", url.QueryEscape(c.postId), 1)
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
	ret := new(Post)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Update a post.",
	//   "httpMethod": "PUT",
	//   "id": "blogger.posts.update",
	//   "parameterOrder": [
	//     "blogId",
	//     "postId"
	//   ],
	//   "parameters": {
	//     "blogId": {
	//       "description": "The ID of the Blog.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "fetchBody": {
	//       "default": "true",
	//       "description": "Whether the body content of the post is included with the result (default: true).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "fetchImages": {
	//       "description": "Whether image URL metadata for each post is included in the returned result (default: false).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxComments": {
	//       "description": "Maximum number of comments to retrieve with the returned post.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "postId": {
	//       "description": "The ID of the Post.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "publish": {
	//       "description": "Whether a publish action should be performed when the post is updated (default: false).",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "revert": {
	//       "description": "Whether a revert action should be performed when the post is updated (default: false).",
	//       "location": "query",
	//       "type": "boolean"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}",
	//   "request": {
	//     "$ref": "Post"
	//   },
	//   "response": {
	//     "$ref": "Post"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.users.get":

type UsersGetCall struct {
	s      *Service
	userId string
	opt_   map[string]interface{}
}

// Get: Gets one user by id.
func (r *UsersService) Get(userId string) *UsersGetCall {
	c := &UsersGetCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	return c
}

func (c *UsersGetCall) Do() (*User, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/{userId}")
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
	ret := new(User)
	if err := json.NewDecoder(res.Body).Decode(ret); err != nil {
		return nil, err
	}
	return ret, nil
	// {
	//   "description": "Gets one user by id.",
	//   "httpMethod": "GET",
	//   "id": "blogger.users.get",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "description": "The ID of the user to get.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/{userId}",
	//   "response": {
	//     "$ref": "User"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger",
	//     "https://www.googleapis.com/auth/blogger.readonly"
	//   ]
	// }

}
