// Package blogger provides access to the Blogger API.
//
// See https://developers.google.com/blogger/docs/2.0/json/getting_started
//
// Usage example:
//
//   import "code.google.com/p/google-api-go-client/blogger/v2"
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

const apiId = "blogger:v2"
const apiName = "blogger"
const apiVersion = "v2"
const basePath = "https://www.googleapis.com/blogger/v2/"

// OAuth2 scopes used by this API.
const (
	// Manage your Blogger account
	BloggerScope = "https://www.googleapis.com/auth/blogger"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Blogs = NewBlogsService(s)
	s.Comments = NewCommentsService(s)
	s.Pages = NewPagesService(s)
	s.Posts = NewPostsService(s)
	s.Users = NewUsersService(s)
	return s, nil
}

type Service struct {
	client   *http.Client
	BasePath string // API endpoint base URL

	Blogs *BlogsService

	Comments *CommentsService

	Pages *PagesService

	Posts *PostsService

	Users *UsersService
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

func NewPagesService(s *Service) *PagesService {
	rs := &PagesService{s: s}
	return rs
}

type PagesService struct {
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
	rs.Blogs = NewUsersBlogsService(s)
	return rs
}

type UsersService struct {
	s *Service

	Blogs *UsersBlogsService
}

func NewUsersBlogsService(s *Service) *UsersBlogsService {
	rs := &UsersBlogsService{s: s}
	return rs
}

type UsersBlogsService struct {
	s *Service
}

type Blog struct {
	// Description: The description of this blog. This is displayed
	// underneath the title.
	Description string `json:"description,omitempty"`

	// Id: The identifier for this resource.
	Id int64 `json:"id,omitempty,string"`

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
	// SelfLink: The URL of the container for posts in this blog.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalItems: The count of posts in this blog.
	TotalItems int64 `json:"totalItems,omitempty"`
}

type BlogList struct {
	// Items: The list of Blogs this user has Authorship or Admin rights
	// over.
	Items []*Blog `json:"items,omitempty"`

	// Kind: The kind of this entity. Always blogger#blogList
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
	Id int64 `json:"id,omitempty,string"`

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
	Id int64 `json:"id,omitempty,string"`
}

type CommentInReplyTo struct {
	// Id: The identified of the parent of this comment.
	Id int64 `json:"id,omitempty,string"`
}

type CommentPost struct {
	// Id: The identifier of the post containing this comment.
	Id int64 `json:"id,omitempty,string"`
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
	Id int64 `json:"id,omitempty,string"`

	// Kind: The kind of this entity. Always blogger#page
	Kind string `json:"kind,omitempty"`

	// Published: RFC 3339 date-time when this Page was published.
	Published string `json:"published,omitempty"`

	// SelfLink: The API REST URL to fetch this resource from.
	SelfLink string `json:"selfLink,omitempty"`

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
	Id int64 `json:"id,omitempty,string"`
}

type PageList struct {
	// Items: The list of Pages for a Blog.
	Items []*Page `json:"items,omitempty"`

	// Kind: The kind of this entity. Always blogger#pageList
	Kind string `json:"kind,omitempty"`
}

type Post struct {
	// Author: The author of this Post.
	Author *PostAuthor `json:"author,omitempty"`

	// Blog: Data about the blog containing this Post.
	Blog *PostBlog `json:"blog,omitempty"`

	// Content: The content of the Post. May contain HTML markup.
	Content string `json:"content,omitempty"`

	// Id: The identifier of this Post.
	Id int64 `json:"id,omitempty,string"`

	// Kind: The kind of this entity. Always blogger#post
	Kind string `json:"kind,omitempty"`

	// Labels: The list of labels this Post was tagged with.
	Labels []string `json:"labels,omitempty"`

	// Published: RFC 3339 date-time when this Post was published.
	Published string `json:"published,omitempty"`

	// Replies: The container of comments on this Post.
	Replies *PostReplies `json:"replies,omitempty"`

	// SelfLink: The API REST URL to fetch this resource from.
	SelfLink string `json:"selfLink,omitempty"`

	// Title: The title of the Post.
	Title string `json:"title,omitempty"`

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
	Id int64 `json:"id,omitempty,string"`
}

type PostReplies struct {
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

	// PrevPageToken: Pagination token to fetch the previous page, if one
	// exists.
	PrevPageToken string `json:"prevPageToken,omitempty"`
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

func (c *BlogsGetCall) Do() (*Blog, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
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
	//     }
	//   },
	//   "path": "blogs/{blogId}",
	//   "response": {
	//     "$ref": "Blog"
	//   },
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

func (c *CommentsGetCall) Do() (*Comment, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
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
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}/comments/{commentId}",
	//   "response": {
	//     "$ref": "Comment"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
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

// List: Retrieves the comments for a blog, possibly filtered.
func (r *CommentsService) List(blogId string, postId string) *CommentsListCall {
	c := &CommentsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.blogId = blogId
	c.postId = postId
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

func (c *CommentsListCall) Do() (*CommentList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
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
	//   "description": "Retrieves the comments for a blog, possibly filtered.",
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
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}/comments",
	//   "response": {
	//     "$ref": "CommentList"
	//   },
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

func (c *PagesGetCall) Do() (*Page, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
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
	//     }
	//   },
	//   "path": "blogs/{blogId}/pages/{pageId}",
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

// List: Retrieves pages for a blog, possibly filtered.
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

func (c *PagesListCall) Do() (*PageList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
	if v, ok := c.opt_["fetchBodies"]; ok {
		params.Set("fetchBodies", fmt.Sprintf("%v", v))
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
	//   "description": "Retrieves pages for a blog, possibly filtered.",
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
	//     }
	//   },
	//   "path": "blogs/{blogId}/pages",
	//   "response": {
	//     "$ref": "PageList"
	//   },
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

func (c *PostsGetCall) Do() (*Post, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
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
	//     "postId": {
	//       "description": "The ID of the post",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts/{postId}",
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

// FetchBodies sets the optional parameter "fetchBodies": Whether the
// body content of posts is included.
func (c *PostsListCall) FetchBodies(fetchBodies bool) *PostsListCall {
	c.opt_["fetchBodies"] = fetchBodies
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of posts to fetch.
func (c *PostsListCall) MaxResults(maxResults int64) *PostsListCall {
	c.opt_["maxResults"] = maxResults
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

func (c *PostsListCall) Do() (*PostList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
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
	//     "fetchBodies": {
	//       "description": "Whether the body content of posts is included.",
	//       "location": "query",
	//       "type": "boolean"
	//     },
	//     "maxResults": {
	//       "description": "Maximum number of posts to fetch.",
	//       "format": "uint32",
	//       "location": "query",
	//       "type": "integer"
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
	//     }
	//   },
	//   "path": "blogs/{blogId}/posts",
	//   "response": {
	//     "$ref": "PostList"
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
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}

// method id "blogger.users.blogs.list":

type UsersBlogsListCall struct {
	s      *Service
	userId string
	opt_   map[string]interface{}
}

// List: Retrieves a list of blogs, possibly filtered.
func (r *UsersBlogsService) List(userId string) *UsersBlogsListCall {
	c := &UsersBlogsListCall{s: r.s, opt_: make(map[string]interface{})}
	c.userId = userId
	return c
}

func (c *UsersBlogsListCall) Do() (*BlogList, error) {
	var body io.Reader = nil
	params := make(url.Values)
	params.Set("alt", "json")
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
	//   "id": "blogger.users.blogs.list",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "description": "ID of the user whose blogs are to be fetched. Either the word 'self' (sans quote marks) or the user's profile identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "users/{userId}/blogs",
	//   "response": {
	//     "$ref": "BlogList"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/blogger"
	//   ]
	// }

}
