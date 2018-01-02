// Package blogger provides access to the Blogger API.
//
// See https://developers.google.com/blogger/docs/2.0/json/getting_started
//
// Usage example:
//
//   import "google.golang.org/api/blogger/v2"
//   ...
//   bloggerService, err := blogger.New(oauthHttpClient)
package blogger // import "google.golang.org/api/blogger/v2"

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
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Blogs *BlogsService

	Comments *CommentsService

	Pages *PagesService

	Posts *PostsService

	Users *UsersService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Description") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Description") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Blog) MarshalJSON() ([]byte, error) {
	type noMethod Blog
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BlogLocale: The locale this Blog is set to.
type BlogLocale struct {
	// Country: The country this blog's locale is set to.
	Country string `json:"country,omitempty"`

	// Language: The language this blog is authored in.
	Language string `json:"language,omitempty"`

	// Variant: The language variant this blog is authored in.
	Variant string `json:"variant,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Country") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Country") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BlogLocale) MarshalJSON() ([]byte, error) {
	type noMethod BlogLocale
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BlogPages: The container of pages in this blog.
type BlogPages struct {
	// SelfLink: The URL of the container for pages in this blog.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalItems: The count of pages in this blog.
	TotalItems int64 `json:"totalItems,omitempty"`

	// ForceSendFields is a list of field names (e.g. "SelfLink") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SelfLink") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BlogPages) MarshalJSON() ([]byte, error) {
	type noMethod BlogPages
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// BlogPosts: The container of posts in this blog.
type BlogPosts struct {
	// SelfLink: The URL of the container for posts in this blog.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalItems: The count of posts in this blog.
	TotalItems int64 `json:"totalItems,omitempty"`

	// ForceSendFields is a list of field names (e.g. "SelfLink") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SelfLink") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BlogPosts) MarshalJSON() ([]byte, error) {
	type noMethod BlogPosts
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type BlogList struct {
	// Items: The list of Blogs this user has Authorship or Admin rights
	// over.
	Items []*Blog `json:"items,omitempty"`

	// Kind: The kind of this entity. Always blogger#blogList
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *BlogList) MarshalJSON() ([]byte, error) {
	type noMethod BlogList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Author") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Author") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Comment) MarshalJSON() ([]byte, error) {
	type noMethod Comment
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CommentAuthor: The author of this Comment.
type CommentAuthor struct {
	// DisplayName: The display name.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The identifier of the Comment creator.
	Id string `json:"id,omitempty"`

	// Image: The comment creator's avatar.
	Image *CommentAuthorImage `json:"image,omitempty"`

	// Url: The URL of the Comment creator's Profile page.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DisplayName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DisplayName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CommentAuthor) MarshalJSON() ([]byte, error) {
	type noMethod CommentAuthor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CommentAuthorImage: The comment creator's avatar.
type CommentAuthorImage struct {
	// Url: The comment creator's avatar URL.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Url") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Url") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CommentAuthorImage) MarshalJSON() ([]byte, error) {
	type noMethod CommentAuthorImage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CommentBlog: Data about the blog containing this comment.
type CommentBlog struct {
	// Id: The identifier of the blog containing this comment.
	Id int64 `json:"id,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CommentBlog) MarshalJSON() ([]byte, error) {
	type noMethod CommentBlog
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CommentInReplyTo: Data about the comment this is in reply to.
type CommentInReplyTo struct {
	// Id: The identified of the parent of this comment.
	Id int64 `json:"id,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CommentInReplyTo) MarshalJSON() ([]byte, error) {
	type noMethod CommentInReplyTo
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// CommentPost: Data about the post containing this comment.
type CommentPost struct {
	// Id: The identifier of the post containing this comment.
	Id int64 `json:"id,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CommentPost) MarshalJSON() ([]byte, error) {
	type noMethod CommentPost
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *CommentList) MarshalJSON() ([]byte, error) {
	type noMethod CommentList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Author") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Author") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Page) MarshalJSON() ([]byte, error) {
	type noMethod Page
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PageAuthor: The author of this Page.
type PageAuthor struct {
	// DisplayName: The display name.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The identifier of the Page creator.
	Id string `json:"id,omitempty"`

	// Image: The page author's avatar.
	Image *PageAuthorImage `json:"image,omitempty"`

	// Url: The URL of the Page creator's Profile page.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DisplayName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DisplayName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PageAuthor) MarshalJSON() ([]byte, error) {
	type noMethod PageAuthor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PageAuthorImage: The page author's avatar.
type PageAuthorImage struct {
	// Url: The page author's avatar URL.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Url") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Url") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PageAuthorImage) MarshalJSON() ([]byte, error) {
	type noMethod PageAuthorImage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PageBlog: Data about the blog containing this Page.
type PageBlog struct {
	// Id: The identifier of the blog containing this page.
	Id int64 `json:"id,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PageBlog) MarshalJSON() ([]byte, error) {
	type noMethod PageBlog
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

type PageList struct {
	// Items: The list of Pages for a Blog.
	Items []*Page `json:"items,omitempty"`

	// Kind: The kind of this entity. Always blogger#pageList
	Kind string `json:"kind,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PageList) MarshalJSON() ([]byte, error) {
	type noMethod PageList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Author") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Author") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *Post) MarshalJSON() ([]byte, error) {
	type noMethod Post
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PostAuthor: The author of this Post.
type PostAuthor struct {
	// DisplayName: The display name.
	DisplayName string `json:"displayName,omitempty"`

	// Id: The identifier of the Post creator.
	Id string `json:"id,omitempty"`

	// Image: The Post author's avatar.
	Image *PostAuthorImage `json:"image,omitempty"`

	// Url: The URL of the Post creator's Profile page.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "DisplayName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "DisplayName") to include
	// in API requests with the JSON null value. By default, fields with
	// empty values are omitted from API requests. However, any field with
	// an empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PostAuthor) MarshalJSON() ([]byte, error) {
	type noMethod PostAuthor
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PostAuthorImage: The Post author's avatar.
type PostAuthorImage struct {
	// Url: The Post author's avatar URL.
	Url string `json:"url,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Url") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Url") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PostAuthorImage) MarshalJSON() ([]byte, error) {
	type noMethod PostAuthorImage
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PostBlog: Data about the blog containing this Post.
type PostBlog struct {
	// Id: The identifier of the Blog that contains this Post.
	Id int64 `json:"id,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "Id") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Id") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PostBlog) MarshalJSON() ([]byte, error) {
	type noMethod PostBlog
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// PostReplies: The container of comments on this Post.
type PostReplies struct {
	// SelfLink: The URL of the comments on this post.
	SelfLink string `json:"selfLink,omitempty"`

	// TotalItems: The count of comments on this post.
	TotalItems int64 `json:"totalItems,omitempty,string"`

	// ForceSendFields is a list of field names (e.g. "SelfLink") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SelfLink") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PostReplies) MarshalJSON() ([]byte, error) {
	type noMethod PostReplies
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Items") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Items") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *PostList) MarshalJSON() ([]byte, error) {
	type noMethod PostList
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
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

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "About") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "About") to include in API
	// requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *User) MarshalJSON() ([]byte, error) {
	type noMethod User
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UserBlogs: The container of blogs for this user.
type UserBlogs struct {
	// SelfLink: The URL of the Blogs for this user.
	SelfLink string `json:"selfLink,omitempty"`

	// ForceSendFields is a list of field names (e.g. "SelfLink") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "SelfLink") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UserBlogs) MarshalJSON() ([]byte, error) {
	type noMethod UserBlogs
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// UserLocale: This user's locale
type UserLocale struct {
	// Country: The user's country setting.
	Country string `json:"country,omitempty"`

	// Language: The user's language setting.
	Language string `json:"language,omitempty"`

	// Variant: The user's language variant setting.
	Variant string `json:"variant,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Country") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`

	// NullFields is a list of field names (e.g. "Country") to include in
	// API requests with the JSON null value. By default, fields with empty
	// values are omitted from API requests. However, any field with an
	// empty value appearing in NullFields will be sent to the server as
	// null. It is an error if a field in this list has a non-empty value.
	// This may be used to include null fields in Patch requests.
	NullFields []string `json:"-"`
}

func (s *UserLocale) MarshalJSON() ([]byte, error) {
	type noMethod UserLocale
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields, s.NullFields)
}

// method id "blogger.blogs.get":

type BlogsGetCall struct {
	s            *Service
	blogId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets one blog by id.
func (r *BlogsService) Get(blogId string) *BlogsGetCall {
	c := &BlogsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.blogId = blogId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *BlogsGetCall) Fields(s ...googleapi.Field) *BlogsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *BlogsGetCall) IfNoneMatch(entityTag string) *BlogsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *BlogsGetCall) Context(ctx context.Context) *BlogsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *BlogsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *BlogsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"blogId": c.blogId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "blogger.blogs.get" call.
// Exactly one of *Blog or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Blog.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *BlogsGetCall) Do(opts ...googleapi.CallOption) (*Blog, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Blog{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
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
	s            *Service
	blogId       string
	postId       string
	commentId    string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets one comment by id.
func (r *CommentsService) Get(blogId string, postId string, commentId string) *CommentsGetCall {
	c := &CommentsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.blogId = blogId
	c.postId = postId
	c.commentId = commentId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CommentsGetCall) Fields(s ...googleapi.Field) *CommentsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CommentsGetCall) IfNoneMatch(entityTag string) *CommentsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CommentsGetCall) Context(ctx context.Context) *CommentsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CommentsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CommentsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}/comments/{commentId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"blogId":    c.blogId,
		"postId":    c.postId,
		"commentId": c.commentId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "blogger.comments.get" call.
// Exactly one of *Comment or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Comment.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CommentsGetCall) Do(opts ...googleapi.CallOption) (*Comment, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Comment{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
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
	s            *Service
	blogId       string
	postId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves the comments for a blog, possibly filtered.
func (r *CommentsService) List(blogId string, postId string) *CommentsListCall {
	c := &CommentsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.blogId = blogId
	c.postId = postId
	return c
}

// FetchBodies sets the optional parameter "fetchBodies": Whether the
// body content of the comments is included.
func (c *CommentsListCall) FetchBodies(fetchBodies bool) *CommentsListCall {
	c.urlParams_.Set("fetchBodies", fmt.Sprint(fetchBodies))
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of comments to include in the result.
func (c *CommentsListCall) MaxResults(maxResults int64) *CommentsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// if request is paged.
func (c *CommentsListCall) PageToken(pageToken string) *CommentsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// StartDate sets the optional parameter "startDate": Earliest date of
// comment to fetch, a date-time with RFC 3339 formatting.
func (c *CommentsListCall) StartDate(startDate string) *CommentsListCall {
	c.urlParams_.Set("startDate", startDate)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CommentsListCall) Fields(s ...googleapi.Field) *CommentsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CommentsListCall) IfNoneMatch(entityTag string) *CommentsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CommentsListCall) Context(ctx context.Context) *CommentsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *CommentsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *CommentsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}/comments")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"blogId": c.blogId,
		"postId": c.postId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "blogger.comments.list" call.
// Exactly one of *CommentList or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *CommentList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CommentsListCall) Do(opts ...googleapi.CallOption) (*CommentList, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &CommentList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
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

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *CommentsListCall) Pages(ctx context.Context, f func(*CommentList) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "blogger.pages.get":

type PagesGetCall struct {
	s            *Service
	blogId       string
	pageId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets one blog page by id.
func (r *PagesService) Get(blogId string, pageId string) *PagesGetCall {
	c := &PagesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.blogId = blogId
	c.pageId = pageId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PagesGetCall) Fields(s ...googleapi.Field) *PagesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PagesGetCall) IfNoneMatch(entityTag string) *PagesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PagesGetCall) Context(ctx context.Context) *PagesGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PagesGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PagesGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/pages/{pageId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"blogId": c.blogId,
		"pageId": c.pageId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "blogger.pages.get" call.
// Exactly one of *Page or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Page.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *PagesGetCall) Do(opts ...googleapi.CallOption) (*Page, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Page{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
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
	s            *Service
	blogId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves pages for a blog, possibly filtered.
func (r *PagesService) List(blogId string) *PagesListCall {
	c := &PagesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.blogId = blogId
	return c
}

// FetchBodies sets the optional parameter "fetchBodies": Whether to
// retrieve the Page bodies.
func (c *PagesListCall) FetchBodies(fetchBodies bool) *PagesListCall {
	c.urlParams_.Set("fetchBodies", fmt.Sprint(fetchBodies))
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PagesListCall) Fields(s ...googleapi.Field) *PagesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PagesListCall) IfNoneMatch(entityTag string) *PagesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PagesListCall) Context(ctx context.Context) *PagesListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PagesListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PagesListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/pages")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"blogId": c.blogId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "blogger.pages.list" call.
// Exactly one of *PageList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *PageList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PagesListCall) Do(opts ...googleapi.CallOption) (*PageList, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &PageList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
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
	s            *Service
	blogId       string
	postId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Get a post by id.
func (r *PostsService) Get(blogId string, postId string) *PostsGetCall {
	c := &PostsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.blogId = blogId
	c.postId = postId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PostsGetCall) Fields(s ...googleapi.Field) *PostsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PostsGetCall) IfNoneMatch(entityTag string) *PostsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PostsGetCall) Context(ctx context.Context) *PostsGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PostsGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PostsGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts/{postId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"blogId": c.blogId,
		"postId": c.postId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "blogger.posts.get" call.
// Exactly one of *Post or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Post.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *PostsGetCall) Do(opts ...googleapi.CallOption) (*Post, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &Post{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
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
	s            *Service
	blogId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves a list of posts, possibly filtered.
func (r *PostsService) List(blogId string) *PostsListCall {
	c := &PostsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.blogId = blogId
	return c
}

// FetchBodies sets the optional parameter "fetchBodies": Whether the
// body content of posts is included.
func (c *PostsListCall) FetchBodies(fetchBodies bool) *PostsListCall {
	c.urlParams_.Set("fetchBodies", fmt.Sprint(fetchBodies))
	return c
}

// MaxResults sets the optional parameter "maxResults": Maximum number
// of posts to fetch.
func (c *PostsListCall) MaxResults(maxResults int64) *PostsListCall {
	c.urlParams_.Set("maxResults", fmt.Sprint(maxResults))
	return c
}

// PageToken sets the optional parameter "pageToken": Continuation token
// if the request is paged.
func (c *PostsListCall) PageToken(pageToken string) *PostsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// StartDate sets the optional parameter "startDate": Earliest post date
// to fetch, a date-time with RFC 3339 formatting.
func (c *PostsListCall) StartDate(startDate string) *PostsListCall {
	c.urlParams_.Set("startDate", startDate)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *PostsListCall) Fields(s ...googleapi.Field) *PostsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *PostsListCall) IfNoneMatch(entityTag string) *PostsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *PostsListCall) Context(ctx context.Context) *PostsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *PostsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *PostsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "blogs/{blogId}/posts")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"blogId": c.blogId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "blogger.posts.list" call.
// Exactly one of *PostList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *PostList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *PostsListCall) Do(opts ...googleapi.CallOption) (*PostList, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &PostList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
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

// Pages invokes f for each page of results.
// A non-nil error returned from f will halt the iteration.
// The provided context supersedes any context provided to the Context method.
func (c *PostsListCall) Pages(ctx context.Context, f func(*PostList) error) error {
	c.ctx_ = ctx
	defer c.PageToken(c.urlParams_.Get("pageToken")) // reset paging to original point
	for {
		x, err := c.Do()
		if err != nil {
			return err
		}
		if err := f(x); err != nil {
			return err
		}
		if x.NextPageToken == "" {
			return nil
		}
		c.PageToken(x.NextPageToken)
	}
}

// method id "blogger.users.get":

type UsersGetCall struct {
	s            *Service
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// Get: Gets one user by id.
func (r *UsersService) Get(userId string) *UsersGetCall {
	c := &UsersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersGetCall) Fields(s ...googleapi.Field) *UsersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersGetCall) IfNoneMatch(entityTag string) *UsersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersGetCall) Context(ctx context.Context) *UsersGetCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersGetCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersGetCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "blogger.users.get" call.
// Exactly one of *User or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *User.ServerResponse.Header or (if a response was returned at all) in
// error.(*googleapi.Error).Header. Use googleapi.IsNotModified to check
// whether the returned error was because http.StatusNotModified was
// returned.
func (c *UsersGetCall) Do(opts ...googleapi.CallOption) (*User, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &User{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
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
	s            *Service
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
	header_      http.Header
}

// List: Retrieves a list of blogs, possibly filtered.
func (r *UsersBlogsService) List(userId string) *UsersBlogsListCall {
	c := &UsersBlogsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UsersBlogsListCall) Fields(s ...googleapi.Field) *UsersBlogsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UsersBlogsListCall) IfNoneMatch(entityTag string) *UsersBlogsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UsersBlogsListCall) Context(ctx context.Context) *UsersBlogsListCall {
	c.ctx_ = ctx
	return c
}

// Header returns an http.Header that can be modified by the caller to
// add HTTP headers to the request.
func (c *UsersBlogsListCall) Header() http.Header {
	if c.header_ == nil {
		c.header_ = make(http.Header)
	}
	return c.header_
}

func (c *UsersBlogsListCall) doRequest(alt string) (*http.Response, error) {
	reqHeaders := make(http.Header)
	for k, v := range c.header_ {
		reqHeaders[k] = v
	}
	reqHeaders.Set("User-Agent", c.s.userAgent())
	if c.ifNoneMatch_ != "" {
		reqHeaders.Set("If-None-Match", c.ifNoneMatch_)
	}
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "users/{userId}/blogs")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	req.Header = reqHeaders
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
	})
	return gensupport.SendRequest(c.ctx_, c.s.client, req)
}

// Do executes the "blogger.users.blogs.list" call.
// Exactly one of *BlogList or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *BlogList.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UsersBlogsListCall) Do(opts ...googleapi.CallOption) (*BlogList, error) {
	gensupport.SetOptions(c.urlParams_, opts...)
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
	ret := &BlogList{
		ServerResponse: googleapi.ServerResponse{
			Header:         res.Header,
			HTTPStatusCode: res.StatusCode,
		},
	}
	target := &ret
	if err := json.NewDecoder(res.Body).Decode(target); err != nil {
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
