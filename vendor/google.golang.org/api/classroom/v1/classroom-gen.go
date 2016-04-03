// Package classroom provides access to the Google Classroom API.
//
// See https://developers.google.com/classroom/
//
// Usage example:
//
//   import "google.golang.org/api/classroom/v1"
//   ...
//   classroomService, err := classroom.New(oauthHttpClient)
package classroom // import "google.golang.org/api/classroom/v1"

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

const apiId = "classroom:v1"
const apiName = "classroom"
const apiVersion = "v1"
const basePath = "https://classroom.googleapis.com/"

// OAuth2 scopes used by this API.
const (
	// Manage your Google Classroom classes
	ClassroomCoursesScope = "https://www.googleapis.com/auth/classroom.courses"

	// View your Google Classroom classes
	ClassroomCoursesReadonlyScope = "https://www.googleapis.com/auth/classroom.courses.readonly"

	// View the email addresses of people in your classes
	ClassroomProfileEmailsScope = "https://www.googleapis.com/auth/classroom.profile.emails"

	// View the profile photos of people in your classes
	ClassroomProfilePhotosScope = "https://www.googleapis.com/auth/classroom.profile.photos"

	// Manage your Google Classroom class rosters
	ClassroomRostersScope = "https://www.googleapis.com/auth/classroom.rosters"

	// View your Google Classroom class rosters
	ClassroomRostersReadonlyScope = "https://www.googleapis.com/auth/classroom.rosters.readonly"
)

func New(client *http.Client) (*Service, error) {
	if client == nil {
		return nil, errors.New("client is nil")
	}
	s := &Service{client: client, BasePath: basePath}
	s.Courses = NewCoursesService(s)
	s.Invitations = NewInvitationsService(s)
	s.UserProfiles = NewUserProfilesService(s)
	return s, nil
}

type Service struct {
	client    *http.Client
	BasePath  string // API endpoint base URL
	UserAgent string // optional additional User-Agent fragment

	Courses *CoursesService

	Invitations *InvitationsService

	UserProfiles *UserProfilesService
}

func (s *Service) userAgent() string {
	if s.UserAgent == "" {
		return googleapi.UserAgent
	}
	return googleapi.UserAgent + " " + s.UserAgent
}

func NewCoursesService(s *Service) *CoursesService {
	rs := &CoursesService{s: s}
	rs.Aliases = NewCoursesAliasesService(s)
	rs.Students = NewCoursesStudentsService(s)
	rs.Teachers = NewCoursesTeachersService(s)
	return rs
}

type CoursesService struct {
	s *Service

	Aliases *CoursesAliasesService

	Students *CoursesStudentsService

	Teachers *CoursesTeachersService
}

func NewCoursesAliasesService(s *Service) *CoursesAliasesService {
	rs := &CoursesAliasesService{s: s}
	return rs
}

type CoursesAliasesService struct {
	s *Service
}

func NewCoursesStudentsService(s *Service) *CoursesStudentsService {
	rs := &CoursesStudentsService{s: s}
	return rs
}

type CoursesStudentsService struct {
	s *Service
}

func NewCoursesTeachersService(s *Service) *CoursesTeachersService {
	rs := &CoursesTeachersService{s: s}
	return rs
}

type CoursesTeachersService struct {
	s *Service
}

func NewInvitationsService(s *Service) *InvitationsService {
	rs := &InvitationsService{s: s}
	return rs
}

type InvitationsService struct {
	s *Service
}

func NewUserProfilesService(s *Service) *UserProfilesService {
	rs := &UserProfilesService{s: s}
	return rs
}

type UserProfilesService struct {
	s *Service
}

// Course: A Course in Classroom.
type Course struct {
	// AlternateLink: Absolute link to this course in the Classroom web UI.
	// Read-only.
	AlternateLink string `json:"alternateLink,omitempty"`

	// CourseState: State of the course. If unspecified, the default state
	// is `PROVISIONED`.
	//
	// Possible values:
	//   "COURSE_STATE_UNSPECIFIED"
	//   "ACTIVE"
	//   "ARCHIVED"
	//   "PROVISIONED"
	//   "DECLINED"
	CourseState string `json:"courseState,omitempty"`

	// CreationTime: Creation time of the course. Specifying this field in a
	// course update mask will result in an error. Read-only.
	CreationTime string `json:"creationTime,omitempty"`

	// Description: Optional description. For example, "We'll be learning
	// about the structure of living creatures from a combination of
	// textbooks, guest lectures, and lab work. Expect to be excited!" If
	// set, this field must be a valid UTF-8 string and no longer than
	// 30,000 characters.
	Description string `json:"description,omitempty"`

	// DescriptionHeading: Optional heading for the description. For
	// example, "Welcome to 10th Grade Biology." If set, this field must be
	// a valid UTF-8 string and no longer than 3600 characters.
	DescriptionHeading string `json:"descriptionHeading,omitempty"`

	// EnrollmentCode: Enrollment code to use when joining this course.
	// Specifying this field in a course update mask will result in an
	// error. Read-only.
	EnrollmentCode string `json:"enrollmentCode,omitempty"`

	// Id: Identifier for this course assigned by Classroom. When creating a
	// course, you may optionally set this identifier to an alias string in
	// the request to create a corresponding alias. The `id` is still
	// assigned by Classroom and cannot be updated after the course is
	// created. Specifying this field in a course update mask will result in
	// an error.
	Id string `json:"id,omitempty"`

	// Name: Name of the course. For example, "10th Grade Biology". The name
	// is required. It must be between 1 and 750 characters and a valid
	// UTF-8 string.
	Name string `json:"name,omitempty"`

	// OwnerId: The identifier of the owner of a course. When specified as a
	// parameter of a create course request, this field is required. The
	// identifier can be one of the following: * the numeric identifier for
	// the user * the email address of the user * the string literal "me",
	// indicating the requesting user This must be set in a create request.
	// Specifying this field in a course update mask will result in an
	// `INVALID_ARGUMENT` error.
	OwnerId string `json:"ownerId,omitempty"`

	// Room: Optional room location. For example, "301". If set, this field
	// must be a valid UTF-8 string and no longer than 650 characters.
	Room string `json:"room,omitempty"`

	// Section: Section of the course. For example, "Period 2". If set, this
	// field must be a valid UTF-8 string and no longer than 2800
	// characters.
	Section string `json:"section,omitempty"`

	// UpdateTime: Time of the most recent update to this course. Specifying
	// this field in a course update mask will result in an error.
	// Read-only.
	UpdateTime string `json:"updateTime,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "AlternateLink") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Course) MarshalJSON() ([]byte, error) {
	type noMethod Course
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// CourseAlias: Alternative identifier for a course. An alias uniquely
// identifies a course. It must be unique within one of the following
// scopes: * domain: A domain-scoped alias is visible to all users
// within the alias creator's domain and can be created only by a domain
// admin. A domain-scoped alias is often used when a course has an
// identifier external to Classroom. * project: A project-scoped alias
// is visible to any request from an application using the Developer
// Console project ID that created the alias and can be created by any
// project. A project-scoped alias is often used when an application has
// alternative identifiers. A random value can also be used to avoid
// duplicate courses in the event of transmission failures, as retrying
// a request will return `ALREADY_EXISTS` if a previous one has
// succeeded.
type CourseAlias struct {
	// Alias: Alias string. The format of the string indicates the desired
	// alias scoping. * `d:` indicates a domain-scoped alias. Example:
	// `d:math_101` * `p:` indicates a project-scoped alias. Example:
	// `p:abc123` This field has a maximum length of 256 characters.
	Alias string `json:"alias,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Alias") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *CourseAlias) MarshalJSON() ([]byte, error) {
	type noMethod CourseAlias
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

// GlobalPermission: Global user permission description.
type GlobalPermission struct {
	// Permission: Permission value.
	//
	// Possible values:
	//   "PERMISSION_UNSPECIFIED"
	//   "CREATE_COURSE"
	Permission string `json:"permission,omitempty"`

	// ForceSendFields is a list of field names (e.g. "Permission") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *GlobalPermission) MarshalJSON() ([]byte, error) {
	type noMethod GlobalPermission
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Invitation: An invitation to join a course.
type Invitation struct {
	// CourseId: Identifier of the course to invite the user to.
	CourseId string `json:"courseId,omitempty"`

	// Id: Identifier assigned by Classroom. Read-only.
	Id string `json:"id,omitempty"`

	// Role: Role to invite the user to have. Must not be
	// `COURSE_ROLE_UNSPECIFIED`.
	//
	// Possible values:
	//   "COURSE_ROLE_UNSPECIFIED"
	//   "STUDENT"
	//   "TEACHER"
	Role string `json:"role,omitempty"`

	// UserId: Identifier of the invited user. When specified as a parameter
	// of a request, this identifier can be set to one of the following: *
	// the numeric identifier for the user * the email address of the user *
	// the string literal "me", indicating the requesting user
	UserId string `json:"userId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CourseId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Invitation) MarshalJSON() ([]byte, error) {
	type noMethod Invitation
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListCourseAliasesResponse: Response when listing course aliases.
type ListCourseAliasesResponse struct {
	// Aliases: The course aliases.
	Aliases []*CourseAlias `json:"aliases,omitempty"`

	// NextPageToken: Token identifying the next page of results to return.
	// If empty, no further results are available.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Aliases") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListCourseAliasesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListCourseAliasesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListCoursesResponse: Response when listing courses.
type ListCoursesResponse struct {
	// Courses: Courses that match the list request.
	Courses []*Course `json:"courses,omitempty"`

	// NextPageToken: Token identifying the next page of results to return.
	// If empty, no further results are available.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Courses") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListCoursesResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListCoursesResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListInvitationsResponse: Response when listing invitations.
type ListInvitationsResponse struct {
	// Invitations: Invitations that match the list request.
	Invitations []*Invitation `json:"invitations,omitempty"`

	// NextPageToken: Token identifying the next page of results to return.
	// If empty, no further results are available.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "Invitations") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *ListInvitationsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListInvitationsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListStudentsResponse: Response when listing students.
type ListStudentsResponse struct {
	// NextPageToken: Token identifying the next page of results to return.
	// If empty, no further results are available.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Students: Students who match the list request.
	Students []*Student `json:"students,omitempty"`

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

func (s *ListStudentsResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListStudentsResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// ListTeachersResponse: Response when listing teachers.
type ListTeachersResponse struct {
	// NextPageToken: Token identifying the next page of results to return.
	// If empty, no further results are available.
	NextPageToken string `json:"nextPageToken,omitempty"`

	// Teachers: Teachers who match the list request.
	Teachers []*Teacher `json:"teachers,omitempty"`

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

func (s *ListTeachersResponse) MarshalJSON() ([]byte, error) {
	type noMethod ListTeachersResponse
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Name: Details of the user's name.
type Name struct {
	// FamilyName: The user's last name. Read-only.
	FamilyName string `json:"familyName,omitempty"`

	// FullName: The user's full name formed by concatenating the first and
	// last name values. Read-only.
	FullName string `json:"fullName,omitempty"`

	// GivenName: The user's first name. Read-only.
	GivenName string `json:"givenName,omitempty"`

	// ForceSendFields is a list of field names (e.g. "FamilyName") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Name) MarshalJSON() ([]byte, error) {
	type noMethod Name
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Student: Student in a course.
type Student struct {
	// CourseId: Identifier of the course. Read-only.
	CourseId string `json:"courseId,omitempty"`

	// Profile: Global user information for the student. Read-only.
	Profile *UserProfile `json:"profile,omitempty"`

	// UserId: Identifier of the user. When specified as a parameter of a
	// request, this identifier can be one of the following: * the numeric
	// identifier for the user * the email address of the user * the string
	// literal "me", indicating the requesting user
	UserId string `json:"userId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CourseId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Student) MarshalJSON() ([]byte, error) {
	type noMethod Student
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// Teacher: Teacher of a course.
type Teacher struct {
	// CourseId: Identifier of the course. Read-only.
	CourseId string `json:"courseId,omitempty"`

	// Profile: Global user information for the teacher. Read-only.
	Profile *UserProfile `json:"profile,omitempty"`

	// UserId: Identifier of the user. When specified as a parameter of a
	// request, this identifier can be one of the following: * the numeric
	// identifier for the user * the email address of the user * the string
	// literal "me", indicating the requesting user
	UserId string `json:"userId,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "CourseId") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *Teacher) MarshalJSON() ([]byte, error) {
	type noMethod Teacher
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// UserProfile: Global information for a user.
type UserProfile struct {
	// EmailAddress: Email address of the user. Read-only.
	EmailAddress string `json:"emailAddress,omitempty"`

	// Id: Identifier of the user. Read-only.
	Id string `json:"id,omitempty"`

	// Name: Name of the user. Read-only.
	Name *Name `json:"name,omitempty"`

	// Permissions: Global permissions of the user. Read-only.
	Permissions []*GlobalPermission `json:"permissions,omitempty"`

	// PhotoUrl: URL of user's profile photo. Read-only.
	PhotoUrl string `json:"photoUrl,omitempty"`

	// ServerResponse contains the HTTP response code and headers from the
	// server.
	googleapi.ServerResponse `json:"-"`

	// ForceSendFields is a list of field names (e.g. "EmailAddress") to
	// unconditionally include in API requests. By default, fields with
	// empty values are omitted from API requests. However, any non-pointer,
	// non-interface field appearing in ForceSendFields will be sent to the
	// server regardless of whether the field is empty or not. This may be
	// used to include empty fields in Patch requests.
	ForceSendFields []string `json:"-"`
}

func (s *UserProfile) MarshalJSON() ([]byte, error) {
	type noMethod UserProfile
	raw := noMethod(*s)
	return gensupport.MarshalJSON(raw, s.ForceSendFields)
}

// method id "classroom.courses.create":

type CoursesCreateCall struct {
	s          *Service
	course     *Course
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Creates a course. The user specified in `ownerId` is the
// owner of the created course and added as a teacher. This method
// returns the following error codes: * `PERMISSION_DENIED` if the
// requesting user is not permitted to create courses or for access
// errors. * `NOT_FOUND` if the primary teacher is not a valid user. *
// `FAILED_PRECONDITION` if the course owner's account is disabled or
// for the following request errors: * UserGroupsMembershipLimitReached
// * `ALREADY_EXISTS` if an alias was specified in the `id` and already
// exists.
func (r *CoursesService) Create(course *Course) *CoursesCreateCall {
	c := &CoursesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.course = course
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesCreateCall) QuotaUser(quotaUser string) *CoursesCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesCreateCall) Fields(s ...googleapi.Field) *CoursesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesCreateCall) Context(ctx context.Context) *CoursesCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.course)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses")
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

// Do executes the "classroom.courses.create" call.
// Exactly one of *Course or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Course.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CoursesCreateCall) Do() (*Course, error) {
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
	ret := &Course{
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
	//   "description": "Creates a course. The user specified in `ownerId` is the owner of the created course and added as a teacher. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to create courses or for access errors. * `NOT_FOUND` if the primary teacher is not a valid user. * `FAILED_PRECONDITION` if the course owner's account is disabled or for the following request errors: * UserGroupsMembershipLimitReached * `ALREADY_EXISTS` if an alias was specified in the `id` and already exists.",
	//   "httpMethod": "POST",
	//   "id": "classroom.courses.create",
	//   "path": "v1/courses",
	//   "request": {
	//     "$ref": "Course"
	//   },
	//   "response": {
	//     "$ref": "Course"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.courses"
	//   ]
	// }

}

// method id "classroom.courses.delete":

type CoursesDeleteCall struct {
	s          *Service
	id         string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a course. This method returns the following error
// codes: * `PERMISSION_DENIED` if the requesting user is not permitted
// to delete the requested course or for access errors. * `NOT_FOUND` if
// no course exists with the requested ID.
func (r *CoursesService) Delete(id string) *CoursesDeleteCall {
	c := &CoursesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesDeleteCall) QuotaUser(quotaUser string) *CoursesDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesDeleteCall) Fields(s ...googleapi.Field) *CoursesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesDeleteCall) Context(ctx context.Context) *CoursesDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"id": c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "classroom.courses.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CoursesDeleteCall) Do() (*Empty, error) {
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
	//   "description": "Deletes a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to delete the requested course or for access errors. * `NOT_FOUND` if no course exists with the requested ID.",
	//   "httpMethod": "DELETE",
	//   "id": "classroom.courses.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "Identifier of the course to delete. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{id}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.courses"
	//   ]
	// }

}

// method id "classroom.courses.get":

type CoursesGetCall struct {
	s            *Service
	id           string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Returns a course. This method returns the following error codes:
// * `PERMISSION_DENIED` if the requesting user is not permitted to
// access the requested course or for access errors. * `NOT_FOUND` if no
// course exists with the requested ID.
func (r *CoursesService) Get(id string) *CoursesGetCall {
	c := &CoursesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesGetCall) QuotaUser(quotaUser string) *CoursesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesGetCall) Fields(s ...googleapi.Field) *CoursesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CoursesGetCall) IfNoneMatch(entityTag string) *CoursesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesGetCall) Context(ctx context.Context) *CoursesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"id": c.id,
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

// Do executes the "classroom.courses.get" call.
// Exactly one of *Course or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Course.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CoursesGetCall) Do() (*Course, error) {
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
	ret := &Course{
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
	//   "description": "Returns a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to access the requested course or for access errors. * `NOT_FOUND` if no course exists with the requested ID.",
	//   "httpMethod": "GET",
	//   "id": "classroom.courses.get",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "Identifier of the course to return. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{id}",
	//   "response": {
	//     "$ref": "Course"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.courses",
	//     "https://www.googleapis.com/auth/classroom.courses.readonly"
	//   ]
	// }

}

// method id "classroom.courses.list":

type CoursesListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Returns a list of courses that the requesting user is permitted
// to view, restricted to those that match the request. This method
// returns the following error codes: * `PERMISSION_DENIED` for access
// errors. * `INVALID_ARGUMENT` if the query argument is malformed. *
// `NOT_FOUND` if any users specified in the query arguments do not
// exist.
func (r *CoursesService) List() *CoursesListCall {
	c := &CoursesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum number of
// items to return. Zero or unspecified indicates that the server may
// assign a maximum. The server may return fewer than the specified
// number of results.
func (c *CoursesListCall) PageSize(pageSize int64) *CoursesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": nextPageToken
// value returned from a previous list call, indicating that the
// subsequent page of results should be returned. The list request must
// be otherwise identical to the one that resulted in this token.
func (c *CoursesListCall) PageToken(pageToken string) *CoursesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesListCall) QuotaUser(quotaUser string) *CoursesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// StudentId sets the optional parameter "studentId": Restricts returned
// courses to those having a student with the specified identifier. The
// identifier can be one of the following: * the numeric identifier for
// the user * the email address of the user * the string literal "me",
// indicating the requesting user
func (c *CoursesListCall) StudentId(studentId string) *CoursesListCall {
	c.urlParams_.Set("studentId", studentId)
	return c
}

// TeacherId sets the optional parameter "teacherId": Restricts returned
// courses to those having a teacher with the specified identifier. The
// identifier can be one of the following: * the numeric identifier for
// the user * the email address of the user * the string literal "me",
// indicating the requesting user
func (c *CoursesListCall) TeacherId(teacherId string) *CoursesListCall {
	c.urlParams_.Set("teacherId", teacherId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesListCall) Fields(s ...googleapi.Field) *CoursesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CoursesListCall) IfNoneMatch(entityTag string) *CoursesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesListCall) Context(ctx context.Context) *CoursesListCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses")
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

// Do executes the "classroom.courses.list" call.
// Exactly one of *ListCoursesResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListCoursesResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *CoursesListCall) Do() (*ListCoursesResponse, error) {
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
	ret := &ListCoursesResponse{
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
	//   "description": "Returns a list of courses that the requesting user is permitted to view, restricted to those that match the request. This method returns the following error codes: * `PERMISSION_DENIED` for access errors. * `INVALID_ARGUMENT` if the query argument is malformed. * `NOT_FOUND` if any users specified in the query arguments do not exist.",
	//   "httpMethod": "GET",
	//   "id": "classroom.courses.list",
	//   "parameters": {
	//     "pageSize": {
	//       "description": "Maximum number of items to return. Zero or unspecified indicates that the server may assign a maximum. The server may return fewer than the specified number of results.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "nextPageToken value returned from a previous list call, indicating that the subsequent page of results should be returned. The list request must be otherwise identical to the one that resulted in this token.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "studentId": {
	//       "description": "Restricts returned courses to those having a student with the specified identifier. The identifier can be one of the following: * the numeric identifier for the user * the email address of the user * the string literal `\"me\"`, indicating the requesting user",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "teacherId": {
	//       "description": "Restricts returned courses to those having a teacher with the specified identifier. The identifier can be one of the following: * the numeric identifier for the user * the email address of the user * the string literal `\"me\"`, indicating the requesting user",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses",
	//   "response": {
	//     "$ref": "ListCoursesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.courses",
	//     "https://www.googleapis.com/auth/classroom.courses.readonly"
	//   ]
	// }

}

// method id "classroom.courses.patch":

type CoursesPatchCall struct {
	s          *Service
	id         string
	course     *Course
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Patch: Updates one or more fields in a course. This method returns
// the following error codes: * `PERMISSION_DENIED` if the requesting
// user is not permitted to modify the requested course or for access
// errors. * `NOT_FOUND` if no course exists with the requested ID. *
// `INVALID_ARGUMENT` if invalid fields are specified in the update mask
// or if no update mask is supplied. * `FAILED_PRECONDITION` for the
// following request errors: * CourseNotModifiable
func (r *CoursesService) Patch(id string, course *Course) *CoursesPatchCall {
	c := &CoursesPatchCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.id = id
	c.course = course
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesPatchCall) QuotaUser(quotaUser string) *CoursesPatchCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UpdateMask sets the optional parameter "updateMask": Mask that
// identifies which fields on the course to update. This field is
// required to do an update. The update will fail if invalid fields are
// specified. The following fields are valid: * `name` * `section` *
// `descriptionHeading` * `description` * `room` * `courseState` When
// set in a query parameter, this field should be specified as
// `updateMask=,,...`
func (c *CoursesPatchCall) UpdateMask(updateMask string) *CoursesPatchCall {
	c.urlParams_.Set("updateMask", updateMask)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesPatchCall) Fields(s ...googleapi.Field) *CoursesPatchCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesPatchCall) Context(ctx context.Context) *CoursesPatchCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesPatchCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.course)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PATCH", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"id": c.id,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "classroom.courses.patch" call.
// Exactly one of *Course or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Course.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CoursesPatchCall) Do() (*Course, error) {
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
	ret := &Course{
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
	//   "description": "Updates one or more fields in a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to modify the requested course or for access errors. * `NOT_FOUND` if no course exists with the requested ID. * `INVALID_ARGUMENT` if invalid fields are specified in the update mask or if no update mask is supplied. * `FAILED_PRECONDITION` for the following request errors: * CourseNotModifiable",
	//   "httpMethod": "PATCH",
	//   "id": "classroom.courses.patch",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "Identifier of the course to update. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "updateMask": {
	//       "description": "Mask that identifies which fields on the course to update. This field is required to do an update. The update will fail if invalid fields are specified. The following fields are valid: * `name` * `section` * `descriptionHeading` * `description` * `room` * `courseState` When set in a query parameter, this field should be specified as `updateMask=,,...`",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{id}",
	//   "request": {
	//     "$ref": "Course"
	//   },
	//   "response": {
	//     "$ref": "Course"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.courses"
	//   ]
	// }

}

// method id "classroom.courses.update":

type CoursesUpdateCall struct {
	s          *Service
	id         string
	course     *Course
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Update: Updates a course. This method returns the following error
// codes: * `PERMISSION_DENIED` if the requesting user is not permitted
// to modify the requested course or for access errors. * `NOT_FOUND` if
// no course exists with the requested ID. * `FAILED_PRECONDITION` for
// the following request errors: * CourseNotModifiable
func (r *CoursesService) Update(id string, course *Course) *CoursesUpdateCall {
	c := &CoursesUpdateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.id = id
	c.course = course
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesUpdateCall) QuotaUser(quotaUser string) *CoursesUpdateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesUpdateCall) Fields(s ...googleapi.Field) *CoursesUpdateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesUpdateCall) Context(ctx context.Context) *CoursesUpdateCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesUpdateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.course)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("PUT", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"id": c.id,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "classroom.courses.update" call.
// Exactly one of *Course or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Course.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CoursesUpdateCall) Do() (*Course, error) {
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
	ret := &Course{
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
	//   "description": "Updates a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to modify the requested course or for access errors. * `NOT_FOUND` if no course exists with the requested ID. * `FAILED_PRECONDITION` for the following request errors: * CourseNotModifiable",
	//   "httpMethod": "PUT",
	//   "id": "classroom.courses.update",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "Identifier of the course to update. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{id}",
	//   "request": {
	//     "$ref": "Course"
	//   },
	//   "response": {
	//     "$ref": "Course"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.courses"
	//   ]
	// }

}

// method id "classroom.courses.aliases.create":

type CoursesAliasesCreateCall struct {
	s           *Service
	courseId    string
	coursealias *CourseAlias
	urlParams_  gensupport.URLParams
	ctx_        context.Context
}

// Create: Creates an alias for a course. This method returns the
// following error codes: * `PERMISSION_DENIED` if the requesting user
// is not permitted to create the alias or for access errors. *
// `NOT_FOUND` if the course does not exist. * `ALREADY_EXISTS` if the
// alias already exists.
func (r *CoursesAliasesService) Create(courseId string, coursealias *CourseAlias) *CoursesAliasesCreateCall {
	c := &CoursesAliasesCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.courseId = courseId
	c.coursealias = coursealias
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesAliasesCreateCall) QuotaUser(quotaUser string) *CoursesAliasesCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesAliasesCreateCall) Fields(s ...googleapi.Field) *CoursesAliasesCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesAliasesCreateCall) Context(ctx context.Context) *CoursesAliasesCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesAliasesCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.coursealias)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{courseId}/aliases")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"courseId": c.courseId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "classroom.courses.aliases.create" call.
// Exactly one of *CourseAlias or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *CourseAlias.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *CoursesAliasesCreateCall) Do() (*CourseAlias, error) {
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
	ret := &CourseAlias{
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
	//   "description": "Creates an alias for a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to create the alias or for access errors. * `NOT_FOUND` if the course does not exist. * `ALREADY_EXISTS` if the alias already exists.",
	//   "httpMethod": "POST",
	//   "id": "classroom.courses.aliases.create",
	//   "parameterOrder": [
	//     "courseId"
	//   ],
	//   "parameters": {
	//     "courseId": {
	//       "description": "Identifier of the course to alias. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{courseId}/aliases",
	//   "request": {
	//     "$ref": "CourseAlias"
	//   },
	//   "response": {
	//     "$ref": "CourseAlias"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.courses"
	//   ]
	// }

}

// method id "classroom.courses.aliases.delete":

type CoursesAliasesDeleteCall struct {
	s          *Service
	courseId   string
	aliasid    string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes an alias of a course. This method returns the
// following error codes: * `PERMISSION_DENIED` if the requesting user
// is not permitted to remove the alias or for access errors. *
// `NOT_FOUND` if the alias does not exist.
func (r *CoursesAliasesService) Delete(courseId string, aliasid string) *CoursesAliasesDeleteCall {
	c := &CoursesAliasesDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.courseId = courseId
	c.aliasid = aliasid
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesAliasesDeleteCall) QuotaUser(quotaUser string) *CoursesAliasesDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesAliasesDeleteCall) Fields(s ...googleapi.Field) *CoursesAliasesDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesAliasesDeleteCall) Context(ctx context.Context) *CoursesAliasesDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesAliasesDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{courseId}/aliases/{alias}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"courseId": c.courseId,
		"alias":    c.aliasid,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "classroom.courses.aliases.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CoursesAliasesDeleteCall) Do() (*Empty, error) {
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
	//   "description": "Deletes an alias of a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to remove the alias or for access errors. * `NOT_FOUND` if the alias does not exist.",
	//   "httpMethod": "DELETE",
	//   "id": "classroom.courses.aliases.delete",
	//   "parameterOrder": [
	//     "courseId",
	//     "alias"
	//   ],
	//   "parameters": {
	//     "alias": {
	//       "description": "Alias to delete. This may not be the Classroom-assigned identifier.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "courseId": {
	//       "description": "Identifier of the course whose alias should be deleted. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{courseId}/aliases/{alias}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.courses"
	//   ]
	// }

}

// method id "classroom.courses.aliases.list":

type CoursesAliasesListCall struct {
	s            *Service
	courseId     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Returns a list of aliases for a course. This method returns the
// following error codes: * `PERMISSION_DENIED` if the requesting user
// is not permitted to access the course or for access errors. *
// `NOT_FOUND` if the course does not exist.
func (r *CoursesAliasesService) List(courseId string) *CoursesAliasesListCall {
	c := &CoursesAliasesListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.courseId = courseId
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum number of
// items to return. Zero or unspecified indicates that the server may
// assign a maximum. The server may return fewer than the specified
// number of results.
func (c *CoursesAliasesListCall) PageSize(pageSize int64) *CoursesAliasesListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": nextPageToken
// value returned from a previous list call, indicating that the
// subsequent page of results should be returned. The list request must
// be otherwise identical to the one that resulted in this token.
func (c *CoursesAliasesListCall) PageToken(pageToken string) *CoursesAliasesListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesAliasesListCall) QuotaUser(quotaUser string) *CoursesAliasesListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesAliasesListCall) Fields(s ...googleapi.Field) *CoursesAliasesListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CoursesAliasesListCall) IfNoneMatch(entityTag string) *CoursesAliasesListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesAliasesListCall) Context(ctx context.Context) *CoursesAliasesListCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesAliasesListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{courseId}/aliases")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"courseId": c.courseId,
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

// Do executes the "classroom.courses.aliases.list" call.
// Exactly one of *ListCourseAliasesResponse or error will be non-nil.
// Any non-2xx status code is an error. Response headers are in either
// *ListCourseAliasesResponse.ServerResponse.Header or (if a response
// was returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *CoursesAliasesListCall) Do() (*ListCourseAliasesResponse, error) {
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
	ret := &ListCourseAliasesResponse{
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
	//   "description": "Returns a list of aliases for a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to access the course or for access errors. * `NOT_FOUND` if the course does not exist.",
	//   "httpMethod": "GET",
	//   "id": "classroom.courses.aliases.list",
	//   "parameterOrder": [
	//     "courseId"
	//   ],
	//   "parameters": {
	//     "courseId": {
	//       "description": "The identifier of the course. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Maximum number of items to return. Zero or unspecified indicates that the server may assign a maximum. The server may return fewer than the specified number of results.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "nextPageToken value returned from a previous list call, indicating that the subsequent page of results should be returned. The list request must be otherwise identical to the one that resulted in this token.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{courseId}/aliases",
	//   "response": {
	//     "$ref": "ListCourseAliasesResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.courses",
	//     "https://www.googleapis.com/auth/classroom.courses.readonly"
	//   ]
	// }

}

// method id "classroom.courses.students.create":

type CoursesStudentsCreateCall struct {
	s          *Service
	courseId   string
	student    *Student
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Adds a user as a student of a course. This method returns the
// following error codes: * `PERMISSION_DENIED` if the requesting user
// is not permitted to create students in this course or for access
// errors. * `NOT_FOUND` if the requested course ID does not exist. *
// `FAILED_PRECONDITION` if the requested user's account is disabled,
// for the following request errors: * CourseMemberLimitReached *
// CourseNotModifiable * UserGroupsMembershipLimitReached *
// `ALREADY_EXISTS` if the user is already a student or teacher in the
// course.
func (r *CoursesStudentsService) Create(courseId string, student *Student) *CoursesStudentsCreateCall {
	c := &CoursesStudentsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.courseId = courseId
	c.student = student
	return c
}

// EnrollmentCode sets the optional parameter "enrollmentCode":
// Enrollment code of the course to create the student in. This code is
// required if userId corresponds to the requesting user; it may be
// omitted if the requesting user has administrative permissions to
// create students for any user.
func (c *CoursesStudentsCreateCall) EnrollmentCode(enrollmentCode string) *CoursesStudentsCreateCall {
	c.urlParams_.Set("enrollmentCode", enrollmentCode)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesStudentsCreateCall) QuotaUser(quotaUser string) *CoursesStudentsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesStudentsCreateCall) Fields(s ...googleapi.Field) *CoursesStudentsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesStudentsCreateCall) Context(ctx context.Context) *CoursesStudentsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesStudentsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.student)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{courseId}/students")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"courseId": c.courseId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "classroom.courses.students.create" call.
// Exactly one of *Student or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Student.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CoursesStudentsCreateCall) Do() (*Student, error) {
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
	ret := &Student{
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
	//   "description": "Adds a user as a student of a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to create students in this course or for access errors. * `NOT_FOUND` if the requested course ID does not exist. * `FAILED_PRECONDITION` if the requested user's account is disabled, for the following request errors: * CourseMemberLimitReached * CourseNotModifiable * UserGroupsMembershipLimitReached * `ALREADY_EXISTS` if the user is already a student or teacher in the course.",
	//   "httpMethod": "POST",
	//   "id": "classroom.courses.students.create",
	//   "parameterOrder": [
	//     "courseId"
	//   ],
	//   "parameters": {
	//     "courseId": {
	//       "description": "Identifier of the course to create the student in. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "enrollmentCode": {
	//       "description": "Enrollment code of the course to create the student in. This code is required if userId corresponds to the requesting user; it may be omitted if the requesting user has administrative permissions to create students for any user.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{courseId}/students",
	//   "request": {
	//     "$ref": "Student"
	//   },
	//   "response": {
	//     "$ref": "Student"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.profile.emails",
	//     "https://www.googleapis.com/auth/classroom.profile.photos",
	//     "https://www.googleapis.com/auth/classroom.rosters"
	//   ]
	// }

}

// method id "classroom.courses.students.delete":

type CoursesStudentsDeleteCall struct {
	s          *Service
	courseId   string
	userId     string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a student of a course. This method returns the
// following error codes: * `PERMISSION_DENIED` if the requesting user
// is not permitted to delete students of this course or for access
// errors. * `NOT_FOUND` if no student of this course has the requested
// ID or if the course does not exist.
func (r *CoursesStudentsService) Delete(courseId string, userId string) *CoursesStudentsDeleteCall {
	c := &CoursesStudentsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.courseId = courseId
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesStudentsDeleteCall) QuotaUser(quotaUser string) *CoursesStudentsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesStudentsDeleteCall) Fields(s ...googleapi.Field) *CoursesStudentsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesStudentsDeleteCall) Context(ctx context.Context) *CoursesStudentsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesStudentsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{courseId}/students/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"courseId": c.courseId,
		"userId":   c.userId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "classroom.courses.students.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CoursesStudentsDeleteCall) Do() (*Empty, error) {
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
	//   "description": "Deletes a student of a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to delete students of this course or for access errors. * `NOT_FOUND` if no student of this course has the requested ID or if the course does not exist.",
	//   "httpMethod": "DELETE",
	//   "id": "classroom.courses.students.delete",
	//   "parameterOrder": [
	//     "courseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "courseId": {
	//       "description": "Identifier of the course. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Identifier of the student to delete. The identifier can be one of the following: * the numeric identifier for the user * the email address of the user * the string literal `\"me\"`, indicating the requesting user",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{courseId}/students/{userId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.rosters"
	//   ]
	// }

}

// method id "classroom.courses.students.get":

type CoursesStudentsGetCall struct {
	s            *Service
	courseId     string
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Returns a student of a course. This method returns the following
// error codes: * `PERMISSION_DENIED` if the requesting user is not
// permitted to view students of this course or for access errors. *
// `NOT_FOUND` if no student of this course has the requested ID or if
// the course does not exist.
func (r *CoursesStudentsService) Get(courseId string, userId string) *CoursesStudentsGetCall {
	c := &CoursesStudentsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.courseId = courseId
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesStudentsGetCall) QuotaUser(quotaUser string) *CoursesStudentsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesStudentsGetCall) Fields(s ...googleapi.Field) *CoursesStudentsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CoursesStudentsGetCall) IfNoneMatch(entityTag string) *CoursesStudentsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesStudentsGetCall) Context(ctx context.Context) *CoursesStudentsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesStudentsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{courseId}/students/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"courseId": c.courseId,
		"userId":   c.userId,
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

// Do executes the "classroom.courses.students.get" call.
// Exactly one of *Student or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Student.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CoursesStudentsGetCall) Do() (*Student, error) {
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
	ret := &Student{
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
	//   "description": "Returns a student of a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to view students of this course or for access errors. * `NOT_FOUND` if no student of this course has the requested ID or if the course does not exist.",
	//   "httpMethod": "GET",
	//   "id": "classroom.courses.students.get",
	//   "parameterOrder": [
	//     "courseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "courseId": {
	//       "description": "Identifier of the course. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Identifier of the student to return. The identifier can be one of the following: * the numeric identifier for the user * the email address of the user * the string literal `\"me\"`, indicating the requesting user",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{courseId}/students/{userId}",
	//   "response": {
	//     "$ref": "Student"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.profile.emails",
	//     "https://www.googleapis.com/auth/classroom.profile.photos",
	//     "https://www.googleapis.com/auth/classroom.rosters",
	//     "https://www.googleapis.com/auth/classroom.rosters.readonly"
	//   ]
	// }

}

// method id "classroom.courses.students.list":

type CoursesStudentsListCall struct {
	s            *Service
	courseId     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Returns a list of students of this course that the requester is
// permitted to view. This method returns the following error codes: *
// `NOT_FOUND` if the course does not exist. * `PERMISSION_DENIED` for
// access errors.
func (r *CoursesStudentsService) List(courseId string) *CoursesStudentsListCall {
	c := &CoursesStudentsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.courseId = courseId
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum number of
// items to return. Zero means no maximum. The server may return fewer
// than the specified number of results.
func (c *CoursesStudentsListCall) PageSize(pageSize int64) *CoursesStudentsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": nextPageToken
// value returned from a previous list call, indicating that the
// subsequent page of results should be returned. The list request must
// be otherwise identical to the one that resulted in this token.
func (c *CoursesStudentsListCall) PageToken(pageToken string) *CoursesStudentsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesStudentsListCall) QuotaUser(quotaUser string) *CoursesStudentsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesStudentsListCall) Fields(s ...googleapi.Field) *CoursesStudentsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CoursesStudentsListCall) IfNoneMatch(entityTag string) *CoursesStudentsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesStudentsListCall) Context(ctx context.Context) *CoursesStudentsListCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesStudentsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{courseId}/students")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"courseId": c.courseId,
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

// Do executes the "classroom.courses.students.list" call.
// Exactly one of *ListStudentsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListStudentsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *CoursesStudentsListCall) Do() (*ListStudentsResponse, error) {
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
	ret := &ListStudentsResponse{
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
	//   "description": "Returns a list of students of this course that the requester is permitted to view. This method returns the following error codes: * `NOT_FOUND` if the course does not exist. * `PERMISSION_DENIED` for access errors.",
	//   "httpMethod": "GET",
	//   "id": "classroom.courses.students.list",
	//   "parameterOrder": [
	//     "courseId"
	//   ],
	//   "parameters": {
	//     "courseId": {
	//       "description": "Identifier of the course. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Maximum number of items to return. Zero means no maximum. The server may return fewer than the specified number of results.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "nextPageToken value returned from a previous list call, indicating that the subsequent page of results should be returned. The list request must be otherwise identical to the one that resulted in this token.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{courseId}/students",
	//   "response": {
	//     "$ref": "ListStudentsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.profile.emails",
	//     "https://www.googleapis.com/auth/classroom.profile.photos",
	//     "https://www.googleapis.com/auth/classroom.rosters",
	//     "https://www.googleapis.com/auth/classroom.rosters.readonly"
	//   ]
	// }

}

// method id "classroom.courses.teachers.create":

type CoursesTeachersCreateCall struct {
	s          *Service
	courseId   string
	teacher    *Teacher
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Creates a teacher of a course. This method returns the
// following error codes: * `PERMISSION_DENIED` if the requesting user
// is not permitted to create teachers in this course or for access
// errors. * `NOT_FOUND` if the requested course ID does not exist. *
// `FAILED_PRECONDITION` if the requested user's account is disabled,
// for the following request errors: * CourseMemberLimitReached *
// CourseNotModifiable * CourseTeacherLimitReached *
// UserGroupsMembershipLimitReached * `ALREADY_EXISTS` if the user is
// already a teacher or student in the course.
func (r *CoursesTeachersService) Create(courseId string, teacher *Teacher) *CoursesTeachersCreateCall {
	c := &CoursesTeachersCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.courseId = courseId
	c.teacher = teacher
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesTeachersCreateCall) QuotaUser(quotaUser string) *CoursesTeachersCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesTeachersCreateCall) Fields(s ...googleapi.Field) *CoursesTeachersCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesTeachersCreateCall) Context(ctx context.Context) *CoursesTeachersCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesTeachersCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.teacher)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{courseId}/teachers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"courseId": c.courseId,
	})
	req.Header.Set("Content-Type", ctype)
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "classroom.courses.teachers.create" call.
// Exactly one of *Teacher or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Teacher.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CoursesTeachersCreateCall) Do() (*Teacher, error) {
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
	ret := &Teacher{
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
	//   "description": "Creates a teacher of a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to create teachers in this course or for access errors. * `NOT_FOUND` if the requested course ID does not exist. * `FAILED_PRECONDITION` if the requested user's account is disabled, for the following request errors: * CourseMemberLimitReached * CourseNotModifiable * CourseTeacherLimitReached * UserGroupsMembershipLimitReached * `ALREADY_EXISTS` if the user is already a teacher or student in the course.",
	//   "httpMethod": "POST",
	//   "id": "classroom.courses.teachers.create",
	//   "parameterOrder": [
	//     "courseId"
	//   ],
	//   "parameters": {
	//     "courseId": {
	//       "description": "Identifier of the course. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{courseId}/teachers",
	//   "request": {
	//     "$ref": "Teacher"
	//   },
	//   "response": {
	//     "$ref": "Teacher"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.profile.emails",
	//     "https://www.googleapis.com/auth/classroom.profile.photos",
	//     "https://www.googleapis.com/auth/classroom.rosters"
	//   ]
	// }

}

// method id "classroom.courses.teachers.delete":

type CoursesTeachersDeleteCall struct {
	s          *Service
	courseId   string
	userId     string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes a teacher of a course. This method returns the
// following error codes: * `PERMISSION_DENIED` if the requesting user
// is not permitted to delete teachers of this course or for access
// errors. * `NOT_FOUND` if no teacher of this course has the requested
// ID or if the course does not exist. * `FAILED_PRECONDITION` if the
// requested ID belongs to the primary teacher of this course.
func (r *CoursesTeachersService) Delete(courseId string, userId string) *CoursesTeachersDeleteCall {
	c := &CoursesTeachersDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.courseId = courseId
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesTeachersDeleteCall) QuotaUser(quotaUser string) *CoursesTeachersDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesTeachersDeleteCall) Fields(s ...googleapi.Field) *CoursesTeachersDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesTeachersDeleteCall) Context(ctx context.Context) *CoursesTeachersDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesTeachersDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{courseId}/teachers/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"courseId": c.courseId,
		"userId":   c.userId,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "classroom.courses.teachers.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CoursesTeachersDeleteCall) Do() (*Empty, error) {
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
	//   "description": "Deletes a teacher of a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to delete teachers of this course or for access errors. * `NOT_FOUND` if no teacher of this course has the requested ID or if the course does not exist. * `FAILED_PRECONDITION` if the requested ID belongs to the primary teacher of this course.",
	//   "httpMethod": "DELETE",
	//   "id": "classroom.courses.teachers.delete",
	//   "parameterOrder": [
	//     "courseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "courseId": {
	//       "description": "Identifier of the course. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Identifier of the teacher to delete. The identifier can be one of the following: * the numeric identifier for the user * the email address of the user * the string literal `\"me\"`, indicating the requesting user",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{courseId}/teachers/{userId}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.rosters"
	//   ]
	// }

}

// method id "classroom.courses.teachers.get":

type CoursesTeachersGetCall struct {
	s            *Service
	courseId     string
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Returns a teacher of a course. This method returns the following
// error codes: * `PERMISSION_DENIED` if the requesting user is not
// permitted to view teachers of this course or for access errors. *
// `NOT_FOUND` if no teacher of this course has the requested ID or if
// the course does not exist.
func (r *CoursesTeachersService) Get(courseId string, userId string) *CoursesTeachersGetCall {
	c := &CoursesTeachersGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.courseId = courseId
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesTeachersGetCall) QuotaUser(quotaUser string) *CoursesTeachersGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesTeachersGetCall) Fields(s ...googleapi.Field) *CoursesTeachersGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CoursesTeachersGetCall) IfNoneMatch(entityTag string) *CoursesTeachersGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesTeachersGetCall) Context(ctx context.Context) *CoursesTeachersGetCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesTeachersGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{courseId}/teachers/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"courseId": c.courseId,
		"userId":   c.userId,
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

// Do executes the "classroom.courses.teachers.get" call.
// Exactly one of *Teacher or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Teacher.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *CoursesTeachersGetCall) Do() (*Teacher, error) {
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
	ret := &Teacher{
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
	//   "description": "Returns a teacher of a course. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to view teachers of this course or for access errors. * `NOT_FOUND` if no teacher of this course has the requested ID or if the course does not exist.",
	//   "httpMethod": "GET",
	//   "id": "classroom.courses.teachers.get",
	//   "parameterOrder": [
	//     "courseId",
	//     "userId"
	//   ],
	//   "parameters": {
	//     "courseId": {
	//       "description": "Identifier of the course. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Identifier of the teacher to return. The identifier can be one of the following: * the numeric identifier for the user * the email address of the user * the string literal `\"me\"`, indicating the requesting user",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{courseId}/teachers/{userId}",
	//   "response": {
	//     "$ref": "Teacher"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.profile.emails",
	//     "https://www.googleapis.com/auth/classroom.profile.photos",
	//     "https://www.googleapis.com/auth/classroom.rosters",
	//     "https://www.googleapis.com/auth/classroom.rosters.readonly"
	//   ]
	// }

}

// method id "classroom.courses.teachers.list":

type CoursesTeachersListCall struct {
	s            *Service
	courseId     string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Returns a list of teachers of this course that the requester is
// permitted to view. This method returns the following error codes: *
// `NOT_FOUND` if the course does not exist. * `PERMISSION_DENIED` for
// access errors.
func (r *CoursesTeachersService) List(courseId string) *CoursesTeachersListCall {
	c := &CoursesTeachersListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.courseId = courseId
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum number of
// items to return. Zero means no maximum. The server may return fewer
// than the specified number of results.
func (c *CoursesTeachersListCall) PageSize(pageSize int64) *CoursesTeachersListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": nextPageToken
// value returned from a previous list call, indicating that the
// subsequent page of results should be returned. The list request must
// be otherwise identical to the one that resulted in this token.
func (c *CoursesTeachersListCall) PageToken(pageToken string) *CoursesTeachersListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *CoursesTeachersListCall) QuotaUser(quotaUser string) *CoursesTeachersListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *CoursesTeachersListCall) Fields(s ...googleapi.Field) *CoursesTeachersListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *CoursesTeachersListCall) IfNoneMatch(entityTag string) *CoursesTeachersListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *CoursesTeachersListCall) Context(ctx context.Context) *CoursesTeachersListCall {
	c.ctx_ = ctx
	return c
}

func (c *CoursesTeachersListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/courses/{courseId}/teachers")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"courseId": c.courseId,
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

// Do executes the "classroom.courses.teachers.list" call.
// Exactly one of *ListTeachersResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListTeachersResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *CoursesTeachersListCall) Do() (*ListTeachersResponse, error) {
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
	ret := &ListTeachersResponse{
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
	//   "description": "Returns a list of teachers of this course that the requester is permitted to view. This method returns the following error codes: * `NOT_FOUND` if the course does not exist. * `PERMISSION_DENIED` for access errors.",
	//   "httpMethod": "GET",
	//   "id": "classroom.courses.teachers.list",
	//   "parameterOrder": [
	//     "courseId"
	//   ],
	//   "parameters": {
	//     "courseId": {
	//       "description": "Identifier of the course. This identifier can be either the Classroom-assigned identifier or an alias.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Maximum number of items to return. Zero means no maximum. The server may return fewer than the specified number of results.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "nextPageToken value returned from a previous list call, indicating that the subsequent page of results should be returned. The list request must be otherwise identical to the one that resulted in this token.",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/courses/{courseId}/teachers",
	//   "response": {
	//     "$ref": "ListTeachersResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.profile.emails",
	//     "https://www.googleapis.com/auth/classroom.profile.photos",
	//     "https://www.googleapis.com/auth/classroom.rosters",
	//     "https://www.googleapis.com/auth/classroom.rosters.readonly"
	//   ]
	// }

}

// method id "classroom.invitations.accept":

type InvitationsAcceptCall struct {
	s          *Service
	id         string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Accept: Accepts an invitation, removing it and adding the invited
// user to the teachers or students (as appropriate) of the specified
// course. Only the invited user may accept an invitation. This method
// returns the following error codes: * `PERMISSION_DENIED` if the
// requesting user is not permitted to accept the requested invitation
// or for access errors. * `FAILED_PRECONDITION` for the following
// request errors: * CourseMemberLimitReached * CourseNotModifiable *
// CourseTeacherLimitReached * UserGroupsMembershipLimitReached *
// `NOT_FOUND` if no invitation exists with the requested ID.
func (r *InvitationsService) Accept(id string) *InvitationsAcceptCall {
	c := &InvitationsAcceptCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *InvitationsAcceptCall) QuotaUser(quotaUser string) *InvitationsAcceptCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InvitationsAcceptCall) Fields(s ...googleapi.Field) *InvitationsAcceptCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InvitationsAcceptCall) Context(ctx context.Context) *InvitationsAcceptCall {
	c.ctx_ = ctx
	return c
}

func (c *InvitationsAcceptCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/invitations/{id}:accept")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("POST", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"id": c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "classroom.invitations.accept" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *InvitationsAcceptCall) Do() (*Empty, error) {
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
	//   "description": "Accepts an invitation, removing it and adding the invited user to the teachers or students (as appropriate) of the specified course. Only the invited user may accept an invitation. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to accept the requested invitation or for access errors. * `FAILED_PRECONDITION` for the following request errors: * CourseMemberLimitReached * CourseNotModifiable * CourseTeacherLimitReached * UserGroupsMembershipLimitReached * `NOT_FOUND` if no invitation exists with the requested ID.",
	//   "httpMethod": "POST",
	//   "id": "classroom.invitations.accept",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "Identifier of the invitation to accept.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/invitations/{id}:accept",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.rosters"
	//   ]
	// }

}

// method id "classroom.invitations.create":

type InvitationsCreateCall struct {
	s          *Service
	invitation *Invitation
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Create: Creates an invitation. Only one invitation for a user and
// course may exist at a time. Delete and re-create an invitation to
// make changes. This method returns the following error codes: *
// `PERMISSION_DENIED` if the requesting user is not permitted to create
// invitations for this course or for access errors. * `NOT_FOUND` if
// the course or the user does not exist. * `FAILED_PRECONDITION` if the
// requested user's account is disabled or if the user already has this
// role or a role with greater permissions. * `ALREADY_EXISTS` if an
// invitation for the specified user and course already exists.
func (r *InvitationsService) Create(invitation *Invitation) *InvitationsCreateCall {
	c := &InvitationsCreateCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.invitation = invitation
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *InvitationsCreateCall) QuotaUser(quotaUser string) *InvitationsCreateCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InvitationsCreateCall) Fields(s ...googleapi.Field) *InvitationsCreateCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InvitationsCreateCall) Context(ctx context.Context) *InvitationsCreateCall {
	c.ctx_ = ctx
	return c
}

func (c *InvitationsCreateCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	body, err := googleapi.WithoutDataWrapper.JSONReader(c.invitation)
	if err != nil {
		return nil, err
	}
	ctype := "application/json"
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/invitations")
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

// Do executes the "classroom.invitations.create" call.
// Exactly one of *Invitation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Invitation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InvitationsCreateCall) Do() (*Invitation, error) {
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
	ret := &Invitation{
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
	//   "description": "Creates an invitation. Only one invitation for a user and course may exist at a time. Delete and re-create an invitation to make changes. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to create invitations for this course or for access errors. * `NOT_FOUND` if the course or the user does not exist. * `FAILED_PRECONDITION` if the requested user's account is disabled or if the user already has this role or a role with greater permissions. * `ALREADY_EXISTS` if an invitation for the specified user and course already exists.",
	//   "httpMethod": "POST",
	//   "id": "classroom.invitations.create",
	//   "path": "v1/invitations",
	//   "request": {
	//     "$ref": "Invitation"
	//   },
	//   "response": {
	//     "$ref": "Invitation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.rosters"
	//   ]
	// }

}

// method id "classroom.invitations.delete":

type InvitationsDeleteCall struct {
	s          *Service
	id         string
	urlParams_ gensupport.URLParams
	ctx_       context.Context
}

// Delete: Deletes an invitation. This method returns the following
// error codes: * `PERMISSION_DENIED` if the requesting user is not
// permitted to delete the requested invitation or for access errors. *
// `NOT_FOUND` if no invitation exists with the requested ID.
func (r *InvitationsService) Delete(id string) *InvitationsDeleteCall {
	c := &InvitationsDeleteCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *InvitationsDeleteCall) QuotaUser(quotaUser string) *InvitationsDeleteCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InvitationsDeleteCall) Fields(s ...googleapi.Field) *InvitationsDeleteCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InvitationsDeleteCall) Context(ctx context.Context) *InvitationsDeleteCall {
	c.ctx_ = ctx
	return c
}

func (c *InvitationsDeleteCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/invitations/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("DELETE", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"id": c.id,
	})
	req.Header.Set("User-Agent", c.s.userAgent())
	if c.ctx_ != nil {
		return ctxhttp.Do(c.ctx_, c.s.client, req)
	}
	return c.s.client.Do(req)
}

// Do executes the "classroom.invitations.delete" call.
// Exactly one of *Empty or error will be non-nil. Any non-2xx status
// code is an error. Response headers are in either
// *Empty.ServerResponse.Header or (if a response was returned at all)
// in error.(*googleapi.Error).Header. Use googleapi.IsNotModified to
// check whether the returned error was because http.StatusNotModified
// was returned.
func (c *InvitationsDeleteCall) Do() (*Empty, error) {
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
	//   "description": "Deletes an invitation. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to delete the requested invitation or for access errors. * `NOT_FOUND` if no invitation exists with the requested ID.",
	//   "httpMethod": "DELETE",
	//   "id": "classroom.invitations.delete",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "Identifier of the invitation to delete.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/invitations/{id}",
	//   "response": {
	//     "$ref": "Empty"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.rosters"
	//   ]
	// }

}

// method id "classroom.invitations.get":

type InvitationsGetCall struct {
	s            *Service
	id           string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Returns an invitation. This method returns the following error
// codes: * `PERMISSION_DENIED` if the requesting user is not permitted
// to view the requested invitation or for access errors. * `NOT_FOUND`
// if no invitation exists with the requested ID.
func (r *InvitationsService) Get(id string) *InvitationsGetCall {
	c := &InvitationsGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.id = id
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *InvitationsGetCall) QuotaUser(quotaUser string) *InvitationsGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InvitationsGetCall) Fields(s ...googleapi.Field) *InvitationsGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InvitationsGetCall) IfNoneMatch(entityTag string) *InvitationsGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InvitationsGetCall) Context(ctx context.Context) *InvitationsGetCall {
	c.ctx_ = ctx
	return c
}

func (c *InvitationsGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/invitations/{id}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"id": c.id,
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

// Do executes the "classroom.invitations.get" call.
// Exactly one of *Invitation or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *Invitation.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *InvitationsGetCall) Do() (*Invitation, error) {
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
	ret := &Invitation{
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
	//   "description": "Returns an invitation. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to view the requested invitation or for access errors. * `NOT_FOUND` if no invitation exists with the requested ID.",
	//   "httpMethod": "GET",
	//   "id": "classroom.invitations.get",
	//   "parameterOrder": [
	//     "id"
	//   ],
	//   "parameters": {
	//     "id": {
	//       "description": "Identifier of the invitation to return.",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/invitations/{id}",
	//   "response": {
	//     "$ref": "Invitation"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.rosters",
	//     "https://www.googleapis.com/auth/classroom.rosters.readonly"
	//   ]
	// }

}

// method id "classroom.invitations.list":

type InvitationsListCall struct {
	s            *Service
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// List: Returns a list of invitations that the requesting user is
// permitted to view, restricted to those that match the list request.
// *Note:* At least one of `user_id` or `course_id` must be supplied.
// Both fields can be supplied. This method returns the following error
// codes: * `PERMISSION_DENIED` for access errors.
func (r *InvitationsService) List() *InvitationsListCall {
	c := &InvitationsListCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	return c
}

// CourseId sets the optional parameter "courseId": Restricts returned
// invitations to those for a course with the specified identifier.
func (c *InvitationsListCall) CourseId(courseId string) *InvitationsListCall {
	c.urlParams_.Set("courseId", courseId)
	return c
}

// PageSize sets the optional parameter "pageSize": Maximum number of
// items to return. Zero means no maximum. The server may return fewer
// than the specified number of results.
func (c *InvitationsListCall) PageSize(pageSize int64) *InvitationsListCall {
	c.urlParams_.Set("pageSize", fmt.Sprint(pageSize))
	return c
}

// PageToken sets the optional parameter "pageToken": nextPageToken
// value returned from a previous list call, indicating that the
// subsequent page of results should be returned. The list request must
// be otherwise identical to the one that resulted in this token.
func (c *InvitationsListCall) PageToken(pageToken string) *InvitationsListCall {
	c.urlParams_.Set("pageToken", pageToken)
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *InvitationsListCall) QuotaUser(quotaUser string) *InvitationsListCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// UserId sets the optional parameter "userId": Restricts returned
// invitations to those for a specific user. The identifier can be one
// of the following: * the numeric identifier for the user * the email
// address of the user * the string literal "me", indicating the
// requesting user
func (c *InvitationsListCall) UserId(userId string) *InvitationsListCall {
	c.urlParams_.Set("userId", userId)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *InvitationsListCall) Fields(s ...googleapi.Field) *InvitationsListCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *InvitationsListCall) IfNoneMatch(entityTag string) *InvitationsListCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *InvitationsListCall) Context(ctx context.Context) *InvitationsListCall {
	c.ctx_ = ctx
	return c
}

func (c *InvitationsListCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/invitations")
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

// Do executes the "classroom.invitations.list" call.
// Exactly one of *ListInvitationsResponse or error will be non-nil. Any
// non-2xx status code is an error. Response headers are in either
// *ListInvitationsResponse.ServerResponse.Header or (if a response was
// returned at all) in error.(*googleapi.Error).Header. Use
// googleapi.IsNotModified to check whether the returned error was
// because http.StatusNotModified was returned.
func (c *InvitationsListCall) Do() (*ListInvitationsResponse, error) {
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
	ret := &ListInvitationsResponse{
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
	//   "description": "Returns a list of invitations that the requesting user is permitted to view, restricted to those that match the list request. *Note:* At least one of `user_id` or `course_id` must be supplied. Both fields can be supplied. This method returns the following error codes: * `PERMISSION_DENIED` for access errors.",
	//   "httpMethod": "GET",
	//   "id": "classroom.invitations.list",
	//   "parameters": {
	//     "courseId": {
	//       "description": "Restricts returned invitations to those for a course with the specified identifier.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "pageSize": {
	//       "description": "Maximum number of items to return. Zero means no maximum. The server may return fewer than the specified number of results.",
	//       "format": "int32",
	//       "location": "query",
	//       "type": "integer"
	//     },
	//     "pageToken": {
	//       "description": "nextPageToken value returned from a previous list call, indicating that the subsequent page of results should be returned. The list request must be otherwise identical to the one that resulted in this token.",
	//       "location": "query",
	//       "type": "string"
	//     },
	//     "userId": {
	//       "description": "Restricts returned invitations to those for a specific user. The identifier can be one of the following: * the numeric identifier for the user * the email address of the user * the string literal `\"me\"`, indicating the requesting user",
	//       "location": "query",
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/invitations",
	//   "response": {
	//     "$ref": "ListInvitationsResponse"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.rosters",
	//     "https://www.googleapis.com/auth/classroom.rosters.readonly"
	//   ]
	// }

}

// method id "classroom.userProfiles.get":

type UserProfilesGetCall struct {
	s            *Service
	userId       string
	urlParams_   gensupport.URLParams
	ifNoneMatch_ string
	ctx_         context.Context
}

// Get: Returns a user profile. This method returns the following error
// codes: * `PERMISSION_DENIED` if the requesting user is not permitted
// to access this user profile or if no profile exists with the
// requested ID or for access errors.
func (r *UserProfilesService) Get(userId string) *UserProfilesGetCall {
	c := &UserProfilesGetCall{s: r.s, urlParams_: make(gensupport.URLParams)}
	c.userId = userId
	return c
}

// QuotaUser sets the optional parameter "quotaUser": Available to use
// for quota purposes for server-side applications. Can be any arbitrary
// string assigned to a user, but should not exceed 40 characters.
func (c *UserProfilesGetCall) QuotaUser(quotaUser string) *UserProfilesGetCall {
	c.urlParams_.Set("quotaUser", quotaUser)
	return c
}

// Fields allows partial responses to be retrieved. See
// https://developers.google.com/gdata/docs/2.0/basics#PartialResponse
// for more information.
func (c *UserProfilesGetCall) Fields(s ...googleapi.Field) *UserProfilesGetCall {
	c.urlParams_.Set("fields", googleapi.CombineFields(s))
	return c
}

// IfNoneMatch sets the optional parameter which makes the operation
// fail if the object's ETag matches the given value. This is useful for
// getting updates only after the object has changed since the last
// request. Use googleapi.IsNotModified to check whether the response
// error from Do is the result of In-None-Match.
func (c *UserProfilesGetCall) IfNoneMatch(entityTag string) *UserProfilesGetCall {
	c.ifNoneMatch_ = entityTag
	return c
}

// Context sets the context to be used in this call's Do method. Any
// pending HTTP request will be aborted if the provided context is
// canceled.
func (c *UserProfilesGetCall) Context(ctx context.Context) *UserProfilesGetCall {
	c.ctx_ = ctx
	return c
}

func (c *UserProfilesGetCall) doRequest(alt string) (*http.Response, error) {
	var body io.Reader = nil
	c.urlParams_.Set("alt", alt)
	urls := googleapi.ResolveRelative(c.s.BasePath, "v1/userProfiles/{userId}")
	urls += "?" + c.urlParams_.Encode()
	req, _ := http.NewRequest("GET", urls, body)
	googleapi.Expand(req.URL, map[string]string{
		"userId": c.userId,
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

// Do executes the "classroom.userProfiles.get" call.
// Exactly one of *UserProfile or error will be non-nil. Any non-2xx
// status code is an error. Response headers are in either
// *UserProfile.ServerResponse.Header or (if a response was returned at
// all) in error.(*googleapi.Error).Header. Use googleapi.IsNotModified
// to check whether the returned error was because
// http.StatusNotModified was returned.
func (c *UserProfilesGetCall) Do() (*UserProfile, error) {
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
	ret := &UserProfile{
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
	//   "description": "Returns a user profile. This method returns the following error codes: * `PERMISSION_DENIED` if the requesting user is not permitted to access this user profile or if no profile exists with the requested ID or for access errors.",
	//   "httpMethod": "GET",
	//   "id": "classroom.userProfiles.get",
	//   "parameterOrder": [
	//     "userId"
	//   ],
	//   "parameters": {
	//     "userId": {
	//       "description": "Identifier of the profile to return. The identifier can be one of the following: * the numeric identifier for the user * the email address of the user * the string literal `\"me\"`, indicating the requesting user",
	//       "location": "path",
	//       "required": true,
	//       "type": "string"
	//     }
	//   },
	//   "path": "v1/userProfiles/{userId}",
	//   "response": {
	//     "$ref": "UserProfile"
	//   },
	//   "scopes": [
	//     "https://www.googleapis.com/auth/classroom.profile.emails",
	//     "https://www.googleapis.com/auth/classroom.profile.photos",
	//     "https://www.googleapis.com/auth/classroom.rosters",
	//     "https://www.googleapis.com/auth/classroom.rosters.readonly"
	//   ]
	// }

}
