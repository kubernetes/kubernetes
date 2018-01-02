package v2

import (
	"net/http"
	"regexp"

	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/api/errcode"
	"github.com/opencontainers/go-digest"
)

var (
	nameParameterDescriptor = ParameterDescriptor{
		Name:        "name",
		Type:        "string",
		Format:      reference.NameRegexp.String(),
		Required:    true,
		Description: `Name of the target repository.`,
	}

	referenceParameterDescriptor = ParameterDescriptor{
		Name:        "reference",
		Type:        "string",
		Format:      reference.TagRegexp.String(),
		Required:    true,
		Description: `Tag or digest of the target manifest.`,
	}

	uuidParameterDescriptor = ParameterDescriptor{
		Name:        "uuid",
		Type:        "opaque",
		Required:    true,
		Description: "A uuid identifying the upload. This field can accept characters that match `[a-zA-Z0-9-_.=]+`.",
	}

	digestPathParameter = ParameterDescriptor{
		Name:        "digest",
		Type:        "path",
		Required:    true,
		Format:      digest.DigestRegexp.String(),
		Description: `Digest of desired blob.`,
	}

	hostHeader = ParameterDescriptor{
		Name:        "Host",
		Type:        "string",
		Description: "Standard HTTP Host Header. Should be set to the registry host.",
		Format:      "<registry host>",
		Examples:    []string{"registry-1.docker.io"},
	}

	authHeader = ParameterDescriptor{
		Name:        "Authorization",
		Type:        "string",
		Description: "An RFC7235 compliant authorization header.",
		Format:      "<scheme> <token>",
		Examples:    []string{"Bearer dGhpcyBpcyBhIGZha2UgYmVhcmVyIHRva2VuIQ=="},
	}

	authChallengeHeader = ParameterDescriptor{
		Name:        "WWW-Authenticate",
		Type:        "string",
		Description: "An RFC7235 compliant authentication challenge header.",
		Format:      `<scheme> realm="<realm>", ..."`,
		Examples: []string{
			`Bearer realm="https://auth.docker.com/", service="registry.docker.com", scopes="repository:library/ubuntu:pull"`,
		},
	}

	contentLengthZeroHeader = ParameterDescriptor{
		Name:        "Content-Length",
		Description: "The `Content-Length` header must be zero and the body must be empty.",
		Type:        "integer",
		Format:      "0",
	}

	dockerUploadUUIDHeader = ParameterDescriptor{
		Name:        "Docker-Upload-UUID",
		Description: "Identifies the docker upload uuid for the current request.",
		Type:        "uuid",
		Format:      "<uuid>",
	}

	digestHeader = ParameterDescriptor{
		Name:        "Docker-Content-Digest",
		Description: "Digest of the targeted content for the request.",
		Type:        "digest",
		Format:      "<digest>",
	}

	linkHeader = ParameterDescriptor{
		Name:        "Link",
		Type:        "link",
		Description: "RFC5988 compliant rel='next' with URL to next result set, if available",
		Format:      `<<url>?n=<last n value>&last=<last entry from response>>; rel="next"`,
	}

	paginationParameters = []ParameterDescriptor{
		{
			Name:        "n",
			Type:        "integer",
			Description: "Limit the number of entries in each response. It not present, all entries will be returned.",
			Format:      "<integer>",
			Required:    false,
		},
		{
			Name:        "last",
			Type:        "string",
			Description: "Result set will include values lexically after last.",
			Format:      "<integer>",
			Required:    false,
		},
	}

	unauthorizedResponseDescriptor = ResponseDescriptor{
		Name:        "Authentication Required",
		StatusCode:  http.StatusUnauthorized,
		Description: "The client is not authenticated.",
		Headers: []ParameterDescriptor{
			authChallengeHeader,
			{
				Name:        "Content-Length",
				Type:        "integer",
				Description: "Length of the JSON response body.",
				Format:      "<length>",
			},
		},
		Body: BodyDescriptor{
			ContentType: "application/json; charset=utf-8",
			Format:      errorsBody,
		},
		ErrorCodes: []errcode.ErrorCode{
			errcode.ErrorCodeUnauthorized,
		},
	}

	repositoryNotFoundResponseDescriptor = ResponseDescriptor{
		Name:        "No Such Repository Error",
		StatusCode:  http.StatusNotFound,
		Description: "The repository is not known to the registry.",
		Headers: []ParameterDescriptor{
			{
				Name:        "Content-Length",
				Type:        "integer",
				Description: "Length of the JSON response body.",
				Format:      "<length>",
			},
		},
		Body: BodyDescriptor{
			ContentType: "application/json; charset=utf-8",
			Format:      errorsBody,
		},
		ErrorCodes: []errcode.ErrorCode{
			ErrorCodeNameUnknown,
		},
	}

	deniedResponseDescriptor = ResponseDescriptor{
		Name:        "Access Denied",
		StatusCode:  http.StatusForbidden,
		Description: "The client does not have required access to the repository.",
		Headers: []ParameterDescriptor{
			{
				Name:        "Content-Length",
				Type:        "integer",
				Description: "Length of the JSON response body.",
				Format:      "<length>",
			},
		},
		Body: BodyDescriptor{
			ContentType: "application/json; charset=utf-8",
			Format:      errorsBody,
		},
		ErrorCodes: []errcode.ErrorCode{
			errcode.ErrorCodeDenied,
		},
	}

	tooManyRequestsDescriptor = ResponseDescriptor{
		Name:        "Too Many Requests",
		StatusCode:  http.StatusTooManyRequests,
		Description: "The client made too many requests within a time interval.",
		Headers: []ParameterDescriptor{
			{
				Name:        "Content-Length",
				Type:        "integer",
				Description: "Length of the JSON response body.",
				Format:      "<length>",
			},
		},
		Body: BodyDescriptor{
			ContentType: "application/json; charset=utf-8",
			Format:      errorsBody,
		},
		ErrorCodes: []errcode.ErrorCode{
			errcode.ErrorCodeTooManyRequests,
		},
	}
)

const (
	manifestBody = `{
   "name": <name>,
   "tag": <tag>,
   "fsLayers": [
      {
         "blobSum": "<digest>"
      },
      ...
    ]
   ],
   "history": <v1 images>,
   "signature": <JWS>
}`

	errorsBody = `{
	"errors:" [
	    {
            "code": <error code>,
            "message": "<error message>",
            "detail": ...
        },
        ...
    ]
}`
)

// APIDescriptor exports descriptions of the layout of the v2 registry API.
var APIDescriptor = struct {
	// RouteDescriptors provides a list of the routes available in the API.
	RouteDescriptors []RouteDescriptor
}{
	RouteDescriptors: routeDescriptors,
}

// RouteDescriptor describes a route specified by name.
type RouteDescriptor struct {
	// Name is the name of the route, as specified in RouteNameXXX exports.
	// These names a should be considered a unique reference for a route. If
	// the route is registered with gorilla, this is the name that will be
	// used.
	Name string

	// Path is a gorilla/mux-compatible regexp that can be used to match the
	// route. For any incoming method and path, only one route descriptor
	// should match.
	Path string

	// Entity should be a short, human-readalbe description of the object
	// targeted by the endpoint.
	Entity string

	// Description should provide an accurate overview of the functionality
	// provided by the route.
	Description string

	// Methods should describe the various HTTP methods that may be used on
	// this route, including request and response formats.
	Methods []MethodDescriptor
}

// MethodDescriptor provides a description of the requests that may be
// conducted with the target method.
type MethodDescriptor struct {

	// Method is an HTTP method, such as GET, PUT or POST.
	Method string

	// Description should provide an overview of the functionality provided by
	// the covered method, suitable for use in documentation. Use of markdown
	// here is encouraged.
	Description string

	// Requests is a slice of request descriptors enumerating how this
	// endpoint may be used.
	Requests []RequestDescriptor
}

// RequestDescriptor covers a particular set of headers and parameters that
// can be carried out with the parent method. Its most helpful to have one
// RequestDescriptor per API use case.
type RequestDescriptor struct {
	// Name provides a short identifier for the request, usable as a title or
	// to provide quick context for the particular request.
	Name string

	// Description should cover the requests purpose, covering any details for
	// this particular use case.
	Description string

	// Headers describes headers that must be used with the HTTP request.
	Headers []ParameterDescriptor

	// PathParameters enumerate the parameterized path components for the
	// given request, as defined in the route's regular expression.
	PathParameters []ParameterDescriptor

	// QueryParameters provides a list of query parameters for the given
	// request.
	QueryParameters []ParameterDescriptor

	// Body describes the format of the request body.
	Body BodyDescriptor

	// Successes enumerates the possible responses that are considered to be
	// the result of a successful request.
	Successes []ResponseDescriptor

	// Failures covers the possible failures from this particular request.
	Failures []ResponseDescriptor
}

// ResponseDescriptor describes the components of an API response.
type ResponseDescriptor struct {
	// Name provides a short identifier for the response, usable as a title or
	// to provide quick context for the particular response.
	Name string

	// Description should provide a brief overview of the role of the
	// response.
	Description string

	// StatusCode specifies the status received by this particular response.
	StatusCode int

	// Headers covers any headers that may be returned from the response.
	Headers []ParameterDescriptor

	// Fields describes any fields that may be present in the response.
	Fields []ParameterDescriptor

	// ErrorCodes enumerates the error codes that may be returned along with
	// the response.
	ErrorCodes []errcode.ErrorCode

	// Body describes the body of the response, if any.
	Body BodyDescriptor
}

// BodyDescriptor describes a request body and its expected content type. For
// the most  part, it should be example json or some placeholder for body
// data in documentation.
type BodyDescriptor struct {
	ContentType string
	Format      string
}

// ParameterDescriptor describes the format of a request parameter, which may
// be a header, path parameter or query parameter.
type ParameterDescriptor struct {
	// Name is the name of the parameter, either of the path component or
	// query parameter.
	Name string

	// Type specifies the type of the parameter, such as string, integer, etc.
	Type string

	// Description provides a human-readable description of the parameter.
	Description string

	// Required means the field is required when set.
	Required bool

	// Format is a specifying the string format accepted by this parameter.
	Format string

	// Regexp is a compiled regular expression that can be used to validate
	// the contents of the parameter.
	Regexp *regexp.Regexp

	// Examples provides multiple examples for the values that might be valid
	// for this parameter.
	Examples []string
}

var routeDescriptors = []RouteDescriptor{
	{
		Name:        RouteNameBase,
		Path:        "/v2/",
		Entity:      "Base",
		Description: `Base V2 API route. Typically, this can be used for lightweight version checks and to validate registry authentication.`,
		Methods: []MethodDescriptor{
			{
				Method:      "GET",
				Description: "Check that the endpoint implements Docker Registry API V2.",
				Requests: []RequestDescriptor{
					{
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
						},
						Successes: []ResponseDescriptor{
							{
								Description: "The API implements V2 protocol and is accessible.",
								StatusCode:  http.StatusOK,
							},
						},
						Failures: []ResponseDescriptor{
							{
								Description: "The registry does not implement the V2 API.",
								StatusCode:  http.StatusNotFound,
							},
							unauthorizedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
				},
			},
		},
	},
	{
		Name:        RouteNameTags,
		Path:        "/v2/{name:" + reference.NameRegexp.String() + "}/tags/list",
		Entity:      "Tags",
		Description: "Retrieve information about tags.",
		Methods: []MethodDescriptor{
			{
				Method:      "GET",
				Description: "Fetch the tags under the repository identified by `name`.",
				Requests: []RequestDescriptor{
					{
						Name:        "Tags",
						Description: "Return all tags for the repository",
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
						},
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
						},
						Successes: []ResponseDescriptor{
							{
								StatusCode:  http.StatusOK,
								Description: "A list of tags for the named repository.",
								Headers: []ParameterDescriptor{
									{
										Name:        "Content-Length",
										Type:        "integer",
										Description: "Length of the JSON response body.",
										Format:      "<length>",
									},
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format: `{
    "name": <name>,
    "tags": [
        <tag>,
        ...
    ]
}`,
								},
							},
						},
						Failures: []ResponseDescriptor{
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
					{
						Name:            "Tags Paginated",
						Description:     "Return a portion of the tags for the specified repository.",
						PathParameters:  []ParameterDescriptor{nameParameterDescriptor},
						QueryParameters: paginationParameters,
						Successes: []ResponseDescriptor{
							{
								StatusCode:  http.StatusOK,
								Description: "A list of tags for the named repository.",
								Headers: []ParameterDescriptor{
									{
										Name:        "Content-Length",
										Type:        "integer",
										Description: "Length of the JSON response body.",
										Format:      "<length>",
									},
									linkHeader,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format: `{
    "name": <name>,
    "tags": [
        <tag>,
        ...
    ],
}`,
								},
							},
						},
						Failures: []ResponseDescriptor{
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
				},
			},
		},
	},
	{
		Name:        RouteNameManifest,
		Path:        "/v2/{name:" + reference.NameRegexp.String() + "}/manifests/{reference:" + reference.TagRegexp.String() + "|" + digest.DigestRegexp.String() + "}",
		Entity:      "Manifest",
		Description: "Create, update, delete and retrieve manifests.",
		Methods: []MethodDescriptor{
			{
				Method:      "GET",
				Description: "Fetch the manifest identified by `name` and `reference` where `reference` can be a tag or digest. A `HEAD` request can also be issued to this endpoint to obtain resource information without receiving all data.",
				Requests: []RequestDescriptor{
					{
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
						},
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
							referenceParameterDescriptor,
						},
						Successes: []ResponseDescriptor{
							{
								Description: "The manifest identified by `name` and `reference`. The contents can be used to identify and resolve resources required to run the specified image.",
								StatusCode:  http.StatusOK,
								Headers: []ParameterDescriptor{
									digestHeader,
								},
								Body: BodyDescriptor{
									ContentType: "<media type of manifest>",
									Format:      manifestBody,
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Description: "The name or reference was invalid.",
								StatusCode:  http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeNameInvalid,
									ErrorCodeTagInvalid,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
				},
			},
			{
				Method:      "PUT",
				Description: "Put the manifest identified by `name` and `reference` where `reference` can be a tag or digest.",
				Requests: []RequestDescriptor{
					{
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
						},
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
							referenceParameterDescriptor,
						},
						Body: BodyDescriptor{
							ContentType: "<media type of manifest>",
							Format:      manifestBody,
						},
						Successes: []ResponseDescriptor{
							{
								Description: "The manifest has been accepted by the registry and is stored under the specified `name` and `tag`.",
								StatusCode:  http.StatusCreated,
								Headers: []ParameterDescriptor{
									{
										Name:        "Location",
										Type:        "url",
										Description: "The canonical location url of the uploaded manifest.",
										Format:      "<url>",
									},
									contentLengthZeroHeader,
									digestHeader,
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Name:        "Invalid Manifest",
								Description: "The received manifest was invalid in some way, as described by the error codes. The client should resolve the issue and retry the request.",
								StatusCode:  http.StatusBadRequest,
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeNameInvalid,
									ErrorCodeTagInvalid,
									ErrorCodeManifestInvalid,
									ErrorCodeManifestUnverified,
									ErrorCodeBlobUnknown,
								},
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
							{
								Name:        "Missing Layer(s)",
								Description: "One or more layers may be missing during a manifest upload. If so, the missing layers will be enumerated in the error response.",
								StatusCode:  http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeBlobUnknown,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format: `{
    "errors:" [{
            "code": "BLOB_UNKNOWN",
            "message": "blob unknown to registry",
            "detail": {
                "digest": "<digest>"
            }
        },
        ...
    ]
}`,
								},
							},
							{
								Name:        "Not allowed",
								Description: "Manifest put is not allowed because the registry is configured as a pull-through cache or for some other reason",
								StatusCode:  http.StatusMethodNotAllowed,
								ErrorCodes: []errcode.ErrorCode{
									errcode.ErrorCodeUnsupported,
								},
							},
						},
					},
				},
			},
			{
				Method:      "DELETE",
				Description: "Delete the manifest identified by `name` and `reference`. Note that a manifest can _only_ be deleted by `digest`.",
				Requests: []RequestDescriptor{
					{
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
						},
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
							referenceParameterDescriptor,
						},
						Successes: []ResponseDescriptor{
							{
								StatusCode: http.StatusAccepted,
							},
						},
						Failures: []ResponseDescriptor{
							{
								Name:        "Invalid Name or Reference",
								Description: "The specified `name` or `reference` were invalid and the delete was unable to proceed.",
								StatusCode:  http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeNameInvalid,
									ErrorCodeTagInvalid,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
							{
								Name:        "Unknown Manifest",
								Description: "The specified `name` or `reference` are unknown to the registry and the delete was unable to proceed. Clients can assume the manifest was already deleted if this response is returned.",
								StatusCode:  http.StatusNotFound,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeNameUnknown,
									ErrorCodeManifestUnknown,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							{
								Name:        "Not allowed",
								Description: "Manifest delete is not allowed because the registry is configured as a pull-through cache or `delete` has been disabled.",
								StatusCode:  http.StatusMethodNotAllowed,
								ErrorCodes: []errcode.ErrorCode{
									errcode.ErrorCodeUnsupported,
								},
							},
						},
					},
				},
			},
		},
	},

	{
		Name:        RouteNameBlob,
		Path:        "/v2/{name:" + reference.NameRegexp.String() + "}/blobs/{digest:" + digest.DigestRegexp.String() + "}",
		Entity:      "Blob",
		Description: "Operations on blobs identified by `name` and `digest`. Used to fetch or delete layers by digest.",
		Methods: []MethodDescriptor{
			{
				Method:      "GET",
				Description: "Retrieve the blob from the registry identified by `digest`. A `HEAD` request can also be issued to this endpoint to obtain resource information without receiving all data.",
				Requests: []RequestDescriptor{
					{
						Name: "Fetch Blob",
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
						},
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
							digestPathParameter,
						},
						Successes: []ResponseDescriptor{
							{
								Description: "The blob identified by `digest` is available. The blob content will be present in the body of the request.",
								StatusCode:  http.StatusOK,
								Headers: []ParameterDescriptor{
									{
										Name:        "Content-Length",
										Type:        "integer",
										Description: "The length of the requested blob content.",
										Format:      "<length>",
									},
									digestHeader,
								},
								Body: BodyDescriptor{
									ContentType: "application/octet-stream",
									Format:      "<blob binary data>",
								},
							},
							{
								Description: "The blob identified by `digest` is available at the provided location.",
								StatusCode:  http.StatusTemporaryRedirect,
								Headers: []ParameterDescriptor{
									{
										Name:        "Location",
										Type:        "url",
										Description: "The location where the layer should be accessible.",
										Format:      "<blob location>",
									},
									digestHeader,
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Description: "There was a problem with the request that needs to be addressed by the client, such as an invalid `name` or `tag`.",
								StatusCode:  http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeNameInvalid,
									ErrorCodeDigestInvalid,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							{
								Description: "The blob, identified by `name` and `digest`, is unknown to the registry.",
								StatusCode:  http.StatusNotFound,
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeNameUnknown,
									ErrorCodeBlobUnknown,
								},
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
					{
						Name:        "Fetch Blob Part",
						Description: "This endpoint may also support RFC7233 compliant range requests. Support can be detected by issuing a HEAD request. If the header `Accept-Range: bytes` is returned, range requests can be used to fetch partial content.",
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
							{
								Name:        "Range",
								Type:        "string",
								Description: "HTTP Range header specifying blob chunk.",
								Format:      "bytes=<start>-<end>",
							},
						},
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
							digestPathParameter,
						},
						Successes: []ResponseDescriptor{
							{
								Description: "The blob identified by `digest` is available. The specified chunk of blob content will be present in the body of the request.",
								StatusCode:  http.StatusPartialContent,
								Headers: []ParameterDescriptor{
									{
										Name:        "Content-Length",
										Type:        "integer",
										Description: "The length of the requested blob chunk.",
										Format:      "<length>",
									},
									{
										Name:        "Content-Range",
										Type:        "byte range",
										Description: "Content range of blob chunk.",
										Format:      "bytes <start>-<end>/<size>",
									},
								},
								Body: BodyDescriptor{
									ContentType: "application/octet-stream",
									Format:      "<blob binary data>",
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Description: "There was a problem with the request that needs to be addressed by the client, such as an invalid `name` or `tag`.",
								StatusCode:  http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeNameInvalid,
									ErrorCodeDigestInvalid,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							{
								StatusCode: http.StatusNotFound,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeNameUnknown,
									ErrorCodeBlobUnknown,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							{
								Description: "The range specification cannot be satisfied for the requested content. This can happen when the range is not formatted correctly or if the range is outside of the valid size of the content.",
								StatusCode:  http.StatusRequestedRangeNotSatisfiable,
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
				},
			},
			{
				Method:      "DELETE",
				Description: "Delete the blob identified by `name` and `digest`",
				Requests: []RequestDescriptor{
					{
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
						},
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
							digestPathParameter,
						},
						Successes: []ResponseDescriptor{
							{
								StatusCode: http.StatusAccepted,
								Headers: []ParameterDescriptor{
									{
										Name:        "Content-Length",
										Type:        "integer",
										Description: "0",
										Format:      "0",
									},
									digestHeader,
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Name:       "Invalid Name or Digest",
								StatusCode: http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeDigestInvalid,
									ErrorCodeNameInvalid,
								},
							},
							{
								Description: "The blob, identified by `name` and `digest`, is unknown to the registry.",
								StatusCode:  http.StatusNotFound,
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeNameUnknown,
									ErrorCodeBlobUnknown,
								},
							},
							{
								Description: "Blob delete is not allowed because the registry is configured as a pull-through cache or `delete` has been disabled",
								StatusCode:  http.StatusMethodNotAllowed,
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
								ErrorCodes: []errcode.ErrorCode{
									errcode.ErrorCodeUnsupported,
								},
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
				},
			},

			// TODO(stevvooe): We may want to add a PUT request here to
			// kickoff an upload of a blob, integrated with the blob upload
			// API.
		},
	},

	{
		Name:        RouteNameBlobUpload,
		Path:        "/v2/{name:" + reference.NameRegexp.String() + "}/blobs/uploads/",
		Entity:      "Initiate Blob Upload",
		Description: "Initiate a blob upload. This endpoint can be used to create resumable uploads or monolithic uploads.",
		Methods: []MethodDescriptor{
			{
				Method:      "POST",
				Description: "Initiate a resumable blob upload. If successful, an upload location will be provided to complete the upload. Optionally, if the `digest` parameter is present, the request body will be used to complete the upload in a single request.",
				Requests: []RequestDescriptor{
					{
						Name:        "Initiate Monolithic Blob Upload",
						Description: "Upload a blob identified by the `digest` parameter in single request. This upload will not be resumable unless a recoverable error is returned.",
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
							{
								Name:   "Content-Length",
								Type:   "integer",
								Format: "<length of blob>",
							},
						},
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
						},
						QueryParameters: []ParameterDescriptor{
							{
								Name:        "digest",
								Type:        "query",
								Format:      "<digest>",
								Regexp:      digest.DigestRegexp,
								Description: `Digest of uploaded blob. If present, the upload will be completed, in a single request, with contents of the request body as the resulting blob.`,
							},
						},
						Body: BodyDescriptor{
							ContentType: "application/octect-stream",
							Format:      "<binary data>",
						},
						Successes: []ResponseDescriptor{
							{
								Description: "The blob has been created in the registry and is available at the provided location.",
								StatusCode:  http.StatusCreated,
								Headers: []ParameterDescriptor{
									{
										Name:   "Location",
										Type:   "url",
										Format: "<blob location>",
									},
									contentLengthZeroHeader,
									dockerUploadUUIDHeader,
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Name:       "Invalid Name or Digest",
								StatusCode: http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeDigestInvalid,
									ErrorCodeNameInvalid,
								},
							},
							{
								Name:        "Not allowed",
								Description: "Blob upload is not allowed because the registry is configured as a pull-through cache or for some other reason",
								StatusCode:  http.StatusMethodNotAllowed,
								ErrorCodes: []errcode.ErrorCode{
									errcode.ErrorCodeUnsupported,
								},
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
					{
						Name:        "Initiate Resumable Blob Upload",
						Description: "Initiate a resumable blob upload with an empty request body.",
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
							contentLengthZeroHeader,
						},
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
						},
						Successes: []ResponseDescriptor{
							{
								Description: "The upload has been created. The `Location` header must be used to complete the upload. The response should be identical to a `GET` request on the contents of the returned `Location` header.",
								StatusCode:  http.StatusAccepted,
								Headers: []ParameterDescriptor{
									contentLengthZeroHeader,
									{
										Name:        "Location",
										Type:        "url",
										Format:      "/v2/<name>/blobs/uploads/<uuid>",
										Description: "The location of the created upload. Clients should use the contents verbatim to complete the upload, adding parameters where required.",
									},
									{
										Name:        "Range",
										Format:      "0-0",
										Description: "Range header indicating the progress of the upload. When starting an upload, it will return an empty range, since no content has been received.",
									},
									dockerUploadUUIDHeader,
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Name:       "Invalid Name or Digest",
								StatusCode: http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeDigestInvalid,
									ErrorCodeNameInvalid,
								},
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
					{
						Name:        "Mount Blob",
						Description: "Mount a blob identified by the `mount` parameter from another repository.",
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
							contentLengthZeroHeader,
						},
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
						},
						QueryParameters: []ParameterDescriptor{
							{
								Name:        "mount",
								Type:        "query",
								Format:      "<digest>",
								Regexp:      digest.DigestRegexp,
								Description: `Digest of blob to mount from the source repository.`,
							},
							{
								Name:        "from",
								Type:        "query",
								Format:      "<repository name>",
								Regexp:      reference.NameRegexp,
								Description: `Name of the source repository.`,
							},
						},
						Successes: []ResponseDescriptor{
							{
								Description: "The blob has been mounted in the repository and is available at the provided location.",
								StatusCode:  http.StatusCreated,
								Headers: []ParameterDescriptor{
									{
										Name:   "Location",
										Type:   "url",
										Format: "<blob location>",
									},
									contentLengthZeroHeader,
									dockerUploadUUIDHeader,
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Name:       "Invalid Name or Digest",
								StatusCode: http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeDigestInvalid,
									ErrorCodeNameInvalid,
								},
							},
							{
								Name:        "Not allowed",
								Description: "Blob mount is not allowed because the registry is configured as a pull-through cache or for some other reason",
								StatusCode:  http.StatusMethodNotAllowed,
								ErrorCodes: []errcode.ErrorCode{
									errcode.ErrorCodeUnsupported,
								},
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
				},
			},
		},
	},

	{
		Name:        RouteNameBlobUploadChunk,
		Path:        "/v2/{name:" + reference.NameRegexp.String() + "}/blobs/uploads/{uuid:[a-zA-Z0-9-_.=]+}",
		Entity:      "Blob Upload",
		Description: "Interact with blob uploads. Clients should never assemble URLs for this endpoint and should only take it through the `Location` header on related API requests. The `Location` header and its parameters should be preserved by clients, using the latest value returned via upload related API calls.",
		Methods: []MethodDescriptor{
			{
				Method:      "GET",
				Description: "Retrieve status of upload identified by `uuid`. The primary purpose of this endpoint is to resolve the current status of a resumable upload.",
				Requests: []RequestDescriptor{
					{
						Description: "Retrieve the progress of the current upload, as reported by the `Range` header.",
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
						},
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
							uuidParameterDescriptor,
						},
						Successes: []ResponseDescriptor{
							{
								Name:        "Upload Progress",
								Description: "The upload is known and in progress. The last received offset is available in the `Range` header.",
								StatusCode:  http.StatusNoContent,
								Headers: []ParameterDescriptor{
									{
										Name:        "Range",
										Type:        "header",
										Format:      "0-<offset>",
										Description: "Range indicating the current progress of the upload.",
									},
									contentLengthZeroHeader,
									dockerUploadUUIDHeader,
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Description: "There was an error processing the upload and it must be restarted.",
								StatusCode:  http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeDigestInvalid,
									ErrorCodeNameInvalid,
									ErrorCodeBlobUploadInvalid,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							{
								Description: "The upload is unknown to the registry. The upload must be restarted.",
								StatusCode:  http.StatusNotFound,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeBlobUploadUnknown,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
				},
			},
			{
				Method:      "PATCH",
				Description: "Upload a chunk of data for the specified upload.",
				Requests: []RequestDescriptor{
					{
						Name:        "Stream upload",
						Description: "Upload a stream of data to upload without completing the upload.",
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
							uuidParameterDescriptor,
						},
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
						},
						Body: BodyDescriptor{
							ContentType: "application/octet-stream",
							Format:      "<binary data>",
						},
						Successes: []ResponseDescriptor{
							{
								Name:        "Data Accepted",
								Description: "The stream of data has been accepted and the current progress is available in the range header. The updated upload location is available in the `Location` header.",
								StatusCode:  http.StatusNoContent,
								Headers: []ParameterDescriptor{
									{
										Name:        "Location",
										Type:        "url",
										Format:      "/v2/<name>/blobs/uploads/<uuid>",
										Description: "The location of the upload. Clients should assume this changes after each request. Clients should use the contents verbatim to complete the upload, adding parameters where required.",
									},
									{
										Name:        "Range",
										Type:        "header",
										Format:      "0-<offset>",
										Description: "Range indicating the current progress of the upload.",
									},
									contentLengthZeroHeader,
									dockerUploadUUIDHeader,
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Description: "There was an error processing the upload and it must be restarted.",
								StatusCode:  http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeDigestInvalid,
									ErrorCodeNameInvalid,
									ErrorCodeBlobUploadInvalid,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							{
								Description: "The upload is unknown to the registry. The upload must be restarted.",
								StatusCode:  http.StatusNotFound,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeBlobUploadUnknown,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
					{
						Name:        "Chunked upload",
						Description: "Upload a chunk of data to specified upload without completing the upload. The data will be uploaded to the specified Content Range.",
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
							uuidParameterDescriptor,
						},
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
							{
								Name:        "Content-Range",
								Type:        "header",
								Format:      "<start of range>-<end of range, inclusive>",
								Required:    true,
								Description: "Range of bytes identifying the desired block of content represented by the body. Start must the end offset retrieved via status check plus one. Note that this is a non-standard use of the `Content-Range` header.",
							},
							{
								Name:        "Content-Length",
								Type:        "integer",
								Format:      "<length of chunk>",
								Description: "Length of the chunk being uploaded, corresponding the length of the request body.",
							},
						},
						Body: BodyDescriptor{
							ContentType: "application/octet-stream",
							Format:      "<binary chunk>",
						},
						Successes: []ResponseDescriptor{
							{
								Name:        "Chunk Accepted",
								Description: "The chunk of data has been accepted and the current progress is available in the range header. The updated upload location is available in the `Location` header.",
								StatusCode:  http.StatusNoContent,
								Headers: []ParameterDescriptor{
									{
										Name:        "Location",
										Type:        "url",
										Format:      "/v2/<name>/blobs/uploads/<uuid>",
										Description: "The location of the upload. Clients should assume this changes after each request. Clients should use the contents verbatim to complete the upload, adding parameters where required.",
									},
									{
										Name:        "Range",
										Type:        "header",
										Format:      "0-<offset>",
										Description: "Range indicating the current progress of the upload.",
									},
									contentLengthZeroHeader,
									dockerUploadUUIDHeader,
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Description: "There was an error processing the upload and it must be restarted.",
								StatusCode:  http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeDigestInvalid,
									ErrorCodeNameInvalid,
									ErrorCodeBlobUploadInvalid,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							{
								Description: "The upload is unknown to the registry. The upload must be restarted.",
								StatusCode:  http.StatusNotFound,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeBlobUploadUnknown,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							{
								Description: "The `Content-Range` specification cannot be accepted, either because it does not overlap with the current progress or it is invalid.",
								StatusCode:  http.StatusRequestedRangeNotSatisfiable,
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
				},
			},
			{
				Method:      "PUT",
				Description: "Complete the upload specified by `uuid`, optionally appending the body as the final chunk.",
				Requests: []RequestDescriptor{
					{
						Description: "Complete the upload, providing all the data in the body, if necessary. A request without a body will just complete the upload with previously uploaded content.",
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
							{
								Name:        "Content-Length",
								Type:        "integer",
								Format:      "<length of data>",
								Description: "Length of the data being uploaded, corresponding to the length of the request body. May be zero if no data is provided.",
							},
						},
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
							uuidParameterDescriptor,
						},
						QueryParameters: []ParameterDescriptor{
							{
								Name:        "digest",
								Type:        "string",
								Format:      "<digest>",
								Regexp:      digest.DigestRegexp,
								Required:    true,
								Description: `Digest of uploaded blob.`,
							},
						},
						Body: BodyDescriptor{
							ContentType: "application/octet-stream",
							Format:      "<binary data>",
						},
						Successes: []ResponseDescriptor{
							{
								Name:        "Upload Complete",
								Description: "The upload has been completed and accepted by the registry. The canonical location will be available in the `Location` header.",
								StatusCode:  http.StatusNoContent,
								Headers: []ParameterDescriptor{
									{
										Name:        "Location",
										Type:        "url",
										Format:      "<blob location>",
										Description: "The canonical location of the blob for retrieval",
									},
									{
										Name:        "Content-Range",
										Type:        "header",
										Format:      "<start of range>-<end of range, inclusive>",
										Description: "Range of bytes identifying the desired block of content represented by the body. Start must match the end of offset retrieved via status check. Note that this is a non-standard use of the `Content-Range` header.",
									},
									contentLengthZeroHeader,
									digestHeader,
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Description: "There was an error processing the upload and it must be restarted.",
								StatusCode:  http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeDigestInvalid,
									ErrorCodeNameInvalid,
									ErrorCodeBlobUploadInvalid,
									errcode.ErrorCodeUnsupported,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							{
								Description: "The upload is unknown to the registry. The upload must be restarted.",
								StatusCode:  http.StatusNotFound,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeBlobUploadUnknown,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
				},
			},
			{
				Method:      "DELETE",
				Description: "Cancel outstanding upload processes, releasing associated resources. If this is not called, the unfinished uploads will eventually timeout.",
				Requests: []RequestDescriptor{
					{
						Description: "Cancel the upload specified by `uuid`.",
						PathParameters: []ParameterDescriptor{
							nameParameterDescriptor,
							uuidParameterDescriptor,
						},
						Headers: []ParameterDescriptor{
							hostHeader,
							authHeader,
							contentLengthZeroHeader,
						},
						Successes: []ResponseDescriptor{
							{
								Name:        "Upload Deleted",
								Description: "The upload has been successfully deleted.",
								StatusCode:  http.StatusNoContent,
								Headers: []ParameterDescriptor{
									contentLengthZeroHeader,
								},
							},
						},
						Failures: []ResponseDescriptor{
							{
								Description: "An error was encountered processing the delete. The client may ignore this error.",
								StatusCode:  http.StatusBadRequest,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeNameInvalid,
									ErrorCodeBlobUploadInvalid,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							{
								Description: "The upload is unknown to the registry. The client may ignore this error and assume the upload has been deleted.",
								StatusCode:  http.StatusNotFound,
								ErrorCodes: []errcode.ErrorCode{
									ErrorCodeBlobUploadUnknown,
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format:      errorsBody,
								},
							},
							unauthorizedResponseDescriptor,
							repositoryNotFoundResponseDescriptor,
							deniedResponseDescriptor,
							tooManyRequestsDescriptor,
						},
					},
				},
			},
		},
	},
	{
		Name:        RouteNameCatalog,
		Path:        "/v2/_catalog",
		Entity:      "Catalog",
		Description: "List a set of available repositories in the local registry cluster. Does not provide any indication of what may be available upstream. Applications can only determine if a repository is available but not if it is not available.",
		Methods: []MethodDescriptor{
			{
				Method:      "GET",
				Description: "Retrieve a sorted, json list of repositories available in the registry.",
				Requests: []RequestDescriptor{
					{
						Name:        "Catalog Fetch",
						Description: "Request an unabridged list of repositories available.  The implementation may impose a maximum limit and return a partial set with pagination links.",
						Successes: []ResponseDescriptor{
							{
								Description: "Returns the unabridged list of repositories as a json response.",
								StatusCode:  http.StatusOK,
								Headers: []ParameterDescriptor{
									{
										Name:        "Content-Length",
										Type:        "integer",
										Description: "Length of the JSON response body.",
										Format:      "<length>",
									},
								},
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format: `{
	"repositories": [
		<name>,
		...
	]
}`,
								},
							},
						},
					},
					{
						Name:            "Catalog Fetch Paginated",
						Description:     "Return the specified portion of repositories.",
						QueryParameters: paginationParameters,
						Successes: []ResponseDescriptor{
							{
								StatusCode: http.StatusOK,
								Body: BodyDescriptor{
									ContentType: "application/json; charset=utf-8",
									Format: `{
	"repositories": [
		<name>,
		...
	]
	"next": "<url>?last=<name>&n=<last value of n>"
}`,
								},
								Headers: []ParameterDescriptor{
									{
										Name:        "Content-Length",
										Type:        "integer",
										Description: "Length of the JSON response body.",
										Format:      "<length>",
									},
									linkHeader,
								},
							},
						},
					},
				},
			},
		},
	},
}

var routeDescriptorsMap map[string]RouteDescriptor

func init() {
	routeDescriptorsMap = make(map[string]RouteDescriptor, len(routeDescriptors))

	for _, descriptor := range routeDescriptors {
		routeDescriptorsMap[descriptor.Name] = descriptor
	}
}
