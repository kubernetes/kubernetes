Change history of go-restful
=
2016-11-26
- Default change! now use CurlyRouter (was RouterJSR311)
- Default change! no more caching of request content
- Default change! do not recover from panics

2016-09-22
- fix the DefaultRequestContentType feature

2016-02-14
- take the qualify factor of the Accept header mediatype into account when deciding the contentype of the response
- add constructors for custom entity accessors for xml and json 

2015-09-27
- rename new WriteStatusAnd... to WriteHeaderAnd... for consistency

2015-09-25
- fixed problem with changing Header after WriteHeader (issue 235)

2015-09-14
- changed behavior of WriteHeader (immediate write) and WriteEntity (no status write)
- added support for custom EntityReaderWriters.

2015-08-06
- add support for reading entities from compressed request content
- use sync.Pool for compressors of http response and request body
- add Description to Parameter for documentation in Swagger UI

2015-03-20
- add configurable logging

2015-03-18
- if not specified, the Operation is derived from the Route function

2015-03-17
- expose Parameter creation functions
- make trace logger an interface
- fix OPTIONSFilter
- customize rendering of ServiceError
- JSR311 router now handles wildcards
- add Notes to Route

2014-11-27
- (api add) PrettyPrint per response. (as proposed in #167)

2014-11-12
- (api add) ApiVersion(.) for documentation in Swagger UI

2014-11-10
- (api change) struct fields tagged with "description" show up in Swagger UI

2014-10-31
- (api change) ReturnsError -> Returns
- (api add)    RouteBuilder.Do(aBuilder) for DRY use of RouteBuilder
- fix swagger nested structs
- sort Swagger response messages by code

2014-10-23
- (api add) ReturnsError allows you to document Http codes in swagger
- fixed problem with greedy CurlyRouter
- (api add) Access-Control-Max-Age in CORS
- add tracing functionality (injectable) for debugging purposes
- support JSON parse 64bit int 
- fix empty parameters for swagger
- WebServicesUrl is now optional for swagger
- fixed duplicate AccessControlAllowOrigin in CORS
- (api change) expose ServeMux in container
- (api add) added AllowedDomains in CORS
- (api add) ParameterNamed for detailed documentation

2014-04-16
- (api add) expose constructor of Request for testing.

2014-06-27
- (api add) ParameterNamed gives access to a Parameter definition and its data (for further specification).
- (api add) SetCacheReadEntity allow scontrol over whether or not the request body is being cached (default true for compatibility reasons).

2014-07-03
- (api add) CORS can be configured with a list of allowed domains

2014-03-12
- (api add) Route path parameters can use wildcard or regular expressions. (requires CurlyRouter)

2014-02-26
- (api add) Request now provides information about the matched Route, see method SelectedRoutePath 

2014-02-17
- (api change) renamed parameter constants (go-lint checks)

2014-01-10
 - (api add) support for CloseNotify, see http://golang.org/pkg/net/http/#CloseNotifier

2014-01-07
 - (api change) Write* methods in Response now return the error or nil.
 - added example of serving HTML from a Go template.
 - fixed comparing Allowed headers in CORS (is now case-insensitive)

2013-11-13
 - (api add) Response knows how many bytes are written to the response body.

2013-10-29
 - (api add) RecoverHandler(handler RecoverHandleFunction) to change how panic recovery is handled. Default behavior is to log and return a stacktrace. This may be a security issue as it exposes sourcecode information.

2013-10-04
 - (api add) Response knows what HTTP status has been written
 - (api add) Request can have attributes (map of string->interface, also called request-scoped variables

2013-09-12
 - (api change) Router interface simplified
 - Implemented CurlyRouter, a Router that does not use|allow regular expressions in paths

2013-08-05
 - add OPTIONS support
 - add CORS support

2013-08-27
 - fixed some reported issues (see github)
 - (api change) deprecated use of WriteError; use WriteErrorString instead

2014-04-15
 - (fix) v1.0.1 tag: fix Issue 111: WriteErrorString

2013-08-08
 - (api add) Added implementation Container: a WebServices collection with its own http.ServeMux allowing multiple endpoints per program. Existing uses of go-restful will register their services to the DefaultContainer.
 - (api add) the swagger package has be extended to have a UI per container.
 - if panic is detected then a small stack trace is printed (thanks to runner-mei)
 - (api add) WriteErrorString to Response

Important API changes:

 - (api remove) package variable DoNotRecover no longer works ; use restful.DefaultContainer.DoNotRecover(true) instead.
 - (api remove) package variable EnableContentEncoding no longer works ; use restful.DefaultContainer.EnableContentEncoding(true) instead.
 
 
2013-07-06

 - (api add) Added support for response encoding (gzip and deflate(zlib)). This feature is disabled on default (for backwards compatibility). Use restful.EnableContentEncoding = true in your initialization to enable this feature.

2013-06-19

 - (improve) DoNotRecover option, moved request body closer, improved ReadEntity

2013-06-03

 - (api change) removed Dispatcher interface, hide PathExpression
 - changed receiver names of type functions to be more idiomatic Go

2013-06-02

 - (optimize) Cache the RegExp compilation of Paths.

2013-05-22
	
 - (api add) Added support for request/response filter functions

2013-05-18


 - (api add) Added feature to change the default Http Request Dispatch function (travis cline)
 - (api change) Moved Swagger Webservice to swagger package (see example restful-user)

[2012-11-14 .. 2013-05-18>
 
 - See https://github.com/emicklei/go-restful/commits

2012-11-14

 - Initial commit


