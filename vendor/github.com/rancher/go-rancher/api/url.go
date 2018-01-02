package api

import (
	"bytes"
	"fmt"
	"net/http"
	"net/url"
	"strings"

	"github.com/rancher/go-rancher/client"
)

const (
	DEFAULT_OVERRIDE_URL_HEADER       = "X-API-request-url"
	DEFAULT_OVERRIDE_CLIENT_IP_HEADER = "X-API-client-ip"
	FORWARDED_FOR_HEADER              = "X-Forwarded-For"
	FORWARDED_HOST_HEADER             = "X-Forwarded-Host"
	FORWARDED_PROTO_HEADER            = "X-Forwarded-Proto"
	FORWARDED_PORT_HEADER             = "X-Forwarded-Port"
	SELF                              = "self"
	COLLECTION                        = "collection"
	LATEST                            = "latest"
	HTML                              = "html"
	JSON                              = "json"
)

var (
	allowedFormats = map[string]bool{
		HTML: true,
		JSON: true,
	}
)

type UrlBuilder interface {
	//ActionLink(resource client.Resource, name string) string
	Current() string
	Collection(resourceType string) string
	Link(resource client.Resource, name string) string
	ReferenceLink(resource client.Resource) string
	ReferenceByIdLink(resourceType string, id string) string
	Version(version string) string
}

func NewUrlBuilder(r *http.Request, schemas *client.Schemas) (UrlBuilder, error) {
	requestUrl := parseRequestUrl(r)
	responseUrlBase, err := parseResponseUrlBase(requestUrl, r)
	if err != nil {
		return nil, err
	}

	builder := &urlBuilder{
		schemas:         schemas,
		requestUrl:      requestUrl,
		responseUrlBase: responseUrlBase,
		apiVersion:      parseRequestVersion(r),
	}

	return builder, err
}

type urlBuilder struct {
	schemas *client.Schemas
	//r               *http.Request
	requestUrl      string
	responseUrlBase string
	//query           string
	apiVersion string
	subContext string
}

func (u *urlBuilder) Current() string {
	return u.requestUrl
}

func (u *urlBuilder) Collection(resourceType string) string {
	plural := u.getPluralName(resourceType)
	return u.constructBasicUrl(plural)
}

func (u *urlBuilder) Link(resource client.Resource, name string) string {
	if name == "" {
		return ""
	}

	return u.constructBasicUrl(u.getPluralName(resource.Type), resource.Id, strings.ToLower(name))
}

func (u *urlBuilder) ReferenceLink(resource client.Resource) string {
	return u.ReferenceByIdLink(resource.Type, resource.Id)
}

func (u *urlBuilder) ReferenceByIdLink(resourceType string, id string) string {
	return u.constructBasicUrl(u.getPluralName(resourceType), id)
}

func (u *urlBuilder) Version(version string) string {
	return fmt.Sprintf("%s/%s", u.responseUrlBase, version)
}

func (u *urlBuilder) constructBasicUrl(parts ...string) string {
	buffer := bytes.Buffer{}

	buffer.WriteString(u.responseUrlBase)
	buffer.WriteString("/")
	buffer.WriteString(u.apiVersion)
	buffer.WriteString(u.subContext)

	for _, part := range parts {
		if part == "" {
			return ""
		}
		buffer.WriteString("/")
		buffer.WriteString(part)
	}

	return buffer.String()
}

func (u *urlBuilder) getPluralName(resourceType string) string {
	schema := u.schemas.Schema(resourceType)
	if schema.PluralName == "" {
		return strings.ToLower(resourceType)
	}
	return strings.ToLower(schema.PluralName)
}

// Constructs the request URL based off of standard headers in the request, falling back to the HttpServletRequest.getRequestURL()
// if the headers aren't available. Here is the ordered list of how we'll attempt to construct the URL:
//  - x-api-request-url
//  - x-forwarded-proto://x-forwarded-host:x-forwarded-port/HttpServletRequest.getRequestURI()
//  - x-forwarded-proto://x-forwarded-host/HttpServletRequest.getRequestURI()
//  - x-forwarded-proto://host:x-forwarded-port/HttpServletRequest.getRequestURI()
//  - x-forwarded-proto://host/HttpServletRequest.getRequestURI() request.getRequestURL()
//
// Additional notes:
//  - With x-api-request-url, the query string is passed, it will be dropped to match the other formats.
//  - If the x-forwarded-host/host header has a port and x-forwarded-port has been passed, x-forwarded-port will be used.
func parseRequestUrl(r *http.Request) string {
	// Get url from custom x-api-request-url header
	requestUrl := getOverrideHeader(r, DEFAULT_OVERRIDE_URL_HEADER, "")
	if requestUrl != "" {
		return strings.SplitN(requestUrl, "?", 2)[0]
	}

	// Get url from standard headers
	requestUrl = getUrlFromStandardHeaders(r)
	if requestUrl != "" {
		return requestUrl
	}

	// Use incoming url
	return fmt.Sprintf("http://%s%s", r.Host, r.URL.Path)
}

func getUrlFromStandardHeaders(r *http.Request) string {
	xForwardedProto := getOverrideHeader(r, FORWARDED_PROTO_HEADER, "")
	if xForwardedProto == "" {
		return ""
	}

	host := getOverrideHeader(r, FORWARDED_HOST_HEADER, "")
	if host == "" {
		host = r.Host
	}

	if host == "" {
		return ""
	}

	port := getOverrideHeader(r, FORWARDED_PORT_HEADER, "")
	if port == "443" || port == "80" {
		port = "" // Don't include default ports in url
	}

	if port != "" && strings.Contains(host, ":") {
		// Have to strip the port that is in the host. Handle IPv6, which has this format: [::1]:8080
		if (strings.HasPrefix(host, "[") && strings.Contains(host, "]:")) || !strings.HasPrefix(host, "[") {
			host = host[0:strings.LastIndex(host, ":")]
		}
	}

	if port != "" {
		port = ":" + port
	}

	return fmt.Sprintf("%s://%s%s%s", xForwardedProto, host, port, r.URL.Path)
}

func getOverrideHeader(r *http.Request, header string, defaultValue string) string {
	// Need to handle comma separated hosts in X-Forwarded-For
	value := r.Header.Get(header)
	if value != "" {
		return strings.TrimSpace(strings.Split(value, ",")[0])
	}
	return defaultValue
}

func parseResponseUrlBase(requestUrl string, r *http.Request) (string, error) {
	path := r.URL.Path

	index := strings.LastIndex(requestUrl, path)
	if index == -1 {
		// Fallback, if we can't find path in requestUrl, then we just assume the base is the root of the web request
		u, err := url.Parse(requestUrl)
		if err != nil {
			return "", err
		}

		buffer := bytes.Buffer{}
		buffer.WriteString(u.Scheme)
		buffer.WriteString("://")
		buffer.WriteString(u.Host)
		return buffer.String(), nil
	} else {
		return requestUrl[0:index], nil
	}
}

func parseRequestVersion(r *http.Request) string {
	path := r.URL.Path

	if !strings.HasPrefix(path, "/") || len(path) < 2 {
		return ""
	}

	return strings.Split(path, "/")[1]
}

func parseResponseType(r *http.Request) ApiResponseWriter {
	format := r.URL.Query().Get("_format")

	if format != "" {
		format = strings.ToLower(strings.TrimSpace(format))
	}

	/* Format specified */
	if format != "" && allowedFormats[format] {
		switch {
		case format == HTML:
			return &HtmlWriter{r: r}
		case format == JSON:
			return &JsonWriter{r: r}
		}
	}

	// User agent has Mozilla and browser accepts */*
	if IsBrowser(r, true) {
		return &HtmlWriter{r: r}
	} else {
		return &JsonWriter{r: r}
	}
}

func IsBrowser(r *http.Request, checkAccepts bool) bool {
	accepts := r.Header.Get("Accept")
	userAgent := r.Header.Get("User-Agent")

	if accepts == "" || !checkAccepts {
		accepts = "*/*"
	}

	accepts = strings.ToLower(accepts)

	// User agent has Mozilla and browser accepts */*
	return strings.Contains(strings.ToLower(userAgent), "mozilla") && strings.Contains(accepts, "*/*")
}
