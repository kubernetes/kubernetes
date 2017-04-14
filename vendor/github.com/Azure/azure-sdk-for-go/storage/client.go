// Package storage provides clients for Microsoft Azure Storage Services.
package storage

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"runtime"
	"strconv"
	"strings"
)

const (
	// DefaultBaseURL is the domain name used for storage requests when a
	// default client is created.
	DefaultBaseURL = "core.windows.net"

	// DefaultAPIVersion is the  Azure Storage API version string used when a
	// basic client is created.
	DefaultAPIVersion = "2015-02-21"

	defaultUseHTTPS = true

	// StorageEmulatorAccountName is the fixed storage account used by Azure Storage Emulator
	StorageEmulatorAccountName = "devstoreaccount1"

	// StorageEmulatorAccountKey is the the fixed storage account used by Azure Storage Emulator
	StorageEmulatorAccountKey = "Eby8vdM02xNOcqFlqUwJPLlmEtlCDXJ1OUzFT50uSRZ6IFsuFq2UVErCz4I6tq/K1SZFPTOtr/KBHBeksoGMGw=="

	blobServiceName  = "blob"
	tableServiceName = "table"
	queueServiceName = "queue"
	fileServiceName  = "file"

	storageEmulatorBlob  = "127.0.0.1:10000"
	storageEmulatorTable = "127.0.0.1:10002"
	storageEmulatorQueue = "127.0.0.1:10001"

	userAgentHeader = "User-Agent"
)

// Client is the object that needs to be constructed to perform
// operations on the storage account.
type Client struct {
	// HTTPClient is the http.Client used to initiate API
	// requests.  If it is nil, http.DefaultClient is used.
	HTTPClient *http.Client

	accountName      string
	accountKey       []byte
	useHTTPS         bool
	UseSharedKeyLite bool
	baseURL          string
	apiVersion       string
	userAgent        string
}

type storageResponse struct {
	statusCode int
	headers    http.Header
	body       io.ReadCloser
}

type odataResponse struct {
	storageResponse
	odata odataErrorMessage
}

// AzureStorageServiceError contains fields of the error response from
// Azure Storage Service REST API. See https://msdn.microsoft.com/en-us/library/azure/dd179382.aspx
// Some fields might be specific to certain calls.
type AzureStorageServiceError struct {
	Code                      string `xml:"Code"`
	Message                   string `xml:"Message"`
	AuthenticationErrorDetail string `xml:"AuthenticationErrorDetail"`
	QueryParameterName        string `xml:"QueryParameterName"`
	QueryParameterValue       string `xml:"QueryParameterValue"`
	Reason                    string `xml:"Reason"`
	StatusCode                int
	RequestID                 string
}

type odataErrorMessageMessage struct {
	Lang  string `json:"lang"`
	Value string `json:"value"`
}

type odataErrorMessageInternal struct {
	Code    string                   `json:"code"`
	Message odataErrorMessageMessage `json:"message"`
}

type odataErrorMessage struct {
	Err odataErrorMessageInternal `json:"odata.error"`
}

// UnexpectedStatusCodeError is returned when a storage service responds with neither an error
// nor with an HTTP status code indicating success.
type UnexpectedStatusCodeError struct {
	allowed []int
	got     int
}

func (e UnexpectedStatusCodeError) Error() string {
	s := func(i int) string { return fmt.Sprintf("%d %s", i, http.StatusText(i)) }

	got := s(e.got)
	expected := []string{}
	for _, v := range e.allowed {
		expected = append(expected, s(v))
	}
	return fmt.Sprintf("storage: status code from service response is %s; was expecting %s", got, strings.Join(expected, " or "))
}

// Got is the actual status code returned by Azure.
func (e UnexpectedStatusCodeError) Got() int {
	return e.got
}

// NewBasicClient constructs a Client with given storage service name and
// key.
func NewBasicClient(accountName, accountKey string) (Client, error) {
	if accountName == StorageEmulatorAccountName {
		return NewEmulatorClient()
	}
	return NewClient(accountName, accountKey, DefaultBaseURL, DefaultAPIVersion, defaultUseHTTPS)

}

//NewEmulatorClient contructs a Client intended to only work with Azure
//Storage Emulator
func NewEmulatorClient() (Client, error) {
	return NewClient(StorageEmulatorAccountName, StorageEmulatorAccountKey, DefaultBaseURL, DefaultAPIVersion, false)
}

// NewClient constructs a Client. This should be used if the caller wants
// to specify whether to use HTTPS, a specific REST API version or a custom
// storage endpoint than Azure Public Cloud.
func NewClient(accountName, accountKey, blobServiceBaseURL, apiVersion string, useHTTPS bool) (Client, error) {
	var c Client
	if accountName == "" {
		return c, fmt.Errorf("azure: account name required")
	} else if accountKey == "" {
		return c, fmt.Errorf("azure: account key required")
	} else if blobServiceBaseURL == "" {
		return c, fmt.Errorf("azure: base storage service url required")
	}

	key, err := base64.StdEncoding.DecodeString(accountKey)
	if err != nil {
		return c, fmt.Errorf("azure: malformed storage account key: %v", err)
	}

	c = Client{
		accountName:      accountName,
		accountKey:       key,
		useHTTPS:         useHTTPS,
		baseURL:          blobServiceBaseURL,
		apiVersion:       apiVersion,
		UseSharedKeyLite: false,
	}
	c.userAgent = c.getDefaultUserAgent()
	return c, nil
}

func (c Client) getDefaultUserAgent() string {
	return fmt.Sprintf("Go/%s (%s-%s) Azure-SDK-For-Go/%s storage-dataplane/%s",
		runtime.Version(),
		runtime.GOARCH,
		runtime.GOOS,
		sdkVersion,
		c.apiVersion,
	)
}

// AddToUserAgent adds an extension to the current user agent
func (c *Client) AddToUserAgent(extension string) error {
	if extension != "" {
		c.userAgent = fmt.Sprintf("%s %s", c.userAgent, extension)
		return nil
	}
	return fmt.Errorf("Extension was empty, User Agent stayed as %s", c.userAgent)
}

// protectUserAgent is used in funcs that include extraheaders as a parameter.
// It prevents the User-Agent header to be overwritten, instead if it happens to
// be present, it gets added to the current User-Agent. Use it before getStandardHeaders
func (c *Client) protectUserAgent(extraheaders map[string]string) map[string]string {
	if v, ok := extraheaders[userAgentHeader]; ok {
		c.AddToUserAgent(v)
		delete(extraheaders, userAgentHeader)
	}
	return extraheaders
}

func (c Client) getBaseURL(service string) string {
	scheme := "http"
	if c.useHTTPS {
		scheme = "https"
	}
	host := ""
	if c.accountName == StorageEmulatorAccountName {
		switch service {
		case blobServiceName:
			host = storageEmulatorBlob
		case tableServiceName:
			host = storageEmulatorTable
		case queueServiceName:
			host = storageEmulatorQueue
		}
	} else {
		host = fmt.Sprintf("%s.%s.%s", c.accountName, service, c.baseURL)
	}

	u := &url.URL{
		Scheme: scheme,
		Host:   host}
	return u.String()
}

func (c Client) getEndpoint(service, path string, params url.Values) string {
	u, err := url.Parse(c.getBaseURL(service))
	if err != nil {
		// really should not be happening
		panic(err)
	}

	// API doesn't accept path segments not starting with '/'
	if !strings.HasPrefix(path, "/") {
		path = fmt.Sprintf("/%v", path)
	}

	if c.accountName == StorageEmulatorAccountName {
		path = fmt.Sprintf("/%v%v", StorageEmulatorAccountName, path)
	}

	u.Path = path
	u.RawQuery = params.Encode()
	return u.String()
}

// GetBlobService returns a BlobStorageClient which can operate on the blob
// service of the storage account.
func (c Client) GetBlobService() BlobStorageClient {
	b := BlobStorageClient{
		client: c,
	}
	b.client.AddToUserAgent(blobServiceName)
	b.auth = sharedKey
	if c.UseSharedKeyLite {
		b.auth = sharedKeyLite
	}
	return b
}

// GetQueueService returns a QueueServiceClient which can operate on the queue
// service of the storage account.
func (c Client) GetQueueService() QueueServiceClient {
	q := QueueServiceClient{
		client: c,
	}
	q.client.AddToUserAgent(queueServiceName)
	q.auth = sharedKey
	if c.UseSharedKeyLite {
		q.auth = sharedKeyLite
	}
	return q
}

// GetTableService returns a TableServiceClient which can operate on the table
// service of the storage account.
func (c Client) GetTableService() TableServiceClient {
	t := TableServiceClient{
		client: c,
	}
	t.client.AddToUserAgent(tableServiceName)
	t.auth = sharedKeyForTable
	if c.UseSharedKeyLite {
		t.auth = sharedKeyLiteForTable
	}
	return t
}

// GetFileService returns a FileServiceClient which can operate on the file
// service of the storage account.
func (c Client) GetFileService() FileServiceClient {
	f := FileServiceClient{
		client: c,
	}
	f.client.AddToUserAgent(fileServiceName)
	f.auth = sharedKey
	if c.UseSharedKeyLite {
		f.auth = sharedKeyLite
	}
	return f
}

func (c Client) getStandardHeaders() map[string]string {
	return map[string]string{
		userAgentHeader: c.userAgent,
		"x-ms-version":  c.apiVersion,
		"x-ms-date":     currentTimeRfc1123Formatted(),
	}
}

func (c Client) exec(verb, url string, headers map[string]string, body io.Reader, auth authentication) (*storageResponse, error) {
	headers, err := c.addAuthorizationHeader(verb, url, headers, auth)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(verb, url, body)
	if err != nil {
		return nil, errors.New("azure/storage: error creating request: " + err.Error())
	}

	if clstr, ok := headers["Content-Length"]; ok {
		// content length header is being signed, but completely ignored by golang.
		// instead we have to use the ContentLength property on the request struct
		// (see https://golang.org/src/net/http/request.go?s=18140:18370#L536 and
		// https://golang.org/src/net/http/transfer.go?s=1739:2467#L49)
		req.ContentLength, err = strconv.ParseInt(clstr, 10, 64)
		if err != nil {
			return nil, err
		}
	}
	for k, v := range headers {
		req.Header.Add(k, v)
	}

	httpClient := c.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}
	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}

	statusCode := resp.StatusCode
	if statusCode >= 400 && statusCode <= 505 {
		var respBody []byte
		respBody, err = readResponseBody(resp)
		if err != nil {
			return nil, err
		}

		requestID := resp.Header.Get("x-ms-request-id")
		if len(respBody) == 0 {
			// no error in response body, might happen in HEAD requests
			err = serviceErrFromStatusCode(resp.StatusCode, resp.Status, requestID)
		} else {
			// response contains storage service error object, unmarshal
			storageErr, errIn := serviceErrFromXML(respBody, resp.StatusCode, requestID)
			if err != nil { // error unmarshaling the error response
				err = errIn
			}
			err = storageErr
		}
		return &storageResponse{
			statusCode: resp.StatusCode,
			headers:    resp.Header,
			body:       ioutil.NopCloser(bytes.NewReader(respBody)), /* restore the body */
		}, err
	}

	return &storageResponse{
		statusCode: resp.StatusCode,
		headers:    resp.Header,
		body:       resp.Body}, nil
}

func (c Client) execInternalJSON(verb, url string, headers map[string]string, body io.Reader, auth authentication) (*odataResponse, error) {
	headers, err := c.addAuthorizationHeader(verb, url, headers, auth)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(verb, url, body)
	for k, v := range headers {
		req.Header.Add(k, v)
	}

	httpClient := c.HTTPClient
	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	resp, err := httpClient.Do(req)
	if err != nil {
		return nil, err
	}

	respToRet := &odataResponse{}
	respToRet.body = resp.Body
	respToRet.statusCode = resp.StatusCode
	respToRet.headers = resp.Header

	statusCode := resp.StatusCode
	if statusCode >= 400 && statusCode <= 505 {
		var respBody []byte
		respBody, err = readResponseBody(resp)
		if err != nil {
			return nil, err
		}

		if len(respBody) == 0 {
			// no error in response body, might happen in HEAD requests
			err = serviceErrFromStatusCode(resp.StatusCode, resp.Status, resp.Header.Get("x-ms-request-id"))
			return respToRet, err
		}
		// try unmarshal as odata.error json
		err = json.Unmarshal(respBody, &respToRet.odata)
		return respToRet, err
	}

	return respToRet, nil
}

func readResponseBody(resp *http.Response) ([]byte, error) {
	defer resp.Body.Close()
	out, err := ioutil.ReadAll(resp.Body)
	if err == io.EOF {
		err = nil
	}
	return out, err
}

func serviceErrFromXML(body []byte, statusCode int, requestID string) (AzureStorageServiceError, error) {
	var storageErr AzureStorageServiceError
	if err := xml.Unmarshal(body, &storageErr); err != nil {
		return storageErr, err
	}
	storageErr.StatusCode = statusCode
	storageErr.RequestID = requestID
	return storageErr, nil
}

func serviceErrFromStatusCode(code int, status string, requestID string) AzureStorageServiceError {
	return AzureStorageServiceError{
		StatusCode: code,
		Code:       status,
		RequestID:  requestID,
		Message:    "no response body was available for error status code",
	}
}

func (e AzureStorageServiceError) Error() string {
	return fmt.Sprintf("storage: service returned error: StatusCode=%d, ErrorCode=%s, ErrorMessage=%s, RequestId=%s, QueryParameterName=%s, QueryParameterValue=%s",
		e.StatusCode, e.Code, e.Message, e.RequestID, e.QueryParameterName, e.QueryParameterValue)
}

// checkRespCode returns UnexpectedStatusError if the given response code is not
// one of the allowed status codes; otherwise nil.
func checkRespCode(respCode int, allowed []int) error {
	for _, v := range allowed {
		if respCode == v {
			return nil
		}
	}
	return UnexpectedStatusCodeError{allowed, respCode}
}
