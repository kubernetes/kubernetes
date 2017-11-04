// Package storage provides clients for Microsoft Azure Storage Services.
package storage

import (
	"bufio"
	"bytes"
	"encoding/base64"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"mime"
	"mime/multipart"
	"net/http"
	"net/url"
	"regexp"
	"runtime"
	"strings"
	"time"

	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
)

const (
	// DefaultBaseURL is the domain name used for storage requests in the
	// public cloud when a default client is created.
	DefaultBaseURL = "core.windows.net"

	// DefaultAPIVersion is the Azure Storage API version string used when a
	// basic client is created.
	DefaultAPIVersion = "2016-05-31"

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

	userDefinedMetadataHeaderPrefix = "x-ms-meta-"
)

var (
	validStorageAccount = regexp.MustCompile("^[0-9a-z]{3,24}$")
)

// Sender sends a request
type Sender interface {
	Send(*Client, *http.Request) (*http.Response, error)
}

// DefaultSender is the default sender for the client. It implements
// an automatic retry strategy.
type DefaultSender struct {
	RetryAttempts    int
	RetryDuration    time.Duration
	ValidStatusCodes []int
	attempts         int // used for testing
}

// Send is the default retry strategy in the client
func (ds *DefaultSender) Send(c *Client, req *http.Request) (resp *http.Response, err error) {
	b := []byte{}
	if req.Body != nil {
		b, err = ioutil.ReadAll(req.Body)
		if err != nil {
			return resp, err
		}
	}

	for attempts := 0; attempts < ds.RetryAttempts; attempts++ {
		if len(b) > 0 {
			req.Body = ioutil.NopCloser(bytes.NewBuffer(b))
		}
		resp, err = c.HTTPClient.Do(req)
		if err != nil || !autorest.ResponseHasStatusCode(resp, ds.ValidStatusCodes...) {
			return resp, err
		}
		autorest.DelayForBackoff(ds.RetryDuration, attempts, req.Cancel)
		ds.attempts = attempts
	}
	ds.attempts++
	return resp, err
}

// Client is the object that needs to be constructed to perform
// operations on the storage account.
type Client struct {
	// HTTPClient is the http.Client used to initiate API
	// requests. http.DefaultClient is used when creating a
	// client.
	HTTPClient *http.Client

	// Sender is an interface that sends the request. Clients are
	// created with a DefaultSender. The DefaultSender has an
	// automatic retry strategy built in. The Sender can be customized.
	Sender Sender

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
	odata odataErrorWrapper
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
	Lang                      string
	StatusCode                int
	RequestID                 string
	Date                      string
	APIVersion                string
}

type odataErrorMessage struct {
	Lang  string `json:"lang"`
	Value string `json:"value"`
}

type odataError struct {
	Code    string            `json:"code"`
	Message odataErrorMessage `json:"message"`
}

type odataErrorWrapper struct {
	Err odataError `json:"odata.error"`
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

// NewBasicClientOnSovereignCloud constructs a Client with given storage service name and
// key in the referenced cloud.
func NewBasicClientOnSovereignCloud(accountName, accountKey string, env azure.Environment) (Client, error) {
	if accountName == StorageEmulatorAccountName {
		return NewEmulatorClient()
	}
	return NewClient(accountName, accountKey, env.StorageEndpointSuffix, DefaultAPIVersion, defaultUseHTTPS)
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
	if !IsValidStorageAccount(accountName) {
		return c, fmt.Errorf("azure: account name is not valid: it must be between 3 and 24 characters, and only may contain numbers and lowercase letters: %v", accountName)
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
		HTTPClient:       http.DefaultClient,
		accountName:      accountName,
		accountKey:       key,
		useHTTPS:         useHTTPS,
		baseURL:          blobServiceBaseURL,
		apiVersion:       apiVersion,
		UseSharedKeyLite: false,
		Sender: &DefaultSender{
			RetryAttempts: 5,
			ValidStatusCodes: []int{
				http.StatusRequestTimeout,      // 408
				http.StatusInternalServerError, // 500
				http.StatusBadGateway,          // 502
				http.StatusServiceUnavailable,  // 503
				http.StatusGatewayTimeout,      // 504
			},
			RetryDuration: time.Second * 5,
		},
	}
	c.userAgent = c.getDefaultUserAgent()
	return c, nil
}

// IsValidStorageAccount checks if the storage account name is valid.
// See https://docs.microsoft.com/en-us/azure/storage/storage-create-storage-account
func IsValidStorageAccount(account string) bool {
	return validStorageAccount.MatchString(account)
}

func (c Client) getDefaultUserAgent() string {
	return fmt.Sprintf("Go/%s (%s-%s) azure-storage-go/%s api-version/%s",
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

func (c Client) getBaseURL(service string) *url.URL {
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

	return &url.URL{
		Scheme: scheme,
		Host:   host,
	}
}

func (c Client) getEndpoint(service, path string, params url.Values) string {
	u := c.getBaseURL(service)

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

	for k, v := range headers {
		req.Header.Add(k, v)
	}

	resp, err := c.Sender.Send(&c, req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 && resp.StatusCode <= 505 {
		var respBody []byte
		respBody, err = readAndCloseBody(resp.Body)
		if err != nil {
			return nil, err
		}

		requestID, date, version := getDebugHeaders(resp.Header)
		if len(respBody) == 0 {
			// no error in response body, might happen in HEAD requests
			err = serviceErrFromStatusCode(resp.StatusCode, resp.Status, requestID, date, version)
		} else {
			storageErr := AzureStorageServiceError{
				StatusCode: resp.StatusCode,
				RequestID:  requestID,
				Date:       date,
				APIVersion: version,
			}
			// response contains storage service error object, unmarshal
			if resp.Header.Get("Content-Type") == "application/xml" {
				errIn := serviceErrFromXML(respBody, &storageErr)
				if err != nil { // error unmarshaling the error response
					err = errIn
				}
			} else {
				errIn := serviceErrFromJSON(respBody, &storageErr)
				if err != nil { // error unmarshaling the error response
					err = errIn
				}
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

func (c Client) execInternalJSONCommon(verb, url string, headers map[string]string, body io.Reader, auth authentication) (*odataResponse, *http.Request, *http.Response, error) {
	headers, err := c.addAuthorizationHeader(verb, url, headers, auth)
	if err != nil {
		return nil, nil, nil, err
	}

	req, err := http.NewRequest(verb, url, body)
	for k, v := range headers {
		req.Header.Add(k, v)
	}

	resp, err := c.Sender.Send(&c, req)
	if err != nil {
		return nil, nil, nil, err
	}

	respToRet := &odataResponse{}
	respToRet.body = resp.Body
	respToRet.statusCode = resp.StatusCode
	respToRet.headers = resp.Header

	statusCode := resp.StatusCode
	if statusCode >= 400 && statusCode <= 505 {
		var respBody []byte
		respBody, err = readAndCloseBody(resp.Body)
		if err != nil {
			return nil, nil, nil, err
		}

		requestID, date, version := getDebugHeaders(resp.Header)
		if len(respBody) == 0 {
			// no error in response body, might happen in HEAD requests
			err = serviceErrFromStatusCode(resp.StatusCode, resp.Status, requestID, date, version)
			return respToRet, req, resp, err
		}
		// try unmarshal as odata.error json
		err = json.Unmarshal(respBody, &respToRet.odata)
	}

	return respToRet, req, resp, err
}

func (c Client) execInternalJSON(verb, url string, headers map[string]string, body io.Reader, auth authentication) (*odataResponse, error) {
	respToRet, _, _, err := c.execInternalJSONCommon(verb, url, headers, body, auth)
	return respToRet, err
}

func (c Client) execBatchOperationJSON(verb, url string, headers map[string]string, body io.Reader, auth authentication) (*odataResponse, error) {
	// execute common query, get back generated request, response etc... for more processing.
	respToRet, req, resp, err := c.execInternalJSONCommon(verb, url, headers, body, auth)
	if err != nil {
		return nil, err
	}

	// return the OData in the case of executing batch commands.
	// In this case we need to read the outer batch boundary and contents.
	// Then we read the changeset information within the batch
	var respBody []byte
	respBody, err = readAndCloseBody(resp.Body)
	if err != nil {
		return nil, err
	}

	// outer multipart body
	_, batchHeader, err := mime.ParseMediaType(resp.Header["Content-Type"][0])
	if err != nil {
		return nil, err
	}

	// batch details.
	batchBoundary := batchHeader["boundary"]
	batchPartBuf, changesetBoundary, err := genBatchReader(batchBoundary, respBody)
	if err != nil {
		return nil, err
	}

	// changeset details.
	err = genChangesetReader(req, respToRet, batchPartBuf, changesetBoundary)
	if err != nil {
		return nil, err
	}

	return respToRet, nil
}

func genChangesetReader(req *http.Request, respToRet *odataResponse, batchPartBuf io.Reader, changesetBoundary string) error {
	changesetMultiReader := multipart.NewReader(batchPartBuf, changesetBoundary)
	changesetPart, err := changesetMultiReader.NextPart()
	if err != nil {
		return err
	}

	changesetPartBufioReader := bufio.NewReader(changesetPart)
	changesetResp, err := http.ReadResponse(changesetPartBufioReader, req)
	if err != nil {
		return err
	}

	if changesetResp.StatusCode != http.StatusNoContent {
		changesetBody, err := readAndCloseBody(changesetResp.Body)
		err = json.Unmarshal(changesetBody, &respToRet.odata)
		if err != nil {
			return err
		}
		respToRet.statusCode = changesetResp.StatusCode
	}

	return nil
}

func genBatchReader(batchBoundary string, respBody []byte) (io.Reader, string, error) {
	respBodyString := string(respBody)
	respBodyReader := strings.NewReader(respBodyString)

	// reading batchresponse
	batchMultiReader := multipart.NewReader(respBodyReader, batchBoundary)
	batchPart, err := batchMultiReader.NextPart()
	if err != nil {
		return nil, "", err
	}
	batchPartBufioReader := bufio.NewReader(batchPart)

	_, changesetHeader, err := mime.ParseMediaType(batchPart.Header.Get("Content-Type"))
	if err != nil {
		return nil, "", err
	}
	changesetBoundary := changesetHeader["boundary"]
	return batchPartBufioReader, changesetBoundary, nil
}

func readAndCloseBody(body io.ReadCloser) ([]byte, error) {
	defer body.Close()
	out, err := ioutil.ReadAll(body)
	if err == io.EOF {
		err = nil
	}
	return out, err
}

func serviceErrFromXML(body []byte, storageErr *AzureStorageServiceError) error {
	if err := xml.Unmarshal(body, storageErr); err != nil {
		storageErr.Message = fmt.Sprintf("Response body could no be unmarshaled: %v. Body: %v.", err, string(body))
		return err
	}
	return nil
}

func serviceErrFromJSON(body []byte, storageErr *AzureStorageServiceError) error {
	odataError := odataErrorWrapper{}
	if err := json.Unmarshal(body, &odataError); err != nil {
		storageErr.Message = fmt.Sprintf("Response body could no be unmarshaled: %v. Body: %v.", err, string(body))
		return err
	}
	storageErr.Code = odataError.Err.Code
	storageErr.Message = odataError.Err.Message.Value
	storageErr.Lang = odataError.Err.Message.Lang
	return nil
}

func serviceErrFromStatusCode(code int, status string, requestID, date, version string) AzureStorageServiceError {
	return AzureStorageServiceError{
		StatusCode: code,
		Code:       status,
		RequestID:  requestID,
		Date:       date,
		APIVersion: version,
		Message:    "no response body was available for error status code",
	}
}

func (e AzureStorageServiceError) Error() string {
	return fmt.Sprintf("storage: service returned error: StatusCode=%d, ErrorCode=%s, ErrorMessage=%s, RequestInitiated=%s, RequestId=%s, API Version=%s, QueryParameterName=%s, QueryParameterValue=%s",
		e.StatusCode, e.Code, e.Message, e.Date, e.RequestID, e.APIVersion, e.QueryParameterName, e.QueryParameterValue)
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

func (c Client) addMetadataToHeaders(h map[string]string, metadata map[string]string) map[string]string {
	metadata = c.protectUserAgent(metadata)
	for k, v := range metadata {
		h[userDefinedMetadataHeaderPrefix+k] = v
	}
	return h
}

func getDebugHeaders(h http.Header) (requestID, date, version string) {
	requestID = h.Get("x-ms-request-id")
	version = h.Get("x-ms-version")
	date = h.Get("Date")
	return
}
