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
	"regexp"
	"sort"
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
)

// Client is the object that needs to be constructed to perform
// operations on the storage account.
type Client struct {
	// HTTPClient is the http.Client used to initiate API
	// requests.  If it is nil, http.DefaultClient is used.
	HTTPClient *http.Client

	accountName string
	accountKey  []byte
	useHTTPS    bool
	baseURL     string
	apiVersion  string
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

	return Client{
		accountName: accountName,
		accountKey:  key,
		useHTTPS:    useHTTPS,
		baseURL:     blobServiceBaseURL,
		apiVersion:  apiVersion,
	}, nil
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
	return BlobStorageClient{c}
}

// GetQueueService returns a QueueServiceClient which can operate on the queue
// service of the storage account.
func (c Client) GetQueueService() QueueServiceClient {
	return QueueServiceClient{c}
}

// GetTableService returns a TableServiceClient which can operate on the table
// service of the storage account.
func (c Client) GetTableService() TableServiceClient {
	return TableServiceClient{c}
}

// GetFileService returns a FileServiceClient which can operate on the file
// service of the storage account.
func (c Client) GetFileService() FileServiceClient {
	return FileServiceClient{c}
}

func (c Client) createAuthorizationHeader(canonicalizedString string) string {
	signature := c.computeHmac256(canonicalizedString)
	return fmt.Sprintf("%s %s:%s", "SharedKey", c.getCanonicalizedAccountName(), signature)
}

func (c Client) getAuthorizationHeader(verb, url string, headers map[string]string) (string, error) {
	canonicalizedResource, err := c.buildCanonicalizedResource(url)
	if err != nil {
		return "", err
	}

	canonicalizedString := c.buildCanonicalizedString(verb, headers, canonicalizedResource)
	return c.createAuthorizationHeader(canonicalizedString), nil
}

func (c Client) getStandardHeaders() map[string]string {
	return map[string]string{
		"x-ms-version": c.apiVersion,
		"x-ms-date":    currentTimeRfc1123Formatted(),
	}
}

func (c Client) getCanonicalizedAccountName() string {
	// since we may be trying to access a secondary storage account, we need to
	// remove the -secondary part of the storage name
	return strings.TrimSuffix(c.accountName, "-secondary")
}

func (c Client) buildCanonicalizedHeader(headers map[string]string) string {
	cm := make(map[string]string)

	for k, v := range headers {
		headerName := strings.TrimSpace(strings.ToLower(k))
		match, _ := regexp.MatchString("x-ms-", headerName)
		if match {
			cm[headerName] = v
		}
	}

	if len(cm) == 0 {
		return ""
	}

	keys := make([]string, 0, len(cm))
	for key := range cm {
		keys = append(keys, key)
	}

	sort.Strings(keys)

	ch := ""

	for i, key := range keys {
		if i == len(keys)-1 {
			ch += fmt.Sprintf("%s:%s", key, cm[key])
		} else {
			ch += fmt.Sprintf("%s:%s\n", key, cm[key])
		}
	}
	return ch
}

func (c Client) buildCanonicalizedResourceTable(uri string) (string, error) {
	errMsg := "buildCanonicalizedResourceTable error: %s"
	u, err := url.Parse(uri)
	if err != nil {
		return "", fmt.Errorf(errMsg, err.Error())
	}

	cr := "/" + c.getCanonicalizedAccountName()

	if len(u.Path) > 0 {
		cr += u.EscapedPath()
	}

	return cr, nil
}

func (c Client) buildCanonicalizedResource(uri string) (string, error) {
	errMsg := "buildCanonicalizedResource error: %s"
	u, err := url.Parse(uri)
	if err != nil {
		return "", fmt.Errorf(errMsg, err.Error())
	}

	cr := "/" + c.getCanonicalizedAccountName()

	if len(u.Path) > 0 {
		// Any portion of the CanonicalizedResource string that is derived from
		// the resource's URI should be encoded exactly as it is in the URI.
		// -- https://msdn.microsoft.com/en-gb/library/azure/dd179428.aspx
		cr += u.EscapedPath()
	}

	params, err := url.ParseQuery(u.RawQuery)
	if err != nil {
		return "", fmt.Errorf(errMsg, err.Error())
	}

	if len(params) > 0 {
		cr += "\n"
		keys := make([]string, 0, len(params))
		for key := range params {
			keys = append(keys, key)
		}

		sort.Strings(keys)

		for i, key := range keys {
			if len(params[key]) > 1 {
				sort.Strings(params[key])
			}

			if i == len(keys)-1 {
				cr += fmt.Sprintf("%s:%s", key, strings.Join(params[key], ","))
			} else {
				cr += fmt.Sprintf("%s:%s\n", key, strings.Join(params[key], ","))
			}
		}
	}

	return cr, nil
}

func (c Client) buildCanonicalizedString(verb string, headers map[string]string, canonicalizedResource string) string {
	contentLength := headers["Content-Length"]
	if contentLength == "0" {
		contentLength = ""
	}
	canonicalizedString := fmt.Sprintf("%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s\n%s",
		verb,
		headers["Content-Encoding"],
		headers["Content-Language"],
		contentLength,
		headers["Content-MD5"],
		headers["Content-Type"],
		headers["Date"],
		headers["If-Modified-Since"],
		headers["If-Match"],
		headers["If-None-Match"],
		headers["If-Unmodified-Since"],
		headers["Range"],
		c.buildCanonicalizedHeader(headers),
		canonicalizedResource)

	return canonicalizedString
}

func (c Client) exec(verb, url string, headers map[string]string, body io.Reader) (*storageResponse, error) {
	authHeader, err := c.getAuthorizationHeader(verb, url, headers)
	if err != nil {
		return nil, err
	}
	headers["Authorization"] = authHeader
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

		if len(respBody) == 0 {
			// no error in response body
			err = fmt.Errorf("storage: service returned without a response body (%s)", resp.Status)
		} else {
			// response contains storage service error object, unmarshal
			storageErr, errIn := serviceErrFromXML(respBody, resp.StatusCode, resp.Header.Get("x-ms-request-id"))
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

func (c Client) execInternalJSON(verb, url string, headers map[string]string, body io.Reader) (*odataResponse, error) {
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
			// no error in response body
			err = fmt.Errorf("storage: service returned without a response body (%d)", resp.StatusCode)
			return respToRet, err
		}
		// try unmarshal as odata.error json
		err = json.Unmarshal(respBody, &respToRet.odata)
		return respToRet, err
	}

	return respToRet, nil
}

func (c Client) createSharedKeyLite(url string, headers map[string]string) (string, error) {
	can, err := c.buildCanonicalizedResourceTable(url)

	if err != nil {
		return "", err
	}
	strToSign := headers["x-ms-date"] + "\n" + can

	hmac := c.computeHmac256(strToSign)
	return fmt.Sprintf("SharedKeyLite %s:%s", c.accountName, hmac), nil
}

func (c Client) execTable(verb, url string, headers map[string]string, body io.Reader) (*odataResponse, error) {
	var err error
	headers["Authorization"], err = c.createSharedKeyLite(url, headers)
	if err != nil {
		return nil, err
	}

	return c.execInternalJSON(verb, url, headers, body)
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
