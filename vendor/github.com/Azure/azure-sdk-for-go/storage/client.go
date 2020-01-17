// Package storage provides clients for Microsoft Azure Storage Services.
package storage

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"bufio"
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
	"strconv"
	"strings"
	"time"

	"github.com/Azure/azure-sdk-for-go/version"
	"github.com/Azure/go-autorest/autorest"
	"github.com/Azure/go-autorest/autorest/azure"
)

const (
	// DefaultBaseURL is the domain name used for storage requests in the
	// public cloud when a default client is created.
	DefaultBaseURL = "core.windows.net"

	// DefaultAPIVersion is the Azure Storage API version string used when a
	// basic client is created.
	DefaultAPIVersion = "2018-03-28"

	defaultUseHTTPS      = true
	defaultRetryAttempts = 5
	defaultRetryDuration = time.Second * 5

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

	connectionStringAccountName      = "accountname"
	connectionStringAccountKey       = "accountkey"
	connectionStringEndpointSuffix   = "endpointsuffix"
	connectionStringEndpointProtocol = "defaultendpointsprotocol"

	connectionStringBlobEndpoint  = "blobendpoint"
	connectionStringFileEndpoint  = "fileendpoint"
	connectionStringQueueEndpoint = "queueendpoint"
	connectionStringTableEndpoint = "tableendpoint"
	connectionStringSAS           = "sharedaccesssignature"
)

var (
	validStorageAccount     = regexp.MustCompile("^[0-9a-z]{3,24}$")
	defaultValidStatusCodes = []int{
		http.StatusRequestTimeout,      // 408
		http.StatusInternalServerError, // 500
		http.StatusBadGateway,          // 502
		http.StatusServiceUnavailable,  // 503
		http.StatusGatewayTimeout,      // 504
	}
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
	rr := autorest.NewRetriableRequest(req)
	for attempts := 0; attempts < ds.RetryAttempts; attempts++ {
		err = rr.Prepare()
		if err != nil {
			return resp, err
		}
		resp, err = c.HTTPClient.Do(rr.Request())
		if err != nil || !autorest.ResponseHasStatusCode(resp, ds.ValidStatusCodes...) {
			return resp, err
		}
		drainRespBody(resp)
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
	sasClient        bool
	accountSASToken  url.Values
}

type odataResponse struct {
	resp  *http.Response
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
	inner   error
}

func (e UnexpectedStatusCodeError) Error() string {
	s := func(i int) string { return fmt.Sprintf("%d %s", i, http.StatusText(i)) }

	got := s(e.got)
	expected := []string{}
	for _, v := range e.allowed {
		expected = append(expected, s(v))
	}
	return fmt.Sprintf("storage: status code from service response is %s; was expecting %s.  Inner error: %+v", got, strings.Join(expected, " or "), e.inner)
}

// Got is the actual status code returned by Azure.
func (e UnexpectedStatusCodeError) Got() int {
	return e.got
}

// Inner returns any inner error info.
func (e UnexpectedStatusCodeError) Inner() error {
	return e.inner
}

// NewClientFromConnectionString creates a Client from the connection string.
func NewClientFromConnectionString(input string) (Client, error) {
	// build a map of connection string key/value pairs
	parts := map[string]string{}
	for _, pair := range strings.Split(input, ";") {
		if pair == "" {
			continue
		}

		equalDex := strings.IndexByte(pair, '=')
		if equalDex <= 0 {
			return Client{}, fmt.Errorf("Invalid connection segment %q", pair)
		}

		value := strings.TrimSpace(pair[equalDex+1:])
		key := strings.TrimSpace(strings.ToLower(pair[:equalDex]))
		parts[key] = value
	}

	// TODO: validate parameter sets?

	if parts[connectionStringAccountName] == StorageEmulatorAccountName {
		return NewEmulatorClient()
	}

	if parts[connectionStringSAS] != "" {
		endpoint := ""
		if parts[connectionStringBlobEndpoint] != "" {
			endpoint = parts[connectionStringBlobEndpoint]
		} else if parts[connectionStringFileEndpoint] != "" {
			endpoint = parts[connectionStringFileEndpoint]
		} else if parts[connectionStringQueueEndpoint] != "" {
			endpoint = parts[connectionStringQueueEndpoint]
		} else {
			endpoint = parts[connectionStringTableEndpoint]
		}

		return NewAccountSASClientFromEndpointToken(endpoint, parts[connectionStringSAS])
	}

	useHTTPS := defaultUseHTTPS
	if parts[connectionStringEndpointProtocol] != "" {
		useHTTPS = parts[connectionStringEndpointProtocol] == "https"
	}

	return NewClient(parts[connectionStringAccountName], parts[connectionStringAccountKey],
		parts[connectionStringEndpointSuffix], DefaultAPIVersion, useHTTPS)
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
func NewClient(accountName, accountKey, serviceBaseURL, apiVersion string, useHTTPS bool) (Client, error) {
	var c Client
	if !IsValidStorageAccount(accountName) {
		return c, fmt.Errorf("azure: account name is not valid: it must be between 3 and 24 characters, and only may contain numbers and lowercase letters: %v", accountName)
	} else if accountKey == "" {
		return c, fmt.Errorf("azure: account key required")
	} else if serviceBaseURL == "" {
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
		baseURL:          serviceBaseURL,
		apiVersion:       apiVersion,
		sasClient:        false,
		UseSharedKeyLite: false,
		Sender: &DefaultSender{
			RetryAttempts:    defaultRetryAttempts,
			ValidStatusCodes: defaultValidStatusCodes,
			RetryDuration:    defaultRetryDuration,
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

// NewAccountSASClient contructs a client that uses accountSAS authorization
// for its operations.
func NewAccountSASClient(account string, token url.Values, env azure.Environment) Client {
	return newSASClient(account, env.StorageEndpointSuffix, token)
}

// NewAccountSASClientFromEndpointToken constructs a client that uses accountSAS authorization
// for its operations using the specified endpoint and SAS token.
func NewAccountSASClientFromEndpointToken(endpoint string, sasToken string) (Client, error) {
	u, err := url.Parse(endpoint)
	if err != nil {
		return Client{}, err
	}
	_, err = url.ParseQuery(sasToken)
	if err != nil {
		return Client{}, err
	}
	u.RawQuery = sasToken
	return newSASClientFromURL(u)
}

func newSASClient(accountName, baseURL string, sasToken url.Values) Client {
	c := Client{
		HTTPClient: http.DefaultClient,
		apiVersion: DefaultAPIVersion,
		sasClient:  true,
		Sender: &DefaultSender{
			RetryAttempts:    defaultRetryAttempts,
			ValidStatusCodes: defaultValidStatusCodes,
			RetryDuration:    defaultRetryDuration,
		},
		accountName:     accountName,
		baseURL:         baseURL,
		accountSASToken: sasToken,
		useHTTPS:        defaultUseHTTPS,
	}
	c.userAgent = c.getDefaultUserAgent()
	// Get API version and protocol from token
	c.apiVersion = sasToken.Get("sv")
	if spr := sasToken.Get("spr"); spr != "" {
		c.useHTTPS = spr == "https"
	}
	return c
}

func newSASClientFromURL(u *url.URL) (Client, error) {
	// the host name will look something like this
	// - foo.blob.core.windows.net
	// "foo" is the account name
	// "core.windows.net" is the baseURL

	// find the first dot to get account name
	i1 := strings.IndexByte(u.Host, '.')
	if i1 < 0 {
		return Client{}, fmt.Errorf("failed to find '.' in %s", u.Host)
	}

	// now find the second dot to get the base URL
	i2 := strings.IndexByte(u.Host[i1+1:], '.')
	if i2 < 0 {
		return Client{}, fmt.Errorf("failed to find '.' in %s", u.Host[i1+1:])
	}

	sasToken := u.Query()
	c := newSASClient(u.Host[:i1], u.Host[i1+i2+2:], sasToken)
	if spr := sasToken.Get("spr"); spr == "" {
		// infer from URL if not in the query params set
		c.useHTTPS = u.Scheme == "https"
	}
	return c, nil
}

func (c Client) isServiceSASClient() bool {
	return c.sasClient && c.accountSASToken == nil
}

func (c Client) isAccountSASClient() bool {
	return c.sasClient && c.accountSASToken != nil
}

func (c Client) getDefaultUserAgent() string {
	return fmt.Sprintf("Go/%s (%s-%s) azure-storage-go/%s api-version/%s",
		runtime.Version(),
		runtime.GOARCH,
		runtime.GOOS,
		version.Number,
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

// AccountSASTokenOptions includes options for constructing
// an account SAS token.
// https://docs.microsoft.com/en-us/rest/api/storageservices/constructing-an-account-sas
type AccountSASTokenOptions struct {
	APIVersion    string
	Services      Services
	ResourceTypes ResourceTypes
	Permissions   Permissions
	Start         time.Time
	Expiry        time.Time
	IP            string
	UseHTTPS      bool
}

// Services specify services accessible with an account SAS.
type Services struct {
	Blob  bool
	Queue bool
	Table bool
	File  bool
}

// ResourceTypes specify the resources accesible with an
// account SAS.
type ResourceTypes struct {
	Service   bool
	Container bool
	Object    bool
}

// Permissions specifies permissions for an accountSAS.
type Permissions struct {
	Read    bool
	Write   bool
	Delete  bool
	List    bool
	Add     bool
	Create  bool
	Update  bool
	Process bool
}

// GetAccountSASToken creates an account SAS token
// See https://docs.microsoft.com/en-us/rest/api/storageservices/constructing-an-account-sas
func (c Client) GetAccountSASToken(options AccountSASTokenOptions) (url.Values, error) {
	if options.APIVersion == "" {
		options.APIVersion = c.apiVersion
	}

	if options.APIVersion < "2015-04-05" {
		return url.Values{}, fmt.Errorf("account SAS does not support API versions prior to 2015-04-05. API version : %s", options.APIVersion)
	}

	// build services string
	services := ""
	if options.Services.Blob {
		services += "b"
	}
	if options.Services.Queue {
		services += "q"
	}
	if options.Services.Table {
		services += "t"
	}
	if options.Services.File {
		services += "f"
	}

	// build resources string
	resources := ""
	if options.ResourceTypes.Service {
		resources += "s"
	}
	if options.ResourceTypes.Container {
		resources += "c"
	}
	if options.ResourceTypes.Object {
		resources += "o"
	}

	// build permissions string
	permissions := ""
	if options.Permissions.Read {
		permissions += "r"
	}
	if options.Permissions.Write {
		permissions += "w"
	}
	if options.Permissions.Delete {
		permissions += "d"
	}
	if options.Permissions.List {
		permissions += "l"
	}
	if options.Permissions.Add {
		permissions += "a"
	}
	if options.Permissions.Create {
		permissions += "c"
	}
	if options.Permissions.Update {
		permissions += "u"
	}
	if options.Permissions.Process {
		permissions += "p"
	}

	// build start time, if exists
	start := ""
	if options.Start != (time.Time{}) {
		start = options.Start.UTC().Format(time.RFC3339)
	}

	// build expiry time
	expiry := options.Expiry.UTC().Format(time.RFC3339)

	protocol := "https,http"
	if options.UseHTTPS {
		protocol = "https"
	}

	stringToSign := strings.Join([]string{
		c.accountName,
		permissions,
		services,
		resources,
		start,
		expiry,
		options.IP,
		protocol,
		options.APIVersion,
		"",
	}, "\n")
	signature := c.computeHmac256(stringToSign)

	sasParams := url.Values{
		"sv":  {options.APIVersion},
		"ss":  {services},
		"srt": {resources},
		"sp":  {permissions},
		"se":  {expiry},
		"spr": {protocol},
		"sig": {signature},
	}
	if start != "" {
		sasParams.Add("st", start)
	}
	if options.IP != "" {
		sasParams.Add("sip", options.IP)
	}

	return sasParams, nil
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

func (c Client) exec(verb, url string, headers map[string]string, body io.Reader, auth authentication) (*http.Response, error) {
	headers, err := c.addAuthorizationHeader(verb, url, headers, auth)
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequest(verb, url, body)
	if err != nil {
		return nil, errors.New("azure/storage: error creating request: " + err.Error())
	}

	// http.NewRequest() will automatically set req.ContentLength for a handful of types
	// otherwise we will handle here.
	if req.ContentLength < 1 {
		if clstr, ok := headers["Content-Length"]; ok {
			if cl, err := strconv.ParseInt(clstr, 10, 64); err == nil {
				req.ContentLength = cl
			}
		}
	}

	for k, v := range headers {
		req.Header[k] = append(req.Header[k], v) // Must bypass case munging present in `Add` by using map functions directly. See https://github.com/Azure/azure-sdk-for-go/issues/645
	}

	if c.isAccountSASClient() {
		// append the SAS token to the query params
		v := req.URL.Query()
		v = mergeParams(v, c.accountSASToken)
		req.URL.RawQuery = v.Encode()
	}

	resp, err := c.Sender.Send(&c, req)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode >= 400 && resp.StatusCode <= 505 {
		return resp, getErrorFromResponse(resp)
	}

	return resp, nil
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

	respToRet := &odataResponse{resp: resp}

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
		respToRet.resp = changesetResp
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

// reads the response body then closes it
func drainRespBody(resp *http.Response) {
	io.Copy(ioutil.Discard, resp.Body)
	resp.Body.Close()
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
func checkRespCode(resp *http.Response, allowed []int) error {
	for _, v := range allowed {
		if resp.StatusCode == v {
			return nil
		}
	}
	err := getErrorFromResponse(resp)
	return UnexpectedStatusCodeError{
		allowed: allowed,
		got:     resp.StatusCode,
		inner:   err,
	}
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

func getErrorFromResponse(resp *http.Response) error {
	respBody, err := readAndCloseBody(resp.Body)
	if err != nil {
		return err
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
	return err
}
