package common

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http"
	"net/url"
	"time"
	"strings"

	"github.com/denverdino/aliyungo/util"
)

// A Client represents a client of ECS services
type Client struct {
	AccessKeyId     string //Access Key Id
	AccessKeySecret string //Access Key Secret
	debug           bool
	httpClient      *http.Client
	endpoint        string
	version         string
}

// NewClient creates a new instance of ECS client
func (client *Client) Init(endpoint, version, accessKeyId, accessKeySecret string) {
	client.AccessKeyId = accessKeyId
	client.AccessKeySecret = accessKeySecret + "&"
	client.debug = false
	client.httpClient = &http.Client{}
	client.endpoint = endpoint
	client.version = version
}

// SetEndpoint sets custom endpoint
func (client *Client) SetEndpoint(endpoint string) {
	client.endpoint = endpoint
}

// SetEndpoint sets custom version
func (client *Client) SetVersion(version string) {
	client.version = version
}

// SetAccessKeyId sets new AccessKeyId
func (client *Client) SetAccessKeyId(id string) {
	client.AccessKeyId = id
}

// SetAccessKeySecret sets new AccessKeySecret
func (client *Client) SetAccessKeySecret(secret string) {
	client.AccessKeySecret = secret + "&"
}

// SetDebug sets debug mode to log the request/response message
func (client *Client) SetDebug(debug bool) {
	client.debug = debug
}

// Invoke sends the raw HTTP request for ECS services
func (client *Client) Invoke(action string, args interface{}, response interface{}) error {

	request := Request{}
	request.init(client.version, action, client.AccessKeyId)

	query := util.ConvertToQueryValues(request)
	util.SetQueryValues(args, &query)

	// Sign request
	signature := util.CreateSignatureForRequest(ECSRequestMethod, &query, client.AccessKeySecret)

	// Generate the request URL
	requestURL := client.endpoint + "?" + query.Encode() + "&Signature=" + url.QueryEscape(signature)

	httpReq, err := http.NewRequest(ECSRequestMethod, requestURL, nil)

	if err != nil {
		return GetClientError(err)
	}

	// TODO move to util and add build val flag
	httpReq.Header.Set("X-SDK-Client", `AliyunGO/`+Version)

	t0 := time.Now()
	httpResp, err := client.httpClient.Do(httpReq)
	t1 := time.Now()
	if err != nil {
		return GetClientError(err)
	}
	statusCode := httpResp.StatusCode

	if client.debug {
		log.Printf("Invoke %s %s %d (%v)", ECSRequestMethod, requestURL, statusCode, t1.Sub(t0))
	}

	defer httpResp.Body.Close()
	body, err := ioutil.ReadAll(httpResp.Body)

	if err != nil {
		return GetClientError(err)
	}

	if client.debug {
		var prettyJSON bytes.Buffer
		err = json.Indent(&prettyJSON, body, "", "    ")
		log.Println(string(prettyJSON.Bytes()))
	}

	if statusCode >= 400 && statusCode <= 599 {
		errorResponse := ErrorResponse{}
		err = json.Unmarshal(body, &errorResponse)
		ecsError := &Error{
			ErrorResponse: errorResponse,
			StatusCode:    statusCode,
		}
		return ecsError
	}

	err = json.Unmarshal(body, response)
	//log.Printf("%++v", response)
	if err != nil {
		return GetClientError(err)
	}

	return nil
}

// Invoke sends the raw HTTP request for ECS services
//改进了一下上面那个方法，可以使用各种Http方法
func (client *Client) InvokeByAnyMethod(method, action string, args interface{}, response interface{}) error {

	request := Request{}
	request.init(client.version, action, client.AccessKeyId)

	data := util.ConvertToQueryValues(request)
	util.SetQueryValues(args, &data)

	// Sign request
	signature := util.CreateSignatureForRequest(method, &data, client.AccessKeySecret)

	data.Add("Signature", signature)

	// Generate the request URL
	var (
		httpReq *http.Request
		err error
	)
	if method == http.MethodGet {
		requestURL := client.endpoint + "?" + data.Encode()
		httpReq, err = http.NewRequest(method, requestURL, nil)
	} else {
		httpReq, err = http.NewRequest(method, client.endpoint, strings.NewReader(data.Encode()))
		httpReq.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	}

	if err != nil {
		return GetClientError(err)
	}

	// TODO move to util and add build val flag
	httpReq.Header.Set("X-SDK-Client", `AliyunGO/` + Version)

	t0 := time.Now()
	httpResp, err := client.httpClient.Do(httpReq)
	t1 := time.Now()
	if err != nil {
		return GetClientError(err)
	}
	statusCode := httpResp.StatusCode

	if client.debug {
		log.Printf("Invoke %s %s %d (%v) %v", ECSRequestMethod, client.endpoint, statusCode, t1.Sub(t0), data.Encode())
	}

	defer httpResp.Body.Close()
	body, err := ioutil.ReadAll(httpResp.Body)

	if err != nil {
		return GetClientError(err)
	}

	if client.debug {
		var prettyJSON bytes.Buffer
		err = json.Indent(&prettyJSON, body, "", "    ")
		log.Println(string(prettyJSON.Bytes()))
	}

	if statusCode >= 400 && statusCode <= 599 {
		errorResponse := ErrorResponse{}
		err = json.Unmarshal(body, &errorResponse)
		ecsError := &Error{
			ErrorResponse: errorResponse,
			StatusCode:    statusCode,
		}
		return ecsError
	}

	err = json.Unmarshal(body, response)
	//log.Printf("%++v", response)
	if err != nil {
		return GetClientError(err)
	}

	return nil
}

// GenerateClientToken generates the Client Token with random string
func (client *Client) GenerateClientToken() string {
	return util.CreateRandomString()
}

func GetClientErrorFromString(str string) error {
	return &Error{
		ErrorResponse: ErrorResponse{
			Code:    "AliyunGoClientFailure",
			Message: str,
		},
		StatusCode: -1,
	}
}

func GetClientError(err error) error {
	return GetClientErrorFromString(err.Error())
}
