package client

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"time"

	"github.com/gorilla/websocket"
	"github.com/pkg/errors"
)

const (
	SELF       = "self"
	COLLECTION = "collection"
)

var (
	debug  = false
	dialer = &websocket.Dialer{}
)

type ClientOpts struct {
	Url       string
	AccessKey string
	SecretKey string
	Timeout   time.Duration
}

type ApiError struct {
	StatusCode int
	Url        string
	Msg        string
	Status     string
	Body       string
}

func (e *ApiError) Error() string {
	return e.Msg
}

func IsNotFound(err error) bool {
	apiError, ok := err.(*ApiError)
	if !ok {
		return false
	}

	return apiError.StatusCode == http.StatusNotFound
}

func newApiError(resp *http.Response, url string) *ApiError {
	contents, err := ioutil.ReadAll(resp.Body)
	var body string
	if err != nil {
		body = "Unreadable body."
	} else {
		body = string(contents)
	}

	data := map[string]interface{}{}
	if json.Unmarshal(contents, &data) == nil {
		delete(data, "id")
		delete(data, "links")
		delete(data, "actions")
		delete(data, "type")
		delete(data, "status")
		buf := &bytes.Buffer{}
		for k, v := range data {
			if v == nil {
				continue
			}
			if buf.Len() > 0 {
				buf.WriteString(", ")
			}
			fmt.Fprintf(buf, "%s=%v", k, v)
		}
		body = buf.String()
	}
	formattedMsg := fmt.Sprintf("Bad response statusCode [%d]. Status [%s]. Body: [%s] from [%s]",
		resp.StatusCode, resp.Status, body, url)
	return &ApiError{
		Url:        url,
		Msg:        formattedMsg,
		StatusCode: resp.StatusCode,
		Status:     resp.Status,
		Body:       body,
	}
}

func contains(array []string, item string) bool {
	for _, check := range array {
		if check == item {
			return true
		}
	}

	return false
}

func appendFilters(urlString string, filters map[string]interface{}) (string, error) {
	if len(filters) == 0 {
		return urlString, nil
	}

	u, err := url.Parse(urlString)
	if err != nil {
		return "", err
	}

	q := u.Query()
	for k, v := range filters {
		if l, ok := v.([]string); ok {
			for _, v := range l {
				q.Add(k, v)
			}
		} else {
			q.Add(k, fmt.Sprintf("%v", v))
		}
	}

	u.RawQuery = q.Encode()
	return u.String(), nil
}

func setupRancherBaseClient(rancherClient *RancherBaseClientImpl, opts *ClientOpts) error {
	if opts.Timeout == 0 {
		opts.Timeout = time.Second * 10
	}
	client := &http.Client{Timeout: opts.Timeout}
	req, err := http.NewRequest("GET", opts.Url, nil)
	if err != nil {
		return err
	}

	req.SetBasicAuth(opts.AccessKey, opts.SecretKey)

	resp, err := client.Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return newApiError(resp, opts.Url)
	}

	schemasUrls := resp.Header.Get("X-API-Schemas")
	if len(schemasUrls) == 0 {
		return errors.New("Failed to find schema at [" + opts.Url + "]")
	}

	if schemasUrls != opts.Url {
		req, err = http.NewRequest("GET", schemasUrls, nil)
		req.SetBasicAuth(opts.AccessKey, opts.SecretKey)
		if err != nil {
			return err
		}

		resp, err = client.Do(req)
		if err != nil {
			return err
		}

		defer resp.Body.Close()

		if resp.StatusCode != 200 {
			return newApiError(resp, opts.Url)
		}
	}

	var schemas Schemas
	bytes, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	err = json.Unmarshal(bytes, &schemas)
	if err != nil {
		return err
	}

	rancherClient.Opts = opts
	rancherClient.Schemas = &schemas

	for _, schema := range schemas.Data {
		rancherClient.Types[schema.Id] = schema
	}

	return nil
}

func NewListOpts() *ListOpts {
	return &ListOpts{
		Filters: map[string]interface{}{},
	}
}

func (rancherClient *RancherBaseClientImpl) setupRequest(req *http.Request) {
	req.SetBasicAuth(rancherClient.Opts.AccessKey, rancherClient.Opts.SecretKey)
}

func (rancherClient *RancherBaseClientImpl) newHttpClient() *http.Client {
	if rancherClient.Opts.Timeout == 0 {
		rancherClient.Opts.Timeout = time.Second * 10
	}
	return &http.Client{Timeout: rancherClient.Opts.Timeout}
}

func (rancherClient *RancherBaseClientImpl) doDelete(url string) error {
	client := rancherClient.newHttpClient()
	req, err := http.NewRequest("DELETE", url, nil)
	if err != nil {
		return err
	}

	rancherClient.setupRequest(req)

	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	io.Copy(ioutil.Discard, resp.Body)

	if resp.StatusCode >= 300 {
		return newApiError(resp, url)
	}

	return nil
}

func (rancherClient *RancherBaseClientImpl) Websocket(url string, headers map[string][]string) (*websocket.Conn, *http.Response, error) {
	return dialer.Dial(url, http.Header(headers))
}

func (rancherClient *RancherBaseClientImpl) doGet(url string, opts *ListOpts, respObject interface{}) error {
	if opts == nil {
		opts = NewListOpts()
	}
	url, err := appendFilters(url, opts.Filters)
	if err != nil {
		return err
	}

	if debug {
		fmt.Println("GET " + url)
	}

	client := rancherClient.newHttpClient()
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return err
	}

	rancherClient.setupRequest(req)

	resp, err := client.Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		return newApiError(resp, url)
	}

	byteContent, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if debug {
		fmt.Println("Response <= " + string(byteContent))
	}

	if err := json.Unmarshal(byteContent, respObject); err != nil {
		return errors.Wrap(err, fmt.Sprintf("Failed to parse: %s", byteContent))
	}

	return nil
}

func (rancherClient *RancherBaseClientImpl) List(schemaType string, opts *ListOpts, respObject interface{}) error {
	return rancherClient.doList(schemaType, opts, respObject)
}

func (rancherClient *RancherBaseClientImpl) doList(schemaType string, opts *ListOpts, respObject interface{}) error {
	schema, ok := rancherClient.Types[schemaType]
	if !ok {
		return errors.New("Unknown schema type [" + schemaType + "]")
	}

	if !contains(schema.CollectionMethods, "GET") {
		return errors.New("Resource type [" + schemaType + "] is not listable")
	}

	collectionUrl, ok := schema.Links[COLLECTION]
	if !ok {
		return errors.New("Failed to find collection URL for [" + schemaType + "]")
	}

	return rancherClient.doGet(collectionUrl, opts, respObject)
}

func (rancherClient *RancherBaseClientImpl) Post(url string, createObj interface{}, respObject interface{}) error {
	return rancherClient.doModify("POST", url, createObj, respObject)
}

func (rancherClient *RancherBaseClientImpl) GetLink(resource Resource, link string, respObject interface{}) error {
	url := resource.Links[link]
	if url == "" {
		return fmt.Errorf("Failed to find link: %s", link)
	}

	return rancherClient.doGet(url, &ListOpts{}, respObject)
}

func (rancherClient *RancherBaseClientImpl) doModify(method string, url string, createObj interface{}, respObject interface{}) error {
	bodyContent, err := json.Marshal(createObj)
	if err != nil {
		return err
	}

	if debug {
		fmt.Println(method + " " + url)
		fmt.Println("Request => " + string(bodyContent))
	}

	client := rancherClient.newHttpClient()
	req, err := http.NewRequest(method, url, bytes.NewBuffer(bodyContent))
	if err != nil {
		return err
	}

	rancherClient.setupRequest(req)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Content-Length", string(len(bodyContent)))

	resp, err := client.Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		return newApiError(resp, url)
	}

	byteContent, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if len(byteContent) > 0 {
		if debug {
			fmt.Println("Response <= " + string(byteContent))
		}
		return json.Unmarshal(byteContent, respObject)
	}

	return nil
}

func (rancherClient *RancherBaseClientImpl) Create(schemaType string, createObj interface{}, respObject interface{}) error {
	return rancherClient.doCreate(schemaType, createObj, respObject)
}

func (rancherClient *RancherBaseClientImpl) doCreate(schemaType string, createObj interface{}, respObject interface{}) error {
	if createObj == nil {
		createObj = map[string]string{}
	}
	if respObject == nil {
		respObject = &map[string]interface{}{}
	}
	schema, ok := rancherClient.Types[schemaType]
	if !ok {
		return errors.New("Unknown schema type [" + schemaType + "]")
	}

	if !contains(schema.CollectionMethods, "POST") {
		return errors.New("Resource type [" + schemaType + "] is not creatable")
	}

	var collectionUrl string
	collectionUrl, ok = schema.Links[COLLECTION]
	if !ok {
		// return errors.New("Failed to find collection URL for [" + schemaType + "]")
		// This is a hack to address https://github.com/rancher/cattle/issues/254
		re := regexp.MustCompile("schemas.*")
		collectionUrl = re.ReplaceAllString(schema.Links[SELF], schema.PluralName)
	}

	return rancherClient.doModify("POST", collectionUrl, createObj, respObject)
}

func (rancherClient *RancherBaseClientImpl) Update(schemaType string, existing *Resource, updates interface{}, respObject interface{}) error {
	return rancherClient.doUpdate(schemaType, existing, updates, respObject)
}

func (rancherClient *RancherBaseClientImpl) doUpdate(schemaType string, existing *Resource, updates interface{}, respObject interface{}) error {
	if existing == nil {
		return errors.New("Existing object is nil")
	}

	selfUrl, ok := existing.Links[SELF]
	if !ok {
		return errors.New(fmt.Sprintf("Failed to find self URL of [%v]", existing))
	}

	if updates == nil {
		updates = map[string]string{}
	}

	if respObject == nil {
		respObject = &map[string]interface{}{}
	}

	schema, ok := rancherClient.Types[schemaType]
	if !ok {
		return errors.New("Unknown schema type [" + schemaType + "]")
	}

	if !contains(schema.ResourceMethods, "PUT") {
		return errors.New("Resource type [" + schemaType + "] is not updatable")
	}

	return rancherClient.doModify("PUT", selfUrl, updates, respObject)
}

func (rancherClient *RancherBaseClientImpl) ById(schemaType string, id string, respObject interface{}) error {
	return rancherClient.doById(schemaType, id, respObject)
}

func (rancherClient *RancherBaseClientImpl) doById(schemaType string, id string, respObject interface{}) error {
	schema, ok := rancherClient.Types[schemaType]
	if !ok {
		return errors.New("Unknown schema type [" + schemaType + "]")
	}

	if !contains(schema.ResourceMethods, "GET") {
		return errors.New("Resource type [" + schemaType + "] can not be looked up by ID")
	}

	collectionUrl, ok := schema.Links[COLLECTION]
	if !ok {
		return errors.New("Failed to find collection URL for [" + schemaType + "]")
	}

	err := rancherClient.doGet(collectionUrl+"/"+id, nil, respObject)
	//TODO check for 404 and return nil, nil
	return err
}

func (rancherClient *RancherBaseClientImpl) Delete(existing *Resource) error {
	if existing == nil {
		return nil
	}
	return rancherClient.doResourceDelete(existing.Type, existing)
}

func (rancherClient *RancherBaseClientImpl) doResourceDelete(schemaType string, existing *Resource) error {
	schema, ok := rancherClient.Types[schemaType]
	if !ok {
		return errors.New("Unknown schema type [" + schemaType + "]")
	}

	if !contains(schema.ResourceMethods, "DELETE") {
		return errors.New("Resource type [" + schemaType + "] can not be deleted")
	}

	selfUrl, ok := existing.Links[SELF]
	if !ok {
		return errors.New(fmt.Sprintf("Failed to find self URL of [%v]", existing))
	}

	return rancherClient.doDelete(selfUrl)
}

func (rancherClient *RancherBaseClientImpl) Reload(existing *Resource, output interface{}) error {
	selfUrl, ok := existing.Links[SELF]
	if !ok {
		return errors.New(fmt.Sprintf("Failed to find self URL of [%v]", existing))
	}

	return rancherClient.doGet(selfUrl, NewListOpts(), output)
}

func (rancherClient *RancherBaseClientImpl) Action(schemaType string, action string,
	existing *Resource, inputObject, respObject interface{}) error {
	return rancherClient.doAction(schemaType, action, existing, inputObject, respObject)
}

func (rancherClient *RancherBaseClientImpl) doAction(schemaType string, action string,
	existing *Resource, inputObject, respObject interface{}) error {

	if existing == nil {
		return errors.New("Existing object is nil")
	}

	actionUrl, ok := existing.Actions[action]
	if !ok {
		return errors.New(fmt.Sprintf("Action [%v] not available on [%v]", action, existing))
	}

	_, ok = rancherClient.Types[schemaType]
	if !ok {
		return errors.New("Unknown schema type [" + schemaType + "]")
	}

	var input io.Reader

	if inputObject != nil {
		bodyContent, err := json.Marshal(inputObject)
		if err != nil {
			return err
		}
		if debug {
			fmt.Println("Request => " + string(bodyContent))
		}
		input = bytes.NewBuffer(bodyContent)
	}

	client := rancherClient.newHttpClient()
	req, err := http.NewRequest("POST", actionUrl, input)
	if err != nil {
		return err
	}

	rancherClient.setupRequest(req)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Content-Length", "0")

	resp, err := client.Do(req)
	if err != nil {
		return err
	}

	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		return newApiError(resp, actionUrl)
	}

	byteContent, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if debug {
		fmt.Println("Response <= " + string(byteContent))
	}

	return json.Unmarshal(byteContent, respObject)
}

func init() {
	debug = os.Getenv("RANCHER_CLIENT_DEBUG") == "true"
	if debug {
		fmt.Println("Rancher client debug on")
	}
}
