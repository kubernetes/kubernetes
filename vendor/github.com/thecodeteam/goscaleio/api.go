package goscaleio

import (
	"bytes"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"regexp"
	"strings"
	"time"

	types "github.com/thecodeteam/goscaleio/types/v1"
	log "github.com/sirupsen/logrus"
)

type Client struct {
	Token         string
	SIOEndpoint   url.URL
	Http          http.Client
	Insecure      string
	ShowBody      bool
	configConnect *ConfigConnect
}

type Cluster struct {
}

type ConfigConnect struct {
	Endpoint string
	Version  string
	Username string
	Password string
}

type ClientPersistent struct {
	configConnect *ConfigConnect
	client        *Client
}

func (client *Client) getVersion() (string, error) {
	endpoint := client.SIOEndpoint
	endpoint.Path = "/api/version"

	req := client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth("", client.Token)

	resp, err := client.retryCheckResp(&client.Http, req)
	if err != nil {
		return "", fmt.Errorf("problem getting response: %v", err)
	}
	defer resp.Body.Close()

	bs, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", errors.New("error reading body")
	}

	version := string(bs)

	if client.ShowBody {
		log.WithField("body", version).Debug(
			"printing version message body")
	}

	version = strings.TrimRight(version, `"`)
	version = strings.TrimLeft(version, `"`)

	versionRX := regexp.MustCompile(`^(\d+?\.\d+?).*$`)
	if m := versionRX.FindStringSubmatch(version); len(m) > 0 {
		return m[1], nil
	}
	return version, nil
}

func (client *Client) updateVersion() error {

	version, err := client.getVersion()
	if err != nil {
		return err
	}
	client.configConnect.Version = version

	return nil
}

func (client *Client) Authenticate(configConnect *ConfigConnect) (Cluster, error) {

	configConnect.Version = client.configConnect.Version
	client.configConnect = configConnect

	endpoint := client.SIOEndpoint
	endpoint.Path += "/login"

	req := client.NewRequest(map[string]string{}, "GET", endpoint, nil)
	req.SetBasicAuth(configConnect.Username, configConnect.Password)

	httpClient := &client.Http
	resp, errBody, err := client.checkResp(httpClient.Do(req))
	if errBody == nil && err != nil {
		return Cluster{}, err
	} else if errBody != nil && err != nil {
		if resp == nil {
			return Cluster{}, errors.New("Problem getting response from endpoint")
		}
		return Cluster{}, errors.New(errBody.Message)
	}
	defer resp.Body.Close()

	bs, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return Cluster{}, errors.New("error reading body")
	}

	token := string(bs)

	if client.ShowBody {
		log.WithField("body", token).Debug(
			"printing authentication message body")
	}

	token = strings.TrimRight(token, `"`)
	token = strings.TrimLeft(token, `"`)
	client.Token = token

	if client.configConnect.Version == "" {
		err = client.updateVersion()
		if err != nil {
			return Cluster{}, errors.New("error getting version of ScaleIO")
		}
	}

	return Cluster{}, nil
}

//https://github.com/chrislusf/teeproxy/blob/master/teeproxy.go
type nopCloser struct {
	io.Reader
}

func (nopCloser) Close() error { return nil }

func DuplicateRequest(request *http.Request) (request1 *http.Request, request2 *http.Request) {
	request1 = &http.Request{
		Method:        request.Method,
		URL:           request.URL,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        request.Header,
		Host:          request.Host,
		ContentLength: request.ContentLength,
	}
	request2 = &http.Request{
		Method:        request.Method,
		URL:           request.URL,
		Proto:         "HTTP/1.1",
		ProtoMajor:    1,
		ProtoMinor:    1,
		Header:        request.Header,
		Host:          request.Host,
		ContentLength: request.ContentLength,
	}

	if request.Body != nil {
		b1 := new(bytes.Buffer)
		b2 := new(bytes.Buffer)
		w := io.MultiWriter(b1, b2)
		io.Copy(w, request.Body)
		request1.Body = nopCloser{b1}
		request2.Body = nopCloser{b2}

		defer request.Body.Close()
	}

	return
}

func (client *Client) retryCheckResp(httpClient *http.Client, req *http.Request) (*http.Response, error) {

	req1, req2 := DuplicateRequest(req)
	resp, errBody, err := client.checkResp(httpClient.Do(req1))
	if errBody == nil && err != nil {
		return &http.Response{}, err
	} else if errBody != nil && err != nil {
		if resp == nil {
			return nil, errors.New("Problem getting response from endpoint")
		}

		if resp.StatusCode == 401 && errBody.MajorErrorCode == 0 {
			_, err := client.Authenticate(client.configConnect)
			if err != nil {
				return nil, fmt.Errorf("Error re-authenticating: %s", err)
			}

			ioutil.ReadAll(resp.Body)
			resp.Body.Close()

			req2.SetBasicAuth("", client.Token)
			resp, errBody, err = client.checkResp(httpClient.Do(req2))
			if err != nil {
				return &http.Response{}, errors.New(errBody.Message)
			}
		} else {
			return &http.Response{}, errors.New(errBody.Message)
		}
	}

	return resp, nil
}

func (client *Client) checkResp(resp *http.Response, err error) (*http.Response, *types.Error, error) {
	if err != nil {
		return resp, &types.Error{}, err
	}

	switch i := resp.StatusCode; {
	// Valid request, return the response.
	case i == 200 || i == 201 || i == 202 || i == 204:
		return resp, &types.Error{}, nil
	// Invalid request, parse the XML error returned and return it.
	case i == 400 || i == 401 || i == 403 || i == 404 || i == 405 || i == 406 || i == 409 || i == 415 || i == 500 || i == 503 || i == 504:
		errBody, err := client.parseErr(resp)
		return resp, errBody, err
	// Unhandled response.
	default:
		return nil, &types.Error{}, fmt.Errorf("unhandled API response, please report this issue, status code: %s", resp.Status)
	}
}

func (client *Client) decodeBody(resp *http.Response, out interface{}) error {

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return err
	}

	if client.ShowBody {
		var prettyJSON bytes.Buffer
		_ = json.Indent(&prettyJSON, body, "", "  ")
		log.WithField("body", prettyJSON.String()).Debug(
			"print decoded body")
	}

	if err = json.Unmarshal(body, &out); err != nil {
		return err
	}

	return nil
}

func (client *Client) parseErr(resp *http.Response) (*types.Error, error) {

	errBody := new(types.Error)

	// if there was an error decoding the body, just return that
	if err := client.decodeBody(resp, errBody); err != nil {
		return &types.Error{}, fmt.Errorf("error parsing error body for non-200 request: %s", err)
	}

	return errBody, fmt.Errorf("API (%d) Error: %d: %s", resp.StatusCode, errBody.MajorErrorCode, errBody.Message)
}

func (c *Client) NewRequest(params map[string]string, method string, u url.URL, body io.Reader) *http.Request {

	if log.GetLevel() == log.DebugLevel && c.ShowBody && body != nil {
		buf := new(bytes.Buffer)
		buf.ReadFrom(body)
		log.WithField("body", buf.String()).Debug("print new request body")
	}

	p := url.Values{}

	for k, v := range params {
		p.Add(k, v)
	}

	u.RawQuery = p.Encode()

	req, _ := http.NewRequest(method, u.String(), body)

	return req

}

func NewClient() (client *Client, err error) {
	return NewClientWithArgs(
		os.Getenv("GOSCALEIO_ENDPOINT"),
		os.Getenv("GOSCALEIO_VERSION"),
		os.Getenv("GOSCALEIO_INSECURE") == "true",
		os.Getenv("GOSCALEIO_USECERTS") == "true")
}

func NewClientWithArgs(
	endpoint string,
	version string,
	insecure,
	useCerts bool) (client *Client, err error) {

	fields := map[string]interface{}{
		"endpoint": endpoint,
		"insecure": insecure,
		"useCerts": useCerts,
		"version":  version,
	}

	var uri *url.URL

	if endpoint != "" {
		uri, err = url.ParseRequestURI(endpoint)
		if err != nil {
			return &Client{},
				withFieldsE(fields, "error parsing endpoint", err)
		}
	} else {
		return &Client{},
			withFields(fields, "endpoint is required")
	}

	client = &Client{
		SIOEndpoint: *uri,
		Http: http.Client{
			Transport: &http.Transport{
				TLSHandshakeTimeout: 120 * time.Second,
				TLSClientConfig: &tls.Config{
					InsecureSkipVerify: insecure,
				},
			},
		},
	}

	if useCerts {
		pool := x509.NewCertPool()
		pool.AppendCertsFromPEM(pemCerts)

		client.Http.Transport = &http.Transport{
			TLSHandshakeTimeout: 120 * time.Second,
			TLSClientConfig: &tls.Config{
				RootCAs:            pool,
				InsecureSkipVerify: insecure,
			},
		}
	}

	client.configConnect = &ConfigConnect{
		Version: version,
	}

	return client, nil
}

func GetLink(links []*types.Link, rel string) (*types.Link, error) {
	for _, link := range links {
		if link.Rel == rel {
			return link, nil
		}
	}

	return &types.Link{}, errors.New("Couldn't find link")
}

func withFields(fields map[string]interface{}, message string) error {
	return withFieldsE(fields, message, nil)
}

func withFieldsE(
	fields map[string]interface{}, message string, inner error) error {

	if fields == nil {
		fields = make(map[string]interface{})
	}

	if inner != nil {
		fields["inner"] = inner
	}

	x := 0
	l := len(fields)

	var b bytes.Buffer
	for k, v := range fields {
		if x < l-1 {
			b.WriteString(fmt.Sprintf("%s=%v,", k, v))
		} else {
			b.WriteString(fmt.Sprintf("%s=%v", k, v))
		}
		x = x + 1
	}

	return newf("%s %s", message, b.String())
}

func newf(format string, a ...interface{}) error {
	return errors.New(fmt.Sprintf(format, a))
}
