package qingcloud

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"sync"
	"time"

	"github.com/hashicorp/go-cleanhttp"
)

// Client 客户端
type Client struct {
	zone            string
	accessKeyID     string
	secretAccessKey string
	params          Params
	commonParams    Params
	l               sync.Mutex
	httpCli         *http.Client
}

// NewClient 创建新的客户端
func NewClient() *Client {
	return &Client{
		params:       Params{},
		commonParams: Params{},
		httpCli:      cleanhttp.DefaultPooledClient(),
	}
}

// ConnectToZone 建立连接
func (c *Client) ConnectToZone(zone string, id string, secret string) {
	c.zone = zone
	c.accessKeyID = id
	c.secretAccessKey = secret
	c.commonParams = []*Param{
		{"version", 1},
		{"signature_method", "HmacSHA256"},
		{"signature_version", 1},
		{"access_key_id", id},
		{"zone", zone},
	}

}

func (c *Client) addAction(action string) {
	c.params = append(c.params, &Param{
		"action", action,
	})

}

func (c *Client) addTimeStamp() {
	c.params = append(c.params, &Param{
		"time_stamp", time.Now().UTC().Format("2006-01-02T15:04:05Z"),
	})
}

func (c *Client) getURL(httpMethod string) (string, string) {
	for i := range c.commonParams {
		c.params = append(c.params, c.commonParams[i])
	}
	sortParamsByKey(c.params)
	urlEscapeParams(c.params)
	url := generateURLByParams(c.params)
	return url, genSignature(genSignatureURL(httpMethod, `/iaas/`, url), c.secretAccessKey)
}

// Get 获取数据
func (c *Client) Get(action string, params Params, response interface{}) error {
	c.l.Lock()
	defer c.l.Unlock()
	result, err := c.get(action, params)
	if err != nil {
		return fmt.Errorf("Get Error %s , %#v   %s", err, params, string(result))
	}

	var errCode CommonResponse
	err = json.Unmarshal(result, &errCode)
	if err != nil {
		return fmt.Errorf("Get Error Unmashl %s , %#v   %s", err, params, string(result))
	}

	if errCode.RetCode != 0 {
		return errors.New(errCode.Message)
	}

	err = json.Unmarshal(result, &response)
	if err != nil {
		return fmt.Errorf("Get Error Unmashl to Response %s , %#v   %s", err, params, string(result))
	}

	return nil
}

// Post 发送数据
func (c *Client) Post(action string, params Params, response interface{}) error {
	c.l.Lock()
	defer c.l.Unlock()
	result, err := c.post(action, params)
	if err != nil {
		return err
	}
	var errCode CommonResponse
	json.Unmarshal(result, &errCode)
	if errCode.RetCode != 0 {
		return errors.New(errCode.Message)
	}

	err = json.Unmarshal(result, &response)
	if err != nil {
		return err
	}
	return nil
}

// TODO: fix this
func (c *Client) post(action string, params Params) ([]byte, error) {
	var p []*Param
	c.params = p

	for i := range params {
		c.params = append(c.params, params[i])
	}
	c.addAction(action)
	c.addTimeStamp()

	_url, _sig := c.getURL("Post")
	url := fmt.Sprintf("https://api.qingcloud.com/iaas/?%v&signature=%v", _url, _sig)

	res, err := c.httpCli.Post(url, "application/json;utf-8", nil)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	return ioutil.ReadAll(res.Body)
}

func (c *Client) get(action string, params Params) ([]byte, error) {
	var _p []*Param
	c.params = _p

	for i := range params {
		c.params = append(c.params, params[i])
	}

	c.addAction(action)
	c.addTimeStamp()

	_url, _sig := c.getURL("GET")
	url := fmt.Sprintf("https://api.qingcloud.com/iaas/?%v&signature=%v", _url, _sig)
	res, err := c.httpCli.Get(url)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	return ioutil.ReadAll(res.Body)
}
