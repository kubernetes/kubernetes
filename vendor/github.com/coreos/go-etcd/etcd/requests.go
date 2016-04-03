package etcd

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"path"
	"strings"
	"sync"
	"time"
)

// Errors introduced by handling requests
var (
	ErrRequestCancelled = errors.New("sending request is cancelled")
)

type RawRequest struct {
	Method       string
	RelativePath string
	Values       url.Values
	Cancel       <-chan bool
}

// NewRawRequest returns a new RawRequest
func NewRawRequest(method, relativePath string, values url.Values, cancel <-chan bool) *RawRequest {
	return &RawRequest{
		Method:       method,
		RelativePath: relativePath,
		Values:       values,
		Cancel:       cancel,
	}
}

// getCancelable issues a cancelable GET request
func (c *Client) getCancelable(key string, options Options,
	cancel <-chan bool) (*RawResponse, error) {
	logger.Debugf("get %s [%s]", key, c.cluster.pick())
	p := keyToPath(key)

	str, err := options.toParameters(VALID_GET_OPTIONS)
	if err != nil {
		return nil, err
	}
	p += str

	req := NewRawRequest("GET", p, nil, cancel)
	resp, err := c.SendRequest(req)

	if err != nil {
		return nil, err
	}

	return resp, nil
}

// get issues a GET request
func (c *Client) get(key string, options Options) (*RawResponse, error) {
	return c.getCancelable(key, options, nil)
}

// put issues a PUT request
func (c *Client) put(key string, value string, ttl uint64,
	options Options) (*RawResponse, error) {

	logger.Debugf("put %s, %s, ttl: %d, [%s]", key, value, ttl, c.cluster.pick())
	p := keyToPath(key)

	str, err := options.toParameters(VALID_PUT_OPTIONS)
	if err != nil {
		return nil, err
	}
	p += str

	req := NewRawRequest("PUT", p, buildValues(value, ttl), nil)
	resp, err := c.SendRequest(req)

	if err != nil {
		return nil, err
	}

	return resp, nil
}

// post issues a POST request
func (c *Client) post(key string, value string, ttl uint64) (*RawResponse, error) {
	logger.Debugf("post %s, %s, ttl: %d, [%s]", key, value, ttl, c.cluster.pick())
	p := keyToPath(key)

	req := NewRawRequest("POST", p, buildValues(value, ttl), nil)
	resp, err := c.SendRequest(req)

	if err != nil {
		return nil, err
	}

	return resp, nil
}

// delete issues a DELETE request
func (c *Client) delete(key string, options Options) (*RawResponse, error) {
	logger.Debugf("delete %s [%s]", key, c.cluster.pick())
	p := keyToPath(key)

	str, err := options.toParameters(VALID_DELETE_OPTIONS)
	if err != nil {
		return nil, err
	}
	p += str

	req := NewRawRequest("DELETE", p, nil, nil)
	resp, err := c.SendRequest(req)

	if err != nil {
		return nil, err
	}

	return resp, nil
}

// SendRequest sends a HTTP request and returns a Response as defined by etcd
func (c *Client) SendRequest(rr *RawRequest) (*RawResponse, error) {
	var req *http.Request
	var resp *http.Response
	var httpPath string
	var err error
	var respBody []byte

	var numReqs = 1

	checkRetry := c.CheckRetry
	if checkRetry == nil {
		checkRetry = DefaultCheckRetry
	}

	cancelled := make(chan bool, 1)
	reqLock := new(sync.Mutex)

	if rr.Cancel != nil {
		cancelRoutine := make(chan bool)
		defer close(cancelRoutine)

		go func() {
			select {
			case <-rr.Cancel:
				cancelled <- true
				logger.Debug("send.request is cancelled")
			case <-cancelRoutine:
				return
			}

			// Repeat canceling request until this thread is stopped
			// because we have no idea about whether it succeeds.
			for {
				reqLock.Lock()
				c.httpClient.Transport.(*http.Transport).CancelRequest(req)
				reqLock.Unlock()

				select {
				case <-time.After(100 * time.Millisecond):
				case <-cancelRoutine:
					return
				}
			}
		}()
	}

	// If we connect to a follower and consistency is required, retry until
	// we connect to a leader
	sleep := 25 * time.Millisecond
	maxSleep := time.Second

	for attempt := 0; ; attempt++ {
		if attempt > 0 {
			select {
			case <-cancelled:
				return nil, ErrRequestCancelled
			case <-time.After(sleep):
				sleep = sleep * 2
				if sleep > maxSleep {
					sleep = maxSleep
				}
			}
		}

		logger.Debug("Connecting to etcd: attempt ", attempt+1, " for ", rr.RelativePath)

		// get httpPath if not set
		if httpPath == "" {
			httpPath = c.getHttpPath(rr.RelativePath)
		}

		// Return a cURL command if curlChan is set
		if c.cURLch != nil {
			command := fmt.Sprintf("curl -X %s %s", rr.Method, httpPath)
			for key, value := range rr.Values {
				command += fmt.Sprintf(" -d %s=%s", key, value[0])
			}
			if c.credentials != nil {
				command += fmt.Sprintf(" -u %s", c.credentials.username)
			}
			c.sendCURL(command)
		}

		logger.Debug("send.request.to ", httpPath, " | method ", rr.Method)

		req, err := func() (*http.Request, error) {
			reqLock.Lock()
			defer reqLock.Unlock()

			if rr.Values == nil {
				if req, err = http.NewRequest(rr.Method, httpPath, nil); err != nil {
					return nil, err
				}
			} else {
				body := strings.NewReader(rr.Values.Encode())
				if req, err = http.NewRequest(rr.Method, httpPath, body); err != nil {
					return nil, err
				}

				req.Header.Set("Content-Type",
					"application/x-www-form-urlencoded; param=value")
			}
			return req, nil
		}()

		if err != nil {
			return nil, err
		}

		if c.credentials != nil {
			req.SetBasicAuth(c.credentials.username, c.credentials.password)
		}

		resp, err = c.httpClient.Do(req)
		// clear previous httpPath
		httpPath = ""
		defer func() {
			if resp != nil {
				resp.Body.Close()
			}
		}()

		// If the request was cancelled, return ErrRequestCancelled directly
		select {
		case <-cancelled:
			return nil, ErrRequestCancelled
		default:
		}

		numReqs++

		// network error, change a machine!
		if err != nil {
			logger.Debug("network error: ", err.Error())
			lastResp := http.Response{}
			if checkErr := checkRetry(c.cluster, numReqs, lastResp, err); checkErr != nil {
				return nil, checkErr
			}

			c.cluster.failure()
			continue
		}

		// if there is no error, it should receive response
		logger.Debug("recv.response.from ", httpPath)

		if validHttpStatusCode[resp.StatusCode] {
			// try to read byte code and break the loop
			respBody, err = ioutil.ReadAll(resp.Body)
			if err == nil {
				logger.Debug("recv.success ", httpPath)
				break
			}
			// ReadAll error may be caused due to cancel request
			select {
			case <-cancelled:
				return nil, ErrRequestCancelled
			default:
			}

			if err == io.ErrUnexpectedEOF {
				// underlying connection was closed prematurely, probably by timeout
				// TODO: empty body or unexpectedEOF can cause http.Transport to get hosed;
				// this allows the client to detect that and take evasive action. Need
				// to revisit once code.google.com/p/go/issues/detail?id=8648 gets fixed.
				respBody = []byte{}
				break
			}
		}

		if resp.StatusCode == http.StatusTemporaryRedirect {
			u, err := resp.Location()

			if err != nil {
				logger.Warning(err)
			} else {
				// set httpPath for following redirection
				httpPath = u.String()
			}
			resp.Body.Close()
			continue
		}

		if checkErr := checkRetry(c.cluster, numReqs, *resp,
			errors.New("Unexpected HTTP status code")); checkErr != nil {
			return nil, checkErr
		}
		resp.Body.Close()
	}

	r := &RawResponse{
		StatusCode: resp.StatusCode,
		Body:       respBody,
		Header:     resp.Header,
	}

	return r, nil
}

// DefaultCheckRetry defines the retrying behaviour for bad HTTP requests
// If we have retried 2 * machine number, stop retrying.
// If status code is InternalServerError, sleep for 200ms.
func DefaultCheckRetry(cluster *Cluster, numReqs int, lastResp http.Response,
	err error) error {

	if numReqs > 2*len(cluster.Machines) {
		errStr := fmt.Sprintf("failed to propose on members %v twice [last error: %v]", cluster.Machines, err)
		return newError(ErrCodeEtcdNotReachable, errStr, 0)
	}

	if isEmptyResponse(lastResp) {
		// always retry if it failed to get response from one machine
		return nil
	}
	if !shouldRetry(lastResp) {
		body := []byte("nil")
		if lastResp.Body != nil {
			if b, err := ioutil.ReadAll(lastResp.Body); err == nil {
				body = b
			}
		}
		errStr := fmt.Sprintf("unhandled http status [%s] with body [%s]", http.StatusText(lastResp.StatusCode), body)
		return newError(ErrCodeUnhandledHTTPStatus, errStr, 0)
	}
	// sleep some time and expect leader election finish
	time.Sleep(time.Millisecond * 200)
	logger.Warning("bad response status code ", lastResp.StatusCode)
	return nil
}

func isEmptyResponse(r http.Response) bool { return r.StatusCode == 0 }

// shouldRetry returns whether the reponse deserves retry.
func shouldRetry(r http.Response) bool {
	// TODO: only retry when the cluster is in leader election
	// We cannot do it exactly because etcd doesn't support it well.
	return r.StatusCode == http.StatusInternalServerError
}

func (c *Client) getHttpPath(s ...string) string {
	fullPath := c.cluster.pick() + "/" + version
	for _, seg := range s {
		fullPath = fullPath + "/" + seg
	}
	return fullPath
}

// buildValues builds a url.Values map according to the given value and ttl
func buildValues(value string, ttl uint64) url.Values {
	v := url.Values{}

	if value != "" {
		v.Set("value", value)
	}

	if ttl > 0 {
		v.Set("ttl", fmt.Sprintf("%v", ttl))
	}

	return v
}

// convert key string to http path exclude version, including URL escaping
// for example: key[foo] -> path[keys/foo]
// key[/%z] -> path[keys/%25z]
// key[/] -> path[keys/]
func keyToPath(key string) string {
	// URL-escape our key, except for slashes
	p := strings.Replace(url.QueryEscape(path.Join("keys", key)), "%2F", "/", -1)

	// corner case: if key is "/" or "//" ect
	// path join will clear the tailing "/"
	// we need to add it back
	if p == "keys" {
		p = "keys/"
	}

	return p
}
