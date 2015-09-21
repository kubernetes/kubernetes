package etcd

import (
	"errors"
	"fmt"
	"io/ioutil"
	"math/rand"
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
	logger.Debugf("get %s [%s]", key, c.cluster.Leader)
	p := keyToPath(key)

	// If consistency level is set to STRONG, append
	// the `consistent` query string.
	if c.config.Consistency == STRONG_CONSISTENCY {
		options["consistent"] = true
	}

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

	logger.Debugf("put %s, %s, ttl: %d, [%s]", key, value, ttl, c.cluster.Leader)
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
	logger.Debugf("post %s, %s, ttl: %d, [%s]", key, value, ttl, c.cluster.Leader)
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
	logger.Debugf("delete %s [%s]", key, c.cluster.Leader)
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

		logger.Debug("Connecting to etcd: attempt", attempt+1, "for", rr.RelativePath)

		if rr.Method == "GET" && c.config.Consistency == WEAK_CONSISTENCY {
			// If it's a GET and consistency level is set to WEAK,
			// then use a random machine.
			httpPath = c.getHttpPath(true, rr.RelativePath)
		} else {
			// Else use the leader.
			httpPath = c.getHttpPath(false, rr.RelativePath)
		}

		// Return a cURL command if curlChan is set
		if c.cURLch != nil {
			command := fmt.Sprintf("curl -X %s %s", rr.Method, httpPath)
			for key, value := range rr.Values {
				command += fmt.Sprintf(" -d %s=%s", key, value[0])
			}
			c.sendCURL(command)
		}

		logger.Debug("send.request.to ", httpPath, " | method ", rr.Method)

		reqLock.Lock()
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
		reqLock.Unlock()

		resp, err = c.httpClient.Do(req)
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
			logger.Debug("network error:", err.Error())
			lastResp := http.Response{}
			if checkErr := checkRetry(c.cluster, numReqs, lastResp, err); checkErr != nil {
				return nil, checkErr
			}

			c.cluster.switchLeader(attempt % len(c.cluster.Machines))
			continue
		}

		// if there is no error, it should receive response
		logger.Debug("recv.response.from", httpPath)

		if validHttpStatusCode[resp.StatusCode] {
			// try to read byte code and break the loop
			respBody, err = ioutil.ReadAll(resp.Body)
			if err == nil {
				logger.Debug("recv.success.", httpPath)
				break
			}
			// ReadAll error may be caused due to cancel request
			select {
			case <-cancelled:
				return nil, ErrRequestCancelled
			default:
			}
		}

		// if resp is TemporaryRedirect, set the new leader and retry
		if resp.StatusCode == http.StatusTemporaryRedirect {
			u, err := resp.Location()

			if err != nil {
				logger.Warning(err)
			} else {
				// Update cluster leader based on redirect location
				// because it should point to the leader address
				c.cluster.updateLeaderFromURL(u)
				logger.Debug("recv.response.relocate", u.String())
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

	if numReqs >= 2*len(cluster.Machines) {
		return newError(ErrCodeEtcdNotReachable,
			"Tried to connect to each peer twice and failed", 0)
	}

	code := lastResp.StatusCode
	if code == http.StatusInternalServerError {
		time.Sleep(time.Millisecond * 200)

	}

	logger.Warning("bad response status code", code)
	return nil
}

func (c *Client) getHttpPath(random bool, s ...string) string {
	var machine string
	if random {
		machine = c.cluster.Machines[rand.Intn(len(c.cluster.Machines))]
	} else {
		machine = c.cluster.Leader
	}

	fullPath := machine + "/" + version
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

// convert key string to http path exclude version
// for example: key[foo] -> path[keys/foo]
// key[/] -> path[keys/]
func keyToPath(key string) string {
	p := path.Join("keys", key)

	// corner case: if key is "/" or "//" ect
	// path join will clear the tailing "/"
	// we need to add it back
	if p == "keys" {
		p = "keys/"
	}

	return p
}
