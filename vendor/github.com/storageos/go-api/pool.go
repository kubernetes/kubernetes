package storageos

import (
	"encoding/json"
	"errors"
	"io/ioutil"
	"net/http"
	"strconv"

	"github.com/storageos/go-api/types"
)

var (

	// PoolAPIPrefix is a partial path to the HTTP endpoint.
	PoolAPIPrefix = "pools"

	// ErrNoSuchPool is the error returned when the pool does not exist.
	ErrNoSuchPool = errors.New("no such pool")

	// ErrPoolInUse is the error returned when the pool requested to be removed is still in use.
	ErrPoolInUse = errors.New("pool in use and cannot be removed")
)

// PoolList returns the list of available pools.
func (c *Client) PoolList(opts types.ListOptions) ([]types.Pool, error) {
	path := PoolAPIPrefix + "?" + queryString(opts)
	resp, err := c.do("GET", path, doOptions{context: opts.Context})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var pools []types.Pool
	if err := json.NewDecoder(resp.Body).Decode(&pools); err != nil {
		return nil, err
	}
	return pools, nil
}

// PoolCreate creates a pool on the server and returns its unique id.
func (c *Client) PoolCreate(opts types.PoolCreateOptions) (string, error) {
	resp, err := c.do("POST", PoolAPIPrefix, doOptions{
		data:    opts,
		context: opts.Context,
	})
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	out, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	return strconv.Unquote(string(out))
}

// Pool returns a pool by its reference.
func (c *Client) Pool(ref string) (*types.Pool, error) {
	resp, err := c.do("GET", PoolAPIPrefix+"/"+ref, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, ErrNoSuchPool
		}
		return nil, err
	}
	defer resp.Body.Close()
	var pool types.Pool
	if err := json.NewDecoder(resp.Body).Decode(&pool); err != nil {
		return nil, err
	}
	return &pool, nil
}

// PoolDelete removes a pool by its reference.
func (c *Client) PoolDelete(ref string) error {
	resp, err := c.do("DELETE", PoolAPIPrefix+"/"+ref, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok {
			if e.Status == http.StatusNotFound {
				return ErrNoSuchPool
			}
			if e.Status == http.StatusConflict {
				return ErrPoolInUse
			}
		}
		return nil
	}
	defer resp.Body.Close()
	return nil
}
