package storageos

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"

	"github.com/storageos/go-api/types"
)

var (
	// PolicyAPIPrefix is a partial path to the HTTP endpoint.
	PolicyAPIPrefix = "policies"
	// ErrNoSuchPolicy is the error returned when the policy does not exist.
	ErrNoSuchPolicy = errors.New("no such policy")
)

// nopMarshaler is an alias to a []byte that implements json.Marshaler
// it bypasses the base64 encoded string representation that json will give byte slices.
// It should only be used to wrap []byte types containing pre-rendered valid json that will later
// (out of the caller's control) be run through json.Marshal
type nopMarshaler []byte

func (n *nopMarshaler) MarshalJSON() ([]byte, error) {
	return *n, nil
}

// PolicyCreate creates a policy on the server.
func (c *Client) PolicyCreate(ctx context.Context, jsonl []byte) error {
	nopm := nopMarshaler(jsonl)
	_, err := c.do("POST", PolicyAPIPrefix, doOptions{
		data:    &nopm,
		context: ctx,
		headers: map[string]string{"Content-Type": "application/x-jsonlines"},
	})
	return err
}

// Policy returns a policy on the server by ID.
func (c *Client) Policy(id string) (*types.Policy, error) {
	path := fmt.Sprintf("%s/%s", PolicyAPIPrefix, id)
	resp, err := c.do("GET", path, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, ErrNoSuchPolicy
		}
		return nil, err
	}
	defer resp.Body.Close()

	var policy *types.Policy
	if err := json.NewDecoder(resp.Body).Decode(&policy); err != nil {
		return nil, err
	}
	return policy, nil
}

// PolicyList returns the list of policies on the server.
func (c *Client) PolicyList(opts types.ListOptions) (types.PolicySet, error) {
	listOpts := doOptions{
		fieldSelector: opts.FieldSelector,
		labelSelector: opts.LabelSelector,
		namespace:     opts.Namespace,
		context:       opts.Context,
	}

	if opts.LabelSelector != "" {
		query := url.Values{}
		query.Add("labelSelector", opts.LabelSelector)
		listOpts.values = query
	}

	resp, err := c.do("GET", PolicyAPIPrefix, listOpts)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var policies types.PolicySet
	if err := json.NewDecoder(resp.Body).Decode(&policies); err != nil {
		return nil, err
	}
	return policies, nil
}

// PolicyDelete deletes a policy on the server by ID.
func (c *Client) PolicyDelete(opts types.DeleteOptions) error {
	resp, err := c.do("DELETE", PolicyAPIPrefix+"/"+opts.Name, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok {
			if e.Status == http.StatusNotFound {
				return ErrNoSuchPolicy
			}
		}
		return err
	}
	defer resp.Body.Close()
	return nil
}
