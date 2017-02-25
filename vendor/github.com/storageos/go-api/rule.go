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

	// RuleAPIPrefix is a partial path to the HTTP endpoint.
	RuleAPIPrefix = "rules"

	// ErrNoSuchRule is the error returned when the rule does not exist.
	ErrNoSuchRule = errors.New("no such rule")

	// ErrRuleInUse is the error returned when the rule requested to be removed is still in use.
	ErrRuleInUse = errors.New("rule in use and cannot be removed")
)

// RuleList returns the list of available rules.
func (c *Client) RuleList(opts types.ListOptions) ([]types.Rule, error) {
	path := RuleAPIPrefix + "?" + queryString(opts)
	resp, err := c.do("GET", path, doOptions{context: opts.Context})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var rules []types.Rule
	if err := json.NewDecoder(resp.Body).Decode(&rules); err != nil {
		return nil, err
	}
	return rules, nil
}

// RuleCreate creates a rule on the server and returns its unique id.
func (c *Client) RuleCreate(opts types.RuleCreateOptions) (string, error) {
	resp, err := c.do("POST", RuleAPIPrefix, doOptions{
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

// Rule returns a rule by its reference.
func (c *Client) Rule(ref string) (*types.Rule, error) {
	resp, err := c.do("GET", RuleAPIPrefix+"/"+ref, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, ErrNoSuchRule
		}
		return nil, err
	}
	defer resp.Body.Close()
	var rule types.Rule
	if err := json.NewDecoder(resp.Body).Decode(&rule); err != nil {
		return nil, err
	}
	return &rule, nil
}

// RuleDelete removes a rule by its reference.
func (c *Client) RuleDelete(ref string) error {
	resp, err := c.do("DELETE", RuleAPIPrefix+"/"+ref, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok {
			if e.Status == http.StatusNotFound {
				return ErrNoSuchRule
			}
			if e.Status == http.StatusConflict {
				return ErrRuleInUse
			}
		}
		return nil
	}
	defer resp.Body.Close()
	return nil
}
