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

	// TemplateAPIPrefix is a partial path to the HTTP endpoint.
	TemplateAPIPrefix = "/templates"

	// ErrNoSuchTemplate is the error returned when the template does not exist.
	ErrNoSuchTemplate = errors.New("no such template")

	// ErrTemplateInUse is the error returned when the template requested to be removed is still in use.
	ErrTemplateInUse = errors.New("template in use and cannot be removed")
)

// TemplateList returns the list of available templates.
func (c *Client) TemplateList(opts types.ListOptions) ([]types.Template, error) {
	path := TemplateAPIPrefix + "?" + queryString(opts)
	resp, err := c.do("GET", path, doOptions{context: opts.Context})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	var templates []types.Template
	if err := json.NewDecoder(resp.Body).Decode(&templates); err != nil {
		return nil, err
	}
	return templates, nil
}

// TemplateCreate creates a template on the server and returns the new object.
func (c *Client) TemplateCreate(opts types.TemplateCreateOptions) (string, error) {
	resp, err := c.do("POST", TemplateAPIPrefix, doOptions{
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

// Template returns a template by its reference.
func (c *Client) Template(ref string) (*types.Template, error) {
	resp, err := c.do("GET", TemplateAPIPrefix+"/"+ref, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok && e.Status == http.StatusNotFound {
			return nil, ErrNoSuchTemplate
		}
		return nil, err
	}
	defer resp.Body.Close()
	var template types.Template
	if err := json.NewDecoder(resp.Body).Decode(&template); err != nil {
		return nil, err
	}
	return &template, nil
}

// TemplateDelete removes a template by its reference.
func (c *Client) TemplateDelete(ref string) error {
	resp, err := c.do("DELETE", TemplateAPIPrefix+"/"+ref, doOptions{})
	if err != nil {
		if e, ok := err.(*Error); ok {
			if e.Status == http.StatusNotFound {
				return ErrNoSuchTemplate
			}
			if e.Status == http.StatusConflict {
				return ErrTemplateInUse
			}
		}
		return nil
	}
	defer resp.Body.Close()
	return nil
}
