package api

import (
	"fmt"

	"github.com/fatih/structs"
	"github.com/mitchellh/mapstructure"
)

func (c *Sys) ListAuth() (map[string]*AuthMount, error) {
	r := c.c.NewRequest("GET", "/v1/sys/auth")
	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result map[string]interface{}
	err = resp.DecodeJSON(&result)
	if err != nil {
		return nil, err
	}

	mounts := map[string]*AuthMount{}
	for k, v := range result {
		switch v.(type) {
		case map[string]interface{}:
		default:
			continue
		}
		var res AuthMount
		err = mapstructure.Decode(v, &res)
		if err != nil {
			return nil, err
		}
		// Not a mount, some other api.Secret data
		if res.Type == "" {
			continue
		}
		mounts[k] = &res
	}

	return mounts, nil
}

// DEPRECATED: Use EnableAuthWithOptions instead
func (c *Sys) EnableAuth(path, authType, desc string) error {
	return c.EnableAuthWithOptions(path, &EnableAuthOptions{
		Type:        authType,
		Description: desc,
	})
}

func (c *Sys) EnableAuthWithOptions(path string, options *EnableAuthOptions) error {
	body := structs.Map(options)

	r := c.c.NewRequest("POST", fmt.Sprintf("/v1/sys/auth/%s", path))
	if err := r.SetJSONBody(body); err != nil {
		return err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return nil
}

func (c *Sys) DisableAuth(path string) error {
	r := c.c.NewRequest("DELETE", fmt.Sprintf("/v1/sys/auth/%s", path))
	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}

// Structures for the requests/resposne are all down here. They aren't
// individually documentd because the map almost directly to the raw HTTP API
// documentation. Please refer to that documentation for more details.

type EnableAuthOptions struct {
	Type        string `json:"type" structs:"type"`
	Description string `json:"description" structs:"description"`
	Local       bool   `json:"local" structs:"local"`
	PluginName  string `json:"plugin_name,omitempty" structs:"plugin_name,omitempty" mapstructure:"plugin_name"`
}

type AuthMount struct {
	Type        string           `json:"type" structs:"type" mapstructure:"type"`
	Description string           `json:"description" structs:"description" mapstructure:"description"`
	Accessor    string           `json:"accessor" structs:"accessor" mapstructure:"accessor"`
	Config      AuthConfigOutput `json:"config" structs:"config" mapstructure:"config"`
	Local       bool             `json:"local" structs:"local" mapstructure:"local"`
}

type AuthConfigOutput struct {
	DefaultLeaseTTL int    `json:"default_lease_ttl" structs:"default_lease_ttl" mapstructure:"default_lease_ttl"`
	MaxLeaseTTL     int    `json:"max_lease_ttl" structs:"max_lease_ttl" mapstructure:"max_lease_ttl"`
	PluginName      string `json:"plugin_name,omitempty" structs:"plugin_name,omitempty" mapstructure:"plugin_name"`
}
