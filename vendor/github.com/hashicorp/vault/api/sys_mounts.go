package api

import (
	"fmt"

	"github.com/fatih/structs"
	"github.com/mitchellh/mapstructure"
)

func (c *Sys) ListMounts() (map[string]*MountOutput, error) {
	r := c.c.NewRequest("GET", "/v1/sys/mounts")
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

	mounts := map[string]*MountOutput{}
	for k, v := range result {
		switch v.(type) {
		case map[string]interface{}:
		default:
			continue
		}
		var res MountOutput
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

func (c *Sys) Mount(path string, mountInfo *MountInput) error {
	body := structs.Map(mountInfo)

	r := c.c.NewRequest("POST", fmt.Sprintf("/v1/sys/mounts/%s", path))
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

func (c *Sys) Unmount(path string) error {
	r := c.c.NewRequest("DELETE", fmt.Sprintf("/v1/sys/mounts/%s", path))
	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}

func (c *Sys) Remount(from, to string) error {
	body := map[string]interface{}{
		"from": from,
		"to":   to,
	}

	r := c.c.NewRequest("POST", "/v1/sys/remount")
	if err := r.SetJSONBody(body); err != nil {
		return err
	}

	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}

func (c *Sys) TuneMount(path string, config MountConfigInput) error {
	body := structs.Map(config)
	r := c.c.NewRequest("POST", fmt.Sprintf("/v1/sys/mounts/%s/tune", path))
	if err := r.SetJSONBody(body); err != nil {
		return err
	}

	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}

func (c *Sys) MountConfig(path string) (*MountConfigOutput, error) {
	r := c.c.NewRequest("GET", fmt.Sprintf("/v1/sys/mounts/%s/tune", path))

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result MountConfigOutput
	err = resp.DecodeJSON(&result)
	if err != nil {
		return nil, err
	}

	return &result, err
}

type MountInput struct {
	Type        string           `json:"type" structs:"type"`
	Description string           `json:"description" structs:"description"`
	Config      MountConfigInput `json:"config" structs:"config"`
	Local       bool             `json:"local" structs:"local"`
}

type MountConfigInput struct {
	DefaultLeaseTTL string `json:"default_lease_ttl" structs:"default_lease_ttl" mapstructure:"default_lease_ttl"`
	MaxLeaseTTL     string `json:"max_lease_ttl" structs:"max_lease_ttl" mapstructure:"max_lease_ttl"`
	ForceNoCache    bool   `json:"force_no_cache" structs:"force_no_cache" mapstructure:"force_no_cache"`
	PluginName      string `json:"plugin_name,omitempty" structs:"plugin_name,omitempty" mapstructure:"plugin_name"`
}

type MountOutput struct {
	Type        string            `json:"type" structs:"type"`
	Description string            `json:"description" structs:"description"`
	Accessor    string            `json:"accessor" structs:"accessor"`
	Config      MountConfigOutput `json:"config" structs:"config"`
	Local       bool              `json:"local" structs:"local"`
}

type MountConfigOutput struct {
	DefaultLeaseTTL int    `json:"default_lease_ttl" structs:"default_lease_ttl" mapstructure:"default_lease_ttl"`
	MaxLeaseTTL     int    `json:"max_lease_ttl" structs:"max_lease_ttl" mapstructure:"max_lease_ttl"`
	ForceNoCache    bool   `json:"force_no_cache" structs:"force_no_cache" mapstructure:"force_no_cache"`
	PluginName      string `json:"plugin_name,omitempty" structs:"plugin_name,omitempty" mapstructure:"plugin_name"`
}
