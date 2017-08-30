package api

import (
	"fmt"

	"github.com/fatih/structs"
	"github.com/mitchellh/mapstructure"
)

func (c *Sys) AuditHash(path string, input string) (string, error) {
	body := map[string]interface{}{
		"input": input,
	}

	r := c.c.NewRequest("PUT", fmt.Sprintf("/v1/sys/audit-hash/%s", path))
	if err := r.SetJSONBody(body); err != nil {
		return "", err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	type d struct {
		Hash string `json:"hash"`
	}

	var result d
	err = resp.DecodeJSON(&result)
	if err != nil {
		return "", err
	}

	return result.Hash, err
}

func (c *Sys) ListAudit() (map[string]*Audit, error) {
	r := c.c.NewRequest("GET", "/v1/sys/audit")
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

	mounts := map[string]*Audit{}
	for k, v := range result {
		switch v.(type) {
		case map[string]interface{}:
		default:
			continue
		}
		var res Audit
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

// DEPRECATED: Use EnableAuditWithOptions instead
func (c *Sys) EnableAudit(
	path string, auditType string, desc string, opts map[string]string) error {
	return c.EnableAuditWithOptions(path, &EnableAuditOptions{
		Type:        auditType,
		Description: desc,
		Options:     opts,
	})
}

func (c *Sys) EnableAuditWithOptions(path string, options *EnableAuditOptions) error {
	body := structs.Map(options)

	r := c.c.NewRequest("PUT", fmt.Sprintf("/v1/sys/audit/%s", path))
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

func (c *Sys) DisableAudit(path string) error {
	r := c.c.NewRequest("DELETE", fmt.Sprintf("/v1/sys/audit/%s", path))
	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}

// Structures for the requests/resposne are all down here. They aren't
// individually documented because the map almost directly to the raw HTTP API
// documentation. Please refer to that documentation for more details.

type EnableAuditOptions struct {
	Type        string            `json:"type" structs:"type"`
	Description string            `json:"description" structs:"description"`
	Options     map[string]string `json:"options" structs:"options"`
	Local       bool              `json:"local" structs:"local"`
}

type Audit struct {
	Path        string
	Type        string
	Description string
	Options     map[string]string
	Local       bool
}
