package api

import "time"

func (c *Sys) Rotate() error {
	r := c.c.NewRequest("POST", "/v1/sys/rotate")
	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}

func (c *Sys) KeyStatus() (*KeyStatus, error) {
	r := c.c.NewRequest("GET", "/v1/sys/key-status")
	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	result := new(KeyStatus)
	err = resp.DecodeJSON(result)
	return result, err
}

type KeyStatus struct {
	Term        int       `json:"term"`
	InstallTime time.Time `json:"install_time"`
}
