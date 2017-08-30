package api

import (
	"fmt"
)

// Help reads the help information for the given path.
func (c *Client) Help(path string) (*Help, error) {
	r := c.NewRequest("GET", fmt.Sprintf("/v1/%s", path))
	r.Params.Add("help", "1")
	resp, err := c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result Help
	err = resp.DecodeJSON(&result)
	return &result, err
}

type Help struct {
	Help    string   `json:"help"`
	SeeAlso []string `json:"see_also"`
}
