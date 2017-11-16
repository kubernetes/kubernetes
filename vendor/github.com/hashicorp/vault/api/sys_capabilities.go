package api

import "fmt"

func (c *Sys) CapabilitiesSelf(path string) ([]string, error) {
	return c.Capabilities(c.c.Token(), path)
}

func (c *Sys) Capabilities(token, path string) ([]string, error) {
	body := map[string]string{
		"token": token,
		"path":  path,
	}

	reqPath := "/v1/sys/capabilities"
	if token == c.c.Token() {
		reqPath = fmt.Sprintf("%s-self", reqPath)
	}

	r := c.c.NewRequest("POST", reqPath)
	if err := r.SetJSONBody(body); err != nil {
		return nil, err
	}

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

	var capabilities []string
	capabilitiesRaw := result["capabilities"].([]interface{})
	for _, capability := range capabilitiesRaw {
		capabilities = append(capabilities, capability.(string))
	}
	return capabilities, nil
}
