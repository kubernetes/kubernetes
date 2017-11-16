package api

import "fmt"

func (c *Sys) ListPolicies() ([]string, error) {
	r := c.c.NewRequest("GET", "/v1/sys/policy")
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

	var ok bool
	if _, ok = result["policies"]; !ok {
		return nil, fmt.Errorf("policies not found in response")
	}

	listRaw := result["policies"].([]interface{})
	var policies []string

	for _, val := range listRaw {
		policies = append(policies, val.(string))
	}

	return policies, err
}

func (c *Sys) GetPolicy(name string) (string, error) {
	r := c.c.NewRequest("GET", fmt.Sprintf("/v1/sys/policy/%s", name))
	resp, err := c.c.RawRequest(r)
	if resp != nil {
		defer resp.Body.Close()
		if resp.StatusCode == 404 {
			return "", nil
		}
	}
	if err != nil {
		return "", err
	}

	var result map[string]interface{}
	err = resp.DecodeJSON(&result)
	if err != nil {
		return "", err
	}

	var ok bool
	if _, ok = result["rules"]; !ok {
		return "", fmt.Errorf("rules not found in response")
	}

	return result["rules"].(string), nil
}

func (c *Sys) PutPolicy(name, rules string) error {
	body := map[string]string{
		"rules": rules,
	}

	r := c.c.NewRequest("PUT", fmt.Sprintf("/v1/sys/policy/%s", name))
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

func (c *Sys) DeletePolicy(name string) error {
	r := c.c.NewRequest("DELETE", fmt.Sprintf("/v1/sys/policy/%s", name))
	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}

type getPoliciesResp struct {
	Rules string `json:"rules"`
}

type listPoliciesResp struct {
	Policies []string `json:"policies"`
}
