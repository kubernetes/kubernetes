package api

func (c *Sys) Renew(id string, increment int) (*Secret, error) {
	r := c.c.NewRequest("PUT", "/v1/sys/leases/renew")

	body := map[string]interface{}{
		"increment": increment,
		"lease_id":  id,
	}
	if err := r.SetJSONBody(body); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return ParseSecret(resp.Body)
}

func (c *Sys) Revoke(id string) error {
	r := c.c.NewRequest("PUT", "/v1/sys/leases/revoke/"+id)
	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}

func (c *Sys) RevokePrefix(id string) error {
	r := c.c.NewRequest("PUT", "/v1/sys/leases/revoke-prefix/"+id)
	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}

func (c *Sys) RevokeForce(id string) error {
	r := c.c.NewRequest("PUT", "/v1/sys/leases/revoke-force/"+id)
	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}
