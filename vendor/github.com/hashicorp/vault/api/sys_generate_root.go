package api

func (c *Sys) GenerateRootStatus() (*GenerateRootStatusResponse, error) {
	r := c.c.NewRequest("GET", "/v1/sys/generate-root/attempt")
	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result GenerateRootStatusResponse
	err = resp.DecodeJSON(&result)
	return &result, err
}

func (c *Sys) GenerateRootInit(otp, pgpKey string) (*GenerateRootStatusResponse, error) {
	body := map[string]interface{}{
		"otp":     otp,
		"pgp_key": pgpKey,
	}

	r := c.c.NewRequest("PUT", "/v1/sys/generate-root/attempt")
	if err := r.SetJSONBody(body); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result GenerateRootStatusResponse
	err = resp.DecodeJSON(&result)
	return &result, err
}

func (c *Sys) GenerateRootCancel() error {
	r := c.c.NewRequest("DELETE", "/v1/sys/generate-root/attempt")
	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}

func (c *Sys) GenerateRootUpdate(shard, nonce string) (*GenerateRootStatusResponse, error) {
	body := map[string]interface{}{
		"key":   shard,
		"nonce": nonce,
	}

	r := c.c.NewRequest("PUT", "/v1/sys/generate-root/update")
	if err := r.SetJSONBody(body); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result GenerateRootStatusResponse
	err = resp.DecodeJSON(&result)
	return &result, err
}

type GenerateRootStatusResponse struct {
	Nonce            string
	Started          bool
	Progress         int
	Required         int
	Complete         bool
	EncodedRootToken string `json:"encoded_root_token"`
	PGPFingerprint   string `json:"pgp_fingerprint"`
}
