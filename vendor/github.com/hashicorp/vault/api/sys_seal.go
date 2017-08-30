package api

func (c *Sys) SealStatus() (*SealStatusResponse, error) {
	r := c.c.NewRequest("GET", "/v1/sys/seal-status")
	return sealStatusRequest(c, r)
}

func (c *Sys) Seal() error {
	r := c.c.NewRequest("PUT", "/v1/sys/seal")
	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}

func (c *Sys) ResetUnsealProcess() (*SealStatusResponse, error) {
	body := map[string]interface{}{"reset": true}

	r := c.c.NewRequest("PUT", "/v1/sys/unseal")
	if err := r.SetJSONBody(body); err != nil {
		return nil, err
	}

	return sealStatusRequest(c, r)
}

func (c *Sys) Unseal(shard string) (*SealStatusResponse, error) {
	body := map[string]interface{}{"key": shard}

	r := c.c.NewRequest("PUT", "/v1/sys/unseal")
	if err := r.SetJSONBody(body); err != nil {
		return nil, err
	}

	return sealStatusRequest(c, r)
}

func sealStatusRequest(c *Sys, r *Request) (*SealStatusResponse, error) {
	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result SealStatusResponse
	err = resp.DecodeJSON(&result)
	return &result, err
}

type SealStatusResponse struct {
	Sealed      bool   `json:"sealed"`
	T           int    `json:"t"`
	N           int    `json:"n"`
	Progress    int    `json:"progress"`
	Nonce       string `json:"nonce"`
	Version     string `json:"version"`
	ClusterName string `json:"cluster_name,omitempty"`
	ClusterID   string `json:"cluster_id,omitempty"`
}
