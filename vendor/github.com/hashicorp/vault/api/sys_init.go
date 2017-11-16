package api

func (c *Sys) InitStatus() (bool, error) {
	r := c.c.NewRequest("GET", "/v1/sys/init")
	resp, err := c.c.RawRequest(r)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()

	var result InitStatusResponse
	err = resp.DecodeJSON(&result)
	return result.Initialized, err
}

func (c *Sys) Init(opts *InitRequest) (*InitResponse, error) {
	r := c.c.NewRequest("PUT", "/v1/sys/init")
	if err := r.SetJSONBody(opts); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result InitResponse
	err = resp.DecodeJSON(&result)
	return &result, err
}

type InitRequest struct {
	SecretShares      int      `json:"secret_shares"`
	SecretThreshold   int      `json:"secret_threshold"`
	StoredShares      int      `json:"stored_shares"`
	PGPKeys           []string `json:"pgp_keys"`
	RecoveryShares    int      `json:"recovery_shares"`
	RecoveryThreshold int      `json:"recovery_threshold"`
	RecoveryPGPKeys   []string `json:"recovery_pgp_keys"`
	RootTokenPGPKey   string   `json:"root_token_pgp_key"`
}

type InitStatusResponse struct {
	Initialized bool
}

type InitResponse struct {
	Keys            []string `json:"keys"`
	KeysB64         []string `json:"keys_base64"`
	RecoveryKeys    []string `json:"recovery_keys"`
	RecoveryKeysB64 []string `json:"recovery_keys_base64"`
	RootToken       string   `json:"root_token"`
}
