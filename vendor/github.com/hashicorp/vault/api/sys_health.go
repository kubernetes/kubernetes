package api

func (c *Sys) Health() (*HealthResponse, error) {
	r := c.c.NewRequest("GET", "/v1/sys/health")
	// If the code is 400 or above it will automatically turn into an error,
	// but the sys/health API defaults to returning 5xx when not sealed or
	// inited, so we force this code to be something else so we parse correctly
	r.Params.Add("sealedcode", "299")
	r.Params.Add("uninitcode", "299")
	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result HealthResponse
	err = resp.DecodeJSON(&result)
	return &result, err
}

type HealthResponse struct {
	Initialized   bool   `json:"initialized"`
	Sealed        bool   `json:"sealed"`
	Standby       bool   `json:"standby"`
	ServerTimeUTC int64  `json:"server_time_utc"`
	Version       string `json:"version"`
	ClusterName   string `json:"cluster_name,omitempty"`
	ClusterID     string `json:"cluster_id,omitempty"`
}
