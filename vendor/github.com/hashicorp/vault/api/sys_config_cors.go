package api

func (c *Sys) CORSStatus() (*CORSResponse, error) {
	r := c.c.NewRequest("GET", "/v1/sys/config/cors")
	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result CORSResponse
	err = resp.DecodeJSON(&result)
	return &result, err
}

func (c *Sys) ConfigureCORS(req *CORSRequest) (*CORSResponse, error) {
	r := c.c.NewRequest("PUT", "/v1/sys/config/cors")
	if err := r.SetJSONBody(req); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result CORSResponse
	err = resp.DecodeJSON(&result)
	return &result, err
}

func (c *Sys) DisableCORS() (*CORSResponse, error) {
	r := c.c.NewRequest("DELETE", "/v1/sys/config/cors")

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result CORSResponse
	err = resp.DecodeJSON(&result)
	return &result, err

}

type CORSRequest struct {
	AllowedOrigins string `json:"allowed_origins"`
	Enabled        bool   `json:"enabled"`
}

type CORSResponse struct {
	AllowedOrigins string `json:"allowed_origins"`
	Enabled        bool   `json:"enabled"`
}
