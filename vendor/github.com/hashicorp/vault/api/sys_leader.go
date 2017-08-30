package api

func (c *Sys) Leader() (*LeaderResponse, error) {
	r := c.c.NewRequest("GET", "/v1/sys/leader")
	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var result LeaderResponse
	err = resp.DecodeJSON(&result)
	return &result, err
}

type LeaderResponse struct {
	HAEnabled            bool   `json:"ha_enabled"`
	IsSelf               bool   `json:"is_self"`
	LeaderAddress        string `json:"leader_address"`
	LeaderClusterAddress string `json:"leader_cluster_address"`
}
