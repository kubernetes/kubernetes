package api

func (c *Sys) StepDown() error {
	r := c.c.NewRequest("PUT", "/v1/sys/step-down")
	resp, err := c.c.RawRequest(r)
	if err == nil {
		defer resp.Body.Close()
	}
	return err
}
