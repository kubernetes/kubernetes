package client

import (
	"github.com/emccode/libstorage/api/types"
)

func (c *client) API() types.APIClient {
	return c.api
}

func (c *client) OS() types.OSDriver {
	return c.od
}

func (c *client) Storage() types.StorageDriver {
	return c.sd
}

func (c *client) Integration() types.IntegrationDriver {
	return c.id
}

func (c *client) Executor() types.StorageExecutorCLI {
	return c.xli
}
