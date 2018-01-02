package cluster

import (
	apitypes "github.com/docker/docker/api/types"
	types "github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/daemon/cluster/convert"
	swarmapi "github.com/docker/swarmkit/api"
	"golang.org/x/net/context"
)

// GetConfig returns a config from a managed swarm cluster
func (c *Cluster) GetConfig(input string) (types.Config, error) {
	var config *swarmapi.Config

	if err := c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		s, err := getConfig(ctx, state.controlClient, input)
		if err != nil {
			return err
		}
		config = s
		return nil
	}); err != nil {
		return types.Config{}, err
	}
	return convert.ConfigFromGRPC(config), nil
}

// GetConfigs returns all configs of a managed swarm cluster.
func (c *Cluster) GetConfigs(options apitypes.ConfigListOptions) ([]types.Config, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	state := c.currentNodeState()
	if !state.IsActiveManager() {
		return nil, c.errNoManager(state)
	}

	filters, err := newListConfigsFilters(options.Filters)
	if err != nil {
		return nil, err
	}
	ctx, cancel := c.getRequestContext()
	defer cancel()

	r, err := state.controlClient.ListConfigs(ctx,
		&swarmapi.ListConfigsRequest{Filters: filters})
	if err != nil {
		return nil, err
	}

	configs := []types.Config{}

	for _, config := range r.Configs {
		configs = append(configs, convert.ConfigFromGRPC(config))
	}

	return configs, nil
}

// CreateConfig creates a new config in a managed swarm cluster.
func (c *Cluster) CreateConfig(s types.ConfigSpec) (string, error) {
	var resp *swarmapi.CreateConfigResponse
	if err := c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		configSpec := convert.ConfigSpecToGRPC(s)

		r, err := state.controlClient.CreateConfig(ctx,
			&swarmapi.CreateConfigRequest{Spec: &configSpec})
		if err != nil {
			return err
		}
		resp = r
		return nil
	}); err != nil {
		return "", err
	}
	return resp.Config.ID, nil
}

// RemoveConfig removes a config from a managed swarm cluster.
func (c *Cluster) RemoveConfig(input string) error {
	return c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		config, err := getConfig(ctx, state.controlClient, input)
		if err != nil {
			return err
		}

		req := &swarmapi.RemoveConfigRequest{
			ConfigID: config.ID,
		}

		_, err = state.controlClient.RemoveConfig(ctx, req)
		return err
	})
}

// UpdateConfig updates a config in a managed swarm cluster.
// Note: this is not exposed to the CLI but is available from the API only
func (c *Cluster) UpdateConfig(input string, version uint64, spec types.ConfigSpec) error {
	return c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		config, err := getConfig(ctx, state.controlClient, input)
		if err != nil {
			return err
		}

		configSpec := convert.ConfigSpecToGRPC(spec)

		_, err = state.controlClient.UpdateConfig(ctx,
			&swarmapi.UpdateConfigRequest{
				ConfigID: config.ID,
				ConfigVersion: &swarmapi.Version{
					Index: version,
				},
				Spec: &configSpec,
			})
		return err
	})
}
