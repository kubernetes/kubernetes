package cluster

import (
	apitypes "github.com/docker/docker/api/types"
	types "github.com/docker/docker/api/types/swarm"
	"github.com/docker/docker/daemon/cluster/convert"
	swarmapi "github.com/docker/swarmkit/api"
	"golang.org/x/net/context"
)

// GetSecret returns a secret from a managed swarm cluster
func (c *Cluster) GetSecret(input string) (types.Secret, error) {
	var secret *swarmapi.Secret

	if err := c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		s, err := getSecret(ctx, state.controlClient, input)
		if err != nil {
			return err
		}
		secret = s
		return nil
	}); err != nil {
		return types.Secret{}, err
	}
	return convert.SecretFromGRPC(secret), nil
}

// GetSecrets returns all secrets of a managed swarm cluster.
func (c *Cluster) GetSecrets(options apitypes.SecretListOptions) ([]types.Secret, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	state := c.currentNodeState()
	if !state.IsActiveManager() {
		return nil, c.errNoManager(state)
	}

	filters, err := newListSecretsFilters(options.Filters)
	if err != nil {
		return nil, err
	}
	ctx, cancel := c.getRequestContext()
	defer cancel()

	r, err := state.controlClient.ListSecrets(ctx,
		&swarmapi.ListSecretsRequest{Filters: filters})
	if err != nil {
		return nil, err
	}

	secrets := make([]types.Secret, 0, len(r.Secrets))

	for _, secret := range r.Secrets {
		secrets = append(secrets, convert.SecretFromGRPC(secret))
	}

	return secrets, nil
}

// CreateSecret creates a new secret in a managed swarm cluster.
func (c *Cluster) CreateSecret(s types.SecretSpec) (string, error) {
	var resp *swarmapi.CreateSecretResponse
	if err := c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		secretSpec := convert.SecretSpecToGRPC(s)

		r, err := state.controlClient.CreateSecret(ctx,
			&swarmapi.CreateSecretRequest{Spec: &secretSpec})
		if err != nil {
			return err
		}
		resp = r
		return nil
	}); err != nil {
		return "", err
	}
	return resp.Secret.ID, nil
}

// RemoveSecret removes a secret from a managed swarm cluster.
func (c *Cluster) RemoveSecret(input string) error {
	return c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		secret, err := getSecret(ctx, state.controlClient, input)
		if err != nil {
			return err
		}

		req := &swarmapi.RemoveSecretRequest{
			SecretID: secret.ID,
		}

		_, err = state.controlClient.RemoveSecret(ctx, req)
		return err
	})
}

// UpdateSecret updates a secret in a managed swarm cluster.
// Note: this is not exposed to the CLI but is available from the API only
func (c *Cluster) UpdateSecret(input string, version uint64, spec types.SecretSpec) error {
	return c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		secret, err := getSecret(ctx, state.controlClient, input)
		if err != nil {
			return err
		}

		secretSpec := convert.SecretSpecToGRPC(spec)

		_, err = state.controlClient.UpdateSecret(ctx,
			&swarmapi.UpdateSecretRequest{
				SecretID: secret.ID,
				SecretVersion: &swarmapi.Version{
					Index: version,
				},
				Spec: &secretSpec,
			})
		return err
	})
}
