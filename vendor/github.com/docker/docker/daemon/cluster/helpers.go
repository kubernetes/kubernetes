package cluster

import (
	"fmt"

	swarmapi "github.com/docker/swarmkit/api"
	"github.com/pkg/errors"
	"golang.org/x/net/context"
)

func getSwarm(ctx context.Context, c swarmapi.ControlClient) (*swarmapi.Cluster, error) {
	rl, err := c.ListClusters(ctx, &swarmapi.ListClustersRequest{})
	if err != nil {
		return nil, err
	}

	if len(rl.Clusters) == 0 {
		return nil, errors.WithStack(errNoSwarm)
	}

	// TODO: assume one cluster only
	return rl.Clusters[0], nil
}

func getNode(ctx context.Context, c swarmapi.ControlClient, input string) (*swarmapi.Node, error) {
	// GetNode to match via full ID.
	if rg, err := c.GetNode(ctx, &swarmapi.GetNodeRequest{NodeID: input}); err == nil {
		return rg.Node, nil
	}

	// If any error (including NotFound), ListNodes to match via full name.
	rl, err := c.ListNodes(ctx, &swarmapi.ListNodesRequest{
		Filters: &swarmapi.ListNodesRequest_Filters{
			Names: []string{input},
		},
	})
	if err != nil || len(rl.Nodes) == 0 {
		// If any error or 0 result, ListNodes to match via ID prefix.
		rl, err = c.ListNodes(ctx, &swarmapi.ListNodesRequest{
			Filters: &swarmapi.ListNodesRequest_Filters{
				IDPrefixes: []string{input},
			},
		})
	}
	if err != nil {
		return nil, err
	}

	if len(rl.Nodes) == 0 {
		err := fmt.Errorf("node %s not found", input)
		return nil, notFoundError{err}
	}

	if l := len(rl.Nodes); l > 1 {
		return nil, ambiguousResultsError{fmt.Errorf("node %s is ambiguous (%d matches found)", input, l)}
	}

	return rl.Nodes[0], nil
}

func getService(ctx context.Context, c swarmapi.ControlClient, input string, insertDefaults bool) (*swarmapi.Service, error) {
	// GetService to match via full ID.
	if rg, err := c.GetService(ctx, &swarmapi.GetServiceRequest{ServiceID: input, InsertDefaults: insertDefaults}); err == nil {
		return rg.Service, nil
	}

	// If any error (including NotFound), ListServices to match via full name.
	rl, err := c.ListServices(ctx, &swarmapi.ListServicesRequest{
		Filters: &swarmapi.ListServicesRequest_Filters{
			Names: []string{input},
		},
	})
	if err != nil || len(rl.Services) == 0 {
		// If any error or 0 result, ListServices to match via ID prefix.
		rl, err = c.ListServices(ctx, &swarmapi.ListServicesRequest{
			Filters: &swarmapi.ListServicesRequest_Filters{
				IDPrefixes: []string{input},
			},
		})
	}
	if err != nil {
		return nil, err
	}

	if len(rl.Services) == 0 {
		err := fmt.Errorf("service %s not found", input)
		return nil, notFoundError{err}
	}

	if l := len(rl.Services); l > 1 {
		return nil, ambiguousResultsError{fmt.Errorf("service %s is ambiguous (%d matches found)", input, l)}
	}

	if !insertDefaults {
		return rl.Services[0], nil
	}

	rg, err := c.GetService(ctx, &swarmapi.GetServiceRequest{ServiceID: rl.Services[0].ID, InsertDefaults: true})
	if err == nil {
		return rg.Service, nil
	}
	return nil, err
}

func getTask(ctx context.Context, c swarmapi.ControlClient, input string) (*swarmapi.Task, error) {
	// GetTask to match via full ID.
	if rg, err := c.GetTask(ctx, &swarmapi.GetTaskRequest{TaskID: input}); err == nil {
		return rg.Task, nil
	}

	// If any error (including NotFound), ListTasks to match via full name.
	rl, err := c.ListTasks(ctx, &swarmapi.ListTasksRequest{
		Filters: &swarmapi.ListTasksRequest_Filters{
			Names: []string{input},
		},
	})
	if err != nil || len(rl.Tasks) == 0 {
		// If any error or 0 result, ListTasks to match via ID prefix.
		rl, err = c.ListTasks(ctx, &swarmapi.ListTasksRequest{
			Filters: &swarmapi.ListTasksRequest_Filters{
				IDPrefixes: []string{input},
			},
		})
	}
	if err != nil {
		return nil, err
	}

	if len(rl.Tasks) == 0 {
		err := fmt.Errorf("task %s not found", input)
		return nil, notFoundError{err}
	}

	if l := len(rl.Tasks); l > 1 {
		return nil, ambiguousResultsError{fmt.Errorf("task %s is ambiguous (%d matches found)", input, l)}
	}

	return rl.Tasks[0], nil
}

func getSecret(ctx context.Context, c swarmapi.ControlClient, input string) (*swarmapi.Secret, error) {
	// attempt to lookup secret by full ID
	if rg, err := c.GetSecret(ctx, &swarmapi.GetSecretRequest{SecretID: input}); err == nil {
		return rg.Secret, nil
	}

	// If any error (including NotFound), ListSecrets to match via full name.
	rl, err := c.ListSecrets(ctx, &swarmapi.ListSecretsRequest{
		Filters: &swarmapi.ListSecretsRequest_Filters{
			Names: []string{input},
		},
	})
	if err != nil || len(rl.Secrets) == 0 {
		// If any error or 0 result, ListSecrets to match via ID prefix.
		rl, err = c.ListSecrets(ctx, &swarmapi.ListSecretsRequest{
			Filters: &swarmapi.ListSecretsRequest_Filters{
				IDPrefixes: []string{input},
			},
		})
	}
	if err != nil {
		return nil, err
	}

	if len(rl.Secrets) == 0 {
		err := fmt.Errorf("secret %s not found", input)
		return nil, notFoundError{err}
	}

	if l := len(rl.Secrets); l > 1 {
		return nil, ambiguousResultsError{fmt.Errorf("secret %s is ambiguous (%d matches found)", input, l)}
	}

	return rl.Secrets[0], nil
}

func getConfig(ctx context.Context, c swarmapi.ControlClient, input string) (*swarmapi.Config, error) {
	// attempt to lookup config by full ID
	if rg, err := c.GetConfig(ctx, &swarmapi.GetConfigRequest{ConfigID: input}); err == nil {
		return rg.Config, nil
	}

	// If any error (including NotFound), ListConfigs to match via full name.
	rl, err := c.ListConfigs(ctx, &swarmapi.ListConfigsRequest{
		Filters: &swarmapi.ListConfigsRequest_Filters{
			Names: []string{input},
		},
	})
	if err != nil || len(rl.Configs) == 0 {
		// If any error or 0 result, ListConfigs to match via ID prefix.
		rl, err = c.ListConfigs(ctx, &swarmapi.ListConfigsRequest{
			Filters: &swarmapi.ListConfigsRequest_Filters{
				IDPrefixes: []string{input},
			},
		})
	}
	if err != nil {
		return nil, err
	}

	if len(rl.Configs) == 0 {
		err := fmt.Errorf("config %s not found", input)
		return nil, notFoundError{err}
	}

	if l := len(rl.Configs); l > 1 {
		return nil, ambiguousResultsError{fmt.Errorf("config %s is ambiguous (%d matches found)", input, l)}
	}

	return rl.Configs[0], nil
}

func getNetwork(ctx context.Context, c swarmapi.ControlClient, input string) (*swarmapi.Network, error) {
	// GetNetwork to match via full ID.
	if rg, err := c.GetNetwork(ctx, &swarmapi.GetNetworkRequest{NetworkID: input}); err == nil {
		return rg.Network, nil
	}

	// If any error (including NotFound), ListNetworks to match via ID prefix and full name.
	rl, err := c.ListNetworks(ctx, &swarmapi.ListNetworksRequest{
		Filters: &swarmapi.ListNetworksRequest_Filters{
			Names: []string{input},
		},
	})
	if err != nil || len(rl.Networks) == 0 {
		rl, err = c.ListNetworks(ctx, &swarmapi.ListNetworksRequest{
			Filters: &swarmapi.ListNetworksRequest_Filters{
				IDPrefixes: []string{input},
			},
		})
	}
	if err != nil {
		return nil, err
	}

	if len(rl.Networks) == 0 {
		return nil, fmt.Errorf("network %s not found", input)
	}

	if l := len(rl.Networks); l > 1 {
		return nil, ambiguousResultsError{fmt.Errorf("network %s is ambiguous (%d matches found)", input, l)}
	}

	return rl.Networks[0], nil
}
