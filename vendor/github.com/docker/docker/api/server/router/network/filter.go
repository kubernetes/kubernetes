package network

import (
	"fmt"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/runconfig"
)

func filterNetworkByType(nws []types.NetworkResource, netType string) ([]types.NetworkResource, error) {
	retNws := []types.NetworkResource{}
	switch netType {
	case "builtin":
		for _, nw := range nws {
			if runconfig.IsPreDefinedNetwork(nw.Name) {
				retNws = append(retNws, nw)
			}
		}
	case "custom":
		for _, nw := range nws {
			if !runconfig.IsPreDefinedNetwork(nw.Name) {
				retNws = append(retNws, nw)
			}
		}
	default:
		return nil, fmt.Errorf("Invalid filter: 'type'='%s'", netType)
	}
	return retNws, nil
}

// filterNetworks filters network list according to user specified filter
// and returns user chosen networks
func filterNetworks(nws []types.NetworkResource, filter filters.Args) ([]types.NetworkResource, error) {
	// if filter is empty, return original network list
	if filter.Len() == 0 {
		return nws, nil
	}

	displayNet := []types.NetworkResource{}
	for _, nw := range nws {
		if filter.Include("driver") {
			if !filter.ExactMatch("driver", nw.Driver) {
				continue
			}
		}
		if filter.Include("name") {
			if !filter.Match("name", nw.Name) {
				continue
			}
		}
		if filter.Include("id") {
			if !filter.Match("id", nw.ID) {
				continue
			}
		}
		if filter.Include("label") {
			if !filter.MatchKVList("label", nw.Labels) {
				continue
			}
		}
		if filter.Include("scope") {
			if !filter.ExactMatch("scope", nw.Scope) {
				continue
			}
		}
		displayNet = append(displayNet, nw)
	}

	if filter.Include("type") {
		typeNet := []types.NetworkResource{}
		errFilter := filter.WalkValues("type", func(fval string) error {
			passList, err := filterNetworkByType(displayNet, fval)
			if err != nil {
				return err
			}
			typeNet = append(typeNet, passList...)
			return nil
		})
		if errFilter != nil {
			return nil, errFilter
		}
		displayNet = typeNet
	}

	return displayNet, nil
}
