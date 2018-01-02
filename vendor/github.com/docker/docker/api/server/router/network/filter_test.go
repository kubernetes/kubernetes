// +build !windows

package network

import (
	"strings"
	"testing"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
)

func TestFilterNetworks(t *testing.T) {
	networks := []types.NetworkResource{
		{
			Name:   "host",
			Driver: "host",
			Scope:  "local",
		},
		{
			Name:   "bridge",
			Driver: "bridge",
			Scope:  "local",
		},
		{
			Name:   "none",
			Driver: "null",
			Scope:  "local",
		},
		{
			Name:   "myoverlay",
			Driver: "overlay",
			Scope:  "swarm",
		},
		{
			Name:   "mydrivernet",
			Driver: "mydriver",
			Scope:  "local",
		},
		{
			Name:   "mykvnet",
			Driver: "mykvdriver",
			Scope:  "global",
		},
	}

	bridgeDriverFilters := filters.NewArgs()
	bridgeDriverFilters.Add("driver", "bridge")

	overlayDriverFilters := filters.NewArgs()
	overlayDriverFilters.Add("driver", "overlay")

	nonameDriverFilters := filters.NewArgs()
	nonameDriverFilters.Add("driver", "noname")

	customDriverFilters := filters.NewArgs()
	customDriverFilters.Add("type", "custom")

	builtinDriverFilters := filters.NewArgs()
	builtinDriverFilters.Add("type", "builtin")

	invalidDriverFilters := filters.NewArgs()
	invalidDriverFilters.Add("type", "invalid")

	localScopeFilters := filters.NewArgs()
	localScopeFilters.Add("scope", "local")

	swarmScopeFilters := filters.NewArgs()
	swarmScopeFilters.Add("scope", "swarm")

	globalScopeFilters := filters.NewArgs()
	globalScopeFilters.Add("scope", "global")

	testCases := []struct {
		filter      filters.Args
		resultCount int
		err         string
	}{
		{
			filter:      bridgeDriverFilters,
			resultCount: 1,
			err:         "",
		},
		{
			filter:      overlayDriverFilters,
			resultCount: 1,
			err:         "",
		},
		{
			filter:      nonameDriverFilters,
			resultCount: 0,
			err:         "",
		},
		{
			filter:      customDriverFilters,
			resultCount: 3,
			err:         "",
		},
		{
			filter:      builtinDriverFilters,
			resultCount: 3,
			err:         "",
		},
		{
			filter:      invalidDriverFilters,
			resultCount: 0,
			err:         "Invalid filter: 'type'='invalid'",
		},
		{
			filter:      localScopeFilters,
			resultCount: 4,
			err:         "",
		},
		{
			filter:      swarmScopeFilters,
			resultCount: 1,
			err:         "",
		},
		{
			filter:      globalScopeFilters,
			resultCount: 1,
			err:         "",
		},
	}

	for _, testCase := range testCases {
		result, err := filterNetworks(networks, testCase.filter)
		if testCase.err != "" {
			if err == nil {
				t.Fatalf("expect error '%s', got no error", testCase.err)

			} else if !strings.Contains(err.Error(), testCase.err) {
				t.Fatalf("expect error '%s', got '%s'", testCase.err, err)
			}
		} else {
			if err != nil {
				t.Fatalf("expect no error, got error '%s'", err)
			}
			// Make sure result is not nil
			if result == nil {
				t.Fatal("filterNetworks should not return nil")
			}

			if len(result) != testCase.resultCount {
				t.Fatalf("expect '%d' networks, got '%d' networks", testCase.resultCount, len(result))
			}
		}
	}
}
