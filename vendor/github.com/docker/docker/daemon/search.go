package daemon

import (
	"fmt"
	"strconv"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/filters"
	registrytypes "github.com/docker/docker/api/types/registry"
	"github.com/docker/docker/dockerversion"
)

var acceptedSearchFilterTags = map[string]bool{
	"is-automated": true,
	"is-official":  true,
	"stars":        true,
}

// SearchRegistryForImages queries the registry for images matching
// term. authConfig is used to login.
func (daemon *Daemon) SearchRegistryForImages(ctx context.Context, filtersArgs string, term string, limit int,
	authConfig *types.AuthConfig,
	headers map[string][]string) (*registrytypes.SearchResults, error) {

	searchFilters, err := filters.FromParam(filtersArgs)
	if err != nil {
		return nil, err
	}
	if err := searchFilters.Validate(acceptedSearchFilterTags); err != nil {
		return nil, err
	}

	var isAutomated, isOfficial bool
	var hasStarFilter = 0
	if searchFilters.Include("is-automated") {
		if searchFilters.UniqueExactMatch("is-automated", "true") {
			isAutomated = true
		} else if !searchFilters.UniqueExactMatch("is-automated", "false") {
			return nil, fmt.Errorf("Invalid filter 'is-automated=%s'", searchFilters.Get("is-automated"))
		}
	}
	if searchFilters.Include("is-official") {
		if searchFilters.UniqueExactMatch("is-official", "true") {
			isOfficial = true
		} else if !searchFilters.UniqueExactMatch("is-official", "false") {
			return nil, fmt.Errorf("Invalid filter 'is-official=%s'", searchFilters.Get("is-official"))
		}
	}
	if searchFilters.Include("stars") {
		hasStars := searchFilters.Get("stars")
		for _, hasStar := range hasStars {
			iHasStar, err := strconv.Atoi(hasStar)
			if err != nil {
				return nil, fmt.Errorf("Invalid filter 'stars=%s'", hasStar)
			}
			if iHasStar > hasStarFilter {
				hasStarFilter = iHasStar
			}
		}
	}

	unfilteredResult, err := daemon.RegistryService.Search(ctx, term, limit, authConfig, dockerversion.DockerUserAgent(ctx), headers)
	if err != nil {
		return nil, err
	}

	filteredResults := []registrytypes.SearchResult{}
	for _, result := range unfilteredResult.Results {
		if searchFilters.Include("is-automated") {
			if isAutomated != result.IsAutomated {
				continue
			}
		}
		if searchFilters.Include("is-official") {
			if isOfficial != result.IsOfficial {
				continue
			}
		}
		if searchFilters.Include("stars") {
			if result.StarCount < hasStarFilter {
				continue
			}
		}
		filteredResults = append(filteredResults, result)
	}

	return &registrytypes.SearchResults{
		Query:      unfilteredResult.Query,
		NumResults: len(filteredResults),
		Results:    filteredResults,
	}, nil
}
