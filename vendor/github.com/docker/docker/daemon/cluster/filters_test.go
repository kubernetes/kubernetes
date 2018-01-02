package cluster

import (
	"testing"

	"github.com/docker/docker/api/types/filters"
)

func TestNewListSecretsFilters(t *testing.T) {
	validNameFilter := filters.NewArgs()
	validNameFilter.Add("name", "test_name")

	validIDFilter := filters.NewArgs()
	validIDFilter.Add("id", "7c9009d6720f6de3b492f5")

	validLabelFilter := filters.NewArgs()
	validLabelFilter.Add("label", "type=test")
	validLabelFilter.Add("label", "storage=ssd")
	validLabelFilter.Add("label", "memory")

	validNamesFilter := filters.NewArgs()
	validNamesFilter.Add("names", "test_name")

	validAllFilter := filters.NewArgs()
	validAllFilter.Add("name", "nodeName")
	validAllFilter.Add("id", "7c9009d6720f6de3b492f5")
	validAllFilter.Add("label", "type=test")
	validAllFilter.Add("label", "memory")
	validAllFilter.Add("names", "test_name")

	validFilters := []filters.Args{
		validNameFilter,
		validIDFilter,
		validLabelFilter,
		validNamesFilter,
		validAllFilter,
	}

	invalidTypeFilter := filters.NewArgs()
	invalidTypeFilter.Add("nonexist", "aaaa")

	invalidFilters := []filters.Args{
		invalidTypeFilter,
	}

	for _, filter := range validFilters {
		if _, err := newListSecretsFilters(filter); err != nil {
			t.Fatalf("Should get no error, got %v", err)
		}
	}

	for _, filter := range invalidFilters {
		if _, err := newListSecretsFilters(filter); err == nil {
			t.Fatalf("Should get an error for filter %v, while got nil", filter)
		}
	}
}

func TestNewListConfigsFilters(t *testing.T) {
	validNameFilter := filters.NewArgs()
	validNameFilter.Add("name", "test_name")

	validIDFilter := filters.NewArgs()
	validIDFilter.Add("id", "7c9009d6720f6de3b492f5")

	validLabelFilter := filters.NewArgs()
	validLabelFilter.Add("label", "type=test")
	validLabelFilter.Add("label", "storage=ssd")
	validLabelFilter.Add("label", "memory")

	validAllFilter := filters.NewArgs()
	validAllFilter.Add("name", "nodeName")
	validAllFilter.Add("id", "7c9009d6720f6de3b492f5")
	validAllFilter.Add("label", "type=test")
	validAllFilter.Add("label", "memory")

	validFilters := []filters.Args{
		validNameFilter,
		validIDFilter,
		validLabelFilter,
		validAllFilter,
	}

	invalidTypeFilter := filters.NewArgs()
	invalidTypeFilter.Add("nonexist", "aaaa")

	invalidFilters := []filters.Args{
		invalidTypeFilter,
	}

	for _, filter := range validFilters {
		if _, err := newListConfigsFilters(filter); err != nil {
			t.Fatalf("Should get no error, got %v", err)
		}
	}

	for _, filter := range invalidFilters {
		if _, err := newListConfigsFilters(filter); err == nil {
			t.Fatalf("Should get an error for filter %v, while got nil", filter)
		}
	}
}
