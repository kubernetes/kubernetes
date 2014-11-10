package aggregator

import "github.com/GoogleCloudPlatform/kubernetes/pkg/scaler/types"

type Node struct {
	Name     string
	Capacity types.Resource
	Usage    types.Resource
}

type Aggregator interface {
	GetClusterInfo() ([]Node, error)
}
