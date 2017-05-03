package actuator

import (
	"github.com/GoogleCloudPlatform/kubernetes/pkg/scaler/types"
)

type NodeShape struct {
	Name     string
	Capacity types.Resource
}
type Actuator interface {
	GetNodeShapes() ([]NodeShape, error)
	CreateNewNodes(nodeShape []NodeShape) error
}
