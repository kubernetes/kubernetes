package actuator

import (
// Import the actuator api once its ready.
)

type realActuator struct {
}

func (self *realActuator) GetNodeShapes() ([]NodeShape, error) {
	return []NodeShape{}, nil
}

func (self *realActuator) CreateNewNodes(nodeShape []NodeShape) error {
	return nil
}

func New() Actuator {
	return &realActuator{}
}
