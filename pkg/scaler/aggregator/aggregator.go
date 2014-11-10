package aggregator

import (
// import the aggregator API once its ready.
)

type realAggregator struct {
}

func (self *realAggregator) GetClusterInfo() ([]Node, error) {
	return []Node{}, nil
}

func New() Aggregator {
	return &realAggregator{}
}
