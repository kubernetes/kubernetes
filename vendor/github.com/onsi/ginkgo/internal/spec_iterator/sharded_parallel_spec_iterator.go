package spec_iterator

import "github.com/onsi/ginkgo/internal/spec"

type ShardedParallelIterator struct {
	specs    []*spec.Spec
	index    int
	maxIndex int
}

func NewShardedParallelIterator(specs []*spec.Spec, total int, node int) *ShardedParallelIterator {
	startIndex, count := ParallelizedIndexRange(len(specs), total, node)

	return &ShardedParallelIterator{
		specs:    specs,
		index:    startIndex,
		maxIndex: startIndex + count,
	}
}

func (s *ShardedParallelIterator) Next() (*spec.Spec, error) {
	if s.index >= s.maxIndex {
		return nil, ErrClosed
	}

	spec := s.specs[s.index]
	s.index += 1
	return spec, nil
}

func (s *ShardedParallelIterator) NumberOfSpecsPriorToIteration() int {
	return len(s.specs)
}

func (s *ShardedParallelIterator) NumberOfSpecsToProcessIfKnown() (int, bool) {
	return s.maxIndex - s.index, true
}

func (s *ShardedParallelIterator) NumberOfSpecsThatWillBeRunIfKnown() (int, bool) {
	count := 0
	for i := s.index; i < s.maxIndex; i += 1 {
		if !s.specs[i].Skipped() && !s.specs[i].Pending() {
			count += 1
		}
	}
	return count, true
}
