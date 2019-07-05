package spec_iterator

import (
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/onsi/ginkgo/internal/spec"
)

type ParallelIterator struct {
	specs  []*spec.Spec
	host   string
	client *http.Client
}

func NewParallelIterator(specs []*spec.Spec, host string) *ParallelIterator {
	return &ParallelIterator{
		specs:  specs,
		host:   host,
		client: &http.Client{},
	}
}

func (s *ParallelIterator) Next() (*spec.Spec, error) {
	resp, err := s.client.Get(s.host + "/counter")
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unexpected status code %d", resp.StatusCode)
	}

	var counter Counter
	err = json.NewDecoder(resp.Body).Decode(&counter)
	if err != nil {
		return nil, err
	}

	if counter.Index >= len(s.specs) {
		return nil, ErrClosed
	}

	return s.specs[counter.Index], nil
}

func (s *ParallelIterator) NumberOfSpecsPriorToIteration() int {
	return len(s.specs)
}

func (s *ParallelIterator) NumberOfSpecsToProcessIfKnown() (int, bool) {
	return -1, false
}

func (s *ParallelIterator) NumberOfSpecsThatWillBeRunIfKnown() (int, bool) {
	return -1, false
}
