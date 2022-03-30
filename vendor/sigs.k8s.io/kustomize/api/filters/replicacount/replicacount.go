package replicacount

import (
	"strconv"

	"sigs.k8s.io/kustomize/api/filters/fieldspec"
	"sigs.k8s.io/kustomize/api/filters/filtersutil"
	"sigs.k8s.io/kustomize/api/types"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// Filter updates/sets replicas fields using the fieldSpecs
type Filter struct {
	Replica   types.Replica   `json:"replica,omitempty" yaml:"replica,omitempty"`
	FieldSpec types.FieldSpec `json:"fieldSpec,omitempty" yaml:"fieldSpec,omitempty"`

	trackableSetter filtersutil.TrackableSetter
}

var _ kio.Filter = Filter{}
var _ kio.TrackableFilter = &Filter{}

// WithMutationTracker registers a callback which will be invoked each time a field is mutated
func (rc *Filter) WithMutationTracker(callback func(key, value, tag string, node *yaml.RNode)) {
	rc.trackableSetter.WithMutationTracker(callback)
}

func (rc Filter) Filter(nodes []*yaml.RNode) ([]*yaml.RNode, error) {
	return kio.FilterAll(yaml.FilterFunc(rc.run)).Filter(nodes)
}

func (rc Filter) run(node *yaml.RNode) (*yaml.RNode, error) {
	err := node.PipeE(fieldspec.Filter{
		FieldSpec:  rc.FieldSpec,
		SetValue:   rc.set,
		CreateKind: yaml.ScalarNode, // replicas is a ScalarNode
		CreateTag:  yaml.NodeTagInt,
	})
	return node, err
}

func (rc Filter) set(node *yaml.RNode) error {
	return rc.trackableSetter.SetEntry("", strconv.FormatInt(rc.Replica.Count, 10), yaml.NodeTagInt)(node)
}
