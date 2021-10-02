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
}

var _ kio.Filter = Filter{}

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
	return filtersutil.SetScalar(strconv.FormatInt(rc.Replica.Count, 10))(node)
}
