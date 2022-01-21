package utils

import (
	"fmt"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/resid"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// MakeResIds returns all of an RNode's current and previous Ids
func MakeResIds(n *yaml.RNode) ([]resid.ResId, error) {
	var result []resid.ResId
	apiVersion := n.Field(yaml.APIVersionField)
	var group, version string
	if apiVersion != nil {
		group, version = resid.ParseGroupVersion(yaml.GetValue(apiVersion.Value))
	}
	result = append(result, resid.NewResIdWithNamespace(
		resid.Gvk{Group: group, Version: version, Kind: n.GetKind()}, n.GetName(), n.GetNamespace()),
	)
	prevIds, err := PrevIds(n)
	if err != nil {
		return nil, err
	}
	result = append(result, prevIds...)
	return result, nil
}

// PrevIds returns all of an RNode's previous Ids
func PrevIds(n *yaml.RNode) ([]resid.ResId, error) {
	var ids []resid.ResId
	// TODO: merge previous names and namespaces into one list of
	//     pairs on one annotation so there is no chance of error
	annotations := n.GetAnnotations()
	if _, ok := annotations[BuildAnnotationPreviousNames]; !ok {
		return nil, nil
	}
	names := strings.Split(annotations[BuildAnnotationPreviousNames], ",")
	ns := strings.Split(annotations[BuildAnnotationPreviousNamespaces], ",")
	kinds := strings.Split(annotations[BuildAnnotationPreviousKinds], ",")
	// This should never happen
	if len(names) != len(ns) || len(names) != len(kinds) {
		return nil, fmt.Errorf(
			"number of previous names, " +
				"number of previous namespaces, " +
				"number of previous kinds not equal")
	}
	for i := range names {
		meta, err := n.GetMeta()
		if err != nil {
			return nil, err
		}
		group, version := resid.ParseGroupVersion(meta.APIVersion)
		gvk := resid.Gvk{
			Group:   group,
			Version: version,
			Kind:    kinds[i],
		}
		ids = append(ids, resid.NewResIdWithNamespace(
			gvk, names[i], ns[i]))
	}
	return ids, nil
}
