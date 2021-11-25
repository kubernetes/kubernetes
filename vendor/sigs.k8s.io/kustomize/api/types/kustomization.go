// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

import (
	"bytes"
	"encoding/json"
	"fmt"

	"sigs.k8s.io/yaml"
)

const (
	KustomizationVersion  = "kustomize.config.k8s.io/v1beta1"
	KustomizationKind     = "Kustomization"
	ComponentVersion      = "kustomize.config.k8s.io/v1alpha1"
	ComponentKind         = "Component"
	MetadataNamespacePath = "metadata/namespace"
)

// Kustomization holds the information needed to generate customized k8s api resources.
type Kustomization struct {
	TypeMeta `json:",inline" yaml:",inline"`

	// MetaData is a pointer to avoid marshalling empty struct
	MetaData *ObjectMeta `json:"metadata,omitempty" yaml:"metadata,omitempty"`

	// OpenAPI contains information about what kubernetes schema to use.
	OpenAPI map[string]string `json:"openapi,omitempty" yaml:"openapi,omitempty"`

	//
	// Operators - what kustomize can do.
	//

	// NamePrefix will prefix the names of all resources mentioned in the kustomization
	// file including generated configmaps and secrets.
	NamePrefix string `json:"namePrefix,omitempty" yaml:"namePrefix,omitempty"`

	// NameSuffix will suffix the names of all resources mentioned in the kustomization
	// file including generated configmaps and secrets.
	NameSuffix string `json:"nameSuffix,omitempty" yaml:"nameSuffix,omitempty"`

	// Namespace to add to all objects.
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`

	// CommonLabels to add to all objects and selectors.
	CommonLabels map[string]string `json:"commonLabels,omitempty" yaml:"commonLabels,omitempty"`

	// Labels to add to all objects but not selectors.
	Labels []Label `json:"labels,omitempty" yaml:"labels,omitempty"`

	// CommonAnnotations to add to all objects.
	CommonAnnotations map[string]string `json:"commonAnnotations,omitempty" yaml:"commonAnnotations,omitempty"`

	// PatchesStrategicMerge specifies the relative path to a file
	// containing a strategic merge patch.  Format documented at
	// https://github.com/kubernetes/community/blob/master/contributors/devel/strategic-merge-patch.md
	// URLs and globs are not supported.
	PatchesStrategicMerge []PatchStrategicMerge `json:"patchesStrategicMerge,omitempty" yaml:"patchesStrategicMerge,omitempty"`

	// JSONPatches is a list of JSONPatch for applying JSON patch.
	// Format documented at https://tools.ietf.org/html/rfc6902
	// and http://jsonpatch.com
	PatchesJson6902 []Patch `json:"patchesJson6902,omitempty" yaml:"patchesJson6902,omitempty"`

	// Patches is a list of patches, where each one can be either a
	// Strategic Merge Patch or a JSON patch.
	// Each patch can be applied to multiple target objects.
	Patches []Patch `json:"patches,omitempty" yaml:"patches,omitempty"`

	// Images is a list of (image name, new name, new tag or digest)
	// for changing image names, tags or digests. This can also be achieved with a
	// patch, but this operator is simpler to specify.
	Images []Image `json:"images,omitempty" yaml:"images,omitempty"`

	// Replacements is a list of replacements, which will copy nodes from a
	// specified source to N specified targets.
	Replacements []ReplacementField `json:"replacements,omitempty" yaml:"replacements,omitempty"`

	// Replicas is a list of {resourcename, count} that allows for simpler replica
	// specification. This can also be done with a patch.
	Replicas []Replica `json:"replicas,omitempty" yaml:"replicas,omitempty"`

	// Vars allow things modified by kustomize to be injected into a
	// kubernetes object specification. A var is a name (e.g. FOO) associated
	// with a field in a specific resource instance.  The field must
	// contain a value of type string/bool/int/float, and defaults to the name field
	// of the instance.  Any appearance of "$(FOO)" in the object
	// spec will be replaced at kustomize build time, after the final
	// value of the specified field has been determined.
	Vars []Var `json:"vars,omitempty" yaml:"vars,omitempty"`

	//
	// Operands - what kustomize operates on.
	//

	// Resources specifies relative paths to files holding YAML representations
	// of kubernetes API objects, or specifications of other kustomizations
	// via relative paths, absolute paths, or URLs.
	Resources []string `json:"resources,omitempty" yaml:"resources,omitempty"`

	// Components specifies relative paths to specifications of other Components
	// via relative paths, absolute paths, or URLs.
	Components []string `json:"components,omitempty" yaml:"components,omitempty"`

	// Crds specifies relative paths to Custom Resource Definition files.
	// This allows custom resources to be recognized as operands, making
	// it possible to add them to the Resources list.
	// CRDs themselves are not modified.
	Crds []string `json:"crds,omitempty" yaml:"crds,omitempty"`

	// Deprecated.
	// Anything that would have been specified here should
	// be specified in the Resources field instead.
	Bases []string `json:"bases,omitempty" yaml:"bases,omitempty"`

	//
	// Generators (operators that create operands)
	//

	// ConfigMapGenerator is a list of configmaps to generate from
	// local data (one configMap per list item).
	// The resulting resource is a normal operand, subject to
	// name prefixing, patching, etc.  By default, the name of
	// the map will have a suffix hash generated from its contents.
	ConfigMapGenerator []ConfigMapArgs `json:"configMapGenerator,omitempty" yaml:"configMapGenerator,omitempty"`

	// SecretGenerator is a list of secrets to generate from
	// local data (one secret per list item).
	// The resulting resource is a normal operand, subject to
	// name prefixing, patching, etc.  By default, the name of
	// the map will have a suffix hash generated from its contents.
	SecretGenerator []SecretArgs `json:"secretGenerator,omitempty" yaml:"secretGenerator,omitempty"`

	// HelmGlobals contains helm configuration that isn't chart specific.
	HelmGlobals *HelmGlobals `json:"helmGlobals,omitempty" yaml:"helmGlobals,omitempty"`

	// HelmCharts is a list of helm chart configuration instances.
	HelmCharts []HelmChart `json:"helmCharts,omitempty" yaml:"helmCharts,omitempty"`

	// HelmChartInflationGenerator is a list of helm chart configurations.
	// Deprecated.  Auto-converted to HelmGlobals and HelmCharts.
	HelmChartInflationGenerator []HelmChartArgs `json:"helmChartInflationGenerator,omitempty" yaml:"helmChartInflationGenerator,omitempty"`

	// GeneratorOptions modify behavior of all ConfigMap and Secret generators.
	GeneratorOptions *GeneratorOptions `json:"generatorOptions,omitempty" yaml:"generatorOptions,omitempty"`

	// Configurations is a list of transformer configuration files
	Configurations []string `json:"configurations,omitempty" yaml:"configurations,omitempty"`

	// Generators is a list of files containing custom generators
	Generators []string `json:"generators,omitempty" yaml:"generators,omitempty"`

	// Transformers is a list of files containing transformers
	Transformers []string `json:"transformers,omitempty" yaml:"transformers,omitempty"`

	// Validators is a list of files containing validators
	Validators []string `json:"validators,omitempty" yaml:"validators,omitempty"`

	// Inventory appends an object that contains the record
	// of all other objects, which can be used in apply, prune and delete
	Inventory *Inventory `json:"inventory,omitempty" yaml:"inventory,omitempty"`

	// BuildMetadata is a list of strings used to toggle different build options
	BuildMetadata []string `json:"buildMetadata,omitempty" yaml:"buildMetadata,omitempty"`
}

// FixKustomizationPostUnmarshalling fixes things
// like empty fields that should not be empty, or
// moving content of deprecated fields to newer
// fields.
func (k *Kustomization) FixKustomizationPostUnmarshalling() {
	if k.Kind == "" {
		k.Kind = KustomizationKind
	}
	if k.APIVersion == "" {
		if k.Kind == ComponentKind {
			k.APIVersion = ComponentVersion
		} else {
			k.APIVersion = KustomizationVersion
		}
	}
	k.Resources = append(k.Resources, k.Bases...)
	k.Bases = nil
	for i, g := range k.ConfigMapGenerator {
		if g.EnvSource != "" {
			k.ConfigMapGenerator[i].EnvSources =
				append(g.EnvSources, g.EnvSource)
			k.ConfigMapGenerator[i].EnvSource = ""
		}
	}
	for i, g := range k.SecretGenerator {
		if g.EnvSource != "" {
			k.SecretGenerator[i].EnvSources =
				append(g.EnvSources, g.EnvSource)
			k.SecretGenerator[i].EnvSource = ""
		}
	}
	charts, globals := SplitHelmParameters(k.HelmChartInflationGenerator)
	if k.HelmGlobals == nil {
		if globals.ChartHome != "" || globals.ConfigHome != "" {
			k.HelmGlobals = &globals
		}
	}
	k.HelmCharts = append(k.HelmCharts, charts...)
	// Wipe it for the fix command.
	k.HelmChartInflationGenerator = nil
}

// FixKustomizationPreMarshalling fixes things
// that should occur after the kustomization file
// has been processed.
func (k *Kustomization) FixKustomizationPreMarshalling() error {
	// PatchesJson6902 should be under the Patches field.
	k.Patches = append(k.Patches, k.PatchesJson6902...)
	k.PatchesJson6902 = nil

	// this fix is not in FixKustomizationPostUnmarshalling because
	// it will break some commands like `create` and `add`. those
	// commands depend on 'commonLabels' field
	if cl := labelFromCommonLabels(k.CommonLabels); cl != nil {
		// check conflicts between commonLabels and labels
		for _, l := range k.Labels {
			for k := range l.Pairs {
				if _, exist := cl.Pairs[k]; exist {
					return fmt.Errorf("label name '%s' exists in both commonLabels and labels", k)
				}
			}
		}
		k.Labels = append(k.Labels, *cl)
		k.CommonLabels = nil
	}

	return nil
}

func (k *Kustomization) EnforceFields() []string {
	var errs []string
	if k.Kind != "" && k.Kind != KustomizationKind && k.Kind != ComponentKind {
		errs = append(errs, "kind should be "+KustomizationKind+" or "+ComponentKind)
	}
	requiredVersion := KustomizationVersion
	if k.Kind == ComponentKind {
		requiredVersion = ComponentVersion
	}
	if k.APIVersion != "" && k.APIVersion != requiredVersion {
		errs = append(errs, "apiVersion for "+k.Kind+" should be "+requiredVersion)
	}
	return errs
}

// Unmarshal replace k with the content in YAML input y
func (k *Kustomization) Unmarshal(y []byte) error {
	j, err := yaml.YAMLToJSON(y)
	if err != nil {
		return err
	}
	dec := json.NewDecoder(bytes.NewReader(j))
	dec.DisallowUnknownFields()
	var nk Kustomization
	err = dec.Decode(&nk)
	if err != nil {
		return err
	}
	*k = nk
	return nil
}
