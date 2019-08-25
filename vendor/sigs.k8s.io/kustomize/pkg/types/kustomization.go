/*
Copyright 2017 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package types holds struct definitions that should find a better home.
package types

import (
	"regexp"

	"sigs.k8s.io/kustomize/pkg/gvk"
	"sigs.k8s.io/kustomize/pkg/image"
)

const (
	KustomizationVersion = "kustomize.config.k8s.io/v1beta1"
	KustomizationKind    = "Kustomization"
)

// TypeMeta copies apimachinery/pkg/apis/meta/v1.TypeMeta
type TypeMeta struct {
	// Kind copies apimachinery/pkg/apis/meta/v1.Typemeta.Kind
	Kind string `json:"kind,omitempty" protobuf:"bytes,1,opt,name=kind"`

	// APIVersion copies apimachinery/pkg/apis/meta/v1.Typemeta.APIVersion
	APIVersion string `json:"apiVersion,omitempty" protobuf:"bytes,2,opt,name=apiVersion"`
}

// Kustomization holds the information needed to generate customized k8s api resources.
type Kustomization struct {
	TypeMeta `json:",inline" yaml:",inline"`

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
	PatchesJson6902 []PatchJson6902 `json:"patchesJson6902,omitempty" yaml:"patchesJson6902,omitempty"`

	// Images is a list of (image name, new name, new tag or digest)
	// for changing image names, tags or digests. This can also be achieved with a
	// patch, but this operator is simpler to specify.
	Images []image.Image `json:"images,omitempty" yaml:"images,omitempty"`

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
	// of kubernetes API objects, or specifcations of other kustomizations
	// via relative paths, absolute paths, or URLs.
	Resources []string `json:"resources,omitempty" yaml:"resources,omitempty"`

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

	// GeneratorOptions modify behavior of all ConfigMap and Secret generators.
	GeneratorOptions *GeneratorOptions `json:"generatorOptions,omitempty" yaml:"generatorOptions,omitempty"`

	// Configurations is a list of transformer configuration files
	Configurations []string `json:"configurations,omitempty" yaml:"configurations,omitempty"`

	// Generators is a list of files containing custom generators
	Generators []string `json:"generators,omitempty" yaml:"generators,omitempty"`

	// Transformers is a list of files containing transformers
	Transformers []string `json:"transformers,omitempty" yaml:"transformers,omitempty"`

	// Inventory appends an object that contains the record
	// of all other objects, which can be used in apply, prune and delete
	Inventory *Inventory `json:"inventory,omitempty" yaml:"inventory:omitempty"`
}

//go:generate stringer -type=GarbagePolicy
type GarbagePolicy int

const (
	GarbageIgnore GarbagePolicy = iota + 1
	GarbageCollect
)

// FixKustomizationPostUnmarshalling fixes things
// like empty fields that should not be empty, or
// moving content of deprecated fields to newer
// fields.
func (k *Kustomization) FixKustomizationPostUnmarshalling() {
	if k.APIVersion == "" {
		k.APIVersion = KustomizationVersion
	}
	if k.Kind == "" {
		k.Kind = KustomizationKind
	}
	// The EnvSource field is deprecated in favor of the list.
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
	for _, b := range k.Bases {
		k.Resources = append(k.Resources, b)
	}
	k.Bases = nil
}

func (k *Kustomization) EnforceFields() []string {
	var errs []string
	if k.APIVersion != "" && k.APIVersion != KustomizationVersion {
		errs = append(errs, "apiVersion should be "+KustomizationVersion)
	}
	if k.Kind != "" && k.Kind != KustomizationKind {
		errs = append(errs, "kind should be "+KustomizationKind)
	}
	return errs
}

// FixKustomizationPreUnmarshalling modies the raw data
// before marshalling - e.g. changes old field names to
// new field names.
func FixKustomizationPreUnmarshalling(data []byte) []byte {
	deprecateFieldsMap := map[string]string{
		"patches:":   "patchesStrategicMerge:",
		"imageTags:": "images:",
	}
	for oldname, newname := range deprecateFieldsMap {
		pattern := regexp.MustCompile(oldname)
		data = pattern.ReplaceAll(data, []byte(newname))
	}
	return data
}

// GeneratorArgs contains arguments common to generators.
type GeneratorArgs struct {
	// Namespace for the configmap, optional
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`

	// Name - actually the partial name - of the generated resource.
	// The full name ends up being something like
	// NamePrefix + this.Name + hash(content of generated resource).
	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	// Behavior of generated resource, must be one of:
	//   'create': create a new one
	//   'replace': replace the existing one
	//   'merge': merge with the existing one
	Behavior string `json:"behavior,omitempty" yaml:"behavior,omitempty"`

	// DataSources for the generator.
	DataSources `json:",inline,omitempty" yaml:",inline,omitempty"`
}

// PluginConfig holds plugin configuration.
type PluginConfig struct {
	// DirectoryPath is an absolute path to a
	// directory containing kustomize plugins.
	// This directory may contain subdirectories
	// further categorizing plugins.
	DirectoryPath string

	// Enabled is true if plugins are enabled.
	Enabled bool
}

// ConfigMapArgs contains the metadata of how to generate a configmap.
type ConfigMapArgs struct {
	// GeneratorArgs for the configmap.
	GeneratorArgs `json:",inline,omitempty" yaml:",inline,omitempty"`
}

// Pair is a key value pair.
type Pair struct {
	Key   string
	Value string
}

// SecretArgs contains the metadata of how to generate a secret.
type SecretArgs struct {
	// GeneratorArgs for the secret.
	GeneratorArgs `json:",inline,omitempty" yaml:",inline,omitempty"`

	// Type of the secret.
	//
	// This is the same field as the secret type field in v1/Secret:
	// It can be "Opaque" (default), or "kubernetes.io/tls".
	//
	// If type is "kubernetes.io/tls", then "literals" or "files" must have exactly two
	// keys: "tls.key" and "tls.crt"
	Type string `json:"type,omitempty" yaml:"type,omitempty"`
}

// DataSources contains some generic sources for configmaps.
type DataSources struct {
	// LiteralSources is a list of literal
	// pair sources. Each literal source should
	// be a key and literal value, e.g. `key=value`
	LiteralSources []string `json:"literals,omitempty" yaml:"literals,omitempty"`

	// FileSources is a list of file "sources" to
	// use in creating a list of key, value pairs.
	// A source takes the form:  [{key}=]{path}
	// If the "key=" part is missing, the key is the
	// path's basename. If they "key=" part is present,
	// it becomes the key (replacing the basename).
	// In either case, the value is the file contents.
	// Specifying a directory will iterate each named
	// file in the directory whose basename is a
	// valid configmap key.
	FileSources []string `json:"files,omitempty" yaml:"files,omitempty"`

	// EnvSources is a list of file paths.
	// The contents of each file should be one
	// key=value pair per line, e.g. a Docker
	// or npm ".env" file or a ".ini" file
	// (wikipedia.org/wiki/INI_file)
	EnvSources []string `json:"envs,omitempty" yaml:"envs,omitempty"`

	// Deprecated.  Use EnvSources instead.
	EnvSource string `json:"env,omitempty" yaml:"env,omitempty"`
}

// GeneratorOptions modify behavior of all ConfigMap and Secret generators.
type GeneratorOptions struct {
	// Labels to add to all generated resources.
	Labels map[string]string `json:"labels,omitempty" yaml:"labels,omitempty"`

	// Annotations to add to all generated resources.
	Annotations map[string]string `json:"annotations,omitempty" yaml:"annotations,omitempty"`

	// DisableNameSuffixHash if true disables the default behavior of adding a
	// suffix to the names of generated resources that is a hash of the
	// resource contents.
	DisableNameSuffixHash bool `json:"disableNameSuffixHash,omitempty" yaml:"disableNameSuffixHash,omitempty"`
}

type PluginType string

func (p PluginType) IsUndefined() bool {
	return p == PluginType("")
}

// KVSource represents a KV plugin backend.
type KVSource struct {
	PluginType PluginType `json:"pluginType,omitempty" yaml:"pluginType,omitempty"`
	Name       string     `json:"name,omitempty" yaml:"name,omitempty"`
	Args       []string   `json:"args,omitempty" yaml:"args,omitempty"`
}

type Inventory struct {
	Type      string   `json:"type,omitempty" yaml:"type,omitempty"`
	ConfigMap NameArgs `json:"configMap,omitempty" yaml:"configMap,omitempty"`
}

type NameArgs struct {
	Name      string `json:"name,omitempty" yaml:"name,omitempty"`
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`
}

// PatchJson6902 represents a json patch for an object
// with format documented https://tools.ietf.org/html/rfc6902.
type PatchJson6902 struct {
	// PatchTarget refers to a Kubernetes object that the json patch will be
	// applied to. It must refer to a Kubernetes resource under the
	// purview of this kustomization. PatchTarget should use the
	// raw name of the object (the name specified in its YAML,
	// before addition of a namePrefix and a nameSuffix).
	Target *PatchTarget `json:"target" yaml:"target"`

	// relative file path for a json patch file inside a kustomization
	Path string `json:"path,omitempty" yaml:"path,omitempty"`
}

// PatchTarget represents the kubernetes object that the patch is applied to
type PatchTarget struct {
	gvk.Gvk   `json:",inline,omitempty" yaml:",inline,omitempty"`
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`
	Name      string `json:"name" yaml:"name"`
}

// PatchStrategicMerge represents a relative path to a
// stategic merge patch with the format
// https://github.com/kubernetes/community/blob/master/contributors/devel/strategic-merge-patch.md
type PatchStrategicMerge string

// Replica specifies a modification to a replica config.
// The number of replicas of a resource whose name matches will be set to count.
// This struct is used by the ReplicaCountTransform, and is meant to supplement
// the existing patch functionality with a simpler syntax for replica configuration.
type Replica struct {
	// The name of the resource to change the replica count
	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	// The number of replicas required.
	Count int64 `json:"count,omitempty" yaml:"count,omitempty"`
}
