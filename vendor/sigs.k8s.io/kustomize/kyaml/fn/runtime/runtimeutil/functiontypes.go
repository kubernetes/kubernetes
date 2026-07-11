// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package runtimeutil

import (
	"fmt"
	"os"
	"sort"
	"strings"

	"sigs.k8s.io/kustomize/kyaml/yaml"
	k8syaml "sigs.k8s.io/yaml"
)

const (
	FunctionAnnotationKey    = "config.kubernetes.io/function"
	oldFunctionAnnotationKey = "config.k8s.io/function"
)

var functionAnnotationKeys = []string{FunctionAnnotationKey, oldFunctionAnnotationKey}

// ContainerNetworkName is a type for network name used in container
type ContainerNetworkName string

const (
	NetworkNameNone ContainerNetworkName = "none"
	NetworkNameHost ContainerNetworkName = "host"
)
const defaultEnvValue string = "true"

// ContainerEnv defines the environment present in a container.
type ContainerEnv struct {
	// EnvVars is a key-value map that will be set as env in container
	EnvVars map[string]string

	// VarsToExport are only env key. Value will be the value in the host system
	VarsToExport []string
}

// GetDockerFlags returns docker run style env flags
func (ce *ContainerEnv) GetDockerFlags() []string {
	envs := ce.EnvVars
	if envs == nil {
		envs = make(map[string]string)
	}

	flags := []string{}
	// return in order to keep consistent among different runs
	keys := []string{}
	for k := range envs {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, key := range keys {
		flags = append(flags, "-e", key+"="+envs[key])
	}

	for _, key := range ce.VarsToExport {
		flags = append(flags, "-e", key)
	}

	return flags
}

// AddKeyValue adds a key-value pair into the envs
func (ce *ContainerEnv) AddKeyValue(key, value string) {
	if ce.EnvVars == nil {
		ce.EnvVars = make(map[string]string)
	}
	ce.EnvVars[key] = value
}

// HasExportedKey returns true if the key is a exported key
func (ce *ContainerEnv) HasExportedKey(key string) bool {
	for _, k := range ce.VarsToExport {
		if k == key {
			return true
		}
	}
	return false
}

// AddKey adds a key into the envs
func (ce *ContainerEnv) AddKey(key string) {
	if !ce.HasExportedKey(key) {
		ce.VarsToExport = append(ce.VarsToExport, key)
	}
}

// Raw returns a slice of string which represents the envs.
// Example: [foo=bar, baz]
func (ce *ContainerEnv) Raw() []string {
	var ret []string
	for k, v := range ce.EnvVars {
		ret = append(ret, k+"="+v)
	}

	ret = append(ret, ce.VarsToExport...)
	return ret
}

// NewContainerEnv returns a pointer to a new ContainerEnv
func NewContainerEnv() *ContainerEnv {
	var ce ContainerEnv
	ce.EnvVars = make(map[string]string)
	// default envs
	ce.EnvVars["LOG_TO_STDERR"] = defaultEnvValue
	ce.EnvVars["STRUCTURED_RESULTS"] = defaultEnvValue
	return &ce
}

// NewContainerEnvFromStringSlice returns a new ContainerEnv pointer with parsing
// input envStr. envStr example: ["foo=bar", "baz"]
func NewContainerEnvFromStringSlice(envStr []string) *ContainerEnv {
	ce := NewContainerEnv()
	for _, e := range envStr {
		parts := strings.SplitN(e, "=", 2)
		if len(parts) == 1 {
			ce.AddKey(e)
		} else {
			ce.AddKeyValue(parts[0], parts[1])
		}
	}
	return ce
}

// FunctionSpec defines a spec for running a function
type FunctionSpec struct {
	DeferFailure bool `json:"deferFailure,omitempty" yaml:"deferFailure,omitempty"`

	// Container is the spec for running a function as a container
	Container ContainerSpec `json:"container,omitempty" yaml:"container,omitempty"`

	// ExecSpec is the spec for running a function as an executable
	Exec ExecSpec `json:"exec,omitempty" yaml:"exec,omitempty"`
}

type ExecSpec struct {
	Path string `json:"path,omitempty" yaml:"path,omitempty"`

	// Args is a slice of args that will be passed as arguments to script
	Args []string `json:"args,omitempty" yaml:"args,omitempty"`

	// Env is a slice of env string that will be exposed to container
	Env []string `json:"envs,omitempty" yaml:"envs,omitempty"`
}

// ContainerSpec defines a spec for running a function as a container
type ContainerSpec struct {
	// Image is the container image to run
	Image string `json:"image,omitempty" yaml:"image,omitempty"`

	// Network defines network specific configuration
	Network bool `json:"network,omitempty" yaml:"network,omitempty"`

	// Mounts are the storage or directories to mount into the container
	StorageMounts []StorageMount `json:"mounts,omitempty" yaml:"mounts,omitempty"`

	// Env is a slice of env string that will be exposed to container
	Env []string `json:"envs,omitempty" yaml:"envs,omitempty"`
}

// StorageMount represents a container's mounted storage option(s)
type StorageMount struct {
	// Type of mount e.g. bind mount, local volume, etc.
	MountType string `json:"type,omitempty" yaml:"type,omitempty"`

	// Source for the storage to be mounted.
	// For named volumes, this is the name of the volume.
	// For anonymous volumes, this field is omitted (empty string).
	// For bind mounts, this is the path to the file or directory on the host.
	Src string `json:"src,omitempty" yaml:"src,omitempty"`

	// The path where the file or directory is mounted in the container.
	DstPath string `json:"dst,omitempty" yaml:"dst,omitempty"`

	// Mount in ReadWrite mode if it's explicitly configured
	// See https://docs.docker.com/storage/bind-mounts/#use-a-read-only-bind-mount
	ReadWriteMode bool `json:"rw,omitempty" yaml:"rw,omitempty"`
}

func (s *StorageMount) String() string {
	mode := ""
	if !s.ReadWriteMode {
		mode = ",readonly"
	}
	return fmt.Sprintf("type=%s,source=%s,target=%s%s", s.MountType, s.Src, s.DstPath, mode)
}

// GetFunctionSpec returns the FunctionSpec for a resource.  Returns
// nil if the resource does not have a FunctionSpec.
//
// The FunctionSpec is read from the resource metadata.annotation
// "config.kubernetes.io/function"
func GetFunctionSpec(n *yaml.RNode) (*FunctionSpec, error) {
	meta, err := n.GetMeta()
	if err != nil {
		return nil, fmt.Errorf("failed to get ResourceMeta: %w", err)
	}

	fn, err := getFunctionSpecFromAnnotation(n, meta)
	if err != nil {
		return nil, err
	}
	if fn != nil {
		return fn, nil
	}

	// legacy function specification for backwards compatibility
	container := meta.Annotations["config.kubernetes.io/container"]
	if container != "" {
		return &FunctionSpec{Container: ContainerSpec{Image: container}}, nil
	}
	return nil, nil
}

// getFunctionSpecFromAnnotation parses the config function from an annotation
// if it is found
func getFunctionSpecFromAnnotation(n *yaml.RNode, meta yaml.ResourceMeta) (*FunctionSpec, error) {
	var fs FunctionSpec
	for _, s := range functionAnnotationKeys {
		fn := meta.Annotations[s]
		if fn != "" {
			if err := k8syaml.UnmarshalStrict([]byte(fn), &fs); err != nil {
				return nil, fmt.Errorf("%s unmarshal error: %w", s, err)
			}
			return &fs, nil
		}
	}
	n, err := n.Pipe(yaml.Lookup("metadata", "configFn"))
	if err != nil {
		return nil, fmt.Errorf("failed to look up metadata.configFn: %w", err)
	}
	if yaml.IsMissingOrNull(n) {
		return nil, nil
	}
	s, err := n.String()
	if err != nil {
		fmt.Fprintf(os.Stderr, "configFn parse error: %v\n", err)
		return nil, fmt.Errorf("configFn parse error: %w", err)
	}
	if err := k8syaml.UnmarshalStrict([]byte(s), &fs); err != nil {
		return nil, fmt.Errorf("%s unmarshal error: %w", "configFn", err)
	}
	return &fs, nil
}

func StringToStorageMount(s string) StorageMount {
	m := make(map[string]string)
	options := strings.Split(s, ",")
	for _, option := range options {
		keyVal := strings.SplitN(option, "=", 2)
		if len(keyVal) == 2 {
			m[keyVal[0]] = keyVal[1]
		}
	}
	var sm StorageMount
	for key, value := range m {
		switch {
		case key == "type":
			sm.MountType = value
		case key == "src" || key == "source":
			sm.Src = value
		case key == "dst" || key == "target":
			sm.DstPath = value
		case key == "rw" && value == "true":
			sm.ReadWriteMode = true
		}
	}
	return sm
}

// IsReconcilerFilter filters Resources based on whether or not they are Reconciler Resource.
// Resources with an apiVersion starting with '*.gcr.io', 'gcr.io' or 'docker.io' are considered
// Reconciler Resources.
type IsReconcilerFilter struct {
	// ExcludeReconcilers if set to true, then Reconcilers will be excluded -- e.g.
	// Resources with a reconcile container through the apiVersion (gcr.io prefix) or
	// through the annotations
	ExcludeReconcilers bool `yaml:"excludeReconcilers,omitempty"`

	// IncludeNonReconcilers if set to true, the NonReconciler will be included.
	IncludeNonReconcilers bool `yaml:"includeNonReconcilers,omitempty"`
}

// Filter implements kio.Filter
func (c *IsReconcilerFilter) Filter(inputs []*yaml.RNode) ([]*yaml.RNode, error) {
	var out []*yaml.RNode
	for i := range inputs {
		functionSpec, err := GetFunctionSpec(inputs[i])
		if err != nil {
			return nil, err
		}
		isFnResource := functionSpec != nil
		if isFnResource && !c.ExcludeReconcilers {
			out = append(out, inputs[i])
		}
		if !isFnResource && c.IncludeNonReconcilers {
			out = append(out, inputs[i])
		}
	}
	return out, nil
}
