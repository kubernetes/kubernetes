// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

// HelmChartArgs contains the metadata of how to generate a secret.
type HelmChartArgs struct {
	ChartName    string `json:"chartName,omitempty" yaml:"chartName,omitempty"`
	ChartVersion string `json:"chartVersion,omitempty" yaml:"chartVersion,omitempty"`
	ChartRepoURL string `json:"chartRepoUrl,omitempty" yaml:"chartRepoUrl,omitempty"`
	ChartHome    string `json:"chartHome,omitempty" yaml:"chartHome,omitempty"`
	// Use chartRelease to keep compatible with old exec plugin
	ChartRepoName    string                 `json:"chartRelease,omitempty" yaml:"chartRelease,omitempty"`
	HelmBin          string                 `json:"helmBin,omitempty" yaml:"helmBin,omitempty"`
	HelmHome         string                 `json:"helmHome,omitempty" yaml:"helmHome,omitempty"`
	Values           string                 `json:"values,omitempty" yaml:"values,omitempty"`
	ValuesLocal      map[string]interface{} `json:"valuesLocal,omitempty" yaml:"valuesLocal,omitempty"`
	ValuesMerge      string                 `json:"valuesMerge,omitempty" yaml:"valuesMerge,omitempty"`
	ReleaseName      string                 `json:"releaseName,omitempty" yaml:"releaseName,omitempty"`
	ReleaseNamespace string                 `json:"releaseNamespace,omitempty" yaml:"releaseNamespace,omitempty"`
	ExtraArgs        []string               `json:"extraArgs,omitempty" yaml:"extraArgs,omitempty"`
}
