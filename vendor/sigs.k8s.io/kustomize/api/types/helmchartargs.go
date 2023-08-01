// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package types

import "path/filepath"

const HelmDefaultHome = "charts"

type HelmGlobals struct {
	// ChartHome is a file path, relative to the kustomization root,
	// to a directory containing a subdirectory for each chart to be
	// included in the kustomization.
	// The default value of this field is "charts".
	// So, for example, kustomize looks for the minecraft chart
	// at {kustomizationRoot}/{ChartHome}/minecraft.
	// If the chart is there at build time, kustomize will use it as found,
	// and not check version numbers or dates.
	// If the chart is not there, kustomize will attempt to pull it
	// using the version number specified in the kustomization file,
	// and put it there.  To suppress the pull attempt, simply assure
	// that the chart is already there.
	ChartHome string `json:"chartHome,omitempty" yaml:"chartHome,omitempty"`

	// ConfigHome defines a value that kustomize should pass to helm via
	// the HELM_CONFIG_HOME environment variable.  kustomize doesn't attempt
	// to read or write this directory.
	// If omitted, {tmpDir}/helm is used, where {tmpDir} is some temporary
	// directory created by kustomize for the benefit of helm.
	// Likewise, kustomize sets
	//   HELM_CACHE_HOME={ConfigHome}/.cache
	//   HELM_DATA_HOME={ConfigHome}/.data
	// for the helm subprocess.
	ConfigHome string `json:"configHome,omitempty" yaml:"configHome,omitempty"`
}

type HelmChart struct {
	// Name is the name of the chart, e.g. 'minecraft'.
	Name string `json:"name,omitempty" yaml:"name,omitempty"`

	// Version is the version of the chart, e.g. '3.1.3'
	Version string `json:"version,omitempty" yaml:"version,omitempty"`

	// Repo is a URL locating the chart on the internet.
	// This is the argument to helm's  `--repo` flag, e.g.
	// `https://itzg.github.io/minecraft-server-charts`.
	Repo string `json:"repo,omitempty" yaml:"repo,omitempty"`

	// ReleaseName replaces RELEASE-NAME in chart template output,
	// making a particular inflation of a chart unique with respect to
	// other inflations of the same chart in a cluster. It's the first
	// argument to the helm `install` and `template` commands, i.e.
	//   helm install {RELEASE-NAME} {chartName}
	//   helm template {RELEASE-NAME} {chartName}
	// If omitted, the flag --generate-name is passed to 'helm template'.
	ReleaseName string `json:"releaseName,omitempty" yaml:"releaseName,omitempty"`

	// Namespace set the target namespace for a release. It is .Release.Namespace
	// in the helm template
	Namespace string `json:"namespace,omitempty" yaml:"namespace,omitempty"`

	// AdditionalValuesFiles are local file paths to values files to be used in
	// addition to either the default values file or the values specified in ValuesFile.
	AdditionalValuesFiles []string `json:"additionalValuesFiles,omitempty" yaml:"additionalValuesFiles,omitempty"`

	// ValuesFile is a local file path to a values file to use _instead of_
	// the default values that accompanied the chart.
	// The default values are in '{ChartHome}/{Name}/values.yaml'.
	ValuesFile string `json:"valuesFile,omitempty" yaml:"valuesFile,omitempty"`

	// ValuesInline holds value mappings specified directly,
	// rather than in a separate file.
	ValuesInline map[string]interface{} `json:"valuesInline,omitempty" yaml:"valuesInline,omitempty"`

	// ValuesMerge specifies how to treat ValuesInline with respect to Values.
	// Legal values: 'merge', 'override', 'replace'.
	// Defaults to 'override'.
	ValuesMerge string `json:"valuesMerge,omitempty" yaml:"valuesMerge,omitempty"`

	// IncludeCRDs specifies if Helm should also generate CustomResourceDefinitions.
	// Defaults to 'false'.
	IncludeCRDs bool `json:"includeCRDs,omitempty" yaml:"includeCRDs,omitempty"` //nolint: tagliatelle

	// SkipHooks sets the --no-hooks flag when calling helm template. This prevents
	// helm from erroneously rendering test templates.
	SkipHooks bool `json:"skipHooks,omitempty" yaml:"skipHooks,omitempty"`

	// ApiVersions is the kubernetes apiversions used for Capabilities.APIVersions
	ApiVersions []string `json:"apiVersions,omitempty" yaml:"apiVersions,omitempty"`

	// NameTemplate is for specifying the name template used to name the release.
	NameTemplate string `json:"nameTemplate,omitempty" yaml:"nameTemplate,omitempty"`

	// SkipTests skips tests from templated output.
	SkipTests bool `json:"skipTests,omitempty" yaml:"skipTests,omitempty"`
}

// HelmChartArgs contains arguments to helm.
// Deprecated.  Use HelmGlobals and HelmChart instead.
type HelmChartArgs struct {
	ChartName        string                 `json:"chartName,omitempty" yaml:"chartName,omitempty"`
	ChartVersion     string                 `json:"chartVersion,omitempty" yaml:"chartVersion,omitempty"`
	ChartRepoURL     string                 `json:"chartRepoUrl,omitempty" yaml:"chartRepoUrl,omitempty"`
	ChartHome        string                 `json:"chartHome,omitempty" yaml:"chartHome,omitempty"`
	ChartRepoName    string                 `json:"chartRepoName,omitempty" yaml:"chartRepoName,omitempty"`
	HelmBin          string                 `json:"helmBin,omitempty" yaml:"helmBin,omitempty"`
	HelmHome         string                 `json:"helmHome,omitempty" yaml:"helmHome,omitempty"`
	Values           string                 `json:"values,omitempty" yaml:"values,omitempty"`
	ValuesLocal      map[string]interface{} `json:"valuesLocal,omitempty" yaml:"valuesLocal,omitempty"`
	ValuesMerge      string                 `json:"valuesMerge,omitempty" yaml:"valuesMerge,omitempty"`
	ReleaseName      string                 `json:"releaseName,omitempty" yaml:"releaseName,omitempty"`
	ReleaseNamespace string                 `json:"releaseNamespace,omitempty" yaml:"releaseNamespace,omitempty"`
	ExtraArgs        []string               `json:"extraArgs,omitempty" yaml:"extraArgs,omitempty"`
}

// SplitHelmParameters splits helm parameters into
// per-chart params and global chart-independent parameters.
func SplitHelmParameters(
	oldArgs []HelmChartArgs) (charts []HelmChart, globals HelmGlobals) {
	for i, old := range oldArgs {
		charts = append(charts, makeHelmChartFromHca(&oldArgs[i]))
		if old.HelmHome != "" {
			// last non-empty wins
			globals.ConfigHome = old.HelmHome
		}
		if old.ChartHome != "" {
			// last non-empty wins
			globals.ChartHome = old.ChartHome
		}
	}
	return charts, globals
}

func makeHelmChartFromHca(old *HelmChartArgs) (c HelmChart) {
	c.Name = old.ChartName
	c.Version = old.ChartVersion
	c.Repo = old.ChartRepoURL
	c.ValuesFile = old.Values
	c.ValuesInline = old.ValuesLocal
	c.ValuesMerge = old.ValuesMerge
	c.ReleaseName = old.ReleaseName
	return
}

func (h HelmChart) AsHelmArgs(absChartHome string) []string {
	args := []string{"template"}
	if h.ReleaseName != "" {
		args = append(args, h.ReleaseName)
	} else {
		// AFAICT, this doesn't work as intended due to a bug in helm.
		// See https://github.com/helm/helm/issues/6019
		// I've tried placing the flag before and after the name argument.
		args = append(args, "--generate-name")
	}
	if h.Name != "" {
		args = append(args, filepath.Join(absChartHome, h.Name))
	}
	if h.Namespace != "" {
		args = append(args, "--namespace", h.Namespace)
	}
	if h.NameTemplate != "" {
		args = append(args, "--name-template", h.NameTemplate)
	}

	if h.ValuesFile != "" {
		args = append(args, "-f", h.ValuesFile)
	}
	for _, valuesFile := range h.AdditionalValuesFiles {
		args = append(args, "-f", valuesFile)
	}

	for _, apiVer := range h.ApiVersions {
		args = append(args, "--api-versions", apiVer)
	}
	if h.IncludeCRDs {
		args = append(args, "--include-crds")
	}
	if h.SkipTests {
		args = append(args, "--skip-tests")
	}
	if h.SkipHooks {
		args = append(args, "--no-hooks")
	}
	return args
}
