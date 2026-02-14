/*
Copyright 2025 The Kubernetes Authors.

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

package v1

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"

	"github.com/blang/semver/v4"
	"go.yaml.in/yaml/v2"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

var (
	labelExpr      = `[a-zA-Z_][a-zA-Z0-9_]*`
	metricNameExpr = `[a-zA-Z_:][a-zA-Z0-9_:]*`
)

// Validate validates a MetricsConfiguration.
func Validate(c *MetricsConfiguration, currentVersion semver.Version, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}
	if c == nil {
		return errs
	}
	errs = append(errs, validateShowHiddenMetricsVersion(currentVersion, c.ShowHiddenMetricsForVersion, fldPath.Child("showHiddenMetricsForVersion"))...)
	errs = append(errs, validateDisabledMetrics(c.DisabledMetrics, fldPath.Child("disabledMetrics"))...)
	errs = append(errs, validateAllowListMapping(c.AllowListMapping, fldPath.Child("allowListMapping"))...)
	errs = append(errs, validateAllowListMappingManifest(c.AllowListMappingManifest, fldPath.Child("allowListMappingManifest"))...)

	return errs
}

func validateAllowListMapping(allowListMapping map[string]string, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}
	allowListMappingKeyRegex := regexp.MustCompile(metricNameExpr + `,` + labelExpr)
	for k := range allowListMapping {
		if allowListMappingKeyRegex.FindString(k) != k {
			return append(errs, field.Invalid(fldPath, allowListMapping, fmt.Sprintf("must have keys with format `metricName,labelName` where metricName matches %q and labelName matches %q", metricNameExpr, labelExpr)))
		}
	}

	return errs
}

// validateAllowListMappingManifest validates the allow list mapping manifest file.
// This function is used to validate the manifest file provided via the flag --allow-metric-labels-manifest, or the configuration file.
// In the former case, the path resolution is relative to the current working directory.
// In the latter case, the path resolution is relative to the configuration file's location, and components are required to pass in the resolved absolute path.
// NOTE: If its the latter case, components are expected to pass in the *absolute* path to the manifest file.
func validateAllowListMappingManifest(allowListMappingManifestPath string, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}
	if allowListMappingManifestPath == "" {
		return errs
	}
	data, err := os.ReadFile(filepath.Clean(allowListMappingManifestPath))
	if err != nil {
		return append(errs, field.Invalid(fldPath, allowListMappingManifestPath, fmt.Errorf("failed to read allow list manifest: %w", err).Error()))
	}
	allowListMapping := make(map[string]string)
	err = yaml.Unmarshal(data, &allowListMapping)
	if err != nil {
		return append(errs, field.Invalid(fldPath, allowListMappingManifestPath, fmt.Errorf("failed to parse allow list manifest: %w", err).Error()))
	}
	allowListMappingErrs := validateAllowListMapping(allowListMapping, fldPath)
	if len(allowListMappingErrs) > 0 {
		return append(errs, field.Invalid(fldPath, allowListMappingManifestPath, fmt.Sprintf("invalid allow list mapping in manifest: %v", allowListMappingErrs)))
	}

	return errs
}

func validateDisabledMetrics(names []string, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}
	metricNameRegex := regexp.MustCompile(`^` + metricNameExpr + `$`)
	for _, name := range names {
		if !metricNameRegex.MatchString(name) {
			return append(errs, field.Invalid(fldPath, names, fmt.Sprintf("must be fully qualified metric names matching %q, got %q", metricNameRegex.String(), name)))
		}
	}

	return errs
}

func validateShowHiddenMetricsVersion(currentVersion semver.Version, targetVersionStr string, fldPath *field.Path) field.ErrorList {
	errs := field.ErrorList{}
	if targetVersionStr == "" {
		return errs
	}

	validVersionStr := fmt.Sprintf("%d.%d", currentVersion.Major, currentVersion.Minor-1)
	if targetVersionStr != validVersionStr {
		return append(errs, field.Invalid(fldPath, targetVersionStr, fmt.Sprintf("must be omitted or have the value '%v'; only the previous minor version is allowed", validVersionStr)))
	}

	return errs
}

// ValidateShowHiddenMetricsVersionForKubeletBackwardCompatOnly validates the ShowHiddenMetricsForVersion field.
// TODO: This is kept here for backward compatibility in Kubelet (as metrics configuration fields were exposed on an individual basis earlier).
// TODO: Revisit this after Kubelet supports the new metrics configuration API: https://github.com/kubernetes/kubernetes/pull/123426
func ValidateShowHiddenMetricsVersionForKubeletBackwardCompatOnly(currentVersion semver.Version, targetVersionStr string) error {
	errs := validateShowHiddenMetricsVersion(currentVersion, targetVersionStr, field.NewPath("showHiddenMetricsForVersion"))
	if len(errs) > 0 {
		return fmt.Errorf("invalid showHiddenMetricsForVersion: %v", errs.ToAggregate().Error())
	}

	return nil
}
