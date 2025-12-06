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

package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"k8s.io/apimachinery/pkg/util/version"
	"k8s.io/component-base/featuregate"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	_ "k8s.io/kubernetes/pkg/features"
)

var (
	sortBy      = flag.String("sort", "name", "Sort by: name, stage, alpha, beta, ga, deprecated")
	reverse     = flag.Bool("reverse", false, "Reverse sort order")
	output      = flag.String("output", "", "Output file path (stdout if empty)")
	filterStage = flag.String("stage", "", "Filter by stage: alpha, beta, ga, deprecated")
	format      = flag.String("format", "markdown", "Output format: markdown, json")
)

// FeatureGateJSON represents a feature gate in JSON output format
type FeatureGateJSON struct {
	Name         string      `json:"name"`
	Stages       []StageJSON `json:"stages"`
	Dependencies []string    `json:"dependencies,omitempty"`
}

// StageJSON represents a stage in the feature gate lifecycle
type StageJSON struct {
	Stage       string `json:"stage"`
	FromVersion string `json:"fromVersion"`
	ToVersion   string `json:"toVersion,omitempty"`
	Default     bool   `json:"defaultValue"`
	Locked      bool   `json:"locked,omitempty"`
}

// stageInfo holds version range info for a single stage entry
type stageInfo struct {
	fromVersion string
	toVersion   string
	defaultOn   bool
	locked      bool
}

// featureInfo holds processed information about a feature gate (for markdown)
type featureInfo struct {
	name         string
	stage        string
	stageOrder   int // for sorting: 1=Alpha, 2=Beta, 3=GA, 4=Deprecated
	stageDisplay string
	alphaStages  []stageInfo
	alphaVersion *version.Version
	betaStages   []stageInfo
	betaVersion  *version.Version
	gaStages     []stageInfo
	gaVersion    *version.Version
	depStages    []stageInfo
	depVersion   *version.Version
	deps         string
	linkCode     string
	linkKEPs     string
}

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags]\n\nFlags:\n", os.Args[0])
		flag.PrintDefaults()
		fmt.Fprintf(os.Stderr, "\nExamples:\n")
		fmt.Fprintf(os.Stderr, "  %s                          # Markdown output (default)\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -format=json             # JSON output\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -sort=stage              # Sort by current stage\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -sort=alpha              # Sort by alpha version\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -sort=ga -reverse        # Sort by GA version, newest first\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -stage=alpha             # Show only alpha features\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -output=features.json    # Write to file\n", os.Args[0])
	}
	flag.Parse()

	var result string
	if *format == "json" {
		result = generateJSON(*filterStage)
	} else {
		result = generateMarkdown(*sortBy, *reverse, *filterStage)
	}

	if *output == "" {
		fmt.Print(result)
	} else {
		dir := filepath.Dir(*output)
		if dir != "" && dir != "." {
			if err := os.MkdirAll(dir, 0755); err != nil {
				fmt.Fprintf(os.Stderr, "failed to create output directory: %v\n", err)
				os.Exit(1)
			}
		}
		if err := os.WriteFile(*output, []byte(result), 0644); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write output file: %v\n", err)
			os.Exit(1)
		}
	}
}

func generateJSON(filterStage string) string {
	// Get all versioned feature specs and dependencies using public methods
	allFeatures := utilfeature.DefaultMutableFeatureGate.GetAllVersioned()
	allDependencies := utilfeature.DefaultMutableFeatureGate.Dependencies()

	// Build feature list
	features := make([]FeatureGateJSON, 0, len(allFeatures))

	for featureName, specs := range allFeatures {
		feature := string(featureName)

		// Sort specs by version to process in order
		sort.Sort(featuregate.VersionedSpecs(specs))

		fg := FeatureGateJSON{
			Name:   feature,
			Stages: make([]StageJSON, 0),
		}

		// Track versions for each stage to compute toVersion
		type stageEntry struct {
			stage       string
			fromVersion string
			defaultVal  bool
			locked      bool
		}
		var entries []stageEntry

		for _, spec := range specs {
			stageName := ""
			switch spec.PreRelease {
			case featuregate.Alpha:
				stageName = "alpha"
			case featuregate.Beta:
				stageName = "beta"
			case featuregate.GA:
				stageName = "stable"
			case featuregate.Deprecated:
				stageName = "deprecated"
			}

			entries = append(entries, stageEntry{
				stage:       stageName,
				fromVersion: spec.Version.String(),
				defaultVal:  spec.Default,
				locked:      spec.LockToDefault,
			})
		}

		// Convert entries to stages with toVersion
		for i, entry := range entries {
			stage := StageJSON{
				Stage:       entry.stage,
				FromVersion: entry.fromVersion,
				Default:     entry.defaultVal,
				Locked:      entry.locked,
			}

			// Find toVersion: look for next entry that represents a change
			// (different stage, different default, or different locked status)
			for j := i + 1; j < len(entries); j++ {
				if entries[j].stage != entry.stage ||
					entries[j].defaultVal != entry.defaultVal ||
					entries[j].locked != entry.locked {
					// toVersion is one minor version before the next entry's fromVersion
					stage.ToVersion = getPreviousMinorVersion(entries[j].fromVersion)
					break
				}
			}

			fg.Stages = append(fg.Stages, stage)
		}

		// Deduplicate stages - keep first occurrence of each stage with merged version range
		fg.Stages = deduplicateStages(fg.Stages)

		// Apply stage filter
		if filterStage != "" {
			hasStage := false
			for _, s := range fg.Stages {
				if strings.EqualFold(s.Stage, filterStage) || (filterStage == "ga" && s.Stage == "stable") {
					hasStage = true
					break
				}
			}
			if !hasStage {
				continue
			}
		}

		// Get dependencies
		deps := allDependencies[featuregate.Feature(feature)]
		if len(deps) > 0 {
			fg.Dependencies = make([]string, len(deps))
			for i, d := range deps {
				fg.Dependencies[i] = string(d)
			}
		}

		features = append(features, fg)
	}

	// Sort by name
	sort.Slice(features, func(i, j int) bool {
		return features[i].Name < features[j].Name
	})

	jsonBytes, err := json.MarshalIndent(features, "", "  ")
	if err != nil {
		fmt.Fprintf(os.Stderr, "failed to marshal JSON: %v\n", err)
		os.Exit(1)
	}

	return string(jsonBytes) + "\n"
}

// getPreviousMinorVersion returns the previous minor version (e.g., "1.30" -> "1.29")
func getPreviousMinorVersion(ver string) string {
	v, err := version.Parse(ver)
	if err != nil {
		return ""
	}
	minor := v.Minor()
	if minor > 0 {
		return fmt.Sprintf("%d.%d", v.Major(), minor-1)
	}
	return ""
}

// deduplicateStages merges consecutive entries with the same stage, default, and locked values.
// If defaultValue or locked changes within the same stage, we keep separate entries to preserve
// the full lifecycle history (e.g., beta disabled in 1.23, beta enabled in 1.24).
func deduplicateStages(stages []StageJSON) []StageJSON {
	if len(stages) == 0 {
		return stages
	}

	result := make([]StageJSON, 0)
	current := stages[0]

	for i := 1; i < len(stages); i++ {
		// Only merge if stage, default, AND locked are all the same
		if stages[i].Stage == current.Stage &&
			stages[i].Default == current.Default &&
			stages[i].Locked == current.Locked {
			// Same stage with same settings, extend the version range
			current.ToVersion = stages[i].ToVersion
		} else {
			// Different stage or settings changed, save current and start new
			result = append(result, current)
			current = stages[i]
		}
	}
	result = append(result, current)

	return result
}

func generateMarkdown(sortBy string, reverseSort bool, filterStage string) string {
	// Get all versioned feature specs and dependencies using public methods
	allFeatures := utilfeature.DefaultMutableFeatureGate.GetAllVersioned()
	allDependencies := utilfeature.DefaultMutableFeatureGate.Dependencies()

	// Statistics counters
	var alphaCount, betaCount, gaCount, deprecatedCount int

	// Build feature info list
	features := make([]featureInfo, 0, len(allFeatures))

	for featureName, specs := range allFeatures {
		feature := string(featureName)
		info := featureInfo{
			name: feature,
		}

		// Sort specs by version to process in order
		sort.Sort(featuregate.VersionedSpecs(specs))

		// First pass: collect all entries with their stage info
		type rawEntry struct {
			version      *version.Version
			verStr       string
			isAlpha      bool
			isBeta       bool
			isGA         bool
			isDeprecated bool
			defaultOn    bool
			locked       bool
		}
		var entries []rawEntry

		for _, spec := range specs {
			entry := rawEntry{
				version:   spec.Version,
				verStr:    spec.Version.String(),
				defaultOn: spec.Default,
				locked:    spec.LockToDefault,
			}
			switch spec.PreRelease {
			case featuregate.Alpha:
				entry.isAlpha = true
			case featuregate.Beta:
				entry.isBeta = true
			case featuregate.GA:
				entry.isGA = true
			case featuregate.Deprecated:
				entry.isDeprecated = true
			}
			entries = append(entries, entry)

			// Track first version and current stage
			switch spec.PreRelease {
			case featuregate.Alpha:
				if info.alphaVersion == nil {
					info.alphaVersion = spec.Version
				}
				info.stage = "Alpha"
				info.stageOrder = 1
				info.stageDisplay = "Alpha"
				if spec.Default {
					info.stageDisplay += " :ballot_box_with_check:"
				}
				if spec.LockToDefault {
					info.stageDisplay += " :closed_lock_with_key:"
				}
			case featuregate.Beta:
				if info.betaVersion == nil {
					info.betaVersion = spec.Version
				}
				info.stage = "Beta"
				info.stageOrder = 2
				info.stageDisplay = "Beta"
				if spec.Default {
					info.stageDisplay += " :ballot_box_with_check:"
				}
				if spec.LockToDefault {
					info.stageDisplay += " :closed_lock_with_key:"
				}
			case featuregate.GA:
				if info.gaVersion == nil {
					info.gaVersion = spec.Version
				}
				info.stage = "GA"
				info.stageOrder = 3
				info.stageDisplay = "GA"
				if spec.Default {
					info.stageDisplay += " :ballot_box_with_check:"
				}
				if spec.LockToDefault {
					info.stageDisplay += " :closed_lock_with_key:"
				}
			case featuregate.Deprecated:
				if info.depVersion == nil {
					info.depVersion = spec.Version
				}
				info.stage = "Deprecated"
				info.stageOrder = 4
				info.stageDisplay = "Deprecated"
				if spec.Default {
					info.stageDisplay += " :ballot_box_with_check:"
				} else {
					info.stageDisplay += " :red_circle:"
				}
				if spec.LockToDefault {
					info.stageDisplay += " :closed_lock_with_key:"
				}
			}
		}

		// Second pass: build stage ranges with toVersion, grouping by stage+default+locked
		for i := 0; i < len(entries); i++ {
			entry := entries[i]
			si := stageInfo{
				fromVersion: entry.verStr,
				defaultOn:   entry.defaultOn,
				locked:      entry.locked,
			}

			// Find toVersion: look for next entry that represents a change
			for j := i + 1; j < len(entries); j++ {
				sameStage := (entry.isAlpha == entries[j].isAlpha &&
					entry.isBeta == entries[j].isBeta &&
					entry.isGA == entries[j].isGA &&
					entry.isDeprecated == entries[j].isDeprecated)
				if !sameStage ||
					entries[j].defaultOn != entry.defaultOn ||
					entries[j].locked != entry.locked {
					si.toVersion = getPreviousMinorVersion(entries[j].verStr)
					break
				}
			}

			// Skip if this is a duplicate (same stage, same default, same locked as previous)
			// by checking if we should merge with previous entry
			if entry.isAlpha {
				if len(info.alphaStages) > 0 {
					last := &info.alphaStages[len(info.alphaStages)-1]
					if last.defaultOn == si.defaultOn && last.locked == si.locked {
						last.toVersion = si.toVersion
						continue
					}
				}
				info.alphaStages = append(info.alphaStages, si)
			} else if entry.isBeta {
				if len(info.betaStages) > 0 {
					last := &info.betaStages[len(info.betaStages)-1]
					if last.defaultOn == si.defaultOn && last.locked == si.locked {
						last.toVersion = si.toVersion
						continue
					}
				}
				info.betaStages = append(info.betaStages, si)
			} else if entry.isGA {
				if len(info.gaStages) > 0 {
					last := &info.gaStages[len(info.gaStages)-1]
					if last.defaultOn == si.defaultOn && last.locked == si.locked {
						last.toVersion = si.toVersion
						continue
					}
				}
				info.gaStages = append(info.gaStages, si)
			} else if entry.isDeprecated {
				if len(info.depStages) > 0 {
					last := &info.depStages[len(info.depStages)-1]
					if last.defaultOn == si.defaultOn && last.locked == si.locked {
						last.toVersion = si.toVersion
						continue
					}
				}
				info.depStages = append(info.depStages, si)
			}
		}

		// Update statistics
		switch info.stage {
		case "Alpha":
			alphaCount++
		case "Beta":
			betaCount++
		case "GA":
			gaCount++
		case "Deprecated":
			deprecatedCount++
		}

		// Apply stage filter
		if filterStage != "" && !strings.EqualFold(info.stage, filterStage) {
			continue
		}

		// Get dependencies for this feature as bullet list (one per line)
		deps := allDependencies[featuregate.Feature(feature)]
		if len(deps) > 0 {
			depItems := make([]string, len(deps))
			for i, d := range deps {
				depItems[i] = "• " + string(d)
			}
			info.deps = strings.Join(depItems, "<br>")
		}

		info.linkCode = fmt.Sprintf("[code](https://cs.k8s.io/?q=%%5Cb%s%%5Cb&i=nope&files=&excludeFiles=CHANGELOG&repos=kubernetes/kubernetes)", feature)
		info.linkKEPs = fmt.Sprintf("[KEPs](https://cs.k8s.io/?q=%%5Cb%s%%5Cb&i=nope&files=&excludeFiles=CHANGELOG&repos=kubernetes/enhancements)", feature)

		features = append(features, info)
	}

	// Sort features based on sortBy parameter
	sortFeatures(features, sortBy, reverseSort)

	// Build markdown output
	var sb strings.Builder
	sb.WriteString("# Kubernetes Feature Gates\n\n")

	if filterStage != "" {
		sb.WriteString(fmt.Sprintf("*Showing only %s features (%d)*\n\n", strings.ToUpper(filterStage), len(features)))
	}

	sb.WriteString("| Feature | Stage | Alpha | Beta | GA | Deprecated | Dependencies | Links |\n")
	sb.WriteString("|---------|-------|-------|------|----|------------|--------------|-------|\n")

	for _, info := range features {
		sb.WriteString(fmt.Sprintf("| %s | %s | %s | %s | %s | %s | %s | %s %s |\n",
			info.name, info.stageDisplay,
			formatStageRanges(info.alphaStages, false),
			formatStageRanges(info.betaStages, false),
			formatStageRanges(info.gaStages, false),
			formatStageRanges(info.depStages, true),
			info.deps, info.linkCode, info.linkKEPs))
	}

	sb.WriteString("\n## Legend\n\n")
	sb.WriteString("| Symbol | Meaning |\n")
	sb.WriteString("|--------|--------|\n")
	sb.WriteString("| :ballot_box_with_check: | Enabled by default |\n")
	sb.WriteString("| :red_circle: | Disabled by default |\n")
	sb.WriteString("| :closed_lock_with_key: | Locked to default (cannot be changed) |\n")

	// Always show statistics at the end
	total := alphaCount + betaCount + gaCount + deprecatedCount
	sb.WriteString("\n## Statistics\n\n")
	sb.WriteString("| Stage | Count | Percentage |\n")
	sb.WriteString("|-------|-------|------------|\n")
	sb.WriteString(fmt.Sprintf("| Alpha | %d | %.1f%% |\n", alphaCount, float64(alphaCount)*100/float64(total)))
	sb.WriteString(fmt.Sprintf("| Beta | %d | %.1f%% |\n", betaCount, float64(betaCount)*100/float64(total)))
	sb.WriteString(fmt.Sprintf("| GA | %d | %.1f%% |\n", gaCount, float64(gaCount)*100/float64(total)))
	sb.WriteString(fmt.Sprintf("| Deprecated | %d | %.1f%% |\n", deprecatedCount, float64(deprecatedCount)*100/float64(total)))
	sb.WriteString(fmt.Sprintf("| **Total** | **%d** | **100%%** |\n", total))

	return sb.String()
}

// formatStageRanges formats a list of stage ranges for markdown display
// Each range shows "fromVersion-toVersion" with indicators for default/locked state
// If showDisabledIndicator is true, shows :red_circle: for disabled (used for Deprecated)
func formatStageRanges(stages []stageInfo, showDisabledIndicator bool) string {
	if len(stages) == 0 {
		return ""
	}

	var parts []string
	for _, s := range stages {
		var versionRange string
		if s.toVersion != "" && s.toVersion != s.fromVersion {
			versionRange = s.fromVersion + "-" + s.toVersion
		} else {
			versionRange = s.fromVersion
		}

		indicator := ""
		if s.defaultOn {
			indicator += " :ballot_box_with_check:"
		} else if showDisabledIndicator {
			indicator += " :red_circle:"
		}
		if s.locked {
			indicator += " :closed_lock_with_key:"
		}

		parts = append(parts, versionRange+indicator)
	}

	return strings.Join(parts, ", ")
}

func sortFeatures(features []featureInfo, sortBy string, reverseSort bool) {
	var less func(i, j int) bool

	switch sortBy {
	case "stage":
		less = func(i, j int) bool {
			if features[i].stageOrder != features[j].stageOrder {
				return features[i].stageOrder < features[j].stageOrder
			}
			return features[i].name < features[j].name
		}
	case "alpha":
		less = func(i, j int) bool {
			return compareVersions(features[i].alphaVersion, features[j].alphaVersion, features[i].name, features[j].name)
		}
	case "beta":
		less = func(i, j int) bool {
			return compareVersions(features[i].betaVersion, features[j].betaVersion, features[i].name, features[j].name)
		}
	case "ga":
		less = func(i, j int) bool {
			return compareVersions(features[i].gaVersion, features[j].gaVersion, features[i].name, features[j].name)
		}
	case "deprecated":
		less = func(i, j int) bool {
			return compareVersions(features[i].depVersion, features[j].depVersion, features[i].name, features[j].name)
		}
	default: // "name"
		less = func(i, j int) bool {
			return features[i].name < features[j].name
		}
	}

	if reverseSort {
		sort.Slice(features, func(i, j int) bool {
			return less(j, i)
		})
	} else {
		sort.Slice(features, less)
	}
}

// compareVersions compares two versions, putting nil versions last
func compareVersions(v1, v2 *version.Version, name1, name2 string) bool {
	if v1 == nil && v2 == nil {
		return name1 < name2
	}
	if v1 == nil {
		return false // nil goes last
	}
	if v2 == nil {
		return true // nil goes last
	}
	if v1.EqualTo(v2) {
		return name1 < name2
	}
	return v1.LessThan(v2)
}
