/*
Copyright 2024 The Kubernetes Authors.

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
	sortBy  = flag.String("sort", "name", "Sort by: name, stage, alpha, beta, ga, deprecated")
	reverse = flag.Bool("reverse", false, "Reverse sort order")
	output  = flag.String("output", "", "Output file path (stdout if empty)")
)

// featureInfo holds processed information about a feature gate
type featureInfo struct {
	name         string
	stage        string
	stageOrder   int // for sorting: 1=Alpha, 2=Beta, 3=GA, 4=Deprecated
	stageDisplay string
	alpha        string
	alphaVersion *version.Version
	beta         string
	betaVersion  *version.Version
	ga           string
	gaVersion    *version.Version
	deprecated   string
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
		fmt.Fprintf(os.Stderr, "  %s                          # Sort by name (default)\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -sort=stage              # Sort by current stage\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -sort=alpha              # Sort by alpha version\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -sort=ga -reverse        # Sort by GA version, newest first\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "  %s -output=features.md      # Write to file\n", os.Args[0])
	}
	flag.Parse()

	markdown := generateMarkdown(*sortBy, *reverse)

	if *output == "" {
		fmt.Print(markdown)
	} else {
		dir := filepath.Dir(*output)
		if dir != "" && dir != "." {
			if err := os.MkdirAll(dir, 0755); err != nil {
				fmt.Fprintf(os.Stderr, "failed to create output directory: %v\n", err)
				os.Exit(1)
			}
		}
		if err := os.WriteFile(*output, []byte(markdown), 0644); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write output file: %v\n", err)
			os.Exit(1)
		}
	}
}

func generateMarkdown(sortBy string, reverseSort bool) string {
	// Get all versioned feature specs and dependencies using public methods
	allFeatures := utilfeature.DefaultMutableFeatureGate.GetAllVersioned()
	allDependencies := utilfeature.DefaultMutableFeatureGate.Dependencies()

	// Build feature info list
	features := make([]featureInfo, 0, len(allFeatures))

	for featureName, specs := range allFeatures {
		feature := string(featureName)
		info := featureInfo{
			name: feature,
		}

		// Sort specs by version to process in order
		sort.Sort(featuregate.VersionedSpecs(specs))

		for _, spec := range specs {
			verStr := spec.Version.String()
			indicator := ""
			if spec.Default {
				indicator += " :ballot_box_with_check:"
			}
			if spec.LockToDefault {
				indicator += " :closed_lock_with_key:"
			}

			switch spec.PreRelease {
			case featuregate.Alpha:
				if len(info.alpha) > 0 {
					info.alpha += ", "
				}
				info.alpha += verStr + indicator
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
				if len(info.beta) > 0 {
					info.beta += ", "
				}
				info.beta += verStr + indicator
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
				if len(info.ga) > 0 {
					info.ga += ", "
				}
				info.ga += verStr + indicator
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
				depIndicator := ""
				if spec.Default {
					depIndicator += " :ballot_box_with_check:"
				} else {
					depIndicator += " :red_circle:"
				}
				if spec.LockToDefault {
					depIndicator += " :closed_lock_with_key:"
				}
				if len(info.deprecated) > 0 {
					info.deprecated += ", "
				}
				info.deprecated += verStr + depIndicator
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
	sb.WriteString("| Feature | Stage | Alpha | Beta | GA | Deprecated | Dependencies | Links |\n")
	sb.WriteString("|---------|-------|-------|------|----|------------|--------------|-------|\n")

	for _, info := range features {
		sb.WriteString(fmt.Sprintf("| %s | %s | %s | %s | %s | %s | %s | %s %s |\n",
			info.name, info.stageDisplay, info.alpha, info.beta, info.ga, info.deprecated, info.deps, info.linkCode, info.linkKEPs))
	}

	sb.WriteString("\n## Legend\n\n")
	sb.WriteString("| Symbol | Meaning |\n")
	sb.WriteString("|--------|--------|\n")
	sb.WriteString("| :ballot_box_with_check: | Enabled by default |\n")
	sb.WriteString("| :red_circle: | Disabled by default |\n")
	sb.WriteString("| :closed_lock_with_key: | Locked to default (cannot be changed) |\n")

	return sb.String()
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
