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
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"

	"k8s.io/component-base/featuregate"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	_ "k8s.io/kubernetes/pkg/features"
)

func main() {
	var outputPath string
	if len(os.Args) == 2 {
		outputPath = os.Args[1]
	} else if len(os.Args) > 2 {
		fmt.Fprintf(os.Stderr, "usage: %s [output file path]\n", os.Args[0])
		os.Exit(1)
	}

	markdown := generateMarkdown()

	if outputPath == "" {
		fmt.Print(markdown)
	} else {
		dir := filepath.Dir(outputPath)
		if dir != "" && dir != "." {
			if err := os.MkdirAll(dir, 0755); err != nil {
				fmt.Fprintf(os.Stderr, "failed to create output directory: %v\n", err)
				os.Exit(1)
			}
		}
		if err := os.WriteFile(outputPath, []byte(markdown), 0644); err != nil {
			fmt.Fprintf(os.Stderr, "failed to write output file: %v\n", err)
			os.Exit(1)
		}
	}
}

func generateMarkdown() string {
	var sb strings.Builder
	sb.WriteString("| Feature | Alpha | Beta | GA | Deprecated | Links |\n")
	sb.WriteString("|---------|-------|------|----|----|----|\n")

	// Get all versioned feature specs using the public method
	allFeatures := utilfeature.DefaultMutableFeatureGate.GetAllVersioned()

	// Extract and sort feature names
	keys := make([]string, 0, len(allFeatures))
	for key := range allFeatures {
		keys = append(keys, string(key))
	}
	sort.Strings(keys)

	for _, feature := range keys {
		specs := allFeatures[featuregate.Feature(feature)]
		alpha, beta, ga, deprecated := "", "", "", ""
		for _, spec := range specs {
			switch spec.PreRelease {
			case featuregate.Alpha:
				if len(alpha) > 0 {
					alpha += ", "
				}
				alpha += spec.Version.String()
				if spec.Default {
					alpha += " :ballot_box_with_check:"
				}
				if spec.LockToDefault {
					alpha += " :closed_lock_with_key:"
				}
			case featuregate.Beta:
				if len(beta) > 0 {
					beta += ", "
				}
				beta += spec.Version.String()
				if spec.Default {
					beta += " :ballot_box_with_check:"
				}
				if spec.LockToDefault {
					beta += " :closed_lock_with_key:"
				}
			case featuregate.GA:
				if len(ga) > 0 {
					ga += ", "
				}
				ga += spec.Version.String()
				if spec.Default {
					ga += " :ballot_box_with_check:"
				}
				if spec.LockToDefault {
					ga += " :closed_lock_with_key:"
				}
			case featuregate.Deprecated:
				depVer := spec.Version.String()
				if spec.Default {
					depVer += " :ballot_box_with_check:"
				} else {
					depVer += " :red_circle:"
				}
				if spec.LockToDefault {
					depVer += " :closed_lock_with_key:"
				}
				if len(deprecated) > 0 {
					deprecated += ", "
					deprecated += depVer
				} else {
					deprecated = depVer
				}
			}
		}
		linkToCode := fmt.Sprintf("[code](https://cs.k8s.io/?q=%%5Cb%s%%5Cb&i=nope&files=&excludeFiles=CHANGELOG&repos=kubernetes/kubernetes)", feature)
		linkToEnhancements := fmt.Sprintf("[KEPs](https://cs.k8s.io/?q=%%5Cb%s%%5Cb&i=nope&files=&excludeFiles=CHANGELOG&repos=kubernetes/enhancements)", feature)
		sb.WriteString(fmt.Sprintf("| %s | %s | %s | %s | %s |%s %s\n", feature, alpha, beta, ga, deprecated, linkToCode, linkToEnhancements))
	}

	sb.WriteString("\n\n Legend: :ballot_box_with_check: - enabled, :red_circle: - disabled\n")
	sb.WriteString("\t\t:closed_lock_with_key: - locked to default\n")

	return sb.String()
}
