package features

import (
	"fmt"
	"os/exec"
	"regexp"
	"sort"
	"strconv"
	"testing"

	"github.com/google/go-cmp/cmp"

	"k8s.io/apimachinery/pkg/util/version"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/component-base/featuregate"
)

func TestHistory(t *testing.T) {
	checkFeatures(t, defaultVersionedKubernetesFeatureGates)
	checkFeatures(t, genericfeatures.GetDefaultVersionedKubernetesFeatureGatesForTest())
}
func checkFeatures(t *testing.T, versionedFeatures map[featuregate.Feature]featuregate.VersionedSpecs) {
	t.Helper()
	featureNames := []featuregate.Feature{}
	for featureName := range versionedFeatures {
		featureNames = append(featureNames, featureName)
	}
	sort.Slice(featureNames, func(i, j int) bool { return featureNames[i] < featureNames[j] })
	for _, featureName := range featureNames {
		specs := versionedFeatures[featureName]

		// Make sure feature didn't exist at spec[0].Version.Minor - 1
		if _, exists := getFeatures(t, fmt.Sprintf("release-1.%d", specs[0].Version.Minor()-1))[featureName]; exists {
			t.Errorf("feature %s existed prior to first spec entry of %s", featureName, specs[0].Version.String())
		}

		// Build history from first version until now
		lastSpec := featuregate.FeatureSpec{}
		specHistory := featuregate.VersionedSpecs{}
		specHistoryError := false
		for minor := int(specs[0].Version.Minor()); minor < 32; minor++ {
			branch := fmt.Sprintf("release-1.%d", minor)
			featuresForVersion := getFeatures(t, branch)
			featureSpecAtVersion, exists := getFeature(featureName, featuresForVersion)
			if exists {
				if featureSpecAtVersion != lastSpec {
					lastSpec = featureSpecAtVersion
					featureSpecAtVersion.Version = version.MustParse(fmt.Sprintf("1.%d", minor))
					specHistory = append(specHistory, featureSpecAtVersion)
				}
			} else {
				t.Errorf("error building history for %s at %s", featureName, branch)
				specHistoryError = true
				break
			}
		}

		// Make sure all specs are accurate at the versions they identify
		for _, spec := range specs {
			minor := spec.Version.Minor()
			branch := fmt.Sprintf("release-1.%d", minor)
			if minor == 32 {
				branch = "master"
			}
			featuresForVersion := getFeatures(t, branch)
			featureSpecAtVersion, exists := getFeature(featureName, featuresForVersion)
			if !exists {
				// Tolerate things not found at master... they might already be versioned and will show up in the PR diff
				if branch != "master" {
					t.Errorf("feature %s not found at 1.%d", featureName, minor)
				}
				continue
			}
			spec.Version = nil
			if featureSpecAtVersion != spec {
				t.Errorf("feature %s at %s doesn't match:\n\t%s: %#v\n\t%"+strconv.Itoa(len(branch))+"s: %#v", featureName, branch, branch, featureSpecAtVersion, "HEAD", spec)
				continue
			}
		}

		if !specHistoryError {
			// Make sure specs accurately identify the exact transition versions
			specsWithout132 := featuregate.VersionedSpecs{}
			for _, spec := range specs {
				if spec.Version.Minor() != 32 {
					specsWithout132 = append(specsWithout132, spec)
				}
			}
			if diff := cmp.Diff(specsWithout132, specHistory, cmp.Comparer(func(a, b *version.Version) bool { return a.String() == b.String() })); diff != "" {
				t.Errorf("unexpected feature %s history diff:\n%s", featureName, diff)
			}
		}
	}
}

func getFeature(name featuregate.Feature, features map[featuregate.Feature]featuregate.FeatureSpec) (featuregate.FeatureSpec, bool) {
	if spec, ok := features[name]; ok {
		return spec, true
	}
	for _, alias := range aliases[name] {
		if spec, ok := features[alias]; ok {
			return spec, true
		}
	}
	return featuregate.FeatureSpec{}, false
}

var aliases = map[featuregate.Feature][]featuregate.Feature{
	// https://github.com/kubernetes/kubernetes/pull/111090/files#diff-71e3b98f9a6bbf5b8421e26a7ba0c079f397cd8d49abacdad943c66a4f44f03dR821
	"UserNamespacesSupport": {"UserNamespacesStatelessPodsSupport"},
	// https://github.com/kubernetes/kubernetes/pull/63437/files#diff-71e3b98f9a6bbf5b8421e26a7ba0c079f397cd8d49abacdad943c66a4f44f03dR162
	"CustomCPUCFSQuotaPeriod": {"CPUCFSQuotaPeriod"},
	// split from DynamicResourceAllocation in 1.29
	// https://github.com/kubernetes/kubernetes/pull/125488/files#diff-71e3b98f9a6bbf5b8421e26a7ba0c079f397cd8d49abacdad943c66a4f44f03dR240
	"DRAControlPlaneController": {"DynamicResourceAllocation"},
	// Incorrect casing in initial merge
	// https://github.com/kubernetes/kubernetes/pull/121456/files#diff-71e3b98f9a6bbf5b8421e26a7ba0c079f397cd8d49abacdad943c66a4f44f03dR718
	"RuntimeClassInImageCriApi": {"RuntimeClassInImageCriAPI"},
}

var featuresForVersion = map[string]map[featuregate.Feature]featuregate.FeatureSpec{
	// Locations >=1.6 were:
	// https://github.com/kubernetes/kubernetes/blob/release-1.6/pkg/features/kube_features.go
	// https://github.com/kubernetes/kubernetes/blob/release-1.6/staging/src/k8s.io/apiserver/pkg/features/kube_features.go

	// File location was different <=1.5, pre-populate with remaining feature gates from those versions

	// https://github.com/kubernetes/kubernetes/blob/release-1.5/pkg/util/config/feature_gate.go#L64
	"release-1.5": map[featuregate.Feature]featuregate.FeatureSpec{
		"AppArmor": featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta},
	},
	// https://github.com/kubernetes/kubernetes/blob/release-1.4/pkg/util/config/feature_gate.go#L55-L59
	"release-1.4": map[featuregate.Feature]featuregate.FeatureSpec{
		"AppArmor": featuregate.FeatureSpec{Default: true, PreRelease: featuregate.Beta},
	},
	// feature gates were added in 1.4
	"release-1.3": map[featuregate.Feature]featuregate.FeatureSpec{},
}

// This holds features the scraping misparses, or which were renamed
var overridesForVersion = map[string]map[featuregate.Feature]featuregate.FeatureSpec{}

func getFeatures(t *testing.T, branch string) map[featuregate.Feature]featuregate.FeatureSpec {
	if featuresForVersion[branch] != nil {
		return featuresForVersion[branch]
	}
	t.Helper()
	files := []string{
		"pkg/features/kube_features.go",
		"staging/src/k8s.io/apiserver/pkg/features/kube_features.go",
	}
	results := map[featuregate.Feature]featuregate.FeatureSpec{}
	for _, file := range files {
		content, err := exec.Command("git", "show", fmt.Sprintf("upstream/%s:%s", branch, file)).CombinedOutput()
		if err != nil {
			t.Fatal(fmt.Errorf("error checking %s: %w", branch, err))
		}
		matches := myExp.FindAllStringSubmatch(string(content), -1)
		for _, match := range matches {
			feature := featuregate.Feature("")
			spec := featuregate.FeatureSpec{}
			for i, name := range myExp.SubexpNames() {
				switch name {
				case "":
					// Skip unnamed groups
				case "name":
					feature = featuregate.Feature(match[i])
				case "default":
					spec.Default = match[i] == "true"
				case "prerelease":
					switch match[i] {
					case "GA":
						spec.PreRelease = featuregate.GA
					case "Beta":
						spec.PreRelease = featuregate.Beta
					case "Alpha":
						spec.PreRelease = featuregate.Alpha
					case "Deprecated":
						spec.PreRelease = featuregate.Deprecated
					default:
						t.Fatalf("unexpected prerelease %s", match[i])
					}
				case "lock":
					spec.LockToDefault = match[i] == "true"
				default:
					t.Fatalf("unexpected name %s", name)
				}
			}
			results[feature] = spec
		}
	}
	// add in manual overrides
	for k, v := range overridesForVersion[branch] {
		results[k] = v
	}
	// fmt.Println(branch, len(results))
	featuresForVersion[branch] = results
	return results
}

var myExp = regexp.MustCompile(`[. \t](?P<name>[A-Za-z0-9]+):\s*\{Default: (?P<default>true|false), PreRelease: .*?(?P<prerelease>GA|Beta|Alpha|Deprecated)(, LockToDefault: (?P<lock>true|false))?`)

func main() {
	match := myExp.FindStringSubmatch("1234.5678.9")
	result := make(map[string]string)
	for i, name := range myExp.SubexpNames() {
		if i != 0 && name != "" {
			result[name] = match[i]
		}
	}
	fmt.Printf("by name: %s %s\n", result["first"], result["second"])
}
