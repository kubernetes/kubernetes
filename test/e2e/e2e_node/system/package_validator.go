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

package system

import (
	"fmt"
	"io/ioutil"
	"os/exec"
	"strings"

	"k8s.io/apimachinery/pkg/util/errors"

	"github.com/blang/semver"
	"github.com/golang/glog"
)

// semVerDotsCount is the number of dots in a valid semantic version.
const semVerDotsCount int = 2

// packageManager is an interface that abstracts the basic operations of a
// package manager.
type packageManager interface {
	// getPackageVersion returns the version of the package given the
	// packageName, or an error if no such package exists.
	getPackageVersion(packageName string) (string, error)
}

// newPackageManager returns the package manager on the running machine, and an
// error if no package managers is available.
func newPackageManager() (packageManager, error) {
	if m, ok := newDPKG(); ok {
		return m, nil
	}
	return nil, fmt.Errorf("failed to find package manager")
}

// dpkg implements packageManager. It uses "dpkg-query" to retrieve package
// information.
type dpkg struct{}

// newDPKG returns a Debian package manager. It returns (nil, false) if no such
// package manager exists on the running machine.
func newDPKG() (packageManager, bool) {
	_, err := exec.LookPath("dpkg-query")
	if err != nil {
		return nil, false
	}
	return dpkg{}, true
}

// getPackageVersion returns the upstream package version for the package given
// the packageName, and an error if no such package exists.
func (_ dpkg) getPackageVersion(packageName string) (string, error) {
	output, err := exec.Command("dpkg-query", "--show", "--showformat='${Version}'", packageName).Output()
	if err != nil {
		return "", fmt.Errorf("dpkg-query failed: %s", err)
	}
	version := extractUpstreamVersion(string(output))
	if version == "" {
		return "", fmt.Errorf("no version information")
	}
	return version, nil
}

// packageValidator implements the Validator interface. It validates packages
// and their versions.
type packageValidator struct {
	reporter      Reporter
	kernelRelease string
	osDistro      string
}

// Name returns the name of the package validator.
func (self *packageValidator) Name() string {
	return "package"
}

// Validate checks packages and their versions against the spec using the
// package manager on the running machine, and returns an error on any
// package/version mismatch.
func (self *packageValidator) Validate(spec SysSpec) (error, error) {
	if len(spec.PackageSpecs) == 0 {
		return nil, nil
	}

	var err error
	if self.kernelRelease, err = getKernelRelease(); err != nil {
		return nil, err
	}
	if self.osDistro, err = getOSDistro(); err != nil {
		return nil, err
	}

	manager, err := newPackageManager()
	if err != nil {
		return nil, err
	}
	specs := applyPackageSpecOverride(spec.PackageSpecs, spec.PackageSpecOverrides, self.osDistro)
	return self.validate(specs, manager)
}

// Validate checks packages and their versions against the packageSpecs using
// the packageManager, and returns an error on any package/version mismatch.
func (self *packageValidator) validate(packageSpecs []PackageSpec, manager packageManager) (error, error) {
	var errs []error
	for _, spec := range packageSpecs {
		// Substitute variables in package name.
		packageName := resolvePackageName(spec.Name, self.kernelRelease)

		nameWithVerRange := fmt.Sprintf("%s (%s)", packageName, spec.VersionRange)

		// Get the version of the package on the running machine.
		version, err := manager.getPackageVersion(packageName)
		if err != nil {
			glog.V(1).Infof("Failed to get the version for the package %q: %s\n", packageName, err)
			errs = append(errs, err)
			self.reporter.Report(nameWithVerRange, "not installed", bad)
			continue
		}

		// Version requirement will not be enforced if version range is
		// not specified in the spec.
		if spec.VersionRange == "" {
			self.reporter.Report(packageName, version, good)
			continue
		}

		// Convert both the version range in the spec and the version returned
		// from package manager to semantic version format, and then check if
		// the version is in the range.
		sv, err := semver.Make(toSemVer(version))
		if err != nil {
			glog.Errorf("Failed to convert %q to semantic version: %s\n", version, err)
			errs = append(errs, err)
			self.reporter.Report(nameWithVerRange, "internal error", bad)
			continue
		}
		versionRange := semver.MustParseRange(toSemVerRange(spec.VersionRange))
		if versionRange(sv) {
			self.reporter.Report(nameWithVerRange, version, good)
		} else {
			errs = append(errs, fmt.Errorf("package \"%s %s\" does not meet the spec \"%s (%s)\"", packageName, sv, packageName, spec.VersionRange))
			self.reporter.Report(nameWithVerRange, version, bad)
		}
	}
	return nil, errors.NewAggregate(errs)
}

// getKernelRelease returns the kernel release of the local machine.
func getKernelRelease() (string, error) {
	output, err := exec.Command("uname", "-r").Output()
	if err != nil {
		return "", fmt.Errorf("failed to get kernel release: %s", err)
	}
	return strings.TrimSpace(string(output)), nil
}

// getOSDistro returns the OS distro of the local machine.
func getOSDistro() (string, error) {
	f := "/etc/lsb-release"
	b, err := ioutil.ReadFile(f)
	if err != nil {
		return "", fmt.Errorf("failed to read %q: %s", f, err)
	}
	content := string(b)
	switch {
	case strings.Contains(content, "Ubuntu"):
		return "ubuntu", nil
	case strings.Contains(content, "Chrome OS"):
		return "cos", nil
	case strings.Contains(content, "CoreOS"):
		return "coreos", nil
	default:
		return "", fmt.Errorf("failed to get OS distro: %s", content)
	}
}

// resolvePackageName substitutes the variables in the packageName with the
// local information.
// E.g., "linux-headers-${KERNEL_RELEASE}" -> "linux-headers-4.4.0-75-generic".
func resolvePackageName(packageName string, kernelRelease string) string {
	packageName = strings.Replace(packageName, "${KERNEL_RELEASE}", kernelRelease, -1)
	return packageName
}

// applyPackageSpecOverride applies the package spec overrides for the given
// osDistro to the packageSpecs and returns the applied result.
func applyPackageSpecOverride(packageSpecs []PackageSpec, overrides []PackageSpecOverride, osDistro string) []PackageSpec {
	var override *PackageSpecOverride
	for _, o := range overrides {
		if o.OSDistro == osDistro {
			override = &o
			break
		}
	}
	if override == nil {
		return packageSpecs
	}

	// Remove packages in the spec that matches the overrides in
	// Subtractions.
	var out []PackageSpec
	subtractions := make(map[string]bool)
	for _, spec := range override.Subtractions {
		subtractions[spec.Name] = true
	}
	for _, spec := range packageSpecs {
		if _, ok := subtractions[spec.Name]; !ok {
			out = append(out, spec)
		}
	}

	// Add packages in the spec that matches the overrides in Additions.
	return append(out, override.Additions...)
}

// extractUpstreamVersion returns the upstream version of the given full
// version in dpkg format. E.g., "1:1.0.6-2ubuntu2.1" -> "1.0.6".
func extractUpstreamVersion(version string) string {
	// The full version is in the format of
	// "[epoch:]upstream_version[-debian_revision]". See
	// https://www.debian.org/doc/debian-policy/ch-controlfields.html#s-f-Version.
	version = strings.Trim(version, " '")
	if i := strings.Index(version, ":"); i != -1 {
		version = version[i+1:]
	}
	if i := strings.Index(version, "-"); i != -1 {
		version = version[:i]
	}
	return version
}

// toSemVerRange converts the input to a semantic version range.
// E.g., ">=1.0"             -> ">=1.0.x"
//       ">=1"               -> ">=1.x"
//       ">=1 <=2.3"         -> ">=1.x <=2.3.x"
//       ">1 || >3.1.0 !4.2" -> ">1.x || >3.1.0 !4.2.x"
func toSemVerRange(input string) string {
	var output []string
	fields := strings.Fields(input)
	for _, f := range fields {
		numDots, hasDigits := 0, false
		for _, c := range f {
			switch {
			case c == '.':
				numDots++
			case c >= '0' && c <= '9':
				hasDigits = true
			}
		}
		if hasDigits && numDots < semVerDotsCount {
			f = strings.TrimRight(f, " ")
			f += ".x"
		}
		output = append(output, f)
	}
	return strings.Join(output, " ")
}

// toSemVer converts the input to a semantic version, and an empty string on
// error.
func toSemVer(version string) string {
	// Remove the first non-digit and non-dot character as well as the ones
	// following it.
	// E.g., "1.8.19p1" -> "1.8.19".
	if i := strings.IndexFunc(version, func(c rune) bool {
		if (c < '0' || c > '9') && c != '.' {
			return true
		}
		return false
	}); i != -1 {
		version = version[:i]
	}

	// Remove the trailing dots if there's any, and then returns an empty
	// string if nothing left.
	version = strings.TrimRight(version, ".")
	if version == "" {
		return ""
	}

	numDots := strings.Count(version, ".")
	switch {
	case numDots < semVerDotsCount:
		// Add minor version and patch version.
		// E.g. "1.18" -> "1.18.0" and "481" -> "481.0.0".
		version += strings.Repeat(".0", semVerDotsCount-numDots)
	case numDots > semVerDotsCount:
		// Remove anything beyond the patch version
		// E.g. "2.0.10.4" -> "2.0.10".
		for numDots != semVerDotsCount {
			if i := strings.LastIndex(version, "."); i != -1 {
				version = version[:i]
				numDots--
			}
		}
	}

	// Remove leading zeros in major/minor/patch version.
	// E.g., "2.02"     -> "2.2"
	//       "8.0.0095" -> "8.0.95"
	var subs []string
	for _, s := range strings.Split(version, ".") {
		s := strings.TrimLeft(s, "0")
		if s == "" {
			s = "0"
		}
		subs = append(subs, s)
	}
	return strings.Join(subs, ".")
}
