package validation

import (
	"fmt"
	"net"
	"net/url"
	"os"
	"strings"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

type ValidationResults struct {
	Warnings field.ErrorList
	Errors   field.ErrorList
}

func (r *ValidationResults) Append(additionalResults ValidationResults) {
	r.AddErrors(additionalResults.Errors...)
	r.AddWarnings(additionalResults.Warnings...)
}

func (r *ValidationResults) AddErrors(errors ...*field.Error) {
	if len(errors) == 0 {
		return
	}
	r.Errors = append(r.Errors, errors...)
}

func (r *ValidationResults) AddWarnings(warnings ...*field.Error) {
	if len(warnings) == 0 {
		return
	}
	r.Warnings = append(r.Warnings, warnings...)
}

func ValidateHostPort(value string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(value) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, ""))
	} else if _, _, err := net.SplitHostPort(value); err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, value, "must be a host:port"))
	}

	return allErrs
}

func ValidateFile(path string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}

	if len(path) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, ""))
	} else if _, err := os.Stat(path); err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, path, fmt.Sprintf("could not read file: %v", err)))
	}

	return allErrs
}

func ValidateSecureURL(urlString string, fldPath *field.Path) (*url.URL, field.ErrorList) {
	url, urlErrs := ValidateURL(urlString, fldPath)
	if len(urlErrs) == 0 && url.Scheme != "https" {
		urlErrs = append(urlErrs, field.Invalid(fldPath, urlString, "must use https scheme"))
	}
	return url, urlErrs
}

func ValidateURL(urlString string, fldPath *field.Path) (*url.URL, field.ErrorList) {
	allErrs := field.ErrorList{}

	urlObj, err := url.Parse(urlString)
	if err != nil {
		allErrs = append(allErrs, field.Invalid(fldPath, urlString, "must be a valid URL"))
		return nil, allErrs
	}
	if len(urlObj.Scheme) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, urlString, "must contain a scheme (e.g. https://)"))
	}
	if len(urlObj.Host) == 0 {
		allErrs = append(allErrs, field.Invalid(fldPath, urlString, "must contain a host"))
	}
	return urlObj, allErrs
}

func ValidateDir(path string, fldPath *field.Path) field.ErrorList {
	allErrs := field.ErrorList{}
	if len(path) == 0 {
		allErrs = append(allErrs, field.Required(fldPath, ""))
	} else {
		fileInfo, err := os.Stat(path)
		if err != nil {
			allErrs = append(allErrs, field.Invalid(fldPath, path, fmt.Sprintf("could not read info: %v", err)))
		} else if !fileInfo.IsDir() {
			allErrs = append(allErrs, field.Invalid(fldPath, path, "not a directory"))
		}
	}

	return allErrs
}

// HostnameMatchSpecCandidates returns a list of match specs that would match the provided hostname
// Returns nil if len(hostname) == 0
func HostnameMatchSpecCandidates(hostname string) []string {
	if len(hostname) == 0 {
		return nil
	}

	// Exact match has priority
	candidates := []string{hostname}

	// Replace successive labels in the name with wildcards, to require an exact match on number of
	// path segments, because certificates cannot wildcard multiple levels of subdomains
	//
	// This is primarily to be consistent with tls.Config#getCertificate implementation
	//
	// It using a cert signed for *.foo.example.com and *.bar.example.com by specifying the name *.*.example.com
	labels := strings.Split(hostname, ".")
	for i := range labels {
		labels[i] = "*"
		candidates = append(candidates, strings.Join(labels, "."))
	}
	return candidates
}

// HostnameMatches returns true if the given hostname is matched by the given matchSpec
func HostnameMatches(hostname string, matchSpec string) bool {
	return sets.NewString(HostnameMatchSpecCandidates(hostname)...).Has(matchSpec)
}
