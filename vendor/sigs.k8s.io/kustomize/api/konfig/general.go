// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package konfig

// RecognizedKustomizationFileNames is a list of file names
// that kustomize recognizes.
// To avoid ambiguity, a kustomization directory may not
// contain more than one match to this list.
func RecognizedKustomizationFileNames() []string {
	return []string{
		"kustomization.yaml",
		"kustomization.yml",
		"Kustomization",
	}
}

func DefaultKustomizationFileName() string {
	return RecognizedKustomizationFileNames()[0]
}

// IfApiMachineryElseKyaml returns true if executing the apimachinery code
// path, else we're executing the kyaml code paths.
func IfApiMachineryElseKyaml(s1, s2 string) string {
	if !FlagEnableKyamlDefaultValue {
		return s1
	}
	return s2
}

const (
	// FlagEnableKyamlDefaultValue is the default value for the --enable_kyaml
	// flag.  This value is also used in unit tests.  See provider.DepProvider.
	//
	// TODO(#3588): Delete this constant.
	//
	// All tests should pass for either true or false values
	// of this constant, without having to check its value.
	// In the cases where there's a different outcome, either decide
	// that the difference is acceptable, or make the difference go away.
	//
	// Historically, tests passed for enable_kyaml == false, i.e. using
	// apimachinery libs.  This doesn't mean the code was better, it just
	// means regression tests preserved those outcomes.
	FlagEnableKyamlDefaultValue = true

	// An environment variable to consult for kustomization
	// configuration data.  See:
	// https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
	XdgConfigHomeEnv = "XDG_CONFIG_HOME"

	// Use this when XdgConfigHomeEnv not defined.
	XdgConfigHomeEnvDefault = ".config"

	// A program name, for use in help, finding the XDG_CONFIG_DIR, etc.
	ProgramName = "kustomize"

	// ConfigAnnoDomain is configuration-related annotation namespace.
	ConfigAnnoDomain = "config.kubernetes.io"

	// If a resource has this annotation, kustomize will drop it.
	IgnoredByKustomizeAnnotation = ConfigAnnoDomain + "/local-config"

	// Label key that indicates the resources are built from Kustomize
	ManagedbyLabelKey = "app.kubernetes.io/managed-by"

	// An environment variable to turn on/off adding the ManagedByLabelKey
	EnableManagedbyLabelEnv = "KUSTOMIZE_ENABLE_MANAGEDBY_LABEL"

	// Label key that indicates the resources are validated by a validator
	ValidatedByLabelKey = "validated-by"
)
