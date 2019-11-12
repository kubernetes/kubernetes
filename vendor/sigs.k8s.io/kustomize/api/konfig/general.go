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

const (
	// An environment variable to consult for kustomization
	// configuration data.  See:
	// https://specifications.freedesktop.org/basedir-spec/basedir-spec-latest.html
	XdgConfigHomeEnv = "XDG_CONFIG_HOME"

	// Use this when XdgConfigHomeEnv not defined.
	XdgConfigHomeEnvDefault = ".config"

	// A program name, for use in help, finding the XDG_CONFIG_DIR, etc.
	ProgramName = "kustomize"
)
