package utils

import "sigs.k8s.io/kustomize/api/konfig"

const (
	BuildAnnotationPreviousKinds      = konfig.ConfigAnnoDomain + "/previousKinds"
	BuildAnnotationPreviousNames      = konfig.ConfigAnnoDomain + "/previousNames"
	BuildAnnotationPrefixes           = konfig.ConfigAnnoDomain + "/prefixes"
	BuildAnnotationSuffixes           = konfig.ConfigAnnoDomain + "/suffixes"
	BuildAnnotationPreviousNamespaces = konfig.ConfigAnnoDomain + "/previousNamespaces"
	BuildAnnotationsRefBy             = konfig.ConfigAnnoDomain + "/refBy"
	BuildAnnotationsGenBehavior       = konfig.ConfigAnnoDomain + "/generatorBehavior"
	BuildAnnotationsGenAddHashSuffix  = konfig.ConfigAnnoDomain + "/needsHashSuffix"

	// the following are only for patches, to specify whether they can change names
	// and kinds of their targets
	BuildAnnotationAllowNameChange = konfig.ConfigAnnoDomain + "/allowNameChange"
	BuildAnnotationAllowKindChange = konfig.ConfigAnnoDomain + "/allowKindChange"

	OriginAnnotation = "config.kubernetes.io/origin"

	Enabled = "enabled"
)
