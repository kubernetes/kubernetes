package testdata

type UnsortedFeature string

const (
	// These are NOT properly sorted and will trigger a linter error
	UnsortedFeatureB UnsortedFeature = "UnsortedFeatureB"
	UnsortedFeatureA UnsortedFeature = "UnsortedFeatureA"
	UnsortedFeatureC UnsortedFeature = "UnsortedFeatureC"
)

var (
	// These are NOT properly sorted and will trigger a linter error
	UnsortedVarFeatureC UnsortedFeature = "VarUnsortedFeatureC"
	UnsortedVarFeatureA UnsortedFeature = "VarUnsortedFeatureA"
	UnsortedVarFeatureB UnsortedFeature = "VarUnsortedFeatureB"
)
