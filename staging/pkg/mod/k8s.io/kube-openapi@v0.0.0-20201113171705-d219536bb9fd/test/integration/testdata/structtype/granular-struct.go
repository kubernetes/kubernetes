package structtype

// +k8s:openapi-gen=true
type GranularStruct struct {
	// +structType=granular
	Field      ContainedStruct
	OtherField int
}
