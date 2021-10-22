package structtype

// +k8s:openapi-gen=true
type AtomicStruct struct {
	// +structType=atomic
	Field      ContainedStruct
	OtherField int
}

// +k8s:openapi-gen=true
type ContainedStruct struct{}

// +k8s:openapi-gen=true
// +structType=atomic
type DeclaredAtomicStruct struct {
	Field int
}
