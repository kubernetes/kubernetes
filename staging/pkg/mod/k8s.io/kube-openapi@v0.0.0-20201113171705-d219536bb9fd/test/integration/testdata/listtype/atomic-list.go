package listtype

// +k8s:openapi-gen=true
type AtomicList struct {
	// +listType=atomic
	Field []string
}
