package listtype

// +k8s:openapi-gen=true
type SetList struct {
	// +listType=set
	Field []string
}
