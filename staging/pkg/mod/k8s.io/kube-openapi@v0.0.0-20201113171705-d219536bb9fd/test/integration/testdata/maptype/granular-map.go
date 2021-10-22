package maptype

// +k8s:openapi-gen=true
type GranularMap struct {
	// +mapType=granular
	KeyValue map[string]string
}
