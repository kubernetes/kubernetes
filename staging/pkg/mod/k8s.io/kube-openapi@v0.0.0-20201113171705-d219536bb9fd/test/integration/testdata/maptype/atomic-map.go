package maptype

// +k8s:openapi-gen=true
type AtomicMap struct {
	// +mapType=atomic
	KeyValue map[string]string
}
