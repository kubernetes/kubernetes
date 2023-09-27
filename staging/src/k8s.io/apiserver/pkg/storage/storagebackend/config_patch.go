package storagebackend

import genericrequest "k8s.io/apiserver/pkg/endpoints/request"

// KcpStorageMetadata holds KCP specific metadata that is used by the reflector to instruct the storage layer how to assign/extract the cluster name
type KcpStorageMetadata struct {
	// IsCRD indicate that the storage deals with CustomResourceDefinition
	IsCRD bool

	// Cluster holds a KCP cluster
	Cluster genericrequest.Cluster
}
