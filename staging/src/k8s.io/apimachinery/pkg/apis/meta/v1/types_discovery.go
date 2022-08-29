package v1

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// APIGroupDiscoveryList is a resource containing a list of APIGroupDiscoveries.
// The shape is chosen to be very similar to "normal" api endpoints and work well with our type system
// This is what is returned from the /discovery/v2 endpoint.
type APIGroupDiscoveryList struct {
	TypeMeta `json:",inline"`
	// ResourceVersion will not be set, because this does not have a replayable ordering among multiple apiservers.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Groups is the list of groups for discovery.
	// +listType=map
	// +listMapKey=name
	Groups []APIGroupDiscovery `json:"groups" protobuf:"bytes,2,rep,name=groups"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// APIGroupDiscovery holds information about which resources are being served for all version of the API Group.
type APIGroupDiscovery struct {
	TypeMeta `json:",inline"`
	// Standard object's metadata.
	// The only field completed will be Name.  For instance, resourceVersion will be empty.
	// Name is the name of the API group whose discovery information is presented here.
	// Name is allowed to be "" to represent the legacy, ungroupified resources.
	// This is included to make the standard deserialization easier.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	// +optional
	ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`
	// Versions are the versions supported in this group.
	// +listType=map
	// +listMapKey=version
	Versions []APIVersionDiscovery `json:"versions,omitempty" protobuf:"bytes,2,rep,name=versions"`
	// Most recently observed status of the APIGroupDiscovery.
	// This data may not be up to date.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#spec-and-status
	// +optional
	Status APIGroupDiscoveryStatus `json:"status,omitempty" protobuf:"bytes,3,opt,name=status"`
}

// APIVersionDiscovery holds a list of APIResourceDiscovery types that are served for a particular version within an API Group.
type APIVersionDiscovery struct {
	// Version is the name of the version within a group version.
	Version string `json:"version" protobuf:"bytes,1,opt,name=version"`
	// Resources is a list of APIResourceDiscovery objects for the corresponding group version.
	// +listType=map
	// +listMapKey=resource
	Resources []APIResourceDiscovery `json:"resources,omitempty" protobuf:"bytes,2,rep,name=resources"`
}

// APIResourceDiscovery provides information about an API resource for discovery.
type APIResourceDiscovery struct {
	// Resource is the plural name of the resource.  This is used in the URL path and is the unique identifier
	// for this resource across all versions.
	// resources with non-"" groups are located at /apis/<APIGroupDiscovery.objectMeta.name>/<APIVersionDiscovery.version>/<APIResourceDiscovery.Resource>
	// resource with "" groups are located at /api/v1/<APIResourceDiscovery.Resource>
	Resource string `json:"resource" protobuf:"bytes,1,opt,name=resource"`
	// ReturnType describes the type of serialization that will be returned from this endpoint.
	ReturnType APIDiscoveryKind `json:"returnType" protobuf:"bytes,2,opt,name=returnType"`
	// Scope indicates the scope of a resource, either Cluster or Namespace
	Scope ResourceScope `json:"scope" protobuf:"bytes,3,opt,name=scope"`
	// SingularName is the singular name of the resource.  This allows clients to handle plural and singular opaquely.
	// The singularName is more correct for reporting status on a single item and both singular and plural are allowed
	// from the kubectl CLI interface.
	SingularName string `json:"singularName" protobuf:"bytes,4,opt,name=singularName"`
	// Verbs is a list of supported kube verbs (this includes get, list, watch, create,
	// update, patch, delete, deletecollection, other)
	Verbs Verbs `json:"verbs" protobuf:"bytes,5,opt,name=verbs"`
	// ShortNames is a list of suggested short names of the resource.
	// +listType=set
	ShortNames []string `json:"shortNames,omitempty" protobuf:"bytes,6,rep,name=shortNames"`
	// Categories is a list of the grouped resources this resource belongs to (e.g. 'all')
	// +listType=set
	Categories []string `json:"categories,omitempty" protobuf:"bytes,7,rep,name=categories"`
	// Subresources is a list of subresources provided by this resource. Subresources are located at /apis/<APIGroupDiscovery.objectMeta.name>/<APIVersionDiscovery.version>/<APIResourceDiscovery.Resource>/name-of-instance/<APIResourceDiscovery.subresources[i].subresource>
	// +listType=map
	// +listMapKey=subresource
	Subresources []APISubresourceDiscovery `json:"subresources,omitempty" protobuf:"bytes,8,rep,name=subresources"`
}

// ResourceScope is an enum defining the different scopes available to a resource.
type ResourceScope string

const (
	ScopeCluster   ResourceScope = "Cluster"
	ScopeNamespace ResourceScope = "Namespace"
)

// APIDiscoveryKind provides the group version kind information for a resource.
type APIDiscoveryKind struct {
	// Group is the group of the serialization.
	Group string `json:"group,omitempty" protobuf:"bytes,1,opt,name=group"`
	// Version is the version of the serialization
	Version string `json:"version,omitempty" protobuf:"bytes,2,opt,name=version"`
	// Kind is the kind of the serialiation
	Kind string `json:"kind" protobuf:"bytes,3,opt,name=kind"`
}

// APISubresourceDiscovery provides information about an API subresource for discovery.
type APISubresourceDiscovery struct {
	// Subresource is the name of the subresource.  This is used in the URL path and is the unique identifier
	// for this resource across all versions.
	Subresource string `json:"subresource" protobuf:"bytes,1,opt,name=subresource"`
	// ReturnType describes the type of serialization that will be returned from this endpoint.
	// Some subresources do not return normal resources, these will have nil return types.
	ReturnType *APIDiscoveryKind `json:"returnType,omitempty" protobuf:"bytes,2,opt,name=returnType"`
	// AcceptedTypes describes the kinds that this endpoint accepts.  It is possible for a subresource to accept multiple kinds.
	// It is also possible for an endpoint to accept no standard types.  Those will have a zero length list.
	// +listType=set
	AcceptedTypes []APIDiscoveryKind `json:"acceptedTypes,omitempty" protobuf:"bytes,3,rep,name=acceptedTypes"`
	// Verbs is a list of supported kube verbs (this includes get, list, watch, create,
	// update, patch, delete, other)
	Verbs Verbs `json:"verbs" protobuf:"bytes,4,opt,name=verbs"`
}

// APIGroupDiscoveryStatus contains the observations of an APIGroupDiscovery
type APIGroupDiscoveryStatus struct {
	// StaleVersions represent the observations of a APIGroupDiscovery's current state.
	// +listType=set
	StaleVersions []string `json:"staleVersions" protobuf:"bytes,1,name=staleVersions"`
	// Represents the observations of a APIGroupDiscovery's current state.
	// Known .status.conditions.type are: "Stale"
	// +patchMergeKey=type
	// +patchStrategy=merge
	// +listType=map
	// +listMapKey=type
	Conditions []Condition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type" protobuf:"bytes,2,rep,name=conditions"`
}
