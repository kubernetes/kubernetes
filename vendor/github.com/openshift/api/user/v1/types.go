package v1

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Upon log in, every user of the system receives a User and Identity resource. Administrators
// may directly manipulate the attributes of the users for their own tracking, or set groups
// via the API. The user name is unique and is chosen based on the value provided by the
// identity provider - if a user already exists with the incoming name, the user name may have
// a number appended to it depending on the configuration of the system.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type User struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// fullName is the full name of user
	FullName string `json:"fullName,omitempty" protobuf:"bytes,2,opt,name=fullName"`

	// identities are the identities associated with this user
	// +optional
	Identities []string `json:"identities,omitempty" protobuf:"bytes,3,rep,name=identities"`

	// groups specifies group names this user is a member of.
	// This field is deprecated and will be removed in a future release.
	// Instead, create a Group object containing the name of this User.
	Groups []string `json:"groups" protobuf:"bytes,4,rep,name=groups"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// UserList is a collection of Users
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type UserList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of users
	Items []User `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Identity records a successful authentication of a user with an identity provider. The
// information about the source of authentication is stored on the identity, and the identity
// is then associated with a single user object. Multiple identities can reference a single
// user. Information retrieved from the authentication provider is stored in the extra field
// using a schema determined by the provider.
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Identity struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// providerName is the source of identity information
	ProviderName string `json:"providerName" protobuf:"bytes,2,opt,name=providerName"`

	// providerUserName uniquely represents this identity in the scope of the provider
	ProviderUserName string `json:"providerUserName" protobuf:"bytes,3,opt,name=providerUserName"`

	// user is a reference to the user this identity is associated with
	// Both Name and UID must be set
	User corev1.ObjectReference `json:"user" protobuf:"bytes,4,opt,name=user"`

	// extra holds extra information about this identity
	Extra map[string]string `json:"extra,omitempty" protobuf:"bytes,5,rep,name=extra"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IdentityList is a collection of Identities
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type IdentityList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of identities
	Items []Identity `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +genclient:nonNamespaced
// +genclient:onlyVerbs=get,create,update,delete
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// UserIdentityMapping maps a user to an identity
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type UserIdentityMapping struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// identity is a reference to an identity
	Identity corev1.ObjectReference `json:"identity,omitempty" protobuf:"bytes,2,opt,name=identity"`
	// user is a reference to a user
	User corev1.ObjectReference `json:"user,omitempty" protobuf:"bytes,3,opt,name=user"`
}

// OptionalNames is an array that may also be left nil to distinguish between set and unset.
// +protobuf.nullable=true
// +protobuf.options.(gogoproto.goproto_stringer)=false
type OptionalNames []string

func (t OptionalNames) String() string {
	return fmt.Sprintf("%v", []string(t))
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Group represents a referenceable set of Users
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type Group struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard object's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// users is the list of users in this group.
	Users OptionalNames `json:"users" protobuf:"bytes,2,rep,name=users"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// GroupList is a collection of Groups
//
// Compatibility level 1: Stable within a major release for a minimum of 12 months or 3 minor releases (whichever is longer).
// +openshift:compatibility-gen:level=1
type GroupList struct {
	metav1.TypeMeta `json:",inline"`

	// metadata is the standard list's metadata.
	// More info: https://git.k8s.io/community/contributors/devel/sig-architecture/api-conventions.md#metadata
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is the list of groups
	Items []Group `json:"items" protobuf:"bytes,2,rep,name=items"`
}
