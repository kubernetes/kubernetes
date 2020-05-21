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
type User struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// FullName is the full name of user
	FullName string `json:"fullName,omitempty" protobuf:"bytes,2,opt,name=fullName"`

	// Identities are the identities associated with this user
	Identities []string `json:"identities" protobuf:"bytes,3,rep,name=identities"`

	// Groups specifies group names this user is a member of.
	// This field is deprecated and will be removed in a future release.
	// Instead, create a Group object containing the name of this User.
	Groups []string `json:"groups" protobuf:"bytes,4,rep,name=groups"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// UserList is a collection of Users
type UserList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of users
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
type Identity struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// ProviderName is the source of identity information
	ProviderName string `json:"providerName" protobuf:"bytes,2,opt,name=providerName"`

	// ProviderUserName uniquely represents this identity in the scope of the provider
	ProviderUserName string `json:"providerUserName" protobuf:"bytes,3,opt,name=providerUserName"`

	// User is a reference to the user this identity is associated with
	// Both Name and UID must be set
	User corev1.ObjectReference `json:"user" protobuf:"bytes,4,opt,name=user"`

	// Extra holds extra information about this identity
	Extra map[string]string `json:"extra,omitempty" protobuf:"bytes,5,rep,name=extra"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// IdentityList is a collection of Identities
type IdentityList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of identities
	Items []Identity `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +genclient:nonNamespaced
// +genclient:onlyVerbs=get,create,update,delete
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// UserIdentityMapping maps a user to an identity
type UserIdentityMapping struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Identity is a reference to an identity
	Identity corev1.ObjectReference `json:"identity,omitempty" protobuf:"bytes,2,opt,name=identity"`
	// User is a reference to a user
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
type Group struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Users is the list of users in this group.
	Users OptionalNames `json:"users" protobuf:"bytes,2,rep,name=users"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// GroupList is a collection of Groups
type GroupList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is the list of groups
	Items []Group `json:"items" protobuf:"bytes,2,rep,name=items"`
}
