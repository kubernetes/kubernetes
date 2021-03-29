package v1

import (
	"fmt"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
)

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// Template contains the inputs needed to produce a Config.
type Template struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// message is an optional instructional message that will
	// be displayed when this template is instantiated.
	// This field should inform the user how to utilize the newly created resources.
	// Parameter substitution will be performed on the message before being
	// displayed so that generated credentials and other parameters can be
	// included in the output.
	Message string `json:"message,omitempty" protobuf:"bytes,2,opt,name=message"`

	// objects is an array of resources to include in this template.
	// If a namespace value is hardcoded in the object, it will be removed
	// during template instantiation, however if the namespace value
	// is, or contains, a ${PARAMETER_REFERENCE}, the resolved
	// value after parameter substitution will be respected and the object
	// will be created in that namespace.
	// +kubebuilder:pruning:PreserveUnknownFields
	Objects []runtime.RawExtension `json:"objects" protobuf:"bytes,3,rep,name=objects"`

	// parameters is an optional array of Parameters used during the
	// Template to Config transformation.
	Parameters []Parameter `json:"parameters,omitempty" protobuf:"bytes,4,rep,name=parameters"`

	// labels is a optional set of labels that are applied to every
	// object during the Template to Config transformation.
	ObjectLabels map[string]string `json:"labels,omitempty" protobuf:"bytes,5,rep,name=labels"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// TemplateList is a list of Template objects.
type TemplateList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// Items is a list of templates
	Items []Template `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// Parameter defines a name/value variable that is to be processed during
// the Template to Config transformation.
type Parameter struct {
	// Name must be set and it can be referenced in Template
	// Items using ${PARAMETER_NAME}. Required.
	Name string `json:"name" protobuf:"bytes,1,opt,name=name"`

	// Optional: The name that will show in UI instead of parameter 'Name'
	DisplayName string `json:"displayName,omitempty" protobuf:"bytes,2,opt,name=displayName"`

	// Description of a parameter. Optional.
	Description string `json:"description,omitempty" protobuf:"bytes,3,opt,name=description"`

	// Value holds the Parameter data. If specified, the generator will be
	// ignored. The value replaces all occurrences of the Parameter ${Name}
	// expression during the Template to Config transformation. Optional.
	Value string `json:"value,omitempty" protobuf:"bytes,4,opt,name=value"`

	// generate specifies the generator to be used to generate random string
	// from an input value specified by From field. The result string is
	// stored into Value field. If empty, no generator is being used, leaving
	// the result Value untouched. Optional.
	//
	// The only supported generator is "expression", which accepts a "from"
	// value in the form of a simple regular expression containing the
	// range expression "[a-zA-Z0-9]", and the length expression "a{length}".
	//
	// Examples:
	//
	// from             | value
	// -----------------------------
	// "test[0-9]{1}x"  | "test7x"
	// "[0-1]{8}"       | "01001100"
	// "0x[A-F0-9]{4}"  | "0xB3AF"
	// "[a-zA-Z0-9]{8}" | "hW4yQU5i"
	//
	Generate string `json:"generate,omitempty" protobuf:"bytes,5,opt,name=generate"`

	// From is an input value for the generator. Optional.
	From string `json:"from,omitempty" protobuf:"bytes,6,opt,name=from"`

	// Optional: Indicates the parameter must have a value.  Defaults to false.
	Required bool `json:"required,omitempty" protobuf:"varint,7,opt,name=required"`
}

// +genclient
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// TemplateInstance requests and records the instantiation of a Template.
// TemplateInstance is part of an experimental API.
type TemplateInstance struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec describes the desired state of this TemplateInstance.
	Spec TemplateInstanceSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`

	// status describes the current state of this TemplateInstance.
	// +optional
	Status TemplateInstanceStatus `json:"status" protobuf:"bytes,3,opt,name=status"`
}

// TemplateInstanceSpec describes the desired state of a TemplateInstance.
type TemplateInstanceSpec struct {
	// template is a full copy of the template for instantiation.
	Template Template `json:"template" protobuf:"bytes,1,opt,name=template"`

	// secret is a reference to a Secret object containing the necessary
	// template parameters.
	Secret *corev1.LocalObjectReference `json:"secret,omitempty" protobuf:"bytes,2,opt,name=secret"`

	// requester holds the identity of the agent requesting the template
	// instantiation.
	// +optional
	Requester *TemplateInstanceRequester `json:"requester" protobuf:"bytes,3,opt,name=requester"`
}

// TemplateInstanceRequester holds the identity of an agent requesting a
// template instantiation.
type TemplateInstanceRequester struct {
	// username uniquely identifies this user among all active users.
	Username string `json:"username,omitempty" protobuf:"bytes,1,opt,name=username"`

	// uid is a unique value that identifies this user across time; if this user is
	// deleted and another user by the same name is added, they will have
	// different UIDs.
	UID string `json:"uid,omitempty" protobuf:"bytes,2,opt,name=uid"`

	// groups represent the groups this user is a part of.
	Groups []string `json:"groups,omitempty" protobuf:"bytes,3,rep,name=groups"`

	// extra holds additional information provided by the authenticator.
	Extra map[string]ExtraValue `json:"extra,omitempty" protobuf:"bytes,4,rep,name=extra"`
}

// ExtraValue masks the value so protobuf can generate
// +protobuf.nullable=true
// +protobuf.options.(gogoproto.goproto_stringer)=false
type ExtraValue []string

func (t ExtraValue) String() string {
	return fmt.Sprintf("%v", []string(t))
}

// TemplateInstanceStatus describes the current state of a TemplateInstance.
type TemplateInstanceStatus struct {
	// conditions represent the latest available observations of a
	// TemplateInstance's current state.
	Conditions []TemplateInstanceCondition `json:"conditions,omitempty" protobuf:"bytes,1,rep,name=conditions"`

	// Objects references the objects created by the TemplateInstance.
	Objects []TemplateInstanceObject `json:"objects,omitempty" protobuf:"bytes,2,rep,name=objects"`
}

// TemplateInstanceCondition contains condition information for a
// TemplateInstance.
type TemplateInstanceCondition struct {
	// Type of the condition, currently Ready or InstantiateFailure.
	Type TemplateInstanceConditionType `json:"type" protobuf:"bytes,1,opt,name=type,casttype=TemplateInstanceConditionType"`
	// Status of the condition, one of True, False or Unknown.
	Status corev1.ConditionStatus `json:"status" protobuf:"bytes,2,opt,name=status"`
	// LastTransitionTime is the last time a condition status transitioned from
	// one state to another.
	LastTransitionTime metav1.Time `json:"lastTransitionTime" protobuf:"bytes,3,opt,name=lastTransitionTime"`
	// Reason is a brief machine readable explanation for the condition's last
	// transition.
	Reason string `json:"reason" protobuf:"bytes,4,opt,name=reason"`
	// Message is a human readable description of the details of the last
	// transition, complementing reason.
	Message string `json:"message" protobuf:"bytes,5,opt,name=message"`
}

// TemplateInstanceConditionType is the type of condition pertaining to a
// TemplateInstance.
type TemplateInstanceConditionType string

const (
	// TemplateInstanceReady indicates the readiness of the template
	// instantiation.
	TemplateInstanceReady TemplateInstanceConditionType = "Ready"
	// TemplateInstanceInstantiateFailure indicates the failure of the template
	// instantiation
	TemplateInstanceInstantiateFailure TemplateInstanceConditionType = "InstantiateFailure"
)

// TemplateInstanceObject references an object created by a TemplateInstance.
type TemplateInstanceObject struct {
	// ref is a reference to the created object.  When used under .spec, only
	// name and namespace are used; these can contain references to parameters
	// which will be substituted following the usual rules.
	Ref corev1.ObjectReference `json:"ref,omitempty" protobuf:"bytes,1,opt,name=ref"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// TemplateInstanceList is a list of TemplateInstance objects.
type TemplateInstanceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of Templateinstances
	Items []TemplateInstance `json:"items" protobuf:"bytes,2,rep,name=items"`
}

// +genclient
// +genclient:nonNamespaced
// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// BrokerTemplateInstance holds the service broker-related state associated with
// a TemplateInstance.  BrokerTemplateInstance is part of an experimental API.
type BrokerTemplateInstance struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// spec describes the state of this BrokerTemplateInstance.
	Spec BrokerTemplateInstanceSpec `json:"spec" protobuf:"bytes,2,opt,name=spec"`
}

// BrokerTemplateInstanceSpec describes the state of a BrokerTemplateInstance.
type BrokerTemplateInstanceSpec struct {
	// templateinstance is a reference to a TemplateInstance object residing
	// in a namespace.
	TemplateInstance corev1.ObjectReference `json:"templateInstance" protobuf:"bytes,1,opt,name=templateInstance"`

	// secret is a reference to a Secret object residing in a namespace,
	// containing the necessary template parameters.
	Secret corev1.ObjectReference `json:"secret" protobuf:"bytes,2,opt,name=secret"`

	// bindingids is a list of 'binding_id's provided during successive bind
	// calls to the template service broker.
	BindingIDs []string `json:"bindingIDs,omitempty" protobuf:"bytes,3,rep,name=bindingIDs"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// BrokerTemplateInstanceList is a list of BrokerTemplateInstance objects.
type BrokerTemplateInstanceList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty" protobuf:"bytes,1,opt,name=metadata"`

	// items is a list of BrokerTemplateInstances
	Items []BrokerTemplateInstance `json:"items" protobuf:"bytes,2,rep,name=items"`
}
