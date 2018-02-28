package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type Foo struct {
	metav1.TypeMeta
	metav1.Object

	// Reference refers to another name.  The kind of object varies by type.  It used to be just one, but it evolved into
	// two different kinds.  When we switched, we added a defaulter.
	Reference string

	// Type is defaulted to Bar, pointer lets you know empty versus unset.
	Type *string
}
