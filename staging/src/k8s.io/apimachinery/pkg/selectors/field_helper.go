package selectors

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
)

// SingleObject returns a ListOptions for watching a single object.
func SingleObject(meta metav1.ObjectMeta) metav1.ListOptions {
	return metav1.ListOptions{
		FieldSelector:   fields.OneTermEqualSelector("metadata.name", meta.Name).String(),
		ResourceVersion: meta.ResourceVersion,
	}
}
