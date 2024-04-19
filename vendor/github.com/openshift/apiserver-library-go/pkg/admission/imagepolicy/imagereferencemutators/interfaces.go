package imagereferencemutators

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	kapi "k8s.io/kubernetes/pkg/apis/core"
)

type ImageMutators interface {
	GetImageReferenceMutator(obj, old runtime.Object) (ImageReferenceMutator, error)
	GetAnnotationAccessor(obj runtime.Object) (AnnotationAccessor, bool)
}

// ImageReferenceMutateFunc is passed a reference representing an image, and may alter
// the Name, Kind, and Namespace fields of the reference. If an error is returned the
// object may still be mutated under the covers.
type ImageReferenceMutateFunc func(ref *kapi.ObjectReference) error

type ImageReferenceMutator interface {
	// Mutate invokes fn on every image reference in the object. If fn returns an error,
	// a field.Error is added to the list to be returned. Mutate does not terminate early
	// if errors are detected.
	Mutate(fn ImageReferenceMutateFunc) field.ErrorList
}

type AnnotationAccessor interface {
	// Annotations returns a map representing annotations. Not mutable.
	Annotations() map[string]string
	// SetAnnotations sets representing annotations onto the object.
	SetAnnotations(map[string]string)
	// TemplateAnnotations returns a map representing annotations on a nested template in the object. Not mutable.
	// If no template is present bool will be false.
	TemplateAnnotations() (map[string]string, bool)
	// SetTemplateAnnotations sets annotations on a nested template in the object.
	// If no template is present bool will be false.
	SetTemplateAnnotations(map[string]string) bool
}

type ContainerMutator interface {
	GetName() string
	GetImage() string
	SetImage(image string)
}

type PodSpecReferenceMutator interface {
	GetContainerByIndex(init bool, i int) (ContainerMutator, bool)
	GetContainerByName(name string) (ContainerMutator, bool)
	GetPath() *field.Path
}
