package v1

const (
	// TemplateInstanceFinalizer is used to clean up the objects created by the template instance,
	// when the template instance is deleted.
	TemplateInstanceFinalizer = "template.openshift.io/finalizer"

	// TemplateInstanceOwner is a label applied to all objects created from a template instance
	// which contains the uid of the template instance.
	TemplateInstanceOwner = "template.openshift.io/template-instance-owner"

	// WaitForReadyAnnotation indicates that the TemplateInstance controller
	// should wait for the object to be ready before reporting the template
	// instantiation complete.
	WaitForReadyAnnotation = "template.alpha.openshift.io/wait-for-ready"
)
