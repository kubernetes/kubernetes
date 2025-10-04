package wmi

// Instance represents an interface for the wmi instance
type Instance interface {
	GetInstance() (*Instance, error)
	GetProperty(name string) (string, error)
	SetProperty(name, value string) (string, error)
	ResetProperty(name string) (string, error)
	Class() *Class
	EmbeddedInstance() (string, error)
	InstanceManager() *InstanceManager
	Equals(*Instance) bool
	Refresh() error
	Commit() error
	Modify() error
	Delete() error
	InstancePath() (string, error)
	InvokeMethod(namespaceName string, methodName string, methodParameters *[]MethodParameter) (MethodResult, error)
	GetRelated(resultClassName string) (*[]Instance, error)
	GetRelatedEx(resultClassName, associatedClassName, resultRole, sourceRole string) (*[]Instance, error)
	GetAssociated(resultClassName, associatedClassName, resultRole, sourceRole string) (*[]Instance, error)
	EnumerateReferencingInstances(associatedClassName, sourceRole string) (*[]Instance, error)
	Dispose()
}
