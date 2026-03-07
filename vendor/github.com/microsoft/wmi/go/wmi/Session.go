package wmi

type SessionStatus int

const (
	Created      SessionStatus = 0
	Connected    SessionStatus = 1
	Disconnected SessionStatus = 2
	Disposed     SessionStatus = 3
)

// Session
type Session interface {
	Connect() (bool, error)
	Dispose()
	TestConnection() bool
	GetProperty(name string) string
	SetProperty(name, value string) string
	ResetProperty(name string) string
	GetClass(namespaceName, className string) (*Class, error)
	GetInstance(namespaceName string, instance *Instance) (*Instance, error)
	EnumerateClasses(namespaceName, className string) (*[]Class, error)
	EnumerateInstances(namespaceName, className string) (*[]Instance, error)
	QueryInstances(namespaceName, queryDislect, queryExpression string) (*[]Instance, error)
	EnumerateReferencingInstances(namespaceName string, sourceInstance Instance, associationClassName, sourceRole string) (*[]Instance, error)
}
