package wmi

// InstanceManager interface
type InstanceManager interface {
	ServerName() string
	Namespace() string
	Credentials() *Credentials
	EnumerateInstances(className string) (*[]Instance, error)
	QueryInstances(query string) (*[]Instance, error)
	QueryInstancesEx(query Query) (*[]Instance, error)

	CreateInstance(className string, propertyValues map[string]string) (*Instance, error)
	GetInstance(className string, propertyValues map[string]string) (*Instance, error)
	GetClass(className string) (*Class, error)
	EnumerateClasses() (*[]Class, error)
	GetInstancesFromPaths(pathArray []string) (*[]Instance, error)
}
