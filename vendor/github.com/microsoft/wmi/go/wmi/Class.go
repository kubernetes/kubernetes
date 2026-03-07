package wmi

// Class
type Class interface {
	ClassName() string
	SuperClassName() string
	ServerName() string
	Namespace() string
	SuperClass() *Class
	Properties() []string
	Qualifiers() []string
	Methods() []string
	MethodParameters(string) []string
	InvokeMethod(string, []string, string) (error, string)
	Dispose()
}
