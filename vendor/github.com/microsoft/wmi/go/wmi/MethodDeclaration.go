package wmi

type MethodDeclaration struct {
	Name       string
	Parameters *[]MethodParameter
	Qualifiers *[]Qualifier
}
