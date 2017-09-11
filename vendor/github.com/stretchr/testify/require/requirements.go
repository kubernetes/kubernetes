package require

// TestingT is an interface wrapper around *testing.T
type TestingT interface {
	Errorf(format string, args ...interface{})
	FailNow()
}

//go:generate go run ../_codegen/main.go -output-package=require -template=require.go.tmpl -include-format-funcs
