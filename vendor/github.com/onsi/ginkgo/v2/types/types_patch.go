package types

type TestSpec interface {
	CodeLocations() []CodeLocation
	Text() string
	AppendText(text string)
}
