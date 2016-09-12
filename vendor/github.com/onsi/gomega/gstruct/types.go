package gstruct

//Options is the type for options passed to some matchers.
type Options int

const (
	//IgnoreExtras tells the matcher to ignore extra elements or fields, rather than triggering a failure.
	IgnoreExtras Options = 1 << iota
	//IgnoreMissing tells the matcher to ignore missing elements or fields, rather than triggering a failure.
	IgnoreMissing
)
