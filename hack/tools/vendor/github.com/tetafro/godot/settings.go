package godot

// Settings contains linter settings.
type Settings struct {
	// Which comments to check (top level declarations, top level, all).
	Scope Scope

	// Regexp for excluding particular comment lines from check.
	Exclude []string

	// Check periods at the end of sentences.
	Period bool

	// Check that first letter of each sentence is capital.
	Capital bool
}

// Scope sets which comments should be checked.
type Scope string

// List of available check scopes.
const (
	// DeclScope is for top level declaration comments.
	DeclScope Scope = "declarations"
	// TopLevelScope is for all top level comments.
	TopLevelScope Scope = "toplevel"
	// AllScope is for all comments.
	AllScope Scope = "all"
)
