package naming

// Resolver dose name resolution and watches for the resolution changes.
type Resolver interface {
	// Get gets a snapshot of the current name resolution results for target.
	Get(target string) map[string]string
	// Watch watches for the name resolution changes on target. It blocks until Stop() is invoked. The watch results are obtained via GetUpdate().
	Watch(target string)
	// GetUpdate returns a name resolution change when watch is triggered. It blocks until it observes a change. The caller needs to call it again to get the next change.
	GetUpdate() (string, string)
	// Stop shuts down the NameResolver.
	Stop()
}
