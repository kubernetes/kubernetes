package context

// WithVersion stores the application version in the context. The new context
// gets a logger to ensure log messages are marked with the application
// version.
func WithVersion(ctx Context, version string) Context {
	ctx = WithValue(ctx, "version", version)
	// push a new logger onto the stack
	return WithLogger(ctx, GetLogger(ctx, "version"))
}

// GetVersion returns the application version from the context. An empty
// string may returned if the version was not set on the context.
func GetVersion(ctx Context) string {
	return GetStringValue(ctx, "version")
}
