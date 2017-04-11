// Package context provides several utilities for working with
// golang.org/x/net/context in http requests. Primarily, the focus is on
// logging relevant request information but this package is not limited to
// that purpose.
//
// The easiest way to get started is to get the background context:
//
// 	ctx := context.Background()
//
// The returned context should be passed around your application and be the
// root of all other context instances. If the application has a version, this
// line should be called before anything else:
//
// 	ctx := context.WithVersion(context.Background(), version)
//
// The above will store the version in the context and will be available to
// the logger.
//
// Logging
//
// The most useful aspect of this package is GetLogger. This function takes
// any context.Context interface and returns the current logger from the
// context. Canonical usage looks like this:
//
// 	GetLogger(ctx).Infof("something interesting happened")
//
// GetLogger also takes optional key arguments. The keys will be looked up in
// the context and reported with the logger. The following example would
// return a logger that prints the version with each log message:
//
// 	ctx := context.Context(context.Background(), "version", version)
// 	GetLogger(ctx, "version").Infof("this log message has a version field")
//
// The above would print out a log message like this:
//
// 	INFO[0000] this log message has a version field        version=v2.0.0-alpha.2.m
//
// When used with WithLogger, we gain the ability to decorate the context with
// loggers that have information from disparate parts of the call stack.
// Following from the version example, we can build a new context with the
// configured logger such that we always print the version field:
//
// 	ctx = WithLogger(ctx, GetLogger(ctx, "version"))
//
// Since the logger has been pushed to the context, we can now get the version
// field for free with our log messages. Future calls to GetLogger on the new
// context will have the version field:
//
// 	GetLogger(ctx).Infof("this log message has a version field")
//
// This becomes more powerful when we start stacking loggers. Let's say we
// have the version logger from above but also want a request id. Using the
// context above, in our request scoped function, we place another logger in
// the context:
//
// 	ctx = context.WithValue(ctx, "http.request.id", "unique id") // called when building request context
// 	ctx = WithLogger(ctx, GetLogger(ctx, "http.request.id"))
//
// When GetLogger is called on the new context, "http.request.id" will be
// included as a logger field, along with the original "version" field:
//
// 	INFO[0000] this log message has a version field        http.request.id=unique id version=v2.0.0-alpha.2.m
//
// Note that this only affects the new context, the previous context, with the
// version field, can be used independently. Put another way, the new logger,
// added to the request context, is unique to that context and can have
// request scoped varaibles.
//
// HTTP Requests
//
// This package also contains several methods for working with http requests.
// The concepts are very similar to those described above. We simply place the
// request in the context using WithRequest. This makes the request variables
// available. GetRequestLogger can then be called to get request specific
// variables in a log line:
//
// 	ctx = WithRequest(ctx, req)
// 	GetRequestLogger(ctx).Infof("request variables")
//
// Like above, if we want to include the request data in all log messages in
// the context, we push the logger to a new context and use that one:
//
// 	ctx = WithLogger(ctx, GetRequestLogger(ctx))
//
// The concept is fairly powerful and ensures that calls throughout the stack
// can be traced in log messages. Using the fields like "http.request.id", one
// can analyze call flow for a particular request with a simple grep of the
// logs.
package context
