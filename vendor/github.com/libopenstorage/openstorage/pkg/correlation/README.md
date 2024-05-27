## Correlation logging instructions

There are three ways for using this package. All methods will result in a correlation ID being 
added to the log lines. Each method may make sense in different scenarios. We've provided
all three methods as they can be used interchangeably depending on the developers preference. 

Currently, a correlation context is created as part of a gRPC interceptor for the following openstorage packages:
- The CSI Driver (/csi)
- The SDK Server (/api/server/sdk)

### Method 1: Per-package correlation logger
This method is nice if a package maintainer wants to register a global 
logging hook at a per-package level. This is nice as you only need to
create the logger once per package. However, each log line can be quite
verbose as you must provide WithContext(ctx)

File 1:

```go
package example

var (
    clogger := correlation.NewPackageLogger("test")
)
	
func testFunc(ctx context.Context ) {
    ctx := correlation.WithCorrelationContext(ctx, "source-component")

    clogger.WithContext(ctx).Info("test info log 1")
    testFuncTwo(ctx)
}
```

File 2:

```go
package example

func testFuncTwo(ctx context.Context) {
    clogger.WithContext(ctx).Info("test info log 2")
}
```

### Method 2: Per-function correlation logger
This method is great for reducing the amount of count needed per-line, 
as you do not need to pass in the context at every log line. However,
you must remember to instantiate the logger at every function.

```go
package example

func init() {
    // RegisterComponent registers the package where this function was called from
    // as a given component name.
    correlation.RegisterComponent("example-package")
}

func testFunc() {
    ctx := correlation.WithCorrelationContext(context.Background(), "test_origin")
    clogger := correlation.NewFunctionLogger(ctx)

    clogger.Info("test info log 1")
    testFuncTwo(ctx)
}

func testFuncTwo(ctx context.Context) {
    clogger := correlation.NewFunctionLogger(ctx)

    clogger.Info("test info log")
}
```

### Method 3: Global correlation logging hook
This is a nice catch-all for covering packages where a
package-level or function-level logger is not created. To mark certain packages as 
a component, call `RegisterComponent` from a given package. Each 
log line in this package will have the registered component added. If a package 
is not registered as a component, we'll find the component based on the package.

In main.go, first we should register the global hook:

```go
package main

init() {
    // RegisterGlobalHook will register the correlation logging
    // hook at a global/multi-package level. Note that this is not
    // component-aware and it is better to use NewFunctionLogger
    // or NewPackageLogger for logging.
    correlation.RegisterGlobalHook()

    // RegisterComponent registers the package where this function was called from
    // as a given component name.
    correlation.RegisterComponent("main")
}

func main() {
    ...
}
```

In every other package, we can register components as we see fit.

This will help add extra context in our log lines for those who are unfamiliar with our codebase. 
For example, if a support engineer sees many log line failures with "csi-driver" component,
they'll know who to contact regarding that failure.

The following example shows how a helper package can be registered as a component:

```go
package example

func init() {
    // RegisterComponent registers the package where this function was called from
    // as a given component name.
    correlation.RegisterComponent("example-package")
}

func TestFuncTwo(ctx context.Context) {
    logrus.WithContext(ctx).Info("test info log 2")
}
```
