## nsenter

The `nsenter` package registers a special init constructor that is called before the Go runtime has 
a chance to boot.  This provides us the ability to `setns` on existing namespaces and avoid the issues
that the Go runtime has with multiple threads.  This constructor is only called if this package is 
registered, imported, in your go application and the argv 0 is `nsenter`.
