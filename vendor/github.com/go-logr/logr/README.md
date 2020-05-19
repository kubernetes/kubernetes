# A more minimal logging API for Go

Before you consider this package, please read [this blog post by the inimitable
Dave Cheney](http://dave.cheney.net/2015/11/05/lets-talk-about-logging).  I
really appreciate what he has to say, and it largely aligns with my own
experiences.  Too many choices of levels means inconsistent logs.

This package offers a purely abstract interface, based on these ideas but with
a few twists.  Code can depend on just this interface and have the actual
logging implementation be injected from callers.  Ideally only `main()` knows
what logging implementation is being used.

# Differences from Dave's ideas

The main differences are:

1) Dave basically proposes doing away with the notion of a logging API in favor
of `fmt.Printf()`.  I disagree, especially when you consider things like output
locations, timestamps, file and line decorations, and structured logging.  I
restrict the API to just 2 types of logs: info and error.

Info logs are things you want to tell the user which are not errors.  Error
logs are, well, errors.  If your code receives an `error` from a subordinate
function call and is logging that `error` *and not returning it*, use error
logs.

2) Verbosity-levels on info logs.  This gives developers a chance to indicate
arbitrary grades of importance for info logs, without assigning names with
semantic meaning such as "warning", "trace", and "debug".  Superficially this
may feel very similar, but the primary difference is the lack of semantics.
Because verbosity is a numerical value, it's safe to assume that an app running
with higher verbosity means more (and less important) logs will be generated.

This is a BETA grade API.  I have implemented it for
[glog](https://godoc.org/github.com/golang/glog). Until there is a significant
2nd implementation, I don't really know how it will change.
