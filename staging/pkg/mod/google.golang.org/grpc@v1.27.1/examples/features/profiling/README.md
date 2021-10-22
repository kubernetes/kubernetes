# gRPC-Go Profiling

- Author(s): adtac
- Status: Experimental
- Availability: gRPC-Go >= 1.27
- Last updated: December 17, 2019

gRPC-Go has built-in profiling that can be used to generate a detailed timeline
of the lifecycle of an RPC request. This can be done on the client-side and the
server-side. This directory contains an example client-server implementation
with profiling enabled and some example commands you can run to remotely manage
profiling.

Typically, there are three logically separate parts involved in integrating
profiling into your application:

1. Register the `Profiling` service: this requires a simple code change in your
   application.
1. Enable profiling when required: profiling is disabled by default and must be
   enabled remotely or at server initialization.
1. Download and process profiling data: once your application has collected
   enough profiling data, you must use a bundled command-line application to
   download your data and process it to generate human-friendly visualization.

## Registering the `Profiling` Service

### Server-Side

Typically, you would create and register a server like so (some Go is shortened
in the interest of brevity; please see the `server` subdirectory for a full
implementation):

```go
import (
	"google.golang.org/grpc"
	profsvc "google.golang.org/grpc/profiling/service"
	pb "google.golang.org/grpc/examples/features/proto/echo"
)

type server struct{}

func main() error {
	s := grpc.NewServer()
	pb.RegisterEchoServer(s, &server{})

	// Include this to register a profiling-specific service within your server.
	if err := profsvc.Init(&profsvc.ProfilingConfig{Server: s}); err != nil {
		fmt.Printf("error calling profsvc.Init: %v\n", err)
		return
	}

	lis, _ := net.Listen("tcp", address)
	s.Serve(lis)
}
```

To register your server for profiling, simply call the `profsvc.Init` method
as shown above. The passed `ProfilingConfig` parameter must set the `Server`
field to a server that is being served on a TCP address.

### Client-Side

To register profiling on the client-side, you must create a server to expose
your profiling data in order for it to be retrievable. To do this, it is
recommended that you create a dummy, dedicated server with no service other
than profiling's. See the `client` directory for an example client.

## Enabling/Disabling Profiling

Once profiling is baked into your server (unless otherwise specified, from here
on, the word "server" will be used to refer to a `grpc.Server`, not the
server/client distinction from the previous subsection), you need to enable
profiling. There are three ways to do this -- at initialization, remotely
post-initialization, or programmatically within Go.

### Enabling Profiling at Initialization

To force profiling to start measuring data right from the first RPC, set the
`Enabled` attribute of the `ProfilingConfig` struct to `true` when you are
initializing profiling.

```go
	// Set Enabled: true to turn profiling on at initialization time.
	profsvc.Init(&profsvc.ProfilingConfig{
		Server:  s,
		Enabled: true,
	})
```

### Enabling/Disabling Remotely

Alternatively, you can enable/disable profiling any time after server
initialization by using a bundled command-line tool designed for remote
profiling management. Assuming `example.com:50051` is the address of the server
that you would like to enable profiling in, do the following:

```bash
$ go run google.golang.org/grpc/profiling/cmd \
    -address example.com:50051                \
    -enable-profiling
```

Similarly, running the command with `-disable-profiling` can be used to disable
profiling remotely.


### Enabling/Disabling Within Go

In addition to the remote service that is exposed, you may enable/disable
profiling within your application in Go:

```go
import (
	"google.golang.org/grpc/profiling"
)

func setProfiling(enable bool) {
	profiling.Enable(true)
}
```

The `profiling.Enable` function can be safely accessed and called concurrently.

## Downloading and Processing Profiling Data

Once your server has collected enough profiling data, you may want to download
that data and perform some analysis on the retrieved data. The aforementioned
command-line application within gRPC comes bundled with support for both
operations.

To retrieve profiling data from a remote server, run the following command:

```bash
$ go run google.golang.org/grpc/profiling/cmd \
    -address example.com:50051                \
    -retrieve-snapshot                        \
    -snapshot /path/to/snapshot
```

You must provide a path to `-snapshot` that can be written to. This file will
store the retrieved data in a raw and binary form.

To process this data into a human-consumable such as
[Catapult's trace-viewer format](https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU/preview):

```bash
$ go run google.golang.org/grpc/profiling/cmd \
    -snapshot /path/to/snapshot               \
    -stream-stats-catapult-json /path/to/json
```

This would read the data stored in `/path/to/snapshot` and process it to
generate a JSON format that is understood by Chromium's
[Catapult project](https://chromium.googlesource.com/catapult).
The Catapult project comes with a utility called
[trace-viewer](https://chromium.googlesource.com/catapult/+/HEAD/tracing/README.md),
which can be used to generate human-readable visualizations:

```bash
$ git clone https://chromium.googlesource.com/catapult /path/to/catapult
$ /path/to/catapult/tracing/bin/trace2html /path/to/json --output=/path/to/html
```

When the generated `/path/to/html` file is opened with a browser, you will be
presented with a detailed visualization of the lifecycle of all RPC requests.
To learn more about trace-viewer and how to navigate the generated HTML, see
[this](https://chromium.googlesource.com/catapult/+/HEAD/tracing/README.md).

## Frequently Asked Questions

##### I have multiple `grpc.Server`s in my application. Can I register profiling with just one of them?

You may not call `profsvc.Init` more than once -- all calls except for the
first one will return an error. As a corollary, it is also not possible to
register or enable/disable profiling for just one `grpc.Server` or operation.
That is, you can enable/disable profiling globally for all gRPC operations or
none at all.

##### Is `google.golang.org/grpc/profiling/cmd` the canonical implementation of a client that can talk to the profiling service?

No, the command-line tool is simply provided as a reference implementation and
as a convenience. You are free to write your own tool as long as it can
communicate using the underlying protocol buffers.

##### Is Catapult's `trace-viewer` the only option that is supported?

Currently, yes. However, support for other (or better) visualization tools is
welcome.

##### What is the impact of profiling on application performance?

When turned off, profiling has virtually no impact on the performance (QPS,
latency, memory footprint) of your application. However, when turned on, expect
a 5-10% throughput/latency penalty and double the memory footprint.

Profiling is mostly used by gRPC-Go devs. However, if you foresee using
profiling in production machines, because of the negligible impact of profiling
when turned off, you may want to register/initialize your applications with
profiling (but leave it turned off). This will be useful in the off-chance you
want to debug an application later -- in such an event, you can simply remotely
toggle profiling using the `go run` command previously described to enable
profiling data collection. Once you're confident that enough profiling data has
been measured, you can turn it off again and retrieve the data for
post-processing (see previous section).

##### How many RPCs worth of data is stored by profiling? I'd like to restrict the memory footprint of gRPC's profiling framework to a fixed amount.

By default, at any given time, the last 2<sup>14</sup> RPCs worth of data is
stored by profiling. Newly generated profiling data overwrites older data. Note
that the internal data structure is not strictly LIFO in order to be performant
(but is approximately LIFO). All profiling data is timestamped anyway, so
a LIFO property is unnecessary.

This number is configurable. When registering your server with profiling, you
may specify the number of samples that should be stored, like so:

```go
	// Setting StreamStatsSize: 1024 will make profiling store the last 1024
	// RPCs' data (if profiling is enabled, of course).
	profsvc.Init(&profsvc.ProfilingConfig{
		Server:          s,
		StreamStatsSize: 1024,
	})
```

As an estimate, a typical unary RPC is expected produce ~2-3 KiB of profiling
data in memory. This may be useful in estimating how many RPCs worth of data
you can afford depending on your memory capacity. For more complex RPCs such as
streaming RPCs, each RPC will consume more data. The amount of memory consumed
by profiling is mostly independent of the size of messages your application
handles.

##### The generated visualization is flat and has no flows/arrows. How do I distinguish between different RPCs?

Unfortunately, there isn't any way to do this without some changes to the way
your application is compiled. This is because gRPC's profiling relies on the
Goroutine ID to uniquely identify different components.

To enable this, first apply the following patch to your Go runtime installation
directory:

```diff
diff --git a/src/runtime/runtime2.go b/src/runtime/runtime2.go
--- a/src/runtime/runtime2.go
+++ b/src/runtime/runtime2.go
@@ -392,6 +392,10 @@ type stack struct {
 	hi uintptr
 }
 
+func Goid() int64 {
+	return getg().goid
+}
+
 type g struct {
 	// Stack parameters.
 	// stack describes the actual stack memory: [stack.lo, stack.hi).
```

Then, recompile your application with `-tags grpcgoid` to generate a new
binary. This binary should produce profiling data that is much nicer when
visualized.
