# netns - network namespaces in go #

The netns package provides an ultra-simple interface for handling
network namespaces in go. Changing namespaces requires elevated
privileges, so in most cases this code needs to be run as root.

## Local Build and Test ##

You can use go get command:

    go get github.com/vishvananda/netns

Testing (requires root):

    sudo -E go test github.com/vishvananda/netns

## Example ##

```go
package main

import (
    "fmt"
    "net"
    "runtime"

    "github.com/vishvananda/netns"
)

func main() {
    // Lock the OS Thread so we don't accidentally switch namespaces
    runtime.LockOSThread()
    defer runtime.UnlockOSThread()

    // Save the current network namespace
    origns, _ := netns.Get()
    defer origns.Close()

    // Create a new network namespace
    newns, _ := netns.New()
    defer newns.Close()

    // Do something with the network namespace
    ifaces, _ := net.Interfaces()
    fmt.Printf("Interfaces: %v\n", ifaces)

    // Switch back to the original namespace
    netns.Set(origns)
}

```

## NOTE

The library can be safely used only with Go >= 1.10 due to [golang/go#20676](https://github.com/golang/go/issues/20676).

After locking a goroutine to its current OS thread with `runtime.LockOSThread()`
and changing its network namespace, any new subsequent goroutine won't be
scheduled on that thread while it's locked. Therefore, the new goroutine
will run in a different namespace leading to unexpected results.

See [here](https://www.weave.works/blog/linux-namespaces-golang-followup) for more details.
