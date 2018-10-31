# go-testing-interface

go-testing-interface is a Go library that exports an interface that
`*testing.T` implements as well as a runtime version you can use in its
place.

The purpose of this library is so that you can export test helpers as a
public API without depending on the "testing" package, since you can't
create a `*testing.T` struct manually. This lets you, for example, use the
public testing APIs to generate mock data at runtime, rather than just at
test time.

## Usage & Example

For usage and examples see the [Godoc](http://godoc.org/github.com/mitchellh/go-testing-interface).

Given a test helper written using `go-testing-interface` like this:

    import "github.com/mitchellh/go-testing-interface"

    func TestHelper(t testing.T) {
        t.Fatal("I failed")
    }

You can call the test helper in a real test easily:

    import "testing"

    func TestThing(t *testing.T) {
        TestHelper(t)
    }

You can also call the test helper at runtime if needed:

    import "github.com/mitchellh/go-testing-interface"

    func main() {
        TestHelper(&testing.RuntimeT{})
    }

## Why?!

**Why would I call a test helper that takes a *testing.T at runtime?**

You probably shouldn't. The only use case I've seen (and I've had) for this
is to implement a "dev mode" for a service where the test helpers are used
to populate mock data, create a mock DB, perhaps run service dependencies
in-memory, etc.

Outside of a "dev mode", I've never seen a use case for this and I think
there shouldn't be one since the point of the `testing.T` interface is that
you can fail immediately.
