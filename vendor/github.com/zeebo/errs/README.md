# errs

[![GoDoc](https://godoc.org/github.com/zeebo/errs?status.svg)](https://godoc.org/github.com/zeebo/errs)
[![Sourcegraph](https://sourcegraph.com/github.com/zeebo/errs/-/badge.svg)](https://sourcegraph.com/github.com/zeebo/errs?badge)
[![Go Report Card](https://goreportcard.com/badge/github.com/zeebo/errs)](https://goreportcard.com/report/github.com/zeebo/errs)

errs is a package for making errors friendly and easy.

### Creating Errors

The easiest way to use it, is to use the package level [New][New] function.
It's much like `fmt.Errorf`, but better. For example:

```go
func checkThing() error {
	return errs.New("what's up with %q?", "zeebo")
}
```

Why is it better? Errors come with a stack trace that is only printed
when a `"+"` character is used in the format string. This should retain the
benefits of being able to diagnose where and why errors happen, without all of
the noise of printing a stack trace in every situation. For example:

```go
func doSomeRealWork() {
	err := checkThing()
	if err != nil {
		fmt.Printf("%+v\n", err) // contains stack trace if it's a errs error.
		fmt.Printf("%v\n", err)  // does not contain a stack trace
		return
	}
}
```

### Error Classes

You can create a [Class][Class] of errors and check if any error was created by
that class. The class name is prefixed to all of the errors it creates. For example:

```go
var Unauthorized = errs.Class("unauthorized")

func checkUser(username, password string) error {
	if username != "zeebo" {
		return Unauthorized.New("who is %q?", username)
	}
	if password != "hunter2" {
		return Unauthorized.New("that's not a good password, jerkmo!")
	}
	return nil
}

func handleRequest() {
	if err := checkUser("zeebo", "hunter3"); Unauthorized.Has(err) {
		fmt.Println(err)
	}

	// output:
	// unauthorized: that's not a good password, jerkmo!
}
```

Classes can also [Wrap][ClassWrap] other errors, and errors may be wrapped
multiple times. For example:

```go
var (
	Error        = errs.Class("mypackage")
	Unauthorized = errs.Class("unauthorized")
)

func deep3() error {
	return fmt.Errorf("ouch")
}

func deep2() error {
	return Unauthorized.Wrap(deep3())
}

func deep1() error {
	return Error.Wrap(deep2())
}

func deep() {
	fmt.Println(deep1())

	// output:
	// mypackage: unauthorized: ouch
}
```

In the above example, both `Error.Has(deep1())` and `Unauthorized.Has(deep1())`
would return `true`, and the stack trace would only be recorded once at the
`deep2` call.

In addition, when an error has been wrapped, wrapping it again with the same class will
not do anything. For example:

```go
func doubleWrap() {
	fmt.Println(Error.Wrap(Error.New("foo")))

	// output:
	// mypackage: foo
}
```

This is to make it an easier decision if you should wrap or not (you should).

### Utilities

[Classes][Classes] is a helper function to get a slice of classes that an error
has. The latest wrap is first in the slice. For example:

```go
func getClasses() {
	classes := errs.Classes(deep1())
	fmt.Println(classes[0] == &Error)
	fmt.Println(classes[1] == &Unauthorized)

	// output:
	// true
	// true
}
```

Finally, a helper function, [Unwrap][Unwrap] is provided to get the
wrapped error in cases where you might want to inspect details. For
example:

```go
var Error = errs.Class("mypackage")

func getHandle() (*os.File, error) {
	fh, err := os.Open("neat_things")
	if err != nil {
		return nil, Error.Wrap(err)
	}
	return fh, nil
}

func checkForNeatThings() {
	fh, err := getHandle()
	if os.IsNotExist(errs.Unwrap(err)) {
		panic("no neat things?!")
	}
	if err != nil {
		panic("phew, at least there are neat things, even if i can't see them")
	}
	fh.Close()
}
```

It knows about both the `Unwrap() error` and `Unwrap() []error` methods that are
often used in the community, and will call them as many times as possible.

### Defer

The package also provides [WrapP][WrapP] versions of [Wrap][Wrap] that are useful
in defer contexts. For example:

```go
func checkDefer() (err error) {
	defer Error.WrapP(&err)

	fh, err := os.Open("secret_stash")
	if err != nil {
		return nil, err
	}
	return fh.Close()
}
```

### Groups

[Groups][Group] allow one to collect a set of errors. For example:

```go
func tonsOfErrors() error {
	var group errs.Group
	for _, work := range someWork {
		group.Add(maybeErrors(work))
	}
	return group.Err()
}
```

Some things to note:

- The [Add][GroupAdd] method only adds to the group if the passed in error is non-nil.
- The [Err][GroupErr] method returns an error only if non-nil errors have been added, and
  additionally returns just the error if only one error was added. Thus, we always
  have that if you only call `group.Add(err)`, then `group.Err() == err`.

The returned error will format itself similarly:

```go
func groupFormat() {
	var group errs.Group
	group.Add(errs.New("first"))
	group.Add(errs.New("second"))
	err := group.Err()

	fmt.Printf("%v\n", err)
	fmt.Println()
	fmt.Printf("%+v\n", err)

	// output:
	// first; second
	//
	// group:
	// --- first
	//     ... stack trace
	// --- second
	//     ... stack trace
}
```

### Contributing

errs is released under an MIT License. If you want to contribute, be sure to
add yourself to the list in AUTHORS.

[New]: https://godoc.org/github.com/zeebo/errs#New
[Wrap]: https://godoc.org/github.com/zeebo/errs#Wrap
[WrapP]: https://godoc.org/github.com/zeebo/errs#WrapP
[Class]: https://godoc.org/github.com/zeebo/errs#Class
[ClassNew]: https://godoc.org/github.com/zeebo/errs#Class.New
[ClassWrap]: https://godoc.org/github.com/zeebo/errs#Class.Wrap
[Unwrap]: https://godoc.org/github.com/zeebo/errs#Unwrap
[Classes]: https://godoc.org/github.com/zeebo/errs#Classes
[Group]: https://godoc.org/github.com/zeebo/errs#Group
[GroupAdd]: https://godoc.org/github.com/zeebo/errs#Group.Add
[GroupErr]: https://godoc.org/github.com/zeebo/errs#Group.Err
