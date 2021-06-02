# gotenv

[![Build Status](https://travis-ci.org/subosito/gotenv.svg?branch=master)](https://travis-ci.org/subosito/gotenv)
[![Build status](https://ci.appveyor.com/api/projects/status/wb2e075xkfl0m0v2/branch/master?svg=true)](https://ci.appveyor.com/project/subosito/gotenv/branch/master)
[![Coverage Status](https://badgen.net/codecov/c/github/subosito/gotenv)](https://codecov.io/gh/subosito/gotenv)
[![Go Report Card](https://goreportcard.com/badge/github.com/subosito/gotenv)](https://goreportcard.com/report/github.com/subosito/gotenv)
[![GoDoc](https://godoc.org/github.com/subosito/gotenv?status.svg)](https://godoc.org/github.com/subosito/gotenv)

Load environment variables dynamically in Go.

## Usage

Put the gotenv package on your `import` statement:

```go
import "github.com/subosito/gotenv"
```

To modify your app environment variables, `gotenv` expose 2 main functions:

- `gotenv.Load`
- `gotenv.Apply`

By default, `gotenv.Load` will look for a file called `.env` in the current working directory.

Behind the scene, it will then load `.env` file and export the valid variables to the environment variables. Make sure you call the method as soon as possible to ensure it loads all variables, say, put it on `init()` function.

Once loaded you can use `os.Getenv()` to get the value of the variable.

Let's say you have `.env` file:

```
APP_ID=1234567
APP_SECRET=abcdef
```

Here's the example of your app:

```go
package main

import (
	"github.com/subosito/gotenv"
	"log"
	"os"
)

func init() {
	gotenv.Load()
}

func main() {
	log.Println(os.Getenv("APP_ID"))     // "1234567"
	log.Println(os.Getenv("APP_SECRET")) // "abcdef"
}
```

You can also load other than `.env` file if you wish. Just supply filenames when calling `Load()`. It will load them in order and the first value set for a variable will win.:

```go
gotenv.Load(".env.production", "credentials")
```

While `gotenv.Load` loads entries from `.env` file, `gotenv.Apply` allows you to use any `io.Reader`:

```go
gotenv.Apply(strings.NewReader("APP_ID=1234567"))

log.Println(os.Getenv("APP_ID"))
// Output: "1234567"
```

Both `gotenv.Load` and `gotenv.Apply` **DO NOT** overrides existing environment variables. If you want to override existing ones, you can see section below.

### Environment Overrides

Besides above functions, `gotenv` also provides another functions that overrides existing:

- `gotenv.OverLoad`
- `gotenv.OverApply`


Here's the example of this overrides behavior:

```go
os.Setenv("HELLO", "world")

// NOTE: using Apply existing value will be reserved
gotenv.Apply(strings.NewReader("HELLO=universe"))
fmt.Println(os.Getenv("HELLO"))
// Output: "world"

// NOTE: using OverApply existing value will be overridden
gotenv.OverApply(strings.NewReader("HELLO=universe"))
fmt.Println(os.Getenv("HELLO"))
// Output: "universe"
```

### Throw a Panic

Both `gotenv.Load` and `gotenv.OverLoad` returns an error on something wrong occurred, like your env file is not exist, and so on. To make it easier to use, `gotenv` also provides `gotenv.Must` helper, to let it panic when an error returned.

```go
err := gotenv.Load(".env-is-not-exist")
fmt.Println("error", err)
// error: open .env-is-not-exist: no such file or directory

gotenv.Must(gotenv.Load, ".env-is-not-exist")
// it will throw a panic
// panic: open .env-is-not-exist: no such file or directory
```

### Another Scenario

Just in case you want to parse environment variables from any `io.Reader`, gotenv keeps its `Parse` and `StrictParse` function as public API so you can use that.

```go
// import "strings"

pairs := gotenv.Parse(strings.NewReader("FOO=test\nBAR=$FOO"))
// gotenv.Env{"FOO": "test", "BAR": "test"}

err, pairs = gotenv.StrictParse(strings.NewReader(`FOO="bar"`))
// gotenv.Env{"FOO": "bar"}
```

`Parse` ignores invalid lines and returns `Env` of valid environment variables, while `StrictParse` returns an error for invalid lines.

## Notes

The gotenv package is a Go port of [`dotenv`](https://github.com/bkeepers/dotenv) project with some additions made for Go. For general features, it aims to be compatible as close as possible.
