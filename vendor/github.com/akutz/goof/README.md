# Goof [![Build Status](https://travis-ci.org/akutz/goof.svg)](https://travis-ci.org/akutz/goof)  [![GoDoc](https://godoc.org/github.com/akutz/goof?status.svg)](http://godoc.org/github.com/akutz/goof)

Goof is a drop-in replacement for the go stdlib `errors` package, providing
enhanced error construction capabilities and formatting capabilities.

## Plays Golf
Errors can be created with the standard `New` or `Newf` constructors in order to
create new error instances. However, the methods `WithField`, `WithFields`,
`WithFieldE`, and `WithFieldsE` also exist. These methods enable the
construction of new error instances with additional information about the error
injected at the time and location of when the error was created.

These fields then become available to the
[Golf framework](https://github.com/akutz/golf) in order to extract more
information about an error than just a simple error message.

## Structured Logging
The support for fields as a backing-store in conjunction with Golf support
enable the ability to seamlessly integrate with the structured logging
framework [Logrus](https://github.com/Sirupsen/logrus).

The advantage of this marriage between error objects and a structured logging
framework is that information about an error is stored *with* an error, at the
time of the error's construction. This alleviates the question of whether or
not to log an error and the contextual information surrounding it when an error
is created.

### Logging Errors sans Goof
The file `ex1-nogoof.go` in the `examples/example1-nogoof` folder demonstrates
how traditional error handling and logging looks without Goof:

    package main

    import (
        "fmt"
        "os"

        log "github.com/Sirupsen/logrus"
    )

    func getDividend() int {
        return 2
    }

    func getDivisor() int {
        return 0
    }

    func divide() (int, error) {
        x := getDividend()
        y := getDivisor()

        if y == 0 {
            log.Errorf("error dividing by 0 with %d / %d", x, y)
            return -1, fmt.Errorf("error dividing by 0 with %d / %d", x, y)
        }

        return x / y, nil
    }

    func calculate(op string) error {
        switch op {
        case "divide":
            z, err := divide()
            if err != nil {
                log.Errorf("division error %v", err)
                return fmt.Errorf("division error %v", err)
            }
            fmt.Printf("division = %d\n", z)
            return nil
        }
        return nil
    }

    func main() {
        if err := calculate("divide"); err != nil {
            log.Errorf("calculation error %v", err)
            os.Exit(1)
        }
    }

Running the above example results in the above output:

    $ go run examples/example1-nogoof/ex1-nogoof.go
    ERRO[0000] error dividing by 0 with 2 / 0               
    ERRO[0000] division error error dividing by 0 with 2 / 0
    ERRO[0000] calculation error division error error dividing by 0 with 2 / 0
    exit status 1

In the example above the `main` function asks `calculate` to do division, and
so `calculate` forwards that request to `divide`. The `divide` function then
fetches the dividend and the divisor from some data store via the (undefined)
methods `getDividend` and `getDivisor` and proceeds to perform the operation.

However, if the divisor is zero then a *divide-by-zero* is logged and an error
is returned to `calculate` which in turn logs and returns the error to `main`
which also logs the error.

The problem is neither the `divide` or `calculate` functions should really be
logging anything regarding errors. Error logging should be as centralized as
possible in order to avoid cluttering logs with duplicate information. This
often means logging errors at the outer-most areas of a program.

Yet this choice also means you can, and often do, lose contextual information
about the errors. In this case neither `calculate` or `main` know what the
dividend or divisor were. True, the error object can format a string that
includes that information, but the logging framework
[Logrus](https://github.com/Sirupsen/logrus) articulates a very intelligent
case for structured logging.

### Logging Errors with Goof
Goof on the other hand makes creating errors that can be logged by a structured
logger as simple as can be. Let's revisit the previous example using the file
`ex2-goof.go` in the `examples/example2-goof` folder:

    package main

    import (
        "fmt"
        "os"

        log "github.com/Sirupsen/logrus"
        "github.com/akutz/goof"
    )

    func divide() int, error {
        x := getDividend()
        y := getDivisor()
        if y == 0 {
            return -1, goof.WithFields(goof.Fields{
                "dividend": x,
                "divisor": y,
                }, "divide by zero")
        }

        return x / y
    }

    func calculate(op string) error {
        switch op {
            case "divide":
                if z, err := divide(); err != nil {
                    return err
                } else {
                    fmt.Printf("division = %d\n", z)
                    return nil
                }
        }
    }

    func main() {
        if err := calculate("divide"); err != nil {
            log.Error(err)
            os.Exit(1)
        }
    }

In the refactored example no errors are logged in the `divide` or `calculate`
functions. Instead, an error is created with fields labeled as "divisor" and
"dividend" with those fields set to the values to which they relate. The error
is also created with a brief, but sufficient, message, describing the issue.

This error is then returned all the way to the main function where it is logged
via the structured logging framework Logrus. Because the main function also
instructs Logrus to use the Golf formatter for logging, this is what is emitted
to the console:

    $ go run examples/example2-goof/ex2-goof.go
    ERRO[0000] divide by zero                               dividend=2 divisor=0
    exit status 1

The log output is now much cleaner, concise, and without losing any information
regarding the context of the error and that may be helpful to debugging.
