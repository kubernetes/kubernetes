# Golf [![Build Status](https://travis-ci.org/akutz/golf.svg)](https://travis-ci.org/akutz/golf) [![Coverage Status](https://coveralls.io/repos/akutz/golf/badge.svg?branch=master&service=github)](https://coveralls.io/github/akutz/golf?branch=master) [![GoDoc](https://godoc.org/github.com/akutz/golf?status.svg)](http://godoc.org/github.com/akutz/golf)

Go List Fields (Golf) is a package for the Go language that makes it
incredibly simple to get a list of fields from any type using the Field
Output through Reflective Evaluation function, otherwise known as Fore:

    type Person struct {
        Name string
    }

    p := &Person{"Batman"}

    fields := golf.Fore("hero", p)

    for k,v := range fields {
        fmt.Printf("%s=%v\n", k,v)
    }

The above example will emit the following:

    hero.Name=Batman

A simple example, the true power of Golf is in its configuration
options.