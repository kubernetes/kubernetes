# Contributing #

Thanks for your contribution in advance. No matter what you will contribute to this project, pull request or bug report or feature discussion, it's always highly appreciated.

## New API or feature ##

I want to speak more about how to add new functions to this package.

Package `xstring` is a collection of useful string functions which should be implemented in Go. It's a bit subject to say which function should be included and which should not. I set up following rules in order to make it clear and as objective as possible.

* Rule 1: Only string algorithm, which takes string as input, can be included.
* Rule 2: If a function has been implemented in package `string`, it must not be included.
* Rule 3: If a function is not language neutral, it must not be included.
* Rule 4: If a function is a part of standard library in other languages, it can be included.
* Rule 5: If a function is quite useful in some famous framework or library, it can be included.

New function must be discussed in project issues before submitting any code. If a pull request with new functions is sent without any ref issue, it will be rejected.

## Pull request ##

Pull request is always welcome. Just make sure you have run `go fmt` and all test cases passed before submit.

If the pull request is to add a new API or feature, don't forget to update README.md and add new API in function list.
