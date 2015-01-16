# Contributing

We don't use GitHub pull requests but use Gerrit for code reviews,
similar to the Go project.

1. Sign one of the contributor license agreements below.
2. `go get golang.org/x/review/git-codereview` to install the code reviewing tool.
3. Get the package by running `go get -d golang.org/x/oauth2`.
Make changes and create a change by running `git codereview change <name>`, provide a command message, and use `git codereview mail` to create a Gerrit CL.
Keep amending to the change and mail as your recieve feedback.

For more information about the workflow, see Go's [Contribution Guidelines](https://golang.org/doc/contribute.html).

Before we can accept any pull requests
we have to jump through a couple of legal hurdles,
primarily a Contributor License Agreement (CLA):

- **If you are an individual writing original source code**
  and you're sure you own the intellectual property,
  then you'll need to sign an [individual CLA](http://code.google.com/legal/individual-cla-v1.0.html).
- **If you work for a company that wants to allow you to contribute your work**,
  then you'll need to sign a [corporate CLA](http://code.google.com/legal/corporate-cla-v1.0.html).

You can sign these electronically (just scroll to the bottom).
After that, we'll be able to accept your pull requests.
