# go-git [![GoDoc](https://godoc.org/gopkg.in/src-d/go-git.v4?status.svg)](https://godoc.org/github.com/src-d/go-git) [![Build Status](https://travis-ci.org/src-d/go-git.svg)](https://travis-ci.org/src-d/go-git) [![Build status](https://ci.appveyor.com/api/projects/status/nyidskwifo4py6ub?svg=true)](https://ci.appveyor.com/project/mcuadros/go-git) [![codecov.io](https://codecov.io/github/src-d/go-git/coverage.svg)](https://codecov.io/github/src-d/go-git) [![codebeat badge](https://codebeat.co/badges/b6cb2f73-9e54-483d-89f9-4b95a911f40c)](https://codebeat.co/projects/github-com-src-d-go-git)

A highly extensible git implementation in **pure Go**.

*go-git* aims to reach the completeness of [libgit2](https://libgit2.github.com/) or [jgit](http://www.eclipse.org/jgit/), nowadays covers the **majority** of the plumbing **read operations** and **some** of the main **write operations**, but lacks the main porcelain operations such as merges.

It is **highly extensible**, we have been following the open/close principle in its design to facilitate extensions, mainly focusing the efforts on the persistence of the objects.

### ... is this production ready?

The master branch represents the `v4` of the library, it is currently under active development and is planned to be released in early 2017.

If you are looking for a production ready version, please take a look to the [`v3`](https://github.com/src-d/go-git/tree/v3) which is being used in production at [source{d}](http://sourced.tech) since August 2015 to analyze all GitHub public repositories (i.e. 16M repositories).

We recommend the use of `v4` to develop new projects since it includes much new functionality and provides a more *idiomatic git* API

Installation
------------

The recommended way to install *go-git* is:

```
go get -u gopkg.in/src-d/go-git.v4/...
```


Examples
--------

> Please note that the functions `CheckIfError` and `Info` used in the examples are from the [examples package](https://github.com/src-d/go-git/blob/master/_examples/common.go#L17) just to be used in the examples.


### Basic example

A basic example that mimics the standard `git clone` command

```go
// Clone the given repository to the given directory
Info("git clone https://github.com/src-d/go-git")

_, err := git.PlainClone("/tmp/foo", false, &git.CloneOptions{
    URL:      "https://github.com/src-d/go-git",
    Progress: os.Stdout,
})

CheckIfError(err)
```

Outputs:
```
Counting objects: 4924, done.
Compressing objects: 100% (1333/1333), done.
Total 4924 (delta 530), reused 6 (delta 6), pack-reused 3533
```

### In-memory example

Cloning a repository into memory and printing the history of HEAD, just like `git log` does


```go
// Clones the given repository in memory, creating the remote, the local
// branches and fetching the objects, exactly as:
Info("git clone https://github.com/src-d/go-siva")

r, err := git.Clone(memory.NewStorage(), nil, &git.CloneOptions{
    URL: "https://github.com/src-d/go-siva",
})

CheckIfError(err)

// Gets the HEAD history from HEAD, just like does:
Info("git log")

// ... retrieves the branch pointed by HEAD
ref, err := r.Head()
CheckIfError(err)

// ... retrieves the commit object
commit, err := r.CommitObject(ref.Hash())
CheckIfError(err)

// ... retrieves the commit history
history, err := commit.History()
CheckIfError(err)

// ... just iterates over the commits, printing it
for _, c := range history {
    fmt.Println(c)
}
```

Outputs:
```
commit ded8054fd0c3994453e9c8aacaf48d118d42991e
Author: Santiago M. Mola <santi@mola.io>
Date:   Sat Nov 12 21:18:41 2016 +0100

    index: ReadFrom/WriteTo returns IndexReadError/IndexWriteError. (#9)

commit df707095626f384ce2dc1a83b30f9a21d69b9dfc
Author: Santiago M. Mola <santi@mola.io>
Date:   Fri Nov 11 13:23:22 2016 +0100

    readwriter: fix bug when writing index. (#10)

    When using ReadWriter on an existing siva file, absolute offset for
    index entries was not being calculated correctly.
...
```

You can find this [example](_examples/log/main.go) and many other at the [examples](_examples) folder

Comparison With Git
-------------------

In the [compatibility documentation](COMPATIBILITY.md) you can find a comparison
table of git with go-git.

Contribute
----------

If you are interested on contributing to go-git, open an [issue](https://github.com/src-d/go-git/issues) explaining which missing functionality you want to work in, and we will guide you through the implementation.

License
-------

MIT, see [LICENSE](LICENSE)
