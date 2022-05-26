![go-git logo](https://cdn.rawgit.com/src-d/artwork/02036484/go-git/files/go-git-github-readme-header.png)
[![GoDoc](https://godoc.org/github.com/go-git/go-git/v5?status.svg)](https://pkg.go.dev/github.com/go-git/go-git/v5) [![Build Status](https://github.com/go-git/go-git/workflows/Test/badge.svg)](https://github.com/go-git/go-git/actions) [![Go Report Card](https://goreportcard.com/badge/github.com/go-git/go-git)](https://goreportcard.com/report/github.com/go-git/go-git)

*go-git* is a highly extensible git implementation library written in **pure Go**.

It can be used to manipulate git repositories at low level *(plumbing)* or high level *(porcelain)*, through an idiomatic Go API. It also supports several types of storage, such as in-memory filesystems, or custom implementations, thanks to the [`Storer`](https://pkg.go.dev/github.com/go-git/go-git/v5/plumbing/storer) interface.

It's being actively developed since 2015 and is being used extensively by [Keybase](https://keybase.io/blog/encrypted-git-for-everyone), [Gitea](https://gitea.io/en-us/) or [Pulumi](https://github.com/search?q=org%3Apulumi+go-git&type=Code), and by many other libraries and tools.

Project Status
--------------

After the legal issues with the [`src-d`](https://github.com/src-d) organization, the lack of update for four months and the requirement to make a hard fork, the project is **now back to normality**.

The project is currently actively maintained by individual contributors, including several of the original authors, but also backed by a new company, [gitsight](https://github.com/gitsight), where `go-git` is a critical component used at scale.


Comparison with git
-------------------

*go-git* aims to be fully compatible with [git](https://github.com/git/git), all the *porcelain* operations are implemented to work exactly as *git* does.

*git* is a humongous project with years of development by thousands of contributors, making it challenging for *go-git* to implement all the features. You can find a comparison of *go-git* vs *git* in the [compatibility documentation](COMPATIBILITY.md).


Installation
------------

The recommended way to install *go-git* is:

```go
import "github.com/go-git/go-git/v5" // with go modules enabled (GO111MODULE=on or outside GOPATH)
import "github.com/go-git/go-git" // with go modules disabled
```


Examples
--------

> Please note that the `CheckIfError` and `Info` functions  used in the examples are from the [examples package](https://github.com/go-git/go-git/blob/master/_examples/common.go#L19) just to be used in the examples.


### Basic example

A basic example that mimics the standard `git clone` command

```go
// Clone the given repository to the given directory
Info("git clone https://github.com/go-git/go-git")

_, err := git.PlainClone("/tmp/foo", false, &git.CloneOptions{
    URL:      "https://github.com/go-git/go-git",
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
Info("git clone https://github.com/go-git/go-billy")

r, err := git.Clone(memory.NewStorage(), nil, &git.CloneOptions{
    URL: "https://github.com/go-git/go-billy",
})

CheckIfError(err)

// Gets the HEAD history from HEAD, just like this command:
Info("git log")

// ... retrieves the branch pointed by HEAD
ref, err := r.Head()
CheckIfError(err)


// ... retrieves the commit history
cIter, err := r.Log(&git.LogOptions{From: ref.Hash()})
CheckIfError(err)

// ... just iterates over the commits, printing it
err = cIter.ForEach(func(c *object.Commit) error {
	fmt.Println(c)
	return nil
})
CheckIfError(err)
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

You can find this [example](_examples/log/main.go) and many others in the [examples](_examples) folder.

Contribute
----------

[Contributions](https://github.com/go-git/go-git/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) are more than welcome, if you are interested please take a look to
our [Contributing Guidelines](CONTRIBUTING.md).

License
-------
Apache License Version 2.0, see [LICENSE](LICENSE)
