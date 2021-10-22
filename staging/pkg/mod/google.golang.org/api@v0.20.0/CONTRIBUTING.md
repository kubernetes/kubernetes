# Contributing

1. [Install Go](https://golang.org/dl/).
    1. Ensure that your `GOBIN` directory (by default `$(go env GOPATH)/bin`)
    is in your `PATH`.
    1. Check it's working by running `go version`.
        * If it doesn't work, check the install location, usually
        `/usr/local/go`, is on your `PATH`.

1. Sign one of the
[contributor license agreements](#contributor-license-agreements) below.

1. Run `go get golang.org/x/review/git-codereview` to install the code reviewing tool.

    1. Ensure it's working by running `git codereview` (check your `PATH` if
    not).

    1. If you would like, you may want to set up aliases for `git-codereview`,
    such that `git codereview change` becomes `git change`. See the
    [godoc](https://godoc.org/golang.org/x/review/git-codereview) for details.

        * Should you run into issues with the `git-codereview` tool, please note
        that all error messages will assume that you have set up these aliases.

1. Change to a directory of your choosing and clone the repo.

    ```
    cd ~/code
    git clone https://code.googlesource.com/google-api-go-client
    ```

    * If you have already checked out the source, make sure that the remote
    `git` `origin` is https://code.googlesource.com/google-api-go-client:

        ```
        git remote -v
        # ...
        git remote set-url origin https://code.googlesource.com/google-api-go-client
        ```

    * The project uses [Go Modules](https://blog.golang.org/using-go-modules)
    for dependency management See
    [`gopls`](https://github.com/golang/go/wiki/gopls) for making your editor
    work with modules.

1. Change to the project directory:

    ```
    cd ~/code/google-api-go-client
    ```

1. Make sure your `git` auth is configured correctly by visiting
https://code.googlesource.com, clicking "Generate Password" at the top-right,
and following the directions. Otherwise, `git codereview mail` in the next step
will fail.

1. Now you are ready to make changes. Don't create a new branch or make commits in the traditional
way. Use the following`git codereview` commands to create a commit and create a Gerrit CL:

    ```
    git codereview change <branch-name> # Use this instead of git checkout -b <branch-name>
    # Make changes.
    git add ...
    git codereview change # Use this instead of git commit
    git codereview mail # If this fails, the error message will contain instructions to fix it.
    ```

    * This will create a new `git` branch for you to develop on. Once your
    change is merged, you can delete this branch.

1. As you make changes for code review, ammend the commit and re-mail the
change:

    ```
    # Make more changes.
    git add ...
    git codereview change
    git codereview mail
    ```

    * **Warning**: do not change the `Change-Id` at the bottom of the commit
    message - it's how Gerrit knows which change this is (or if it's new).

    * When you fixes issues from code review, respond to each code review
    message then click **Reply** at the top of the page.

    * Each new mailed amendment will create a new patch set for
    your change in Gerrit. Patch sets can be compared and reviewed.

    * **Note**: if your change includes a breaking change, our breaking change
    detector will cause CI/CD to fail. If your breaking change is acceptable
    in some way, add a `BREAKING_CHANGE_ACCEPTABLE=<reason>` line to the commit
    message to cause the detector not to be run and to make it clear why that is
    acceptable.

1. Finally, add reviewers to your CL when it's ready for review. Reviewers will
not be added automatically. If you're not sure who to add for your code review,
add deklerk@, tbp@, cbro@, and codyoss@.

## Contributor License Agreements

Before we can accept your pull requests you'll need to sign a Contributor
License Agreement (CLA):

- **If you are an individual writing original source code** and **you own the
intellectual property**, then you'll need to sign an [individual CLA][indvcla].
- **If you work for a company that wants to allow you to contribute your
work**, then you'll need to sign a [corporate CLA][corpcla].

You can sign these electronically (just scroll to the bottom). After that,
we'll be able to accept your pull requests.

## Contributor Code of Conduct

As contributors and maintainers of this project,
and in the interest of fostering an open and welcoming community,
we pledge to respect all people who contribute through reporting issues,
posting feature requests, updating documentation,
submitting pull requests or patches, and other activities.

We are committed to making participation in this project
a harassment-free experience for everyone,
regardless of level of experience, gender, gender identity and expression,
sexual orientation, disability, personal appearance,
body size, race, ethnicity, age, religion, or nationality.

Examples of unacceptable behavior by participants include:

* The use of sexualized language or imagery
* Personal attacks
* Trolling or insulting/derogatory comments
* Public or private harassment
* Publishing other's private information,
such as physical or electronic
addresses, without explicit permission
* Other unethical or unprofessional conduct.

Project maintainers have the right and responsibility to remove, edit, or reject
comments, commits, code, wiki edits, issues, and other contributions
that are not aligned to this Code of Conduct.
By adopting this Code of Conduct,
project maintainers commit themselves to fairly and consistently
applying these principles to every aspect of managing this project.
Project maintainers who do not follow or enforce the Code of Conduct
may be permanently removed from the project team.

This code of conduct applies both within project spaces and in public spaces
when an individual is representing the project or its community.

Instances of abusive, harassing, or otherwise unacceptable behavior
may be reported by opening an issue
or contacting one or more of the project maintainers.

This Code of Conduct is adapted from the [Contributor Covenant](http://contributor-covenant.org), version 1.2.0,
available at [http://contributor-covenant.org/version/1/2/0/](http://contributor-covenant.org/version/1/2/0/)

[indvcla]: https://developers.google.com/open-source/cla/individual
[corpcla]: https://developers.google.com/open-source/cla/corporate
