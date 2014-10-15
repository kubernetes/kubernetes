# perigee

Perigee provides a REST client that, while it should be generic enough to use with most any RESTful API, is nonetheless optimized to the needs of the OpenStack APIs.
Perigee grew out of the need to refactor out common API access code from the [gorax](http://github.com/racker/gorax) project.

Several things influenced the name of the project.
Numerous elements of the OpenStack ecosystem are named after astronomical artifacts.
Additionally, perigee occurs when two orbiting bodies are closest to each other.
Perigee seemed appropriate for something aiming to bring OpenStack and other RESTful services closer to the end-user.

**This library is still in the very early stages of development. Unless you want to contribute, it probably isn't what you want**

## Installation and Testing

To install:

```bash
go get github.com/racker/perigee
```

To run unit tests:

```bash
go test github.com/racker/perigee
```

## Contributing

The following guidelines are preliminary, as this project is just starting out.
However, this should serve as a working first-draft.

### Branching

The master branch must always be a valid build.
The `go get` command will not work otherwise.
Therefore, development must occur on a different branch.

When creating a feature branch, do so off the master branch:

```bash
git checkout master
git pull
git checkout -b featureBranch
git checkout -b featureBranch-wip   # optional
```

Perform all your editing and testing in the WIP-branch.
Feel free to make as many commits as you see fit.
You may even open "WIP" pull requests from your feature branch to seek feedback.
WIP pull requests will **never** be merged, however.

To get code merged, you'll need to "squash" your changes into one or more clean commits in the feature branch.
These steps should be followed:

```bash
git checkout featureBranch
git merge --squash featureBranch-wip
git commit -a
git push origin featureBranch
```

You may now open a nice, clean, self-contained pull request from featureBranch to master.

The `git commit -a` command above will open a text editor so that
you may provide a comprehensive description of the changes.

In general, when submitting a pull request against master,
be sure to answer the following questions:

- What is the problem?
- Why is it a problem?
- What is your solution?
- How does your solution work?  (Recommended for non-trivial changes.)
- Why should we use your solution over someone elses?  (Recommended especially if multiple solutions being discussed.)

Remember that monster-sized pull requests are a bear to code-review,
so having helpful commit logs are an absolute must to review changes as quickly as possible.

Finally, (s)he who breaks master is ultimately responsible for fixing master.

### Source Representation

The Go community firmly believes in a consistent representation for all Go source code.
We do too.
Make sure all source code is passed through "go fmt" *before* you create your pull request.

Please note, however, that we fully acknowledge and recognize that we no longer rely upon punch-cards for representing source files.
Therefore, no 80-column limit exists.
However, if a line exceeds 132 columns, you may want to consider splitting the line.

### Unit and Integration Tests

Pull requests that include non-trivial code changes without accompanying unit tests will be flatly rejected.
While we have no way of enforcing this practice,
you can ensure your code is thoroughly tested by always [writing tests first by intention.](http://en.wikipedia.org/wiki/Test-driven_development)

When creating a pull request, if even one test fails, the PR will be rejected.
Make sure all unit tests pass.
Make sure all integration tests pass.

### Documentation

Private functions and methods which are obvious to anyone unfamiliar with gorax needn't be accompanied by documentation.
However, this is a code-smell; if submitting a PR, expect to justify your decision.

Public functions, regardless of how obvious, **must** have accompanying godoc-style documentation.
This is not to suggest you should provide a tome for each function, however.
Sometimes a link to more information is more appropriate, provided the link is stable, reliable, and pertinent.

Changing documentation often results in bizarre diffs in pull requests, due to text often spanning multiple lines.
To work around this, put [one logical thought or sentence on a single line.](http://rhodesmill.org/brandon/2012/one-sentence-per-line/)
While this looks weird in a plain-text editor,
remember that both godoc and HTML viewers will reflow text.
The source code and its comments should be easy to edit with minimal diff pollution.
Let software dedicated to presenting the documentation to human readers deal with its presentation.

## Examples

t.b.d.

