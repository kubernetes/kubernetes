# Contributing to Gomega

Your contributions to Gomega are essential for its long-term maintenance and improvement.  To make a contribution:

- Please **open an issue first** - describe what problem you are trying to solve and give the community a forum for input and feedback ahead of investing time in writing code!
- Ensure adequate test coverage:
    - Make sure to add appropriate unit tests
    - Please run all tests locally (`ginkgo -r -p`) and make sure they go green before submitting the PR
    - Please run following linter locally `go vet ./...` and make sure output does not contain any warnings
- Update the documentation.  In addition to standard `godoc` comments Gomega has extensive documentation on the `gh-pages` branch.  If relevant, please submit a docs PR to that branch alongside your code PR.

If you're a committer, check out RELEASING.md to learn how to cut a release.

Thanks for supporting Gomega!
