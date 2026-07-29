# Contributing to Ginkgo

Your contributions to Ginkgo are essential for its long-term maintenance and improvement.

- Please **open an issue first** - describe what problem you are trying to solve and give the community a forum for input and feedback ahead of investing time in writing code!
- Ensure adequate test coverage:
    - When adding to the Ginkgo library, add unit and/or integration tests (under the `integration` folder).
    - When adding to the Ginkgo CLI, note that there are very few unit tests.  Please add an integration test.
- Run `make` or:
  - Install ginkgo locally via `go install ./...`
  - Make sure all the tests succeed via `ginkgo -r -p`
  - Vet your changes via `go vet ./...`
- Update the documentation. Ginkgo uses `godoc` comments and documentation in `docs/index.md`.  You can run `bundle && bundle exec jekyll serve` in the `docs` directory to preview your changes.

Thanks for supporting Ginkgo!
