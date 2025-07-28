# Release Process

## Create a `Version Release` issue

Create a `Version Release` issue to track the release process.

## Semantic Convention Generation

New versions of the [OpenTelemetry Semantic Conventions] mean new versions of the `semconv` package need to be generated.
The `semconv-generate` make target is used for this.

1. Set the `TAG` environment variable to the semantic convention tag you want to generate.
2. Run the `make semconv-generate ...` target from this repository.

For example,

```sh
export TAG="v1.30.0" # Change to the release version you are generating.
make semconv-generate # Uses the exported TAG.
```

This should create a new sub-package of [`semconv`](./semconv).
Ensure things look correct before submitting a pull request to include the addition.

## Breaking changes validation

You can run `make gorelease` that runs [gorelease](https://pkg.go.dev/golang.org/x/exp/cmd/gorelease) to ensure that there are no unwanted changes done in the public API.

You can check/report problems with `gorelease` [here](https://golang.org/issues/26420).

## Verify changes for contrib repository

If the changes in the main repository are going to affect the contrib repository, it is important to verify that the changes are compatible with the contrib repository.

Follow [the steps](https://github.com/open-telemetry/opentelemetry-go-contrib/blob/main/RELEASING.md#verify-otel-changes) in the contrib repository to verify OTel changes.

## Pre-Release

First, decide which module sets will be released and update their versions
in `versions.yaml`.  Commit this change to a new branch.

Update go.mod for submodules to depend on the new release which will happen in the next step.

1. Run the `prerelease` make target. It creates a branch
    `prerelease_<module set>_<new tag>` that will contain all release changes.

    ```
    make prerelease MODSET=<module set>
    ```

2. Verify the changes.

    ```
    git diff ...prerelease_<module set>_<new tag>
    ```

    This should have changed the version for all modules to be `<new tag>`.
    If these changes look correct, merge them into your pre-release branch:

    ```go
    git merge prerelease_<module set>_<new tag>
    ```

3. Update the [Changelog](./CHANGELOG.md).
   - Make sure all relevant changes for this release are included and are in language that non-contributors to the project can understand.
       To verify this, you can look directly at the commits since the `<last tag>`.

       ```
       git --no-pager log --pretty=oneline "<last tag>..HEAD"
       ```

   - Move all the `Unreleased` changes into a new section following the title scheme (`[<new tag>] - <date of release>`).
   - Make sure the new section is under the comment for released section, like `<!-- Released section -->`, so it is protected from being overwritten in the future.
   - Update all the appropriate links at the bottom.

4. Push the changes to upstream and create a Pull Request on GitHub.
    Be sure to include the curated changes from the [Changelog](./CHANGELOG.md) in the description.

## Tag

Once the Pull Request with all the version changes has been approved and merged it is time to tag the merged commit.

***IMPORTANT***: It is critical you use the same tag that you used in the Pre-Release step!
Failure to do so will leave things in a broken state. As long as you do not
change `versions.yaml` between pre-release and this step, things should be fine.

***IMPORTANT***: [There is currently no way to remove an incorrectly tagged version of a Go module](https://github.com/golang/go/issues/34189).
It is critical you make sure the version you push upstream is correct.
[Failure to do so will lead to minor emergencies and tough to work around](https://github.com/open-telemetry/opentelemetry-go/issues/331).

1. For each module set that will be released, run the `add-tags` make target
    using the `<commit-hash>` of the commit on the main branch for the merged Pull Request.

    ```
    make add-tags MODSET=<module set> COMMIT=<commit hash>
    ```

    It should only be necessary to provide an explicit `COMMIT` value if the
    current `HEAD` of your working directory is not the correct commit.

2. Push tags to the upstream remote (not your fork: `github.com/open-telemetry/opentelemetry-go.git`).
    Make sure you push all sub-modules as well.

    ```
    git push upstream <new tag>
    git push upstream <submodules-path/new tag>
    ...
    ```

## Release

Finally create a Release for the new `<new tag>` on GitHub.
The release body should include all the release notes from the Changelog for this release.

## Post-Release

### Contrib Repository

Once verified be sure to [make a release for the `contrib` repository](https://github.com/open-telemetry/opentelemetry-go-contrib/blob/main/RELEASING.md) that uses this release.

### Website Documentation

Update the [Go instrumentation documentation] in the OpenTelemetry website under [content/en/docs/languages/go].
Importantly, bump any package versions referenced to be the latest one you just released and ensure all code examples still compile and are accurate.

[OpenTelemetry Semantic Conventions]: https://github.com/open-telemetry/semantic-conventions
[Go instrumentation documentation]: https://opentelemetry.io/docs/languages/go/
[content/en/docs/languages/go]: https://github.com/open-telemetry/opentelemetry.io/tree/main/content/en/docs/languages/go

### Close the milestone

Once a release is made, ensure all issues that were fixed and PRs that were merged as part of this release are added to the corresponding milestone.
This helps track what changes were included in each release.

- To find issues that haven't been included in a milestone, use this [GitHub search query](https://github.com/open-telemetry/opentelemetry-go/issues?q=is%3Aissue%20no%3Amilestone%20is%3Aclosed%20sort%3Aupdated-desc%20reason%3Acompleted%20-label%3AStale%20linked%3Apr)
- To find merged PRs that haven't been included in a milestone, use this [GitHub search query](https://github.com/open-telemetry/opentelemetry-go/pulls?q=is%3Apr+no%3Amilestone+is%3Amerged).

Once all related issues and PRs have been added to the milestone, close the milestone.

### Demo Repository

Bump the dependencies in the following Go services:

- [`accounting`](https://github.com/open-telemetry/opentelemetry-demo/tree/main/src/accounting)
- [`checkoutservice`](https://github.com/open-telemetry/opentelemetry-demo/tree/main/src/checkout)
- [`productcatalogservice`](https://github.com/open-telemetry/opentelemetry-demo/tree/main/src/product-catalog)

### Close the `Version Release` issue

Once the todo list in the `Version Release` issue is complete, close the issue.
