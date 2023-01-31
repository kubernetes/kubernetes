# Release Process

## Semantic Convention Generation

New versions of the [OpenTelemetry specification] mean new versions of the `semconv` package need to be generated.
The `semconv-generate` make target is used for this.

1. Checkout a local copy of the [OpenTelemetry specification] to the desired release tag.
2. Run the `make semconv-generate ...` target from this repository.

For example,

```sh
export TAG="v1.7.0" # Change to the release version you are generating.
export OTEL_SPEC_REPO="/absolute/path/to/opentelemetry-specification"
git -C "$OTEL_SPEC_REPO" checkout "tags/$TAG"
make semconv-generate # Uses the exported TAG and OTEL_SPEC_REPO.
```

This should create a new sub-package of [`semconv`](./semconv).
Ensure things look correct before submitting a pull request to include the addition.

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

## Verify Examples

After releasing verify that examples build outside of the repository.

```
./verify_examples.sh
```

The script copies examples into a different directory removes any `replace` declarations in `go.mod` and builds them.
This ensures they build with the published release, not the local copy.

## Post-Release

### Contrib Repository

Once verified be sure to [make a release for the `contrib` repository](https://github.com/open-telemetry/opentelemetry-go-contrib/blob/main/RELEASING.md) that uses this release.

### Website Documentation

Update [the documentation](./website_docs) for [the OpenTelemetry website](https://opentelemetry.io/docs/go/).
Importantly, bump any package versions referenced to be the latest one you just released and ensure all code examples still compile and are accurate.

[OpenTelemetry specification]: https://github.com/open-telemetry/opentelemetry-specification
