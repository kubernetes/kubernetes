# How to move a utility pkg from other kubernetes repos

It has 2 steps to move a pkg from other Kubernetes repos to `k8s.io/utils` repo:
- copy the pkg to `k8s.io/utils` repo
- update the import paths and `vendor/` in the repos that refer this pkg

## Copy the pkg to `k8s.io/utils` repo

Copying should preserve all the git history associated with it.
[Here](http://gbayer.com/development/moving-files-from-one-git-repository-to-another-preserving-history/) is a working approach.
Note: You may need to use `--allow-unrelated-histories` if you get error when running `git pull` following the post above.

Then, you may need to restructure the package to make sure it has the following structure.

    .
    ├── doc.go                  # Description for this package
    ├── <utilname1>.go          # utility go file
    ├── <utilname>_test.go      # go unit tests
    └── testing                 # All the testing framework
        └── fake_<utilname>.go  # Testing framework go file

[#5](https://github.com/kubernetes/utils/pull/5) is an example for this step.

## Update the repos that refer the pkg

You should update the import paths.
Then follow [this doc](https://github.com/kubernetes/community/blob/master/contributors/devel/godep.md) to update `vendor/` and `Godeps/`.

You may want to run `make bazel-test` to make sure all new references work.

[kubernetes/kubernetes#49234](https://github.com/kubernetes/kubernetes/pull/49234) is an example for this step.
