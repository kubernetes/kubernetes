This directory is the staging area for packages that have been split to their
own repository. The content here will be periodically published to respective
top-level k8s.io repositories.

The staged content is copied from the main repo, i.e., k8s.io/kubernetes, with
directory rearrangement and necessary rewritings. To sync the content with the
latest code in your local k8s.io/kubernetes, you need to run
`hack/godep-restore.sh` in k8s root directory, then run
`hack/update-staging-client-go.sh`.

The vendor/k8s.io directory contains symlinks pointing to this staging area, so
to use the packages in the staging area, you can import it as
"vendor/client-go/<package-name>", as if the package were vendored. Packages
will be vendored from k8s.io/<package-name> for real after the test matrix is
converted to vendor k8s components.
