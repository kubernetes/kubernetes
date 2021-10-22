# How to Release this Repo

1. Determine the current release version with `git tag -l`. It should look
   something like `vX.Y.Z`. We'll call the current version `$CV` and the new
   version `$NV`.
1. On master, run `git log $CV..` to list all the changes since the last
   release.
1. `cd internal/version && go generate && cd ../..`
1. Edit `CHANGES.md` to include a summary of the changes.
1. Mail the CL containing the `CHANGES.md` changes. When the CL is approved,
   submit it.
1. Without submitting any other CLs:
   a. Switch to master.
   b. `git pull`
   c. Tag the repo with the next version: `git tag $NV`.
   d. Push the tag: `git push origin $NV`.
1. Update [the releases page](https://github.com/googleapis/google-api-go-client/releases)
   with the new release, copying the contents of the CHANGES.md.

# Auto-regeneration

A nightly cron job re-generates all of the clients. It runs `make all`.

Releases are manually tagged (see above).

See:

* [Code for the cronjob, in the discovery-artifact-manager repo](https://github.com/googleapis/discovery-artifact-manager/blob/master/server/tasks/google_api_go_client.py)
* [List of cronjobs](https://console.cloud.google.com/appengine/cronjobs?project=discovery-artifact-manager)
* [App Engine stderr logs](https://console.cloud.google.com/logs/viewer?project=discovery-artifact-manager)
