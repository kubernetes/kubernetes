# Cutting a release

Until we have a proper setup for building this automatically with every binary
release, here are the steps for making a release.  We make releases when they
are ready, not on every PR.

1. Build the container for testing: `make container PREFIX=<your-docker-hub> TAG=rc`

2. Manually deploy this to your own cluster by updating the replication
   controller and deleting the running pod(s).

3. Verify it works.

4. Update the TAG version in `Makefile` and update the `Changelog`.  Update the
   `*.yaml.in` to point to the new tag.  Send a PR but mark it as "DO NOT MERGE".

5. Once the PR is approved, build the container for real: `make container`.

6. Push the container: `make push`.

7. Manually deploy this to your own cluster by updating the replication
   controller and deleting the running pod(s).

8. Verify it works.

9. Allow the PR to be merged.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/cluster/addons/dns/kube2sky/RELEASES.md?pixel)]()
