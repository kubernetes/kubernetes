# Cutting a release

Until we have a proper setup for building this automatically with every binary
release, here are the steps for making a release. We make releases when they
are ready, not on every PR.

1. Build the container for testing: 

```
make release
cd build/kube-dns
make container PREFIX=<your-docker-hub> TAG=rc
```

2. Manually deploy this to your own cluster by updating the replication
   controller and deleting the running pod(s).

3. Verify it works.

4. Update the TAG version in `Makefile` and update the `Changelog`. Update the
   `*.yaml.in` to point to the new tag. Send a PR but mark it as "DO NOT MERGE".

5. Once the PR is approved, build and push the container for real **for all architectures**:

	```console
	# Build for linux/amd64 (default)
	$ make push ARCH=amd64
	# ---> gcr.io/google_containers/kube-dns-amd64:TAG

	$ make push ARCH=arm
	# ---> gcr.io/google_containers/kube-dns-arm:TAG

	$ make push ARCH=arm64
	# ---> gcr.io/google_containers/kube-dns-arm64:TAG

	$ make push ARCH=ppc64le
	# ---> gcr.io/google_containers/kube-dns-ppc64le:TAG
	```

6. Manually deploy this to your own cluster by updating the replication
   controller and deleting the running pod(s).

7. Verify it works.

8. Allow the PR to be merged.


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/build/kube-dns/RELEASES.md?pixel)]()
