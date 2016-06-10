<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->

# How to generate the bps-data-generator container #

This container is maintained as part of the apache bigtop project.

To create it, simply

`git clone https://github.com/apache/bigtop`

and checkout the last exact version (will be updated periodically).

`git checkout -b aNewBranch 2b2392bf135e9f1256bd0b930f05ae5aef8bbdcb`

then, cd to bigtop-bigpetstore/bigpetstore-transaction-queue, and run the docker file, i.e.

`Docker build -t -i jayunit100/bps-transaction-queue`.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/examples/k8petstore/bps-data-generator/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
