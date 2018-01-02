This package copies definitions used with GRPC to represent error conditions
within GRPC data types. These files are licensed under the provisions outlined
at the top of each file.

## `containerd`

This is moved from the [googleapis
project](https://github.com/googleapis/googleapis/tree/master/google/rpc) to
allow us to regenerate these types for use with gogoprotobuf. We can move this
away if google can generate these sensibly.

These files were imported from changes after
7f47d894837ac1701ee555fd5c3d70e5d4a796b1. Updates should not be required.

The other option is to get these into an upstream project, like gogoprotobuf.

Note that the `go_package` option has been changed so that they generate
correctly in a common package in the containerd project.
