## Getting a Binary Release

You can either build a release from sources or download a pre-built release.  If you do not plan on developing Kubernetes itself, we suggest a pre-built release.

### Prebuilt Binary Release

The list of binary releases is available for download from the [GitHub Kubernetes repo release page](https://github.com/GoogleCloudPlatform/kubernetes/releases).

Download the latest release and unpack this tar file on Linux or OS X, cd to the created `kubernetes/` directory, and then follow the getting started guide for your cloud.

### Building from source

Get the Kubernetes source.  If you are simply building a release from source there is no need to set up a full golang environment as all building happens in a Docker container.

Building a release is simple.

```bash
git clone https://github.com/GoogleCloudPlatform/kubernetes.git
cd kubernetes
make release
```

For more details on the release process see the [`build/` directory](../../build)


[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/getting-started-guides/binary_release.md?pixel)]()
