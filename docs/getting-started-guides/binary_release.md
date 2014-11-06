## Getting a Binary Release

You can either build a release from sources or download a pre-built release.  If you don't plan on developing Kubernetes itself, we suggest a pre-built release.

### Prebuilt Binary Release

Soon, we will have a list of numbered and nightly releases.  Until then, you can download a development release/snapshot from [here](http://storage.googleapis.com/kubernetes-releases-56726/devel/kubernetes.tar.gz).

Unpack this tar file on Linux or OS X, cd to the created `kubernetes/` directory, and then follow the getting started guide for your cloud.

### Building from source

Get the Kubernetes source.  If you are simply building a release from source there is no need to set up a full golang environment as all building happens in a Docker container.

**TODO:** Change this to suggest using a numbered release once we have one with the new build scripts.

Building a release is simple.

```bash
git clone https://github.com/GoogleCloudPlatform/kubernetes.git
cd kubernetes
build/release.sh
```

For more details on the release process see the [`build/` directory](../../build)
