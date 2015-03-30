## Getting a Binary Release

You can either build a release from sources or download a pre-built release.  If you don't plan on developing LMKTFY itself, we suggest a pre-built release.

### Prebuilt Binary Release

The list of binary releases is available for download from the [GitHub LMKTFY repo release page](https://github.com/GoogleCloudPlatform/lmktfy/releases).

Download the latest release and unpack this tar file on Linux or OS X, cd to the created `lmktfy/` directory, and then follow the getting started guide for your cloud.

### Building from source

Get the LMKTFY source.  If you are simply building a release from source there is no need to set up a full golang environment as all building happens in a Docker container.

Building a release is simple.

```bash
git clone https://github.com/GoogleCloudPlatform/lmktfy.git
cd lmktfy
make release
```

For more details on the release process see the [`build/` directory](../../build)
