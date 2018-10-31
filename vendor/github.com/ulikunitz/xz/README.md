# Package xz

This Go language package supports the reading and writing of xz
compressed streams. It includes also a gxz command for compressing and
decompressing data. The package is completely written in Go and doesn't
have any dependency on any C code.

The package is currently under development. There might be bugs and APIs
are not considered stable. At this time the package cannot compete with
the xz tool regarding compression speed and size. The algorithms there
have been developed over a long time and are highly optimized. However
there are a number of improvements planned and I'm very optimistic about
parallel compression and decompression. Stay tuned!

# Using the API

The following example program shows how to use the API.

    package main

    import (
        "bytes"
        "io"
        "log"
        "os"

        "github.com/ulikunitz/xz"
    )

    func main() {
        const text = "The quick brown fox jumps over the lazy dog.\n"
        var buf bytes.Buffer
        // compress text
        w, err := xz.NewWriter(&buf)
        if err != nil {
            log.Fatalf("xz.NewWriter error %s", err)
        }
        if _, err := io.WriteString(w, text); err != nil {
            log.Fatalf("WriteString error %s", err)
        }
        if err := w.Close(); err != nil {
            log.Fatalf("w.Close error %s", err)
        }
        // decompress buffer and write output to stdout
        r, err := xz.NewReader(&buf)
        if err != nil {
            log.Fatalf("NewReader error %s", err)
        }
        if _, err = io.Copy(os.Stdout, r); err != nil {
            log.Fatalf("io.Copy error %s", err)
        }
    }

# Using the gxz compression tool

The package includes a gxz command line utility for compression and
decompression.

Use following command for installation:

    $ go get github.com/ulikunitz/xz/cmd/gxz

To test it call the following command.

    $ gxz bigfile

After some time a much smaller file bigfile.xz will replace bigfile.
To decompress it use the following command.

    $ gxz -d bigfile.xz

