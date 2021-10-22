# httpr, a Record/Replay Proxy

httpr is an HTTP proxy that records and replays traffic. It is designed
specifically for Google APIs that use HTTP exclusively. These include the Google
Cloud Storage and BigQuery clients, as well as the clients in the
`github.com/google/google-api-*-client` repos.

If you are writing Go code, you should use the `cloud.google.com/go/httpreplay` package, which
is a simpler way to use the proxy.

## Using a Record/Replay Proxy

A record/replay proxy lets you run an "integration" test that accesses a
backend like a Google service and record the interaction. Subsequent runs of the
test can replay the server's responses without actually contacting the server,
turning the integration test into a fast and inexpensive unit test.

## Usage

First, obtain the `httpr` binary. If you have the Go toolchain, you can run `go
get -u cloud.google.com/go/httpreplay/cmd/httpr`. Otherwise, precompiled
binaries for various architectures and operating systems are available from [the
releases page](https://github.com/googleapis/google-cloud-go/releases).

### Recording

1. Start `httpr` in record mode by passing it the `-record` flag with a filename:
   ```
   httpr -record myclient.replay
   ```
   By default, `httpr` will run on port 8080, and open a control port on 8181.
   You can change these with the `-port` and `-control-port` flags.
   You will want to run `httpr` in the background or in another window.
1. In order for `httpr` to record HTTPS traffic, your client must trust it. It
   does so by installing a CA certificate created by `httpr` during the
   recording session. To obtain the certificate in PEM form, GET the URL
   `http://localhost:8181/authority.cer`. (If you changed the control port, use
   it in place of 8181.)  Consult your language to determine
   how to install the certificate. Note that the certificate is different for each run
   of `httpr`.
1. Arrange for your test program to use `httpr` as a proxy. This may be as
   simple as setting the `HTTPS_PROXY` environment variable.
1. Run your test program, using whatever authentication for your Google API
   clients that you wish.
1. Send `httpr` a SIGINT signal (`kill -2`). `httpr` will write
   the replay file, then exit.

### Replaying

1. Start `httpr` in replay mode, in the background or another window:
   ```
   httpr -replay myclient.replay
   ```
1. Install the CA certificate as described above.
1. Have your test program treat `httpr` as a proxy, as described above.
1. Run your test program. Your Google API clients should use no authentication.

## Tips

You must remove all randomness from your interaction while recording,
so that the replay is fully deterministic.

Note that BigQuery clients choose random values for job IDs and insert ID if you
do not supply them. Either supply your own, or seed the client's random number
generator if possible.

## Examples

Examples of running `httpr` can be found in `examples` under this file's directory.


