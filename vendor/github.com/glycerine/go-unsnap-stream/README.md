go-unsnap-stream
================

This is a small golang library for decoding and encoding the snappy *streaming* format, specified here: https://github.com/google/snappy/blob/master/framing_format.txt

Note that the *streaming or framing format* for snappy is different from snappy itself. Think of it as a train of boxcars: the streaming format breaks your data in chunks, applies snappy to each chunk alone, then puts a thin wrapper around the chunk, and sends it along in turn. You can begin decoding before receiving everything. And memory requirements for decoding are sane.

Strangely, though the streaming format was first proposed in Go[1][2], it was never upated, and I could not locate any other library for Go that would handle the streaming/framed snappy format. Hence this implementation of the spec. There is a command line tool[3] that has a C implementation, but this is the only Go implementation that I am aware of. The reference for the framing/streaming spec seems to be the python implementation[4].

For binary compatibility with the python implementation, one could use the C-snappy compressor/decompressor code directly; using github.com/dgryski/go-csnappy. In fact we did this for a while to verify byte-for-byte compatiblity, as the native Go implementation produces slightly different binary compression (still conformant with the standard of course), which made test-diffs harder, and some have complained about it being slower than the C.

However, while the c-snappy was useful for checking compatibility, it introduced dependencies on external C libraries (both the c-snappy library and the C standard library). Our go binary executable that used the go-unsnap-stream library was no longer standalone, and deployment was painful if not impossible if the target had a different C standard library. So we've gone back to using the snappy-go implementation (entirely in Go) for ease of deployment. See the comments at the top of unsnap.go if you wish to use c-snappy instead.

[1] https://groups.google.com/forum/#!msg/snappy-compression/qvLNe2cSH9s/R19oBC-p7g4J

[2] https://codereview.appspot.com/5167058

[3] https://github.com/kubo/snzip

[4] https://pypi.python.org/pypi/python-snappy