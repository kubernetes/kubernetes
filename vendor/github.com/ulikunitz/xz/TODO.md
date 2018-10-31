# TODO list

## Release v0.6

1. Review encoder and check for lzma improvements under xz.
2. Fix binary tree matcher.
3. Compare compression ratio with xz tool using comparable parameters
   and optimize parameters
4. Do some optimizations
    - rename operation action and make it a simple type of size 8
    - make maxMatches, wordSize parameters
    - stop searching after a certain length is found (parameter sweetLen)

## Release v0.7

1. Optimize code
2. Do statistical analysis to get linear presets.
3. Test sync.Pool compatability for xz and lzma Writer and Reader
3. Fuzz optimized code.

## Release v0.8

1. Support parallel go routines for writing and reading xz files.
2. Support a ReaderAt interface for xz files with small block sizes.
3. Improve compatibility between gxz and xz
4. Provide manual page for gxz

## Release v0.9

1. Improve documentation
2. Fuzz again

## Release v1.0

1. Full functioning gxz
2. Add godoc URL to README.md (godoc.org)
3. Resolve all issues.
4. Define release candidates.
5. Public announcement.

## Package lzma

### Release v0.6

- Rewrite Encoder into a simple greedy one-op-at-a-time encoder
  including
    + simple scan at the dictionary head for the same byte
    + use the killer byte (requiring matches to get longer, the first
      test should be the byte that would make the match longer)


## Optimizations

- There may be a lot of false sharing in lzma.State; check whether this
  can be improved by reorganizing the internal structure of it.
- Check whether batching encoding and decoding improves speed.

### DAG optimizations

- Use full buffer to create minimal bit-length above range encoder.
- Might be too slow (see v0.4)

### Different match finders

- hashes with 2, 3 characters additional to 4 characters
- binary trees with 2-7 characters (uint64 as key, use uint32 as
  pointers into a an array)
- rb-trees with 2-7 characters (uint64 as key, use uint32 as pointers
  into an array with bit-steeling for the colors)

## Release Procedure

- execute goch -l for all packages; probably with lower param like 0.5.
- check orthography with gospell
- Write release notes in doc/relnotes.
- Update README.md
- xb copyright . in xz directory to ensure all new files have Copyright
  header
- VERSION=<version> go generate github.com/ulikunitz/xz/... to update
  version files
- Execute test for Linux/amd64, Linux/x86 and Windows/amd64.
- Update TODO.md - write short log entry
- git checkout master && git merge dev
- git tag -a <version>
- git push

## Log

### 2017-06-05

Release v0.5.4 fixes issues #15 of another problem with the padding size
check for the xz block header. I removed the check completely.

### 2017-02-15

Release v0.5.3 fixes issue #12 regarding the decompression of an empty
XZ stream. Many thanks to Tomasz Kłak, who reported the issue.

### 2016-12-02

Release v0.5.2 became necessary to allow the decoding of xz files with
4-byte padding in the block header. Many thanks to Greg, who reported
the issue.

### 2016-07-23 

Release v0.5.1 became necessary to fix problems with 32-bit platforms.
Many thanks to Bruno Brigas, who reported the issue.

### 2016-07-04

Release v0.5 provides improvements to the compressor and provides support for
the decompression of xz files with multiple xz streams.

### 2016-01-31

Another compression rate increase by checking the byte at length of the
best match first, before checking the whole prefix. This makes the
compressor even faster. We have now a large time budget to beat the
compression ratio of the xz tool. For enwik8 we have now over 40 seconds
to reduce the compressed file size for another 7 MiB.

### 2016-01-30

I simplified the encoder. Speed and compression rate increased
dramatically. A high compression rate affects also the decompression
speed. The approach with the buffer and optimizing for operation
compression rate has not been successful. Going for the maximum length
appears to be the best approach.

### 2016-01-28

The release v0.4 is ready. It provides a working xz implementation,
which is rather slow, but works and is interoperable with the xz tool.
It is an important milestone.

### 2016-01-10

I have the first working implementation of an xz reader and writer. I'm
happy about reaching this milestone.

### 2015-12-02

I'm now ready to implement xz because, I have a working LZMA2
implementation. I decided today that v0.4 will use the slow encoder
using the operations buffer to be able to go back, if I intend to do so.

### 2015-10-21

I have restarted the work on the library. While trying to implement
LZMA2, I discovered that I need to resimplify the encoder and decoder
functions. The option approach is too complicated. Using a limited byte
writer and not caring for written bytes at all and not to try to handle
uncompressed data simplifies the LZMA encoder and decoder much.
Processing uncompressed data and handling limits is a feature of the
LZMA2 format not of LZMA.

I learned an interesting method from the LZO format. If the last copy is
too far away they are moving the head one 2 bytes and not 1 byte to
reduce processing times.

### 2015-08-26

I have now reimplemented the lzma package. The code is reasonably fast,
but can still be optimized. The next step is to implement LZMA2 and then
xz.

### 2015-07-05

Created release v0.3. The version is the foundation for a full xz
implementation that is the target of v0.4.

### 2015-06-11

The gflag package has been developed because I couldn't use flag and
pflag for a fully compatible support of gzip's and lzma's options. It
seems to work now quite nicely.

### 2015-06-05

The overflow issue was interesting to research, however Henry S. Warren
Jr. Hacker's Delight book was very helpful as usual and had the issue
explained perfectly. Fefe's information on his website was based on the
C FAQ and quite bad, because it didn't address the issue of -MININT ==
MININT.

### 2015-06-04

It has been a productive day. I improved the interface of lzma.Reader
and lzma.Writer and fixed the error handling.

### 2015-06-01

By computing the bit length of the LZMA operations I was able to
improve the greedy algorithm implementation. By using an 8 MByte buffer
the compression rate was not as good as for xz but already better then
gzip default. 

Compression is currently slow, but this is something we will be able to
improve over time.

### 2015-05-26

Checked the license of ogier/pflag. The binary lzmago binary should
include the license terms for the pflag library.

I added the endorsement clause as used by Google for the Go sources the
LICENSE file.

### 2015-05-22

The package lzb contains now the basic implementation for creating or
reading LZMA byte streams. It allows the support for the implementation
of the DAG-shortest-path algorithm for the compression function.

### 2015-04-23 

Completed yesterday the lzbase classes. I'm a little bit concerned that
using the components may require too much code, but on the other hand
there is a lot of flexibility.

### 2015-04-22

Implemented Reader and Writer during the Bayern game against Porto. The
second half gave me enough time.

### 2015-04-21

While showering today morning I discovered that the design for OpEncoder
and OpDecoder doesn't work, because encoding/decoding might depend on
the current status of the dictionary. This is not exactly the right way
to start the day.

Therefore we need to keep the Reader and Writer design. This time around
we simplify it by ignoring size limits. These can be added by wrappers
around the Reader and Writer interfaces. The Parameters type isn't
needed anymore.

However I will implement a ReaderState and WriterState type to use
static typing to ensure the right State object is combined with the
right lzbase.Reader and lzbase.Writer.

As a start I have implemented ReaderState and WriterState to ensure
that the state for reading is only used by readers and WriterState only
used by Writers. 

### 2015-04-20

Today I implemented the OpDecoder and tested OpEncoder and OpDecoder.

### 2015-04-08

Came up with a new simplified design for lzbase. I implemented already
the type State that replaces OpCodec.

### 2015-04-06

The new lzma package is now fully usable and lzmago is using it now. The
old lzma package has been completely removed.

### 2015-04-05

Implemented lzma.Reader and tested it.

### 2015-04-04

Implemented baseReader by adapting code form lzma.Reader.

### 2015-04-03

The opCodec has been copied yesterday to lzma2. opCodec has a high
number of dependencies on other files in lzma2. Therefore I had to copy
almost all files from lzma.

### 2015-03-31

Removed only a TODO item. 

However in Francesco Campoy's presentation "Go for Javaneros
(Javaïstes?)" is the the idea that using an embedded field E, all the
methods of E will be defined on T. If E is an interface T satisfies E.

https://talks.golang.org/2014/go4java.slide#51

I have never used this, but it seems to be a cool idea.

### 2015-03-30

Finished the type writerDict and wrote a simple test.

### 2015-03-25

I started to implement the writerDict.

### 2015-03-24

After thinking long about the LZMA2 code and several false starts, I
have now a plan to create a self-sufficient lzma2 package that supports
the classic LZMA format as well as LZMA2. The core idea is to support a
baseReader and baseWriter type that support the basic LZMA stream
without any headers. Both types must support the reuse of dictionaries
and the opCodec.

### 2015-01-10

1. Implemented simple lzmago tool
2. Tested tool against large 4.4G file
    - compression worked correctly; tested decompression with lzma
    - decompression hits a full buffer condition
3. Fixed a bug in the compressor and wrote a test for it
4. Executed full cycle for 4.4 GB file; performance can be improved ;-)

### 2015-01-11

- Release v0.2 because of the working LZMA encoder and decoder
