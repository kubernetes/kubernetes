package tarsum

import (
	"archive/tar"
	"bytes"
	"compress/gzip"
	"crypto/md5"
	"crypto/rand"
	"crypto/sha1"
	"crypto/sha256"
	"crypto/sha512"
	"encoding/hex"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"strings"
	"testing"
)

type testLayer struct {
	filename string
	options  *sizedOptions
	jsonfile string
	gzip     bool
	tarsum   string
	version  Version
	hash     THash
}

var testLayers = []testLayer{
	{
		filename: "testdata/46af0962ab5afeb5ce6740d4d91652e69206fc991fd5328c1a94d364ad00e457/layer.tar",
		jsonfile: "testdata/46af0962ab5afeb5ce6740d4d91652e69206fc991fd5328c1a94d364ad00e457/json",
		version:  Version0,
		tarsum:   "tarsum+sha256:4095cc12fa5fdb1ab2760377e1cd0c4ecdd3e61b4f9b82319d96fcea6c9a41c6"},
	{
		filename: "testdata/46af0962ab5afeb5ce6740d4d91652e69206fc991fd5328c1a94d364ad00e457/layer.tar",
		jsonfile: "testdata/46af0962ab5afeb5ce6740d4d91652e69206fc991fd5328c1a94d364ad00e457/json",
		version:  VersionDev,
		tarsum:   "tarsum.dev+sha256:db56e35eec6ce65ba1588c20ba6b1ea23743b59e81fb6b7f358ccbde5580345c"},
	{
		filename: "testdata/46af0962ab5afeb5ce6740d4d91652e69206fc991fd5328c1a94d364ad00e457/layer.tar",
		jsonfile: "testdata/46af0962ab5afeb5ce6740d4d91652e69206fc991fd5328c1a94d364ad00e457/json",
		gzip:     true,
		tarsum:   "tarsum+sha256:4095cc12fa5fdb1ab2760377e1cd0c4ecdd3e61b4f9b82319d96fcea6c9a41c6"},
	{
		// Tests existing version of TarSum when xattrs are present
		filename: "testdata/xattr/layer.tar",
		jsonfile: "testdata/xattr/json",
		version:  Version0,
		tarsum:   "tarsum+sha256:07e304a8dbcb215b37649fde1a699f8aeea47e60815707f1cdf4d55d25ff6ab4"},
	{
		// Tests next version of TarSum when xattrs are present
		filename: "testdata/xattr/layer.tar",
		jsonfile: "testdata/xattr/json",
		version:  VersionDev,
		tarsum:   "tarsum.dev+sha256:6c58917892d77b3b357b0f9ad1e28e1f4ae4de3a8006bd3beb8beda214d8fd16"},
	{
		filename: "testdata/511136ea3c5a64f264b78b5433614aec563103b4d4702f3ba7d4d2698e22c158/layer.tar",
		jsonfile: "testdata/511136ea3c5a64f264b78b5433614aec563103b4d4702f3ba7d4d2698e22c158/json",
		tarsum:   "tarsum+sha256:c66bd5ec9f87b8f4c6135ca37684618f486a3dd1d113b138d0a177bfa39c2571"},
	{
		options: &sizedOptions{1, 1024 * 1024, false, false}, // a 1mb file (in memory)
		tarsum:  "tarsum+sha256:8bf12d7e67c51ee2e8306cba569398b1b9f419969521a12ffb9d8875e8836738"},
	{
		// this tar has two files with the same path
		filename: "testdata/collision/collision-0.tar",
		tarsum:   "tarsum+sha256:08653904a68d3ab5c59e65ef58c49c1581caa3c34744f8d354b3f575ea04424a"},
	{
		// this tar has the same two files (with the same path), but reversed order. ensuring is has different hash than above
		filename: "testdata/collision/collision-1.tar",
		tarsum:   "tarsum+sha256:b51c13fbefe158b5ce420d2b930eef54c5cd55c50a2ee4abdddea8fa9f081e0d"},
	{
		// this tar has newer of collider-0.tar, ensuring is has different hash
		filename: "testdata/collision/collision-2.tar",
		tarsum:   "tarsum+sha256:381547080919bb82691e995508ae20ed33ce0f6948d41cafbeb70ce20c73ee8e"},
	{
		// this tar has newer of collider-1.tar, ensuring is has different hash
		filename: "testdata/collision/collision-3.tar",
		tarsum:   "tarsum+sha256:f886e431c08143164a676805205979cd8fa535dfcef714db5515650eea5a7c0f"},
	{
		options: &sizedOptions{1, 1024 * 1024, false, false}, // a 1mb file (in memory)
		tarsum:  "tarsum+md5:0d7529ec7a8360155b48134b8e599f53",
		hash:    md5THash,
	},
	{
		options: &sizedOptions{1, 1024 * 1024, false, false}, // a 1mb file (in memory)
		tarsum:  "tarsum+sha1:f1fee39c5925807ff75ef1925e7a23be444ba4df",
		hash:    sha1Hash,
	},
	{
		options: &sizedOptions{1, 1024 * 1024, false, false}, // a 1mb file (in memory)
		tarsum:  "tarsum+sha224:6319390c0b061d639085d8748b14cd55f697cf9313805218b21cf61c",
		hash:    sha224Hash,
	},
	{
		options: &sizedOptions{1, 1024 * 1024, false, false}, // a 1mb file (in memory)
		tarsum:  "tarsum+sha384:a578ce3ce29a2ae03b8ed7c26f47d0f75b4fc849557c62454be4b5ffd66ba021e713b48ce71e947b43aab57afd5a7636",
		hash:    sha384Hash,
	},
	{
		options: &sizedOptions{1, 1024 * 1024, false, false}, // a 1mb file (in memory)
		tarsum:  "tarsum+sha512:e9bfb90ca5a4dfc93c46ee061a5cf9837de6d2fdf82544d6460d3147290aecfabf7b5e415b9b6e72db9b8941f149d5d69fb17a394cbfaf2eac523bd9eae21855",
		hash:    sha512Hash,
	},
}

type sizedOptions struct {
	num      int64
	size     int64
	isRand   bool
	realFile bool
}

// make a tar:
// * num is the number of files the tar should have
// * size is the bytes per file
// * isRand is whether the contents of the files should be a random chunk (otherwise it's all zeros)
// * realFile will write to a TempFile, instead of an in memory buffer
func sizedTar(opts sizedOptions) io.Reader {
	var (
		fh  io.ReadWriter
		err error
	)
	if opts.realFile {
		fh, err = ioutil.TempFile("", "tarsum")
		if err != nil {
			return nil
		}
	} else {
		fh = bytes.NewBuffer([]byte{})
	}
	tarW := tar.NewWriter(fh)
	defer tarW.Close()
	for i := int64(0); i < opts.num; i++ {
		err := tarW.WriteHeader(&tar.Header{
			Name: fmt.Sprintf("/testdata%d", i),
			Mode: 0755,
			Uid:  0,
			Gid:  0,
			Size: opts.size,
		})
		if err != nil {
			return nil
		}
		var rBuf []byte
		if opts.isRand {
			rBuf = make([]byte, 8)
			_, err = rand.Read(rBuf)
			if err != nil {
				return nil
			}
		} else {
			rBuf = []byte{0, 0, 0, 0, 0, 0, 0, 0}
		}

		for i := int64(0); i < opts.size/int64(8); i++ {
			tarW.Write(rBuf)
		}
	}
	return fh
}

func emptyTarSum(gzip bool) (TarSum, error) {
	reader, writer := io.Pipe()
	tarWriter := tar.NewWriter(writer)

	// Immediately close tarWriter and write-end of the
	// Pipe in a separate goroutine so we don't block.
	go func() {
		tarWriter.Close()
		writer.Close()
	}()

	return NewTarSum(reader, !gzip, Version0)
}

// Test errors on NewTarsumForLabel
func TestNewTarSumForLabelInvalid(t *testing.T) {
	reader := strings.NewReader("")

	if _, err := NewTarSumForLabel(reader, true, "invalidlabel"); err == nil {
		t.Fatalf("Expected an error, got nothing.")
	}

	if _, err := NewTarSumForLabel(reader, true, "invalid+sha256"); err == nil {
		t.Fatalf("Expected an error, got nothing.")
	}
	if _, err := NewTarSumForLabel(reader, true, "tarsum.v1+invalid"); err == nil {
		t.Fatalf("Expected an error, got nothing.")
	}
}

func TestNewTarSumForLabel(t *testing.T) {

	layer := testLayers[0]

	reader, err := os.Open(layer.filename)
	if err != nil {
		t.Fatal(err)
	}
	defer reader.Close()

	label := strings.Split(layer.tarsum, ":")[0]
	ts, err := NewTarSumForLabel(reader, false, label)
	if err != nil {
		t.Fatal(err)
	}

	// Make sure it actually worked by reading a little bit of it
	nbByteToRead := 8 * 1024
	dBuf := make([]byte, nbByteToRead)
	_, err = ts.Read(dBuf)
	if err != nil {
		t.Errorf("failed to read %vKB from %s: %s", nbByteToRead, layer.filename, err)
	}
}

// TestEmptyTar tests that tarsum does not fail to read an empty tar
// and correctly returns the hex digest of an empty hash.
func TestEmptyTar(t *testing.T) {
	// Test without gzip.
	ts, err := emptyTarSum(false)
	if err != nil {
		t.Fatal(err)
	}

	zeroBlock := make([]byte, 1024)
	buf := new(bytes.Buffer)

	n, err := io.Copy(buf, ts)
	if err != nil {
		t.Fatal(err)
	}

	if n != int64(len(zeroBlock)) || !bytes.Equal(buf.Bytes(), zeroBlock) {
		t.Fatalf("tarSum did not write the correct number of zeroed bytes: %d", n)
	}

	expectedSum := ts.Version().String() + "+sha256:" + hex.EncodeToString(sha256.New().Sum(nil))
	resultSum := ts.Sum(nil)

	if resultSum != expectedSum {
		t.Fatalf("expected [%s] but got [%s]", expectedSum, resultSum)
	}

	// Test with gzip.
	ts, err = emptyTarSum(true)
	if err != nil {
		t.Fatal(err)
	}
	buf.Reset()

	n, err = io.Copy(buf, ts)
	if err != nil {
		t.Fatal(err)
	}

	bufgz := new(bytes.Buffer)
	gz := gzip.NewWriter(bufgz)
	n, err = io.Copy(gz, bytes.NewBuffer(zeroBlock))
	gz.Close()
	gzBytes := bufgz.Bytes()

	if n != int64(len(zeroBlock)) || !bytes.Equal(buf.Bytes(), gzBytes) {
		t.Fatalf("tarSum did not write the correct number of gzipped-zeroed bytes: %d", n)
	}

	resultSum = ts.Sum(nil)

	if resultSum != expectedSum {
		t.Fatalf("expected [%s] but got [%s]", expectedSum, resultSum)
	}

	// Test without ever actually writing anything.
	if ts, err = NewTarSum(bytes.NewReader([]byte{}), true, Version0); err != nil {
		t.Fatal(err)
	}

	resultSum = ts.Sum(nil)

	if resultSum != expectedSum {
		t.Fatalf("expected [%s] but got [%s]", expectedSum, resultSum)
	}
}

var (
	md5THash   = NewTHash("md5", md5.New)
	sha1Hash   = NewTHash("sha1", sha1.New)
	sha224Hash = NewTHash("sha224", sha256.New224)
	sha384Hash = NewTHash("sha384", sha512.New384)
	sha512Hash = NewTHash("sha512", sha512.New)
)

// Test all the build-in read size : buf8K, buf16K, buf32K and more
func TestTarSumsReadSize(t *testing.T) {
	// Test always on the same layer (that is big enough)
	layer := testLayers[0]

	for i := 0; i < 5; i++ {

		reader, err := os.Open(layer.filename)
		if err != nil {
			t.Fatal(err)
		}
		defer reader.Close()

		ts, err := NewTarSum(reader, false, layer.version)
		if err != nil {
			t.Fatal(err)
		}

		// Read and discard bytes so that it populates sums
		nbByteToRead := (i + 1) * 8 * 1024
		dBuf := make([]byte, nbByteToRead)
		_, err = ts.Read(dBuf)
		if err != nil {
			t.Errorf("failed to read %vKB from %s: %s", nbByteToRead, layer.filename, err)
			continue
		}
	}
}

func TestTarSums(t *testing.T) {
	for _, layer := range testLayers {
		var (
			fh  io.Reader
			err error
		)
		if len(layer.filename) > 0 {
			fh, err = os.Open(layer.filename)
			if err != nil {
				t.Errorf("failed to open %s: %s", layer.filename, err)
				continue
			}
		} else if layer.options != nil {
			fh = sizedTar(*layer.options)
		} else {
			// What else is there to test?
			t.Errorf("what to do with %#v", layer)
			continue
		}
		if file, ok := fh.(*os.File); ok {
			defer file.Close()
		}

		var ts TarSum
		if layer.hash == nil {
			//                           double negatives!
			ts, err = NewTarSum(fh, !layer.gzip, layer.version)
		} else {
			ts, err = NewTarSumHash(fh, !layer.gzip, layer.version, layer.hash)
		}
		if err != nil {
			t.Errorf("%q :: %q", err, layer.filename)
			continue
		}

		// Read variable number of bytes to test dynamic buffer
		dBuf := make([]byte, 1)
		_, err = ts.Read(dBuf)
		if err != nil {
			t.Errorf("failed to read 1B from %s: %s", layer.filename, err)
			continue
		}
		dBuf = make([]byte, 16*1024)
		_, err = ts.Read(dBuf)
		if err != nil {
			t.Errorf("failed to read 16KB from %s: %s", layer.filename, err)
			continue
		}

		// Read and discard remaining bytes
		_, err = io.Copy(ioutil.Discard, ts)
		if err != nil {
			t.Errorf("failed to copy from %s: %s", layer.filename, err)
			continue
		}
		var gotSum string
		if len(layer.jsonfile) > 0 {
			jfh, err := os.Open(layer.jsonfile)
			if err != nil {
				t.Errorf("failed to open %s: %s", layer.jsonfile, err)
				continue
			}
			defer jfh.Close()

			buf, err := ioutil.ReadAll(jfh)
			if err != nil {
				t.Errorf("failed to readAll %s: %s", layer.jsonfile, err)
				continue
			}
			gotSum = ts.Sum(buf)
		} else {
			gotSum = ts.Sum(nil)
		}

		if layer.tarsum != gotSum {
			t.Errorf("expecting [%s], but got [%s]", layer.tarsum, gotSum)
		}
		var expectedHashName string
		if layer.hash != nil {
			expectedHashName = layer.hash.Name()
		} else {
			expectedHashName = DefaultTHash.Name()
		}
		if expectedHashName != ts.Hash().Name() {
			t.Errorf("expecting hash [%v], but got [%s]", expectedHashName, ts.Hash().Name())
		}
	}
}

func TestIteration(t *testing.T) {
	headerTests := []struct {
		expectedSum string // TODO(vbatts) it would be nice to get individual sums of each
		version     Version
		hdr         *tar.Header
		data        []byte
	}{
		{
			"tarsum+sha256:626c4a2e9a467d65c33ae81f7f3dedd4de8ccaee72af73223c4bc4718cbc7bbd",
			Version0,
			&tar.Header{
				Name:     "file.txt",
				Size:     0,
				Typeflag: tar.TypeReg,
				Devminor: 0,
				Devmajor: 0,
			},
			[]byte(""),
		},
		{
			"tarsum.dev+sha256:6ffd43a1573a9913325b4918e124ee982a99c0f3cba90fc032a65f5e20bdd465",
			VersionDev,
			&tar.Header{
				Name:     "file.txt",
				Size:     0,
				Typeflag: tar.TypeReg,
				Devminor: 0,
				Devmajor: 0,
			},
			[]byte(""),
		},
		{
			"tarsum.dev+sha256:b38166c059e11fb77bef30bf16fba7584446e80fcc156ff46d47e36c5305d8ef",
			VersionDev,
			&tar.Header{
				Name:     "another.txt",
				Uid:      1000,
				Gid:      1000,
				Uname:    "slartibartfast",
				Gname:    "users",
				Size:     4,
				Typeflag: tar.TypeReg,
				Devminor: 0,
				Devmajor: 0,
			},
			[]byte("test"),
		},
		{
			"tarsum.dev+sha256:4cc2e71ac5d31833ab2be9b4f7842a14ce595ec96a37af4ed08f87bc374228cd",
			VersionDev,
			&tar.Header{
				Name:     "xattrs.txt",
				Uid:      1000,
				Gid:      1000,
				Uname:    "slartibartfast",
				Gname:    "users",
				Size:     4,
				Typeflag: tar.TypeReg,
				Xattrs: map[string]string{
					"user.key1": "value1",
					"user.key2": "value2",
				},
			},
			[]byte("test"),
		},
		{
			"tarsum.dev+sha256:65f4284fa32c0d4112dd93c3637697805866415b570587e4fd266af241503760",
			VersionDev,
			&tar.Header{
				Name:     "xattrs.txt",
				Uid:      1000,
				Gid:      1000,
				Uname:    "slartibartfast",
				Gname:    "users",
				Size:     4,
				Typeflag: tar.TypeReg,
				Xattrs: map[string]string{
					"user.KEY1": "value1", // adding different case to ensure different sum
					"user.key2": "value2",
				},
			},
			[]byte("test"),
		},
		{
			"tarsum+sha256:c12bb6f1303a9ddbf4576c52da74973c00d14c109bcfa76b708d5da1154a07fa",
			Version0,
			&tar.Header{
				Name:     "xattrs.txt",
				Uid:      1000,
				Gid:      1000,
				Uname:    "slartibartfast",
				Gname:    "users",
				Size:     4,
				Typeflag: tar.TypeReg,
				Xattrs: map[string]string{
					"user.NOT": "CALCULATED",
				},
			},
			[]byte("test"),
		},
	}
	for _, htest := range headerTests {
		s, err := renderSumForHeader(htest.version, htest.hdr, htest.data)
		if err != nil {
			t.Fatal(err)
		}

		if s != htest.expectedSum {
			t.Errorf("expected sum: %q, got: %q", htest.expectedSum, s)
		}
	}

}

func renderSumForHeader(v Version, h *tar.Header, data []byte) (string, error) {
	buf := bytes.NewBuffer(nil)
	// first build our test tar
	tw := tar.NewWriter(buf)
	if err := tw.WriteHeader(h); err != nil {
		return "", err
	}
	if _, err := tw.Write(data); err != nil {
		return "", err
	}
	tw.Close()

	ts, err := NewTarSum(buf, true, v)
	if err != nil {
		return "", err
	}
	tr := tar.NewReader(ts)
	for {
		hdr, err := tr.Next()
		if hdr == nil || err == io.EOF {
			// Signals the end of the archive.
			break
		}
		if err != nil {
			return "", err
		}
		if _, err = io.Copy(ioutil.Discard, tr); err != nil {
			return "", err
		}
	}
	return ts.Sum(nil), nil
}

func Benchmark9kTar(b *testing.B) {
	buf := bytes.NewBuffer([]byte{})
	fh, err := os.Open("testdata/46af0962ab5afeb5ce6740d4d91652e69206fc991fd5328c1a94d364ad00e457/layer.tar")
	if err != nil {
		b.Error(err)
		return
	}
	defer fh.Close()

	n, err := io.Copy(buf, fh)
	if err != nil {
		b.Error(err)
		return
	}

	reader := bytes.NewReader(buf.Bytes())

	b.SetBytes(n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reader.Seek(0, 0)
		ts, err := NewTarSum(reader, true, Version0)
		if err != nil {
			b.Error(err)
			return
		}
		io.Copy(ioutil.Discard, ts)
		ts.Sum(nil)
	}
}

func Benchmark9kTarGzip(b *testing.B) {
	buf := bytes.NewBuffer([]byte{})
	fh, err := os.Open("testdata/46af0962ab5afeb5ce6740d4d91652e69206fc991fd5328c1a94d364ad00e457/layer.tar")
	if err != nil {
		b.Error(err)
		return
	}
	defer fh.Close()

	n, err := io.Copy(buf, fh)
	if err != nil {
		b.Error(err)
		return
	}

	reader := bytes.NewReader(buf.Bytes())

	b.SetBytes(n)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		reader.Seek(0, 0)
		ts, err := NewTarSum(reader, false, Version0)
		if err != nil {
			b.Error(err)
			return
		}
		io.Copy(ioutil.Discard, ts)
		ts.Sum(nil)
	}
}

// this is a single big file in the tar archive
func Benchmark1mbSingleFileTar(b *testing.B) {
	benchmarkTar(b, sizedOptions{1, 1024 * 1024, true, true}, false)
}

// this is a single big file in the tar archive
func Benchmark1mbSingleFileTarGzip(b *testing.B) {
	benchmarkTar(b, sizedOptions{1, 1024 * 1024, true, true}, true)
}

// this is 1024 1k files in the tar archive
func Benchmark1kFilesTar(b *testing.B) {
	benchmarkTar(b, sizedOptions{1024, 1024, true, true}, false)
}

// this is 1024 1k files in the tar archive
func Benchmark1kFilesTarGzip(b *testing.B) {
	benchmarkTar(b, sizedOptions{1024, 1024, true, true}, true)
}

func benchmarkTar(b *testing.B, opts sizedOptions, isGzip bool) {
	var fh *os.File
	tarReader := sizedTar(opts)
	if br, ok := tarReader.(*os.File); ok {
		fh = br
	}
	defer os.Remove(fh.Name())
	defer fh.Close()

	b.SetBytes(opts.size * opts.num)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ts, err := NewTarSum(fh, !isGzip, Version0)
		if err != nil {
			b.Error(err)
			return
		}
		io.Copy(ioutil.Discard, ts)
		ts.Sum(nil)
		fh.Seek(0, 0)
	}
}
