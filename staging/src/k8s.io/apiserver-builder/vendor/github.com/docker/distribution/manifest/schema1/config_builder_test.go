package schema1

import (
	"bytes"
	"compress/gzip"
	"io"
	"reflect"
	"testing"

	"github.com/docker/distribution"
	"github.com/docker/distribution/context"
	"github.com/docker/distribution/digest"
	"github.com/docker/distribution/reference"
	"github.com/docker/libtrust"
)

type mockBlobService struct {
	descriptors map[digest.Digest]distribution.Descriptor
}

func (bs *mockBlobService) Stat(ctx context.Context, dgst digest.Digest) (distribution.Descriptor, error) {
	if descriptor, ok := bs.descriptors[dgst]; ok {
		return descriptor, nil
	}
	return distribution.Descriptor{}, distribution.ErrBlobUnknown
}

func (bs *mockBlobService) Get(ctx context.Context, dgst digest.Digest) ([]byte, error) {
	panic("not implemented")
}

func (bs *mockBlobService) Open(ctx context.Context, dgst digest.Digest) (distribution.ReadSeekCloser, error) {
	panic("not implemented")
}

func (bs *mockBlobService) Put(ctx context.Context, mediaType string, p []byte) (distribution.Descriptor, error) {
	d := distribution.Descriptor{
		Digest:    digest.FromBytes(p),
		Size:      int64(len(p)),
		MediaType: mediaType,
	}
	bs.descriptors[d.Digest] = d
	return d, nil
}

func (bs *mockBlobService) Create(ctx context.Context, options ...distribution.BlobCreateOption) (distribution.BlobWriter, error) {
	panic("not implemented")
}

func (bs *mockBlobService) Resume(ctx context.Context, id string) (distribution.BlobWriter, error) {
	panic("not implemented")
}

func TestEmptyTar(t *testing.T) {
	// Confirm that gzippedEmptyTar expands to 1024 NULL bytes.
	var decompressed [2048]byte
	gzipReader, err := gzip.NewReader(bytes.NewReader(gzippedEmptyTar))
	if err != nil {
		t.Fatalf("NewReader returned error: %v", err)
	}
	n, err := gzipReader.Read(decompressed[:])
	if n != 1024 {
		t.Fatalf("read returned %d bytes; expected 1024", n)
	}
	n, err = gzipReader.Read(decompressed[1024:])
	if n != 0 {
		t.Fatalf("read returned %d bytes; expected 0", n)
	}
	if err != io.EOF {
		t.Fatal("read did not return io.EOF")
	}
	gzipReader.Close()
	for _, b := range decompressed[:1024] {
		if b != 0 {
			t.Fatal("nonzero byte in decompressed tar")
		}
	}

	// Confirm that digestSHA256EmptyTar is the digest of gzippedEmptyTar.
	dgst := digest.FromBytes(gzippedEmptyTar)
	if dgst != digestSHA256GzippedEmptyTar {
		t.Fatalf("digest mismatch for empty tar: expected %s got %s", digestSHA256GzippedEmptyTar, dgst)
	}
}

func TestConfigBuilder(t *testing.T) {
	imgJSON := `{
    "architecture": "amd64",
    "config": {
        "AttachStderr": false,
        "AttachStdin": false,
        "AttachStdout": false,
        "Cmd": [
            "/bin/sh",
            "-c",
            "echo hi"
        ],
        "Domainname": "",
        "Entrypoint": null,
        "Env": [
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "derived=true",
            "asdf=true"
        ],
        "Hostname": "23304fc829f9",
        "Image": "sha256:4ab15c48b859c2920dd5224f92aabcd39a52794c5b3cf088fb3bbb438756c246",
        "Labels": {},
        "OnBuild": [],
        "OpenStdin": false,
        "StdinOnce": false,
        "Tty": false,
        "User": "",
        "Volumes": null,
        "WorkingDir": ""
    },
    "container": "e91032eb0403a61bfe085ff5a5a48e3659e5a6deae9f4d678daa2ae399d5a001",
    "container_config": {
        "AttachStderr": false,
        "AttachStdin": false,
        "AttachStdout": false,
        "Cmd": [
            "/bin/sh",
            "-c",
            "#(nop) CMD [\"/bin/sh\" \"-c\" \"echo hi\"]"
        ],
        "Domainname": "",
        "Entrypoint": null,
        "Env": [
            "PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "derived=true",
            "asdf=true"
        ],
        "Hostname": "23304fc829f9",
        "Image": "sha256:4ab15c48b859c2920dd5224f92aabcd39a52794c5b3cf088fb3bbb438756c246",
        "Labels": {},
        "OnBuild": [],
        "OpenStdin": false,
        "StdinOnce": false,
        "Tty": false,
        "User": "",
        "Volumes": null,
        "WorkingDir": ""
    },
    "created": "2015-11-04T23:06:32.365666163Z",
    "docker_version": "1.9.0-dev",
    "history": [
        {
            "created": "2015-10-31T22:22:54.690851953Z",
            "created_by": "/bin/sh -c #(nop) ADD file:a3bc1e842b69636f9df5256c49c5374fb4eef1e281fe3f282c65fb853ee171c5 in /"
        },
        {
            "created": "2015-10-31T22:22:55.613815829Z",
            "created_by": "/bin/sh -c #(nop) CMD [\"sh\"]"
        },
        {
            "created": "2015-11-04T23:06:30.934316144Z",
            "created_by": "/bin/sh -c #(nop) ENV derived=true",
            "empty_layer": true
        },
        {
            "created": "2015-11-04T23:06:31.192097572Z",
            "created_by": "/bin/sh -c #(nop) ENV asdf=true",
            "empty_layer": true
        },
        {
            "created": "2015-11-04T23:06:32.083868454Z",
            "created_by": "/bin/sh -c dd if=/dev/zero of=/file bs=1024 count=1024"
        },
        {
            "created": "2015-11-04T23:06:32.365666163Z",
            "created_by": "/bin/sh -c #(nop) CMD [\"/bin/sh\" \"-c\" \"echo hi\"]",
            "empty_layer": true
        }
    ],
    "os": "linux",
    "rootfs": {
        "diff_ids": [
            "sha256:c6f988f4874bb0add23a778f753c65efe992244e148a1d2ec2a8b664fb66bbd1",
            "sha256:5f70bf18a086007016e948b04aed3b82103a36bea41755b6cddfaf10ace3c6ef",
            "sha256:13f53e08df5a220ab6d13c58b2bf83a59cbdc2e04d0a3f041ddf4b0ba4112d49"
        ],
        "type": "layers"
    }
}`

	descriptors := []distribution.Descriptor{
		{Digest: digest.Digest("sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4")},
		{Digest: digest.Digest("sha256:86e0e091d0da6bde2456dbb48306f3956bbeb2eae1b5b9a43045843f69fe4aaa")},
		{Digest: digest.Digest("sha256:b4ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4")},
	}

	pk, err := libtrust.GenerateECP256PrivateKey()
	if err != nil {
		t.Fatalf("could not generate key for testing: %v", err)
	}

	bs := &mockBlobService{descriptors: make(map[digest.Digest]distribution.Descriptor)}

	ref, err := reference.ParseNamed("testrepo:testtag")
	if err != nil {
		t.Fatalf("could not parse reference: %v", err)
	}

	builder := NewConfigManifestBuilder(bs, pk, ref, []byte(imgJSON))

	for _, d := range descriptors {
		if err := builder.AppendReference(d); err != nil {
			t.Fatalf("AppendReference returned error: %v", err)
		}
	}

	signed, err := builder.Build(context.Background())
	if err != nil {
		t.Fatalf("Build returned error: %v", err)
	}

	// Check that the gzipped empty layer tar was put in the blob store
	_, err = bs.Stat(context.Background(), digestSHA256GzippedEmptyTar)
	if err != nil {
		t.Fatal("gzipped empty tar was not put in the blob store")
	}

	manifest := signed.(*SignedManifest).Manifest

	if manifest.Versioned.SchemaVersion != 1 {
		t.Fatal("SchemaVersion != 1")
	}
	if manifest.Name != "testrepo" {
		t.Fatal("incorrect name in manifest")
	}
	if manifest.Tag != "testtag" {
		t.Fatal("incorrect tag in manifest")
	}
	if manifest.Architecture != "amd64" {
		t.Fatal("incorrect arch in manifest")
	}

	expectedFSLayers := []FSLayer{
		{BlobSum: digest.Digest("sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4")},
		{BlobSum: digest.Digest("sha256:b4ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4")},
		{BlobSum: digest.Digest("sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4")},
		{BlobSum: digest.Digest("sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4")},
		{BlobSum: digest.Digest("sha256:86e0e091d0da6bde2456dbb48306f3956bbeb2eae1b5b9a43045843f69fe4aaa")},
		{BlobSum: digest.Digest("sha256:a3ed95caeb02ffe68cdd9fd84406680ae93d633cb16422d00e8a7c22955b46d4")},
	}

	if len(manifest.FSLayers) != len(expectedFSLayers) {
		t.Fatalf("wrong number of FSLayers: %d", len(manifest.FSLayers))
	}
	if !reflect.DeepEqual(manifest.FSLayers, expectedFSLayers) {
		t.Fatal("wrong FSLayers list")
	}

	expectedV1Compatibility := []string{
		`{"architecture":"amd64","config":{"AttachStderr":false,"AttachStdin":false,"AttachStdout":false,"Cmd":["/bin/sh","-c","echo hi"],"Domainname":"","Entrypoint":null,"Env":["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin","derived=true","asdf=true"],"Hostname":"23304fc829f9","Image":"sha256:4ab15c48b859c2920dd5224f92aabcd39a52794c5b3cf088fb3bbb438756c246","Labels":{},"OnBuild":[],"OpenStdin":false,"StdinOnce":false,"Tty":false,"User":"","Volumes":null,"WorkingDir":""},"container":"e91032eb0403a61bfe085ff5a5a48e3659e5a6deae9f4d678daa2ae399d5a001","container_config":{"AttachStderr":false,"AttachStdin":false,"AttachStdout":false,"Cmd":["/bin/sh","-c","#(nop) CMD [\"/bin/sh\" \"-c\" \"echo hi\"]"],"Domainname":"","Entrypoint":null,"Env":["PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin","derived=true","asdf=true"],"Hostname":"23304fc829f9","Image":"sha256:4ab15c48b859c2920dd5224f92aabcd39a52794c5b3cf088fb3bbb438756c246","Labels":{},"OnBuild":[],"OpenStdin":false,"StdinOnce":false,"Tty":false,"User":"","Volumes":null,"WorkingDir":""},"created":"2015-11-04T23:06:32.365666163Z","docker_version":"1.9.0-dev","id":"0850bfdeb7b060b1004a09099846c2f023a3f2ecbf33f56b4774384b00ce0323","os":"linux","parent":"74cf9c92699240efdba1903c2748ef57105d5bedc588084c4e88f3bb1c3ef0b0","throwaway":true}`,
		`{"id":"74cf9c92699240efdba1903c2748ef57105d5bedc588084c4e88f3bb1c3ef0b0","parent":"178be37afc7c49e951abd75525dbe0871b62ad49402f037164ee6314f754599d","created":"2015-11-04T23:06:32.083868454Z","container_config":{"Cmd":["/bin/sh -c dd if=/dev/zero of=/file bs=1024 count=1024"]}}`,
		`{"id":"178be37afc7c49e951abd75525dbe0871b62ad49402f037164ee6314f754599d","parent":"b449305a55a283538c4574856a8b701f2a3d5ec08ef8aec47f385f20339a4866","created":"2015-11-04T23:06:31.192097572Z","container_config":{"Cmd":["/bin/sh -c #(nop) ENV asdf=true"]},"throwaway":true}`,
		`{"id":"b449305a55a283538c4574856a8b701f2a3d5ec08ef8aec47f385f20339a4866","parent":"9e3447ca24cb96d86ebd5960cb34d1299b07e0a0e03801d90b9969a2c187dd6e","created":"2015-11-04T23:06:30.934316144Z","container_config":{"Cmd":["/bin/sh -c #(nop) ENV derived=true"]},"throwaway":true}`,
		`{"id":"9e3447ca24cb96d86ebd5960cb34d1299b07e0a0e03801d90b9969a2c187dd6e","parent":"3690474eb5b4b26fdfbd89c6e159e8cc376ca76ef48032a30fa6aafd56337880","created":"2015-10-31T22:22:55.613815829Z","container_config":{"Cmd":["/bin/sh -c #(nop) CMD [\"sh\"]"]}}`,
		`{"id":"3690474eb5b4b26fdfbd89c6e159e8cc376ca76ef48032a30fa6aafd56337880","created":"2015-10-31T22:22:54.690851953Z","container_config":{"Cmd":["/bin/sh -c #(nop) ADD file:a3bc1e842b69636f9df5256c49c5374fb4eef1e281fe3f282c65fb853ee171c5 in /"]}}`,
	}

	if len(manifest.History) != len(expectedV1Compatibility) {
		t.Fatalf("wrong number of history entries: %d", len(manifest.History))
	}
	for i := range expectedV1Compatibility {
		if manifest.History[i].V1Compatibility != expectedV1Compatibility[i] {
			t.Errorf("wrong V1Compatibility %d. expected:\n%s\ngot:\n%s", i, expectedV1Compatibility[i], manifest.History[i].V1Compatibility)
		}
	}
}
