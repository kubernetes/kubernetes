package testsuite

import (
	"bytes"
	"context"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"testing"
	"time"

	"github.com/containerd/containerd/content"
	"github.com/containerd/containerd/namespaces"
	"github.com/containerd/containerd/testutil"
	digest "github.com/opencontainers/go-digest"
	"github.com/pkg/errors"
)

// ContentSuite runs a test suite on the content store given a factory function.
func ContentSuite(t *testing.T, name string, storeFn func(ctx context.Context, root string) (content.Store, func() error, error)) {
	t.Run("Writer", makeTest(t, name, storeFn, checkContentStoreWriter))
	t.Run("UploadStatus", makeTest(t, name, storeFn, checkUploadStatus))
	t.Run("Labels", makeTest(t, name, storeFn, checkLabels))
}

func makeTest(t *testing.T, name string, storeFn func(ctx context.Context, root string) (content.Store, func() error, error), fn func(ctx context.Context, t *testing.T, cs content.Store)) func(t *testing.T) {
	return func(t *testing.T) {
		ctx := namespaces.WithNamespace(context.Background(), name)

		tmpDir, err := ioutil.TempDir("", "content-suite-"+name+"-")
		if err != nil {
			t.Fatal(err)
		}
		defer os.RemoveAll(tmpDir)

		cs, cleanup, err := storeFn(ctx, tmpDir)
		if err != nil {
			t.Fatal(err)
		}
		defer func() {
			if err := cleanup(); err != nil && !t.Failed() {
				t.Fatalf("Cleanup failed: %+v", err)
			}
		}()

		defer testutil.DumpDir(t, tmpDir)
		fn(ctx, t, cs)
	}
}

func checkContentStoreWriter(ctx context.Context, t *testing.T, cs content.Store) {
	c1, d1 := createContent(256, 1)
	w1, err := cs.Writer(ctx, "c1", 0, "")
	if err != nil {
		t.Fatal(err)
	}
	defer w1.Close()

	c2, d2 := createContent(256, 2)
	w2, err := cs.Writer(ctx, "c2", int64(len(c2)), "")
	if err != nil {
		t.Fatal(err)
	}
	defer w2.Close()

	c3, d3 := createContent(256, 3)
	w3, err := cs.Writer(ctx, "c3", 0, d3)
	if err != nil {
		t.Fatal(err)
	}
	defer w3.Close()

	c4, d4 := createContent(256, 4)
	w4, err := cs.Writer(ctx, "c4", int64(len(c4)), d4)
	if err != nil {
		t.Fatal(err)
	}
	defer w4.Close()

	smallbuf := make([]byte, 32)
	for _, s := range []struct {
		content []byte
		digest  digest.Digest
		writer  content.Writer
	}{
		{
			content: c1,
			digest:  d1,
			writer:  w1,
		},
		{
			content: c2,
			digest:  d2,
			writer:  w2,
		},
		{
			content: c3,
			digest:  d3,
			writer:  w3,
		},
		{
			content: c4,
			digest:  d4,
			writer:  w4,
		},
	} {
		n, err := io.CopyBuffer(s.writer, bytes.NewReader(s.content), smallbuf)
		if err != nil {
			t.Fatal(err)
		}

		if n != int64(len(s.content)) {
			t.Fatalf("Unexpected copy length %d, expected %d", n, len(s.content))
		}

		preCommit := time.Now()
		if err := s.writer.Commit(ctx, 0, ""); err != nil {
			t.Fatal(err)
		}
		postCommit := time.Now()

		if s.writer.Digest() != s.digest {
			t.Fatalf("Unexpected commit digest %s, expected %s", s.writer.Digest(), s.digest)
		}

		info := content.Info{
			Digest: s.digest,
			Size:   int64(len(s.content)),
		}
		if err := checkInfo(ctx, cs, s.digest, info, preCommit, postCommit, preCommit, postCommit); err != nil {
			t.Fatalf("Check info failed: %+v", err)
		}
	}
}

func checkUploadStatus(ctx context.Context, t *testing.T, cs content.Store) {
	c1, d1 := createContent(256, 1)

	preStart := time.Now()
	w1, err := cs.Writer(ctx, "c1", 256, d1)
	if err != nil {
		t.Fatal(err)
	}
	defer w1.Close()
	postStart := time.Now()

	d := digest.FromBytes([]byte{})

	expected := content.Status{
		Ref:      "c1",
		Total:    256,
		Expected: d1,
	}
	preUpdate := preStart
	postUpdate := postStart

	if err := checkStatus(w1, expected, d, preStart, postStart, preUpdate, postUpdate); err != nil {
		t.Fatalf("Status check failed: %+v", err)
	}

	// Write first 64 bytes
	preUpdate = time.Now()
	if _, err := w1.Write(c1[:64]); err != nil {
		t.Fatalf("Failed to write: %+v", err)
	}
	postUpdate = time.Now()
	expected.Offset = 64
	d = digest.FromBytes(c1[:64])
	if err := checkStatus(w1, expected, d, preStart, postStart, preUpdate, postUpdate); err != nil {
		t.Fatalf("Status check failed: %+v", err)
	}

	// Write next 128 bytes
	preUpdate = time.Now()
	if _, err := w1.Write(c1[64:192]); err != nil {
		t.Fatalf("Failed to write: %+v", err)
	}
	postUpdate = time.Now()
	expected.Offset = 192
	d = digest.FromBytes(c1[:192])
	if err := checkStatus(w1, expected, d, preStart, postStart, preUpdate, postUpdate); err != nil {
		t.Fatalf("Status check failed: %+v", err)
	}

	// Write last 64 bytes
	preUpdate = time.Now()
	if _, err := w1.Write(c1[192:]); err != nil {
		t.Fatalf("Failed to write: %+v", err)
	}
	postUpdate = time.Now()
	expected.Offset = 256
	if err := checkStatus(w1, expected, d1, preStart, postStart, preUpdate, postUpdate); err != nil {
		t.Fatalf("Status check failed: %+v", err)
	}

	preCommit := time.Now()
	if err := w1.Commit(ctx, 0, ""); err != nil {
		t.Fatalf("Commit failed: %+v", err)
	}
	postCommit := time.Now()

	info := content.Info{
		Digest: d1,
		Size:   256,
	}

	if err := checkInfo(ctx, cs, d1, info, preCommit, postCommit, preCommit, postCommit); err != nil {
		t.Fatalf("Check info failed: %+v", err)
	}
}

func checkLabels(ctx context.Context, t *testing.T, cs content.Store) {
	c1, d1 := createContent(256, 1)

	w1, err := cs.Writer(ctx, "c1", 256, d1)
	if err != nil {
		t.Fatal(err)
	}
	defer w1.Close()

	if _, err := w1.Write(c1); err != nil {
		t.Fatalf("Failed to write: %+v", err)
	}

	labels := map[string]string{
		"k1": "v1",
		"k2": "v2",
	}

	preCommit := time.Now()
	if err := w1.Commit(ctx, 0, "", content.WithLabels(labels)); err != nil {
		t.Fatalf("Commit failed: %+v", err)
	}
	postCommit := time.Now()

	info := content.Info{
		Digest: d1,
		Size:   256,
		Labels: labels,
	}

	if err := checkInfo(ctx, cs, d1, info, preCommit, postCommit, preCommit, postCommit); err != nil {
		t.Fatalf("Check info failed: %+v", err)
	}

	labels["k1"] = "newvalue"
	delete(labels, "k2")
	labels["k3"] = "v3"

	info.Labels = labels
	preUpdate := time.Now()
	if _, err := cs.Update(ctx, info); err != nil {
		t.Fatalf("Update failed: %+v", err)
	}
	postUpdate := time.Now()

	if err := checkInfo(ctx, cs, d1, info, preCommit, postCommit, preUpdate, postUpdate); err != nil {
		t.Fatalf("Check info failed: %+v", err)
	}

	info.Labels = map[string]string{
		"k1": "v1",
	}
	preUpdate = time.Now()
	if _, err := cs.Update(ctx, info, "labels.k3", "labels.k1"); err != nil {
		t.Fatalf("Update failed: %+v", err)
	}
	postUpdate = time.Now()

	if err := checkInfo(ctx, cs, d1, info, preCommit, postCommit, preUpdate, postUpdate); err != nil {
		t.Fatalf("Check info failed: %+v", err)
	}

}

func checkStatus(w content.Writer, expected content.Status, d digest.Digest, preStart, postStart, preUpdate, postUpdate time.Time) error {
	st, err := w.Status()
	if err != nil {
		return errors.Wrap(err, "failed to get status")
	}

	wd := w.Digest()
	if wd != d {
		return errors.Errorf("unexpected digest %v, expected %v", wd, d)
	}

	if st.Ref != expected.Ref {
		return errors.Errorf("unexpected ref %q, expected %q", st.Ref, expected.Ref)
	}

	if st.Offset != expected.Offset {
		return errors.Errorf("unexpected offset %d, expected %d", st.Offset, expected.Offset)
	}

	if st.Total != expected.Total {
		return errors.Errorf("unexpected total %d, expected %d", st.Total, expected.Total)
	}

	// TODO: Add this test once all implementations guarantee this value is held
	//if st.Expected != expected.Expected {
	//	return errors.Errorf("unexpected \"expected digest\" %q, expected %q", st.Expected, expected.Expected)
	//}

	if st.StartedAt.After(postStart) || st.StartedAt.Before(preStart) {
		return errors.Errorf("unexpected started at time %s, expected between %s and %s", st.StartedAt, preStart, postStart)
	}
	if st.UpdatedAt.After(postUpdate) || st.UpdatedAt.Before(preUpdate) {
		return errors.Errorf("unexpected updated at time %s, expected between %s and %s", st.UpdatedAt, preUpdate, postUpdate)
	}

	return nil
}

func checkInfo(ctx context.Context, cs content.Store, d digest.Digest, expected content.Info, c1, c2, u1, u2 time.Time) error {
	info, err := cs.Info(ctx, d)
	if err != nil {
		return errors.Wrap(err, "failed to get info")
	}

	if info.Digest != d {
		return errors.Errorf("unexpected info digest %s, expected %s", info.Digest, d)
	}

	if info.Size != expected.Size {
		return errors.Errorf("unexpected info size %d, expected %d", info.Size, expected.Size)
	}

	if info.CreatedAt.After(c2) || info.CreatedAt.Before(c1) {
		return errors.Errorf("unexpected created at time %s, expected between %s and %s", info.CreatedAt, c1, c2)
	}
	if info.UpdatedAt.After(u2) || info.UpdatedAt.Before(u1) {
		return errors.Errorf("unexpected updated at time %s, expected between %s and %s", info.UpdatedAt, u1, u2)
	}

	if len(info.Labels) != len(expected.Labels) {
		return errors.Errorf("mismatched number of labels\ngot:\n%#v\nexpected:\n%#v", info.Labels, expected.Labels)
	}

	for k, v := range expected.Labels {
		actual := info.Labels[k]
		if v != actual {
			return errors.Errorf("unexpected value for label %q: %q, expected %q", k, actual, v)
		}
	}

	return nil
}

func createContent(size, seed int64) ([]byte, digest.Digest) {
	b, err := ioutil.ReadAll(io.LimitReader(rand.New(rand.NewSource(seed)), size))
	if err != nil {
		panic(err)
	}
	return b, digest.FromBytes(b)
}
