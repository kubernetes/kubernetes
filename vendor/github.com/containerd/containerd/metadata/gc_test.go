package metadata

import (
	"context"
	"io"
	"io/ioutil"
	"math/rand"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/gc"
	"github.com/containerd/containerd/metadata/boltutil"
	digest "github.com/opencontainers/go-digest"
)

func TestGCRoots(t *testing.T) {
	db, cleanup, err := newDatabase()
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	alters := []alterFunc{
		addImage("ns1", "image1", dgst(1), nil),
		addImage("ns1", "image2", dgst(2), labelmap(string(labelGCSnapRef)+"overlay", "sn2")),
		addContent("ns1", dgst(1), nil),
		addContent("ns1", dgst(2), nil),
		addContent("ns1", dgst(3), nil),
		addContent("ns2", dgst(1), nil),
		addContent("ns2", dgst(2), labelmap(string(labelGCRoot), "always")),
		addSnapshot("ns1", "overlay", "sn1", "", nil),
		addSnapshot("ns1", "overlay", "sn2", "", nil),
		addSnapshot("ns1", "overlay", "sn3", "", labelmap(string(labelGCRoot), "always")),
		addLeaseSnapshot("ns2", "l1", "overlay", "sn5"),
		addLeaseSnapshot("ns2", "l2", "overlay", "sn6"),
		addLeaseContent("ns2", "l1", dgst(4)),
		addLeaseContent("ns2", "l2", dgst(5)),
	}

	expected := []gc.Node{
		gcnode(ResourceContent, "ns1", dgst(1).String()),
		gcnode(ResourceContent, "ns1", dgst(2).String()),
		gcnode(ResourceContent, "ns2", dgst(2).String()),
		gcnode(ResourceContent, "ns2", dgst(4).String()),
		gcnode(ResourceContent, "ns2", dgst(5).String()),
		gcnode(ResourceSnapshot, "ns1", "overlay/sn2"),
		gcnode(ResourceSnapshot, "ns1", "overlay/sn3"),
		gcnode(ResourceSnapshot, "ns2", "overlay/sn5"),
		gcnode(ResourceSnapshot, "ns2", "overlay/sn6"),
	}

	if err := db.Update(func(tx *bolt.Tx) error {
		v1bkt, err := tx.CreateBucketIfNotExists(bucketKeyVersion)
		if err != nil {
			return err
		}
		for _, alter := range alters {
			if err := alter(v1bkt); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatalf("Update failed: %+v", err)
	}

	ctx := context.Background()

	checkNodeC(ctx, t, db, expected, func(ctx context.Context, tx *bolt.Tx, nc chan<- gc.Node) error {
		return scanRoots(ctx, tx, nc)
	})
}

func TestGCRemove(t *testing.T) {
	db, cleanup, err := newDatabase()
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	alters := []alterFunc{
		addImage("ns1", "image1", dgst(1), nil),
		addImage("ns1", "image2", dgst(2), labelmap(string(labelGCSnapRef)+"overlay", "sn2")),
		addContent("ns1", dgst(1), nil),
		addContent("ns1", dgst(2), nil),
		addContent("ns1", dgst(3), nil),
		addContent("ns2", dgst(1), nil),
		addContent("ns2", dgst(2), labelmap(string(labelGCRoot), "always")),
		addSnapshot("ns1", "overlay", "sn1", "", nil),
		addSnapshot("ns1", "overlay", "sn2", "", nil),
		addSnapshot("ns1", "overlay", "sn3", "", labelmap(string(labelGCRoot), "always")),
		addSnapshot("ns2", "overlay", "sn1", "", nil),
	}

	all := []gc.Node{
		gcnode(ResourceContent, "ns1", dgst(1).String()),
		gcnode(ResourceContent, "ns1", dgst(2).String()),
		gcnode(ResourceContent, "ns1", dgst(3).String()),
		gcnode(ResourceContent, "ns2", dgst(1).String()),
		gcnode(ResourceContent, "ns2", dgst(2).String()),
		gcnode(ResourceSnapshot, "ns1", "overlay/sn1"),
		gcnode(ResourceSnapshot, "ns1", "overlay/sn2"),
		gcnode(ResourceSnapshot, "ns1", "overlay/sn3"),
		gcnode(ResourceSnapshot, "ns2", "overlay/sn1"),
	}

	var deleted, remaining []gc.Node
	for i, n := range all {
		if i%2 == 0 {
			deleted = append(deleted, n)
		} else {
			remaining = append(remaining, n)
		}
	}

	if err := db.Update(func(tx *bolt.Tx) error {
		v1bkt, err := tx.CreateBucketIfNotExists(bucketKeyVersion)
		if err != nil {
			return err
		}
		for _, alter := range alters {
			if err := alter(v1bkt); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatalf("Update failed: %+v", err)
	}

	ctx := context.Background()

	checkNodes(ctx, t, db, all, func(ctx context.Context, tx *bolt.Tx, fn func(context.Context, gc.Node) error) error {
		return scanAll(ctx, tx, fn)
	})
	if t.Failed() {
		t.Fatal("Scan all failed")
	}

	if err := db.Update(func(tx *bolt.Tx) error {
		for _, n := range deleted {
			if err := remove(ctx, tx, n); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatalf("Update failed: %+v", err)
	}

	checkNodes(ctx, t, db, remaining, func(ctx context.Context, tx *bolt.Tx, fn func(context.Context, gc.Node) error) error {
		return scanAll(ctx, tx, fn)
	})
}

func TestGCRefs(t *testing.T) {
	db, cleanup, err := newDatabase()
	if err != nil {
		t.Fatal(err)
	}
	defer cleanup()

	alters := []alterFunc{
		addContent("ns1", dgst(1), nil),
		addContent("ns1", dgst(2), nil),
		addContent("ns1", dgst(3), nil),
		addContent("ns1", dgst(4), labelmap(string(labelGCContentRef), dgst(1).String())),
		addContent("ns1", dgst(5), labelmap(string(labelGCContentRef)+".anything-1", dgst(2).String(), string(labelGCContentRef)+".anything-2", dgst(3).String())),
		addContent("ns1", dgst(6), labelmap(string(labelGCContentRef)+"bad", dgst(1).String())),
		addContent("ns2", dgst(1), nil),
		addContent("ns2", dgst(2), nil),
		addSnapshot("ns1", "overlay", "sn1", "", nil),
		addSnapshot("ns1", "overlay", "sn2", "sn1", nil),
		addSnapshot("ns1", "overlay", "sn3", "sn2", nil),
		addSnapshot("ns1", "overlay", "sn4", "", labelmap(string(labelGCSnapRef)+"btrfs", "sn1", string(labelGCSnapRef)+"overlay", "sn1")),
		addSnapshot("ns1", "btrfs", "sn1", "", nil),
		addSnapshot("ns2", "overlay", "sn1", "", nil),
		addSnapshot("ns2", "overlay", "sn2", "sn1", nil),
	}

	refs := map[gc.Node][]gc.Node{
		gcnode(ResourceContent, "ns1", dgst(1).String()): nil,
		gcnode(ResourceContent, "ns1", dgst(2).String()): nil,
		gcnode(ResourceContent, "ns1", dgst(3).String()): nil,
		gcnode(ResourceContent, "ns1", dgst(4).String()): {
			gcnode(ResourceContent, "ns1", dgst(1).String()),
		},
		gcnode(ResourceContent, "ns1", dgst(5).String()): {
			gcnode(ResourceContent, "ns1", dgst(2).String()),
			gcnode(ResourceContent, "ns1", dgst(3).String()),
		},
		gcnode(ResourceContent, "ns1", dgst(6).String()): nil,
		gcnode(ResourceContent, "ns2", dgst(1).String()): nil,
		gcnode(ResourceContent, "ns2", dgst(2).String()): nil,
		gcnode(ResourceSnapshot, "ns1", "overlay/sn1"):   nil,
		gcnode(ResourceSnapshot, "ns1", "overlay/sn2"): {
			gcnode(ResourceSnapshot, "ns1", "overlay/sn1"),
		},
		gcnode(ResourceSnapshot, "ns1", "overlay/sn3"): {
			gcnode(ResourceSnapshot, "ns1", "overlay/sn2"),
		},
		gcnode(ResourceSnapshot, "ns1", "overlay/sn4"): {
			gcnode(ResourceSnapshot, "ns1", "btrfs/sn1"),
			gcnode(ResourceSnapshot, "ns1", "overlay/sn1"),
		},
		gcnode(ResourceSnapshot, "ns1", "btrfs/sn1"):   nil,
		gcnode(ResourceSnapshot, "ns2", "overlay/sn1"): nil,
		gcnode(ResourceSnapshot, "ns2", "overlay/sn2"): {
			gcnode(ResourceSnapshot, "ns2", "overlay/sn1"),
		},
	}

	if err := db.Update(func(tx *bolt.Tx) error {
		v1bkt, err := tx.CreateBucketIfNotExists(bucketKeyVersion)
		if err != nil {
			return err
		}
		for _, alter := range alters {
			if err := alter(v1bkt); err != nil {
				return err
			}
		}
		return nil
	}); err != nil {
		t.Fatalf("Update failed: %+v", err)
	}

	ctx := context.Background()

	for n, nodes := range refs {
		checkNodeC(ctx, t, db, nodes, func(ctx context.Context, tx *bolt.Tx, nc chan<- gc.Node) error {
			return references(ctx, tx, n, func(n gc.Node) {
				select {
				case nc <- n:
				case <-ctx.Done():
				}
			})
		})
		if t.Failed() {
			t.Fatalf("Failure scanning %v", n)
		}
	}
}

func newDatabase() (*bolt.DB, func(), error) {
	td, err := ioutil.TempDir("", "gc-roots-")
	if err != nil {
		return nil, nil, err
	}

	db, err := bolt.Open(filepath.Join(td, "test.db"), 0777, nil)
	if err != nil {
		os.RemoveAll(td)
		return nil, nil, err
	}

	return db, func() {
		db.Close()
		os.RemoveAll(td)
	}, nil
}

func checkNodeC(ctx context.Context, t *testing.T, db *bolt.DB, expected []gc.Node, fn func(context.Context, *bolt.Tx, chan<- gc.Node) error) {
	var actual []gc.Node
	nc := make(chan gc.Node)
	done := make(chan struct{})
	go func() {
		defer close(done)
		for n := range nc {
			actual = append(actual, n)
		}
	}()
	if err := db.View(func(tx *bolt.Tx) error {
		defer close(nc)
		return fn(ctx, tx, nc)
	}); err != nil {
		t.Fatal(err)
	}

	<-done
	checkNodesEqual(t, actual, expected)
}

func checkNodes(ctx context.Context, t *testing.T, db *bolt.DB, expected []gc.Node, fn func(context.Context, *bolt.Tx, func(context.Context, gc.Node) error) error) {
	var actual []gc.Node
	scanFn := func(ctx context.Context, n gc.Node) error {
		actual = append(actual, n)
		return nil
	}

	if err := db.View(func(tx *bolt.Tx) error {
		return fn(ctx, tx, scanFn)
	}); err != nil {
		t.Fatal(err)
	}

	checkNodesEqual(t, actual, expected)
}

func checkNodesEqual(t *testing.T, n1, n2 []gc.Node) {
	sort.Sort(nodeList(n1))
	sort.Sort(nodeList(n2))

	if len(n1) != len(n2) {
		t.Errorf("Nodes do not match\n\tExpected:\n\t%v\n\tActual:\n\t%v", n2, n1)
		return
	}

	for i := range n1 {
		if n1[i] != n2[i] {
			t.Errorf("[%d] root does not match expected: expected %v, got %v", i, n2[i], n1[i])
		}
	}
}

type nodeList []gc.Node

func (nodes nodeList) Len() int {
	return len(nodes)
}

func (nodes nodeList) Less(i, j int) bool {
	if nodes[i].Type != nodes[j].Type {
		return nodes[i].Type < nodes[j].Type
	}
	if nodes[i].Namespace != nodes[j].Namespace {
		return nodes[i].Namespace < nodes[j].Namespace
	}
	return nodes[i].Key < nodes[j].Key
}

func (nodes nodeList) Swap(i, j int) {
	nodes[i], nodes[j] = nodes[j], nodes[i]
}

type alterFunc func(bkt *bolt.Bucket) error

func addImage(ns, name string, dgst digest.Digest, labels map[string]string) alterFunc {
	return func(bkt *bolt.Bucket) error {
		ibkt, err := createBuckets(bkt, ns, string(bucketKeyObjectImages), name)
		if err != nil {
			return err
		}

		tbkt, err := ibkt.CreateBucket(bucketKeyTarget)
		if err != nil {
			return err
		}
		if err := tbkt.Put(bucketKeyDigest, []byte(dgst.String())); err != nil {
			return err
		}

		return boltutil.WriteLabels(ibkt, labels)
	}
}

func addSnapshot(ns, snapshotter, name, parent string, labels map[string]string) alterFunc {
	return func(bkt *bolt.Bucket) error {
		sbkt, err := createBuckets(bkt, ns, string(bucketKeyObjectSnapshots), snapshotter, name)
		if err != nil {
			return err
		}
		if parent != "" {
			if err := sbkt.Put(bucketKeyParent, []byte(parent)); err != nil {
				return err
			}
		}
		return boltutil.WriteLabels(sbkt, labels)
	}
}

func addContent(ns string, dgst digest.Digest, labels map[string]string) alterFunc {
	return func(bkt *bolt.Bucket) error {
		cbkt, err := createBuckets(bkt, ns, string(bucketKeyObjectContent), string(bucketKeyObjectBlob), dgst.String())
		if err != nil {
			return err
		}
		return boltutil.WriteLabels(cbkt, labels)
	}
}

func addLeaseSnapshot(ns, lid, snapshotter, name string) alterFunc {
	return func(bkt *bolt.Bucket) error {
		sbkt, err := createBuckets(bkt, ns, string(bucketKeyObjectLeases), lid, string(bucketKeyObjectSnapshots), snapshotter)
		if err != nil {
			return err
		}
		return sbkt.Put([]byte(name), nil)
	}
}

func addLeaseContent(ns, lid string, dgst digest.Digest) alterFunc {
	return func(bkt *bolt.Bucket) error {
		cbkt, err := createBuckets(bkt, ns, string(bucketKeyObjectLeases), lid, string(bucketKeyObjectContent))
		if err != nil {
			return err
		}
		return cbkt.Put([]byte(dgst.String()), nil)
	}
}

func addContainer(ns, name, snapshotter, snapshot string, labels map[string]string) alterFunc {
	return func(bkt *bolt.Bucket) error {
		cbkt, err := createBuckets(bkt, ns, string(bucketKeyObjectContainers), name)
		if err != nil {
			return err
		}
		if err := cbkt.Put(bucketKeySnapshotter, []byte(snapshotter)); err != nil {
			return err
		}
		if err := cbkt.Put(bucketKeySnapshotKey, []byte(snapshot)); err != nil {
			return err
		}
		return boltutil.WriteLabels(cbkt, labels)
	}
}

func createBuckets(bkt *bolt.Bucket, names ...string) (*bolt.Bucket, error) {
	for _, name := range names {
		nbkt, err := bkt.CreateBucketIfNotExists([]byte(name))
		if err != nil {
			return nil, err
		}
		bkt = nbkt
	}
	return bkt, nil
}

func labelmap(kv ...string) map[string]string {
	if len(kv)%2 != 0 {
		panic("bad labels argument")
	}
	l := map[string]string{}
	for i := 0; i < len(kv); i = i + 2 {
		l[kv[i]] = kv[i+1]
	}
	return l
}

func dgst(i int64) digest.Digest {
	r := rand.New(rand.NewSource(i))
	dgstr := digest.SHA256.Digester()
	if _, err := io.CopyN(dgstr.Hash(), r, 256); err != nil {
		panic(err)
	}
	return dgstr.Digest()
}
