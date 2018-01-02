package metadata

import (
	"context"
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/containers"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/filters"
	"github.com/containerd/containerd/namespaces"
	"github.com/containerd/typeurl"
	"github.com/gogo/protobuf/types"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
)

func init() {
	typeurl.Register(&specs.Spec{}, "types.contianerd.io/opencontainers/runtime-spec", "v1", "Spec")
}

func TestContainersList(t *testing.T) {
	ctx, db, cancel := testEnv(t)
	defer cancel()

	spec := &specs.Spec{}
	encoded, err := typeurl.MarshalAny(spec)
	if err != nil {
		t.Fatal(err)
	}

	testset := map[string]*containers.Container{}
	for i := 0; i < 4; i++ {
		id := "container-" + fmt.Sprint(i)
		testset[id] = &containers.Container{
			ID: id,
			Labels: map[string]string{
				"idlabel": id,
				"even":    fmt.Sprint(i%2 == 0),
				"odd":     fmt.Sprint(i%2 != 0),
			},
			Spec:        encoded,
			SnapshotKey: "test-snapshot-key",
			Snapshotter: "snapshotter",
			Runtime: containers.RuntimeInfo{
				Name: "testruntime",
			},
			Image: "test image",
		}

		if err := db.Update(func(tx *bolt.Tx) error {
			store := NewContainerStore(tx)
			now := time.Now()
			result, err := store.Create(ctx, *testset[id])
			if err != nil {
				return err
			}

			checkContainerTimestamps(t, &result, now, true)
			testset[id].UpdatedAt, testset[id].CreatedAt = result.UpdatedAt, result.CreatedAt
			checkContainersEqual(t, &result, testset[id], "ensure that containers were created as expected for list")
			return nil
		}); err != nil {
			t.Fatal(err)
		}
	}

	for _, testcase := range []struct {
		name    string
		filters []string
	}{
		{
			name: "FullSet",
		},
		{
			name:    "FullSetFiltered", // full set, but because we have OR filter
			filters: []string{"labels.even==true", "labels.odd==true"},
		},
		{
			name:    "Even",
			filters: []string{"labels.even==true"},
		},
		{
			name:    "Odd",
			filters: []string{"labels.odd==true"},
		},
		{
			name:    "ByID",
			filters: []string{"id==container-0"},
		},
		{
			name:    "ByIDLabelEven",
			filters: []string{"labels.idlabel==container-0,labels.even==true"},
		},
		{
			name:    "ByRuntime",
			filters: []string{"runtime.name==testruntime"},
		},
	} {
		t.Run(testcase.name, func(t *testing.T) {
			testset := testset
			if len(testcase.filters) > 0 {
				fs, err := filters.ParseAll(testcase.filters...)
				if err != nil {
					t.Fatal(err)
				}

				newtestset := make(map[string]*containers.Container, len(testset))
				for k, v := range testset {
					if fs.Match(adaptContainer(*v)) {
						newtestset[k] = v
					}
				}
				testset = newtestset
			}

			if err := db.View(func(tx *bolt.Tx) error {
				store := NewContainerStore(tx)
				results, err := store.List(ctx, testcase.filters...)
				if err != nil {
					t.Fatal(err)
				}

				if len(results) == 0 { // all tests return a non-empty result set
					t.Fatalf("not results returned")
				}

				if len(results) != len(testset) {
					t.Fatalf("length of result does not match testset: %v != %v", len(results), len(testset))
				}

				for _, result := range results {
					checkContainersEqual(t, &result, testset[result.ID], "list results did not match")
				}

				return nil
			}); err != nil {
				t.Fatal(err)
			}
		})
	}

	// delete everything to test it
	for id := range testset {
		if err := db.Update(func(tx *bolt.Tx) error {
			store := NewContainerStore(tx)
			return store.Delete(ctx, id)
		}); err != nil {
			t.Fatal(err)
		}

		// try it again, get NotFound
		if err := db.Update(func(tx *bolt.Tx) error {
			store := NewContainerStore(tx)
			return store.Delete(ctx, id)
		}); errors.Cause(err) != errdefs.ErrNotFound {
			t.Fatalf("unexpected error %v", err)
		}
	}
}

// TestContainersUpdate ensures that updates are taken in an expected manner.
func TestContainersCreateUpdateDelete(t *testing.T) {
	ctx, db, cancel := testEnv(t)
	defer cancel()

	spec := &specs.Spec{}
	encoded, err := typeurl.MarshalAny(spec)
	if err != nil {
		t.Fatal(err)
	}

	spec.Annotations = map[string]string{"updated": "true"}
	encodedUpdated, err := typeurl.MarshalAny(spec)
	if err != nil {
		t.Fatal(err)
	}

	for _, testcase := range []struct {
		name       string
		original   containers.Container
		createerr  error
		input      containers.Container
		fieldpaths []string
		expected   containers.Container
		cause      error
	}{
		{
			name: "UpdateIDFail",
			original: containers.Container{
				Spec:        encoded,
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
			},
			input: containers.Container{
				ID:   "newid",
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
			},
			fieldpaths: []string{"id"},
			cause:      errdefs.ErrNotFound,
		},
		{
			name: "UpdateRuntimeFail",
			original: containers.Container{
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",
				Spec:        encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
			},
			input: containers.Container{
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntimedifferent",
				},
			},
			fieldpaths: []string{"runtime"},
			cause:      errdefs.ErrInvalidArgument,
		},
		{
			name: "UpdateRuntimeClearFail",
			original: containers.Container{
				Spec:        encoded,
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
			},
			input: containers.Container{
				Spec: encoded,
			},
			fieldpaths: []string{"runtime"},
			cause:      errdefs.ErrInvalidArgument,
		},
		{
			name: "UpdateFail",
			original: containers.Container{
				Spec:        encoded,
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",

				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Image: "test image",
			},
			input: containers.Container{
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",
				// try to clear image field
			},
			cause: errdefs.ErrInvalidArgument,
		},
		{
			name: "UpdateSpec",
			original: containers.Container{
				Spec:        encoded,
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Image: "test image",
			},
			input: containers.Container{
				Spec: encodedUpdated,
			},
			fieldpaths: []string{"spec"},
			expected: containers.Container{
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Spec:        encodedUpdated,
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",
				Image:       "test image",
			},
		},
		{
			name: "UpdateLabel",
			original: containers.Container{
				Labels: map[string]string{
					"foo": "one",
					"bar": "two",
				},
				Spec:        encoded,
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Image: "test image",
			},
			input: containers.Container{
				Labels: map[string]string{
					"bar": "baz",
				},
			},
			fieldpaths: []string{"labels.bar"},
			expected: containers.Container{
				Labels: map[string]string{
					"foo": "one",
					"bar": "baz",
				},
				Spec:        encoded,
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Image: "test image",
			},
		},
		{
			name: "DeleteAllLabels",
			original: containers.Container{
				Labels: map[string]string{
					"foo": "one",
					"bar": "two",
				},
				Spec:        encoded,
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Image: "test image",
			},
			input: containers.Container{
				Labels: nil,
			},
			fieldpaths: []string{"labels"},
			expected: containers.Container{
				Spec:        encoded,
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Image: "test image",
			},
		},
		{
			name: "DeleteLabel",
			original: containers.Container{
				Labels: map[string]string{
					"foo": "one",
					"bar": "two",
				},
				Spec:        encoded,
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Image: "test image",
			},
			input: containers.Container{
				Labels: map[string]string{
					"bar": "",
				},
			},
			fieldpaths: []string{"labels.bar"},
			expected: containers.Container{
				Labels: map[string]string{
					"foo": "one",
				},
				Spec:        encoded,
				SnapshotKey: "test-snapshot-key",
				Snapshotter: "snapshotter",
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Image: "test image",
			},
		},
		{
			name: "UpdateSnapshotKeyImmutable",
			original: containers.Container{
				Spec:        encoded,
				SnapshotKey: "",
				Snapshotter: "",
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
			},
			input: containers.Container{
				SnapshotKey: "something",
				Snapshotter: "something",
			},
			fieldpaths: []string{"snapshotkey", "snapshotter"},
			cause:      errdefs.ErrInvalidArgument,
		},
		{
			name: "SnapshotKeyWithoutSnapshot",
			original: containers.Container{
				Spec:        encoded,
				SnapshotKey: "/nosnapshot",
				Snapshotter: "",
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
			},
			createerr: errdefs.ErrInvalidArgument,
		},
		{
			name: "UpdateExtensionsFull",
			original: containers.Container{
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Extensions: map[string]types.Any{
					"hello": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("hello"),
					},
				},
			},
			input: containers.Container{
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Extensions: map[string]types.Any{
					"hello": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("world"),
					},
				},
			},
			expected: containers.Container{
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Extensions: map[string]types.Any{
					"hello": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("world"),
					},
				},
			},
		},
		{
			name: "UpdateExtensionsNotInFieldpath",
			original: containers.Container{
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Extensions: map[string]types.Any{
					"hello": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("hello"),
					},
				},
			},
			input: containers.Container{
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Extensions: map[string]types.Any{
					"hello": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("world"),
					},
				},
			},
			fieldpaths: []string{"labels"},
			expected: containers.Container{
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Extensions: map[string]types.Any{
					"hello": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("hello"),
					},
				},
			},
		},
		{
			name: "UpdateExtensionsFieldPath",
			original: containers.Container{
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Extensions: map[string]types.Any{
					"hello": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("hello"),
					},
				},
			},
			input: containers.Container{
				Labels: map[string]string{
					"foo": "one",
				},
				Extensions: map[string]types.Any{
					"hello": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("world"),
					},
				},
			},
			fieldpaths: []string{"extensions"},
			expected: containers.Container{
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Extensions: map[string]types.Any{
					"hello": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("world"),
					},
				},
			},
		},
		{
			name: "UpdateExtensionsFieldPathIsolated",
			original: containers.Container{
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Extensions: map[string]types.Any{
					// leaves hello in place.
					"hello": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("hello"),
					},
				},
			},
			input: containers.Container{
				Extensions: map[string]types.Any{
					"hello": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("universe"), // this will be ignored
					},
					"bar": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("foo"), // this will be added
					},
				},
			},
			fieldpaths: []string{"extensions.bar"}, //
			expected: containers.Container{
				Spec: encoded,
				Runtime: containers.RuntimeInfo{
					Name: "testruntime",
				},
				Extensions: map[string]types.Any{
					"hello": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("hello"), // remains as world
					},
					"bar": {
						TypeUrl: "test.update.extensions",
						Value:   []byte("foo"), // this will be added
					},
				},
			},
		},
	} {
		t.Run(testcase.name, func(t *testing.T) {
			testcase.original.ID = testcase.name
			if testcase.input.ID == "" {
				testcase.input.ID = testcase.name
			}
			testcase.expected.ID = testcase.name

			done := errors.New("test complete")
			if err := db.Update(func(tx *bolt.Tx) error {
				var (
					now   = time.Now().UTC()
					store = NewContainerStore(tx)
				)

				result, err := store.Create(ctx, testcase.original)
				if errors.Cause(err) != testcase.createerr {
					if testcase.createerr == nil {
						t.Fatalf("unexpected error: %v", err)
					} else {
						t.Fatalf("cause of %v (cause: %v) != %v", err, errors.Cause(err), testcase.createerr)
					}
				} else if testcase.createerr != nil {
					return done
				}

				checkContainerTimestamps(t, &result, now, true)

				// ensure that createdat is never tampered with
				testcase.original.CreatedAt = result.CreatedAt
				testcase.expected.CreatedAt = result.CreatedAt
				testcase.original.UpdatedAt = result.UpdatedAt
				testcase.expected.UpdatedAt = result.UpdatedAt

				checkContainersEqual(t, &result, &testcase.original, "unexpected result on container update")
				return nil
			}); err != nil {
				if err == done {
					return
				}
				t.Fatal(err)
			}

			if err := db.Update(func(tx *bolt.Tx) error {
				now := time.Now()
				store := NewContainerStore(tx)
				result, err := store.Update(ctx, testcase.input, testcase.fieldpaths...)
				if errors.Cause(err) != testcase.cause {
					if testcase.cause == nil {
						t.Fatalf("unexpected error: %v", err)
					} else {
						t.Fatalf("cause of %v (cause: %v) != %v", err, errors.Cause(err), testcase.cause)
					}
				} else if testcase.cause != nil {
					return done
				}

				checkContainerTimestamps(t, &result, now, false)
				testcase.expected.UpdatedAt = result.UpdatedAt
				checkContainersEqual(t, &result, &testcase.expected, "updated failed to get expected result")
				return nil
			}); err != nil {
				if err == done {
					return
				}
				t.Fatal(err)
			}

			if err := db.View(func(tx *bolt.Tx) error {
				store := NewContainerStore(tx)
				result, err := store.Get(ctx, testcase.original.ID)
				if err != nil {
					t.Fatal(err)
				}

				checkContainersEqual(t, &result, &testcase.expected, "get after failed to get expected result")
				return nil
			}); err != nil {
				t.Fatal(err)
			}

		})
	}
}

func checkContainerTimestamps(t *testing.T, c *containers.Container, now time.Time, oncreate bool) {
	if c.UpdatedAt.IsZero() || c.CreatedAt.IsZero() {
		t.Fatalf("timestamps not set")
	}

	if oncreate {
		if !c.CreatedAt.Equal(c.UpdatedAt) {
			t.Fatal("timestamps should be equal on create")
		}

	} else {
		// ensure that updatedat is always after createdat
		if !c.UpdatedAt.After(c.CreatedAt) {
			t.Fatalf("timestamp for updatedat not after createdat: %v <= %v", c.UpdatedAt, c.CreatedAt)
		}
	}

	if c.UpdatedAt.Before(now) {
		t.Fatal("createdat time incorrect should be after the start of the operation")
	}
}

func checkContainersEqual(t *testing.T, a, b *containers.Container, format string, args ...interface{}) {
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("containers not equal \n\t%v != \n\t%v: "+format, append([]interface{}{a, b}, args...)...)
	}
}

func testEnv(t *testing.T) (context.Context, *bolt.DB, func()) {
	ctx, cancel := context.WithCancel(context.Background())
	ctx = namespaces.WithNamespace(ctx, "testing")

	dirname, err := ioutil.TempDir("", strings.Replace(t.Name(), "/", "_", -1)+"-")
	if err != nil {
		t.Fatal(err)
	}

	db, err := bolt.Open(filepath.Join(dirname, "meta.db"), 0644, nil)
	if err != nil {
		t.Fatal(err)
	}

	return ctx, db, func() {
		db.Close()
		if err := os.RemoveAll(dirname); err != nil {
			t.Log("failed removing temp dir", err)
		}
		cancel()
	}
}
