package metadata

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/filters"
	"github.com/containerd/containerd/images"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

func TestImagesList(t *testing.T) {
	ctx, db, cancel := testEnv(t)
	defer cancel()

	testset := map[string]*images.Image{}
	for i := 0; i < 4; i++ {
		id := "image-" + fmt.Sprint(i)
		testset[id] = &images.Image{
			Name: id,
			Labels: map[string]string{
				"namelabel": id,
				"even":      fmt.Sprint(i%2 == 0),
				"odd":       fmt.Sprint(i%2 != 0),
			},
			Target: ocispec.Descriptor{
				Size:      10,
				MediaType: "application/vnd.containerd.test",
				Digest:    digest.FromString(id),
			},
		}

		if err := db.Update(func(tx *bolt.Tx) error {
			store := NewImageStore(tx)
			now := time.Now()
			result, err := store.Create(ctx, *testset[id])
			if err != nil {
				return err
			}

			checkImageTimestamps(t, &result, now, true)
			testset[id].UpdatedAt, testset[id].CreatedAt = result.UpdatedAt, result.CreatedAt
			checkImagesEqual(t, &result, testset[id], "ensure that containers were created as expected for list")
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
			name:    "ByName",
			filters: []string{"name==image-0"},
		},
		{
			name:    "ByNameLabelEven",
			filters: []string{"labels.namelabel==image-0,labels.even==true"},
		},
		{
			name:    "ByMediaType",
			filters: []string{"target.mediatype~=application/vnd.*"},
		},
	} {
		t.Run(testcase.name, func(t *testing.T) {
			testset := testset
			if len(testcase.filters) > 0 {
				fs, err := filters.ParseAll(testcase.filters...)
				if err != nil {
					t.Fatal(err)
				}

				newtestset := make(map[string]*images.Image, len(testset))
				for k, v := range testset {
					if fs.Match(adaptImage(*v)) {
						newtestset[k] = v
					}
				}
				testset = newtestset
			}

			if err := db.View(func(tx *bolt.Tx) error {
				store := NewImageStore(tx)
				results, err := store.List(ctx, testcase.filters...)
				if err != nil {
					t.Fatal(err)
				}

				if len(results) == 0 { // all tests return a non-empty result set
					t.Fatalf("no results returned")
				}

				if len(results) != len(testset) {
					t.Fatalf("length of result does not match testset: %v != %v", len(results), len(testset))
				}

				for _, result := range results {
					checkImagesEqual(t, &result, testset[result.Name], "list results did not match")
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
			store := NewImageStore(tx)
			return store.Delete(ctx, id)
		}); err != nil {
			t.Fatal(err)
		}

		// try it again, get NotFound
		if err := db.Update(func(tx *bolt.Tx) error {
			store := NewImageStore(tx)
			return store.Delete(ctx, id)
		}); errors.Cause(err) != errdefs.ErrNotFound {
			t.Fatalf("unexpected error %v", err)
		}
	}
}
func TestImagesCreateUpdateDelete(t *testing.T) {
	ctx, db, cancel := testEnv(t)
	defer cancel()

	for _, testcase := range []struct {
		name       string
		original   images.Image
		createerr  error
		input      images.Image
		fieldpaths []string
		expected   images.Image
		cause      error
	}{
		{
			name:     "Touch",
			original: imageBase(),
			input: images.Image{
				Labels: map[string]string{
					"foo": "bar",
					"baz": "boo",
				},
				Target: ocispec.Descriptor{
					Size:      10,
					MediaType: "application/vnd.oci.blab",
				},
			},
			expected: images.Image{
				Labels: map[string]string{
					"foo": "bar",
					"baz": "boo",
				},
				Target: ocispec.Descriptor{
					Size:      10,
					MediaType: "application/vnd.oci.blab",
				},
			},
		},
		{
			name: "NoTarget",
			original: images.Image{
				Labels: map[string]string{
					"foo": "bar",
					"baz": "boo",
				},
				Target: ocispec.Descriptor{},
			},
			createerr: errdefs.ErrInvalidArgument,
		},
		{
			name:     "ReplaceLabels",
			original: imageBase(),
			input: images.Image{
				Labels: map[string]string{
					"for": "bar",
					"boo": "boo",
				},
				Target: ocispec.Descriptor{
					Size:      10,
					MediaType: "application/vnd.oci.blab",
				},
			},
			expected: images.Image{
				Labels: map[string]string{
					"for": "bar",
					"boo": "boo",
				},
				Target: ocispec.Descriptor{
					Size:      10,
					MediaType: "application/vnd.oci.blab",
				},
			},
		},
		{
			name:     "ReplaceLabelsFieldPath",
			original: imageBase(),
			input: images.Image{
				Labels: map[string]string{
					"for": "bar",
					"boo": "boo",
				},
				Target: ocispec.Descriptor{
					Size:      20,                                 // ignored
					MediaType: "application/vnd.oci.blab+ignored", // make sure other stuff is ignored
				},
			},
			fieldpaths: []string{"labels"},
			expected: images.Image{
				Labels: map[string]string{
					"for": "bar",
					"boo": "boo",
				},
				Target: ocispec.Descriptor{
					Size:      10,
					MediaType: "application/vnd.oci.blab",
				},
			},
		},
		{
			name:     "ReplaceLabel",
			original: imageBase(),
			input: images.Image{
				Labels: map[string]string{
					"foo": "baz",
					"baz": "bunk",
				},
				Target: ocispec.Descriptor{
					Size:      20,                                 // ignored
					MediaType: "application/vnd.oci.blab+ignored", // make sure other stuff is ignored
				},
			},
			fieldpaths: []string{"labels.foo"},
			expected: images.Image{
				Labels: map[string]string{
					"foo": "baz",
					"baz": "boo",
				},
				Target: ocispec.Descriptor{
					Size:      10,
					MediaType: "application/vnd.oci.blab",
				},
			},
		},
		{
			name:     "ReplaceTarget", // target must be updated as a unit
			original: imageBase(),
			input: images.Image{
				Target: ocispec.Descriptor{
					Size:      10,
					MediaType: "application/vnd.oci.blab+replaced",
				},
			},
			fieldpaths: []string{"target"},
			expected: images.Image{
				Labels: map[string]string{
					"foo": "bar",
					"baz": "boo",
				},
				Target: ocispec.Descriptor{
					Size:      10,
					MediaType: "application/vnd.oci.blab+replaced",
				},
			},
		},
		{
			name:     "EmptySize",
			original: imageBase(),
			input: images.Image{
				Labels: map[string]string{
					"foo": "bar",
					"baz": "boo",
				},
				Target: ocispec.Descriptor{
					Size:      0,
					MediaType: "application/vnd.oci.blab",
				},
			},
			cause: errdefs.ErrInvalidArgument,
		},
		{
			name: "EmptySizeOnCreate",
			original: images.Image{
				Labels: map[string]string{
					"foo": "bar",
					"baz": "boo",
				},
				Target: ocispec.Descriptor{
					MediaType: "application/vnd.oci.blab",
				},
			},
			createerr: errdefs.ErrInvalidArgument,
		},
		{
			name:     "EmptyMediaType",
			original: imageBase(),
			input: images.Image{
				Labels: map[string]string{
					"foo": "bar",
					"baz": "boo",
				},
				Target: ocispec.Descriptor{
					Size: 10,
				},
			},
			cause: errdefs.ErrInvalidArgument,
		},
		{
			name: "EmptySizeOnCreate",
			original: images.Image{
				Labels: map[string]string{
					"foo": "bar",
					"baz": "boo",
				},
				Target: ocispec.Descriptor{
					Size: 10,
				},
			},
			createerr: errdefs.ErrInvalidArgument,
		},
		{
			name: "TryUpdateNameFail",
			original: images.Image{
				Labels: map[string]string{
					"foo": "bar",
					"baz": "boo",
				},
				Target: ocispec.Descriptor{
					Size:      10,
					MediaType: "application/vnd.oci.blab",
				},
			},
			input: images.Image{
				Name: "test should fail",
				Labels: map[string]string{
					"foo": "bar",
					"baz": "boo",
				},
				Target: ocispec.Descriptor{
					Size:      10,
					MediaType: "application/vnd.oci.blab",
				},
			},
			cause: errdefs.ErrNotFound,
		},
	} {
		t.Run(testcase.name, func(t *testing.T) {
			testcase.original.Name = testcase.name
			if testcase.input.Name == "" {
				testcase.input.Name = testcase.name
			}
			testcase.expected.Name = testcase.name

			if testcase.original.Target.Digest == "" {
				testcase.original.Target.Digest = digest.FromString(testcase.name)
				testcase.input.Target.Digest = testcase.original.Target.Digest
				testcase.expected.Target.Digest = testcase.original.Target.Digest
			}

			done := errors.New("test complete")
			if err := db.Update(func(tx *bolt.Tx) error {
				var (
					store = NewImageStore(tx)
					now   = time.Now()
				)

				created, err := store.Create(ctx, testcase.original)
				if errors.Cause(err) != testcase.createerr {
					if testcase.createerr == nil {
						t.Fatalf("unexpected error: %v", err)
					} else {
						t.Fatalf("cause of %v (cause: %v) != %v", err, errors.Cause(err), testcase.createerr)
					}
				} else if testcase.createerr != nil {
					return done
				}

				checkImageTimestamps(t, &created, now, true)

				testcase.original.CreatedAt = created.CreatedAt
				testcase.expected.CreatedAt = created.CreatedAt
				testcase.original.UpdatedAt = created.UpdatedAt
				testcase.expected.UpdatedAt = created.UpdatedAt

				checkImagesEqual(t, &created, &testcase.original, "unexpected image on creation")
				return nil
			}); err != nil {
				if err == done {
					return
				}
				t.Fatal(err)
			}

			if err := db.Update(func(tx *bolt.Tx) error {
				now := time.Now()
				store := NewImageStore(tx)
				updated, err := store.Update(ctx, testcase.input, testcase.fieldpaths...)
				if errors.Cause(err) != testcase.cause {
					if testcase.cause == nil {
						t.Fatalf("unexpected error: %v", err)
					} else {
						t.Fatalf("cause of %v (cause: %v) != %v", err, errors.Cause(err), testcase.cause)
					}
				} else if testcase.cause != nil {
					return done
				}

				checkImageTimestamps(t, &updated, now, false)
				testcase.expected.UpdatedAt = updated.UpdatedAt
				checkImagesEqual(t, &updated, &testcase.expected, "updated failed to get expected result")
				return nil
			}); err != nil {
				if err == done {
					return
				}
				t.Fatal(err)
			}

			if err := db.View(func(tx *bolt.Tx) error {
				store := NewImageStore(tx)
				result, err := store.Get(ctx, testcase.original.Name)
				if err != nil {
					t.Fatal(err)
				}

				checkImagesEqual(t, &result, &testcase.expected, "get after failed to get expected result")
				return nil
			}); err != nil {
				t.Fatal(err)
			}

		})
	}
}

func imageBase() images.Image {
	return images.Image{
		Labels: map[string]string{
			"foo": "bar",
			"baz": "boo",
		},
		Target: ocispec.Descriptor{
			Size:      10,
			MediaType: "application/vnd.oci.blab",
		},
	}
}

func checkImageTimestamps(t *testing.T, im *images.Image, now time.Time, oncreate bool) {
	t.Helper()
	if im.UpdatedAt.IsZero() || im.CreatedAt.IsZero() {
		t.Fatalf("timestamps not set")
	}

	if oncreate {
		if !im.CreatedAt.Equal(im.UpdatedAt) {
			t.Fatal("timestamps should be equal on create")
		}

	} else {
		// ensure that updatedat is always after createdat
		if !im.UpdatedAt.After(im.CreatedAt) {
			t.Fatalf("timestamp for updatedat not after createdat: %v <= %v", im.UpdatedAt, im.CreatedAt)
		}
	}

	if im.UpdatedAt.Before(now) {
		t.Fatal("createdat time incorrect should be after the start of the operation")
	}
}

func checkImagesEqual(t *testing.T, a, b *images.Image, format string, args ...interface{}) {
	t.Helper()
	if !reflect.DeepEqual(a, b) {
		t.Fatalf("images not equal \n\t%v != \n\t%v: "+format, append([]interface{}{a, b}, args...)...)
	}
}
