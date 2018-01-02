package metadata

import (
	"context"
	"encoding/binary"
	"fmt"
	"strings"
	"time"

	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/filters"
	"github.com/containerd/containerd/images"
	"github.com/containerd/containerd/labels"
	"github.com/containerd/containerd/metadata/boltutil"
	"github.com/containerd/containerd/namespaces"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
)

type imageStore struct {
	tx *bolt.Tx
}

// NewImageStore returns a store backed by a bolt DB
func NewImageStore(tx *bolt.Tx) images.Store {
	return &imageStore{tx: tx}
}

func (s *imageStore) Get(ctx context.Context, name string) (images.Image, error) {
	var image images.Image

	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return images.Image{}, err
	}

	bkt := getImagesBucket(s.tx, namespace)
	if bkt == nil {
		return images.Image{}, errors.Wrapf(errdefs.ErrNotFound, "image %q", name)
	}

	ibkt := bkt.Bucket([]byte(name))
	if ibkt == nil {
		return images.Image{}, errors.Wrapf(errdefs.ErrNotFound, "image %q", name)
	}

	image.Name = name
	if err := readImage(&image, ibkt); err != nil {
		return images.Image{}, errors.Wrapf(err, "image %q", name)
	}

	return image, nil
}

func (s *imageStore) List(ctx context.Context, fs ...string) ([]images.Image, error) {
	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return nil, err
	}

	filter, err := filters.ParseAll(fs...)
	if err != nil {
		return nil, errors.Wrapf(errdefs.ErrInvalidArgument, err.Error())
	}

	bkt := getImagesBucket(s.tx, namespace)
	if bkt == nil {
		return nil, nil // empty store
	}

	var m []images.Image
	if err := bkt.ForEach(func(k, v []byte) error {
		var (
			image = images.Image{
				Name: string(k),
			}
			kbkt = bkt.Bucket(k)
		)

		if err := readImage(&image, kbkt); err != nil {
			return err
		}

		if filter.Match(adaptImage(image)) {
			m = append(m, image)
		}
		return nil
	}); err != nil {
		return nil, err
	}

	return m, nil
}

func (s *imageStore) Create(ctx context.Context, image images.Image) (images.Image, error) {
	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return images.Image{}, err
	}

	if image.Name == "" {
		return images.Image{}, errors.Wrapf(errdefs.ErrInvalidArgument, "image name is required for create")
	}

	if err := validateImage(&image); err != nil {
		return images.Image{}, err
	}

	return image, withImagesBucket(s.tx, namespace, func(bkt *bolt.Bucket) error {
		ibkt, err := bkt.CreateBucket([]byte(image.Name))
		if err != nil {
			if err != bolt.ErrBucketExists {
				return err
			}

			return errors.Wrapf(errdefs.ErrAlreadyExists, "image %q", image.Name)
		}

		image.CreatedAt = time.Now().UTC()
		image.UpdatedAt = image.CreatedAt
		return writeImage(ibkt, &image)
	})
}

func (s *imageStore) Update(ctx context.Context, image images.Image, fieldpaths ...string) (images.Image, error) {
	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return images.Image{}, err
	}

	if image.Name == "" {
		return images.Image{}, errors.Wrapf(errdefs.ErrInvalidArgument, "image name is required for update")
	}

	var updated images.Image
	return updated, withImagesBucket(s.tx, namespace, func(bkt *bolt.Bucket) error {
		ibkt := bkt.Bucket([]byte(image.Name))
		if ibkt == nil {
			return errors.Wrapf(errdefs.ErrNotFound, "image %q", image.Name)
		}

		if err := readImage(&updated, ibkt); err != nil {
			return errors.Wrapf(err, "image %q", image.Name)
		}
		createdat := updated.CreatedAt
		updated.Name = image.Name

		if len(fieldpaths) > 0 {
			for _, path := range fieldpaths {
				if strings.HasPrefix(path, "labels.") {
					if updated.Labels == nil {
						updated.Labels = map[string]string{}
					}

					key := strings.TrimPrefix(path, "labels.")
					updated.Labels[key] = image.Labels[key]
					continue
				}

				switch path {
				case "labels":
					updated.Labels = image.Labels
				case "target":
					// NOTE(stevvooe): While we allow setting individual labels, we
					// only support replacing the target as a unit, since that is
					// commonly pulled as a unit from other sources. It often doesn't
					// make sense to modify the size or digest without touching the
					// mediatype, as well, for example.
					updated.Target = image.Target
				default:
					return errors.Wrapf(errdefs.ErrInvalidArgument, "cannot update %q field on image %q", path, image.Name)
				}
			}
		} else {
			updated = image
		}

		if err := validateImage(&image); err != nil {
			return err
		}

		updated.CreatedAt = createdat
		updated.UpdatedAt = time.Now().UTC()
		return writeImage(ibkt, &updated)
	})
}

func (s *imageStore) Delete(ctx context.Context, name string) error {
	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return err
	}

	return withImagesBucket(s.tx, namespace, func(bkt *bolt.Bucket) error {
		err := bkt.DeleteBucket([]byte(name))
		if err == bolt.ErrBucketNotFound {
			return errors.Wrapf(errdefs.ErrNotFound, "image %q", name)
		}
		return err
	})
}

func validateImage(image *images.Image) error {
	if image.Name == "" {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "image name must not be empty")
	}

	for k, v := range image.Labels {
		if err := labels.Validate(k, v); err != nil {
			return errors.Wrapf(err, "image.Labels")
		}
	}

	return validateTarget(&image.Target)
}

func validateTarget(target *ocispec.Descriptor) error {
	// NOTE(stevvooe): Only validate fields we actually store.

	if err := target.Digest.Validate(); err != nil {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "Target.Digest %q invalid: %v", target.Digest, err)
	}

	if target.Size <= 0 {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "Target.Size must be greater than zero")
	}

	if target.MediaType == "" {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "Target.MediaType must be set")
	}

	return nil
}

func readImage(image *images.Image, bkt *bolt.Bucket) error {
	if err := boltutil.ReadTimestamps(bkt, &image.CreatedAt, &image.UpdatedAt); err != nil {
		return err
	}

	labels, err := boltutil.ReadLabels(bkt)
	if err != nil {
		return err
	}
	image.Labels = labels

	tbkt := bkt.Bucket(bucketKeyTarget)
	if tbkt == nil {
		return errors.New("unable to read target bucket")
	}
	return tbkt.ForEach(func(k, v []byte) error {
		if v == nil {
			return nil // skip it? a bkt maybe?
		}

		// TODO(stevvooe): This is why we need to use byte values for
		// keys, rather than full arrays.
		switch string(k) {
		case string(bucketKeyDigest):
			image.Target.Digest = digest.Digest(v)
		case string(bucketKeyMediaType):
			image.Target.MediaType = string(v)
		case string(bucketKeySize):
			image.Target.Size, _ = binary.Varint(v)
		}

		return nil
	})
}

func writeImage(bkt *bolt.Bucket, image *images.Image) error {
	if err := boltutil.WriteTimestamps(bkt, image.CreatedAt, image.UpdatedAt); err != nil {
		return err
	}

	if err := boltutil.WriteLabels(bkt, image.Labels); err != nil {
		return errors.Wrapf(err, "writing labels for image %v", image.Name)
	}

	// write the target bucket
	tbkt, err := bkt.CreateBucketIfNotExists([]byte(bucketKeyTarget))
	if err != nil {
		return err
	}

	sizeEncoded, err := encodeInt(image.Target.Size)
	if err != nil {
		return err
	}

	for _, v := range [][2][]byte{
		{bucketKeyDigest, []byte(image.Target.Digest)},
		{bucketKeyMediaType, []byte(image.Target.MediaType)},
		{bucketKeySize, sizeEncoded},
	} {
		if err := tbkt.Put(v[0], v[1]); err != nil {
			return err
		}
	}

	return nil
}

func encodeInt(i int64) ([]byte, error) {
	var (
		buf      [binary.MaxVarintLen64]byte
		iEncoded = buf[:]
	)
	iEncoded = iEncoded[:binary.PutVarint(iEncoded, i)]

	if len(iEncoded) == 0 {
		return nil, fmt.Errorf("failed encoding integer = %v", i)
	}
	return iEncoded, nil
}
