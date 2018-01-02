package metadata

import (
	"context"
	"strings"
	"time"

	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/containers"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/filters"
	"github.com/containerd/containerd/identifiers"
	"github.com/containerd/containerd/labels"
	"github.com/containerd/containerd/metadata/boltutil"
	"github.com/containerd/containerd/namespaces"
	"github.com/gogo/protobuf/proto"
	"github.com/gogo/protobuf/types"
	"github.com/pkg/errors"
)

type containerStore struct {
	tx *bolt.Tx
}

// NewContainerStore returns a Store backed by an underlying bolt DB
func NewContainerStore(tx *bolt.Tx) containers.Store {
	return &containerStore{
		tx: tx,
	}
}

func (s *containerStore) Get(ctx context.Context, id string) (containers.Container, error) {
	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return containers.Container{}, err
	}

	bkt := getContainerBucket(s.tx, namespace, id)
	if bkt == nil {
		return containers.Container{}, errors.Wrapf(errdefs.ErrNotFound, "bucket name %q:%q", namespace, id)
	}

	container := containers.Container{ID: id}
	if err := readContainer(&container, bkt); err != nil {
		return containers.Container{}, errors.Wrapf(err, "failed to read container %v", id)
	}

	return container, nil
}

func (s *containerStore) List(ctx context.Context, fs ...string) ([]containers.Container, error) {
	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return nil, err
	}

	filter, err := filters.ParseAll(fs...)
	if err != nil {
		return nil, errors.Wrapf(errdefs.ErrInvalidArgument, err.Error())
	}

	bkt := getContainersBucket(s.tx, namespace)
	if bkt == nil {
		return nil, nil
	}

	var m []containers.Container
	if err := bkt.ForEach(func(k, v []byte) error {
		cbkt := bkt.Bucket(k)
		if cbkt == nil {
			return nil
		}
		container := containers.Container{ID: string(k)}

		if err := readContainer(&container, cbkt); err != nil {
			return errors.Wrap(err, "failed to read container")
		}

		if filter.Match(adaptContainer(container)) {
			m = append(m, container)
		}
		return nil
	}); err != nil {
		return nil, err
	}

	return m, nil
}

func (s *containerStore) Create(ctx context.Context, container containers.Container) (containers.Container, error) {
	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return containers.Container{}, err
	}

	if err := validateContainer(&container); err != nil {
		return containers.Container{}, errors.Wrap(err, "create container failed validation")
	}

	bkt, err := createContainersBucket(s.tx, namespace)
	if err != nil {
		return containers.Container{}, err
	}

	cbkt, err := bkt.CreateBucket([]byte(container.ID))
	if err != nil {
		if err == bolt.ErrBucketExists {
			err = errors.Wrapf(errdefs.ErrAlreadyExists, "content %q", container.ID)
		}
		return containers.Container{}, err
	}

	container.CreatedAt = time.Now().UTC()
	container.UpdatedAt = container.CreatedAt
	if err := writeContainer(cbkt, &container); err != nil {
		return containers.Container{}, errors.Wrap(err, "failed to write container")
	}

	return container, nil
}

func (s *containerStore) Update(ctx context.Context, container containers.Container, fieldpaths ...string) (containers.Container, error) {
	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return containers.Container{}, err
	}

	if container.ID == "" {
		return containers.Container{}, errors.Wrapf(errdefs.ErrInvalidArgument, "must specify a container id")
	}

	bkt := getContainersBucket(s.tx, namespace)
	if bkt == nil {
		return containers.Container{}, errors.Wrapf(errdefs.ErrNotFound, "container %q", container.ID)
	}

	cbkt := bkt.Bucket([]byte(container.ID))
	if cbkt == nil {
		return containers.Container{}, errors.Wrapf(errdefs.ErrNotFound, "container %q", container.ID)
	}

	var updated containers.Container
	if err := readContainer(&updated, cbkt); err != nil {
		return updated, errors.Wrapf(err, "failed to read container from bucket")
	}
	createdat := updated.CreatedAt
	updated.ID = container.ID

	if len(fieldpaths) == 0 {
		// only allow updates to these field on full replace.
		fieldpaths = []string{"labels", "spec", "extensions"}

		// Fields that are immutable must cause an error when no field paths
		// are provided. This allows these fields to become mutable in the
		// future.
		if updated.Image != container.Image {
			return containers.Container{}, errors.Wrapf(errdefs.ErrInvalidArgument, "container.Image field is immutable")
		}

		if updated.SnapshotKey != container.SnapshotKey {
			return containers.Container{}, errors.Wrapf(errdefs.ErrInvalidArgument, "container.SnapshotKey field is immutable")
		}

		if updated.Snapshotter != container.Snapshotter {
			return containers.Container{}, errors.Wrapf(errdefs.ErrInvalidArgument, "container.Snapshotter field is immutable")
		}

		if updated.Runtime.Name != container.Runtime.Name {
			return containers.Container{}, errors.Wrapf(errdefs.ErrInvalidArgument, "container.Runtime.Name field is immutable")
		}
	}

	// apply the field mask. If you update this code, you better follow the
	// field mask rules in field_mask.proto. If you don't know what this
	// is, do not update this code.
	for _, path := range fieldpaths {
		if strings.HasPrefix(path, "labels.") {
			if updated.Labels == nil {
				updated.Labels = map[string]string{}
			}
			key := strings.TrimPrefix(path, "labels.")
			updated.Labels[key] = container.Labels[key]
			continue
		}

		if strings.HasPrefix(path, "extensions.") {
			if updated.Extensions == nil {
				updated.Extensions = map[string]types.Any{}
			}
			key := strings.TrimPrefix(path, "extensions.")
			updated.Extensions[key] = container.Extensions[key]
			continue
		}

		switch path {
		case "labels":
			updated.Labels = container.Labels
		case "spec":
			updated.Spec = container.Spec
		case "extensions":
			updated.Extensions = container.Extensions
		default:
			return containers.Container{}, errors.Wrapf(errdefs.ErrInvalidArgument, "cannot update %q field on %q", path, container.ID)
		}
	}

	if err := validateContainer(&updated); err != nil {
		return containers.Container{}, errors.Wrap(err, "update failed validation")
	}

	updated.CreatedAt = createdat
	updated.UpdatedAt = time.Now().UTC()
	if err := writeContainer(cbkt, &updated); err != nil {
		return containers.Container{}, errors.Wrap(err, "failed to write container")
	}

	return updated, nil
}

func (s *containerStore) Delete(ctx context.Context, id string) error {
	namespace, err := namespaces.NamespaceRequired(ctx)
	if err != nil {
		return err
	}

	bkt := getContainersBucket(s.tx, namespace)
	if bkt == nil {
		return errors.Wrapf(errdefs.ErrNotFound, "cannot delete container %v, bucket not present", id)
	}

	if err := bkt.DeleteBucket([]byte(id)); err == bolt.ErrBucketNotFound {
		return errors.Wrapf(errdefs.ErrNotFound, "container %v", id)
	}
	return err
}

func validateContainer(container *containers.Container) error {
	if err := identifiers.Validate(container.ID); err != nil {
		return errors.Wrapf(err, "container.ID validation error")
	}

	for k := range container.Extensions {
		if k == "" {
			return errors.Wrapf(errdefs.ErrInvalidArgument, "container.Extension keys must not be zero-length")
		}
	}

	// image has no validation
	for k, v := range container.Labels {
		if err := labels.Validate(k, v); err == nil {
			return errors.Wrapf(err, "containers.Labels")
		}
	}

	if container.Runtime.Name == "" {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "container.Runtime.Name must be set")
	}

	if container.Spec == nil {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "container.Spec must be set")
	}

	if container.SnapshotKey != "" && container.Snapshotter == "" {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "container.Snapshotter must be set if container.SnapshotKey is set")
	}

	return nil
}

func readContainer(container *containers.Container, bkt *bolt.Bucket) error {
	labels, err := boltutil.ReadLabels(bkt)
	if err != nil {
		return err
	}
	container.Labels = labels

	if err := boltutil.ReadTimestamps(bkt, &container.CreatedAt, &container.UpdatedAt); err != nil {
		return err
	}

	return bkt.ForEach(func(k, v []byte) error {
		switch string(k) {
		case string(bucketKeyImage):
			container.Image = string(v)
		case string(bucketKeyRuntime):
			rbkt := bkt.Bucket(bucketKeyRuntime)
			if rbkt == nil {
				return nil // skip runtime. should be an error?
			}

			n := rbkt.Get(bucketKeyName)
			if n != nil {
				container.Runtime.Name = string(n)
			}

			obkt := rbkt.Get(bucketKeyOptions)
			if obkt == nil {
				return nil
			}

			var any types.Any
			if err := proto.Unmarshal(obkt, &any); err != nil {
				return err
			}
			container.Runtime.Options = &any
		case string(bucketKeySpec):
			var any types.Any
			if err := proto.Unmarshal(v, &any); err != nil {
				return err
			}
			container.Spec = &any
		case string(bucketKeySnapshotKey):
			container.SnapshotKey = string(v)
		case string(bucketKeySnapshotter):
			container.Snapshotter = string(v)
		case string(bucketKeyExtensions):
			ebkt := bkt.Bucket(bucketKeyExtensions)
			if ebkt == nil {
				return nil
			}

			extensions := make(map[string]types.Any)
			if err := ebkt.ForEach(func(k, v []byte) error {
				var a types.Any
				if err := proto.Unmarshal(v, &a); err != nil {
					return err
				}

				extensions[string(k)] = a
				return nil
			}); err != nil {

				return err
			}

			container.Extensions = extensions
		}

		return nil
	})
}

func writeContainer(bkt *bolt.Bucket, container *containers.Container) error {
	if err := boltutil.WriteTimestamps(bkt, container.CreatedAt, container.UpdatedAt); err != nil {
		return err
	}

	if container.Spec != nil {
		spec, err := container.Spec.Marshal()
		if err != nil {
			return err
		}

		if err := bkt.Put(bucketKeySpec, spec); err != nil {
			return err
		}
	}

	for _, v := range [][2][]byte{
		{bucketKeyImage, []byte(container.Image)},
		{bucketKeySnapshotter, []byte(container.Snapshotter)},
		{bucketKeySnapshotKey, []byte(container.SnapshotKey)},
	} {
		if err := bkt.Put(v[0], v[1]); err != nil {
			return err
		}
	}

	if rbkt := bkt.Bucket(bucketKeyRuntime); rbkt != nil {
		if err := bkt.DeleteBucket(bucketKeyRuntime); err != nil {
			return err
		}
	}

	rbkt, err := bkt.CreateBucket(bucketKeyRuntime)
	if err != nil {
		return err
	}

	if err := rbkt.Put(bucketKeyName, []byte(container.Runtime.Name)); err != nil {
		return err
	}

	if len(container.Extensions) > 0 {
		ebkt, err := bkt.CreateBucketIfNotExists(bucketKeyExtensions)
		if err != nil {
			return err
		}

		for name, ext := range container.Extensions {
			p, err := proto.Marshal(&ext)
			if err != nil {
				return err
			}

			if err := ebkt.Put([]byte(name), p); err != nil {
				return err
			}
		}
	}

	if container.Runtime.Options != nil {
		data, err := proto.Marshal(container.Runtime.Options)
		if err != nil {
			return err
		}

		if err := rbkt.Put(bucketKeyOptions, data); err != nil {
			return err
		}
	}

	return boltutil.WriteLabels(bkt, container.Labels)
}
