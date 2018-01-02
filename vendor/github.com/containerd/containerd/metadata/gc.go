package metadata

import (
	"context"
	"fmt"
	"strings"

	"github.com/boltdb/bolt"
	"github.com/containerd/containerd/gc"
	"github.com/containerd/containerd/log"
	"github.com/pkg/errors"
)

const (
	// ResourceUnknown specifies an unknown resource
	ResourceUnknown gc.ResourceType = iota
	// ResourceContent specifies a content resource
	ResourceContent
	// ResourceSnapshot specifies a snapshot resource
	ResourceSnapshot
	// ResourceContainer specifies a container resource
	ResourceContainer
	// ResourceTask specifies a task resource
	ResourceTask
)

var (
	labelGCRoot       = []byte("containerd.io/gc.root")
	labelGCSnapRef    = []byte("containerd.io/gc.ref.snapshot.")
	labelGCContentRef = []byte("containerd.io/gc.ref.content")
)

func scanRoots(ctx context.Context, tx *bolt.Tx, nc chan<- gc.Node) error {
	v1bkt := tx.Bucket(bucketKeyVersion)
	if v1bkt == nil {
		return nil
	}

	// iterate through each namespace
	v1c := v1bkt.Cursor()

	for k, v := v1c.First(); k != nil; k, v = v1c.Next() {
		if v != nil {
			continue
		}
		nbkt := v1bkt.Bucket(k)
		ns := string(k)

		lbkt := nbkt.Bucket(bucketKeyObjectLeases)
		if lbkt != nil {
			if err := lbkt.ForEach(func(k, v []byte) error {
				if v != nil {
					return nil
				}
				libkt := lbkt.Bucket(k)

				cbkt := libkt.Bucket(bucketKeyObjectContent)
				if cbkt != nil {
					if err := cbkt.ForEach(func(k, v []byte) error {
						select {
						case nc <- gcnode(ResourceContent, ns, string(k)):
						case <-ctx.Done():
							return ctx.Err()
						}
						return nil
					}); err != nil {
						return err
					}
				}

				sbkt := libkt.Bucket(bucketKeyObjectSnapshots)
				if sbkt != nil {
					if err := sbkt.ForEach(func(sk, sv []byte) error {
						if sv != nil {
							return nil
						}
						snbkt := sbkt.Bucket(sk)

						return snbkt.ForEach(func(k, v []byte) error {
							select {
							case nc <- gcnode(ResourceSnapshot, ns, fmt.Sprintf("%s/%s", sk, k)):
							case <-ctx.Done():
								return ctx.Err()
							}
							return nil
						})
					}); err != nil {
						return err
					}
				}

				return nil
			}); err != nil {
				return err
			}
		}

		ibkt := nbkt.Bucket(bucketKeyObjectImages)
		if ibkt != nil {
			if err := ibkt.ForEach(func(k, v []byte) error {
				if v != nil {
					return nil
				}

				target := ibkt.Bucket(k).Bucket(bucketKeyTarget)
				if target != nil {
					contentKey := string(target.Get(bucketKeyDigest))
					select {
					case nc <- gcnode(ResourceContent, ns, contentKey):
					case <-ctx.Done():
						return ctx.Err()
					}
				}
				return sendSnapshotRefs(ns, ibkt.Bucket(k), func(n gc.Node) {
					select {
					case nc <- n:
					case <-ctx.Done():
					}
				})
			}); err != nil {
				return err
			}
		}

		cbkt := nbkt.Bucket(bucketKeyObjectContent)
		if cbkt != nil {
			cbkt = cbkt.Bucket(bucketKeyObjectBlob)
		}
		if cbkt != nil {
			if err := cbkt.ForEach(func(k, v []byte) error {
				if v != nil {
					return nil
				}
				return sendRootRef(ctx, nc, gcnode(ResourceContent, ns, string(k)), cbkt.Bucket(k))
			}); err != nil {
				return err
			}
		}

		cbkt = nbkt.Bucket(bucketKeyObjectContainers)
		if cbkt != nil {
			if err := cbkt.ForEach(func(k, v []byte) error {
				if v != nil {
					return nil
				}
				snapshotter := string(cbkt.Bucket(k).Get(bucketKeySnapshotter))
				if snapshotter != "" {
					ss := string(cbkt.Bucket(k).Get(bucketKeySnapshotKey))
					select {
					case nc <- gcnode(ResourceSnapshot, ns, fmt.Sprintf("%s/%s", snapshotter, ss)):
					case <-ctx.Done():
						return ctx.Err()
					}
				}

				// TODO: Send additional snapshot refs through labels
				return sendSnapshotRefs(ns, cbkt.Bucket(k), func(n gc.Node) {
					select {
					case nc <- n:
					case <-ctx.Done():
					}
				})
			}); err != nil {
				return err
			}
		}

		sbkt := nbkt.Bucket(bucketKeyObjectSnapshots)
		if sbkt != nil {
			if err := sbkt.ForEach(func(sk, sv []byte) error {
				if sv != nil {
					return nil
				}
				snbkt := sbkt.Bucket(sk)

				return snbkt.ForEach(func(k, v []byte) error {
					if v != nil {
						return nil
					}

					return sendRootRef(ctx, nc, gcnode(ResourceSnapshot, ns, fmt.Sprintf("%s/%s", sk, k)), snbkt.Bucket(k))
				})
			}); err != nil {
				return err
			}
		}
	}
	return nil
}

func references(ctx context.Context, tx *bolt.Tx, node gc.Node, fn func(gc.Node)) error {
	if node.Type == ResourceContent {
		bkt := getBucket(tx, bucketKeyVersion, []byte(node.Namespace), bucketKeyObjectContent, bucketKeyObjectBlob, []byte(node.Key))
		if bkt == nil {
			// Node may be created from dead edge
			return nil
		}

		if err := sendSnapshotRefs(node.Namespace, bkt, fn); err != nil {
			return err
		}
		return sendContentRefs(node.Namespace, bkt, fn)
	} else if node.Type == ResourceSnapshot {
		parts := strings.SplitN(node.Key, "/", 2)
		if len(parts) != 2 {
			return errors.Errorf("invalid snapshot gc key %s", node.Key)
		}
		ss := parts[0]
		name := parts[1]

		bkt := getBucket(tx, bucketKeyVersion, []byte(node.Namespace), bucketKeyObjectSnapshots, []byte(ss), []byte(name))
		if bkt == nil {
			getBucket(tx, bucketKeyVersion, []byte(node.Namespace), bucketKeyObjectSnapshots).ForEach(func(k, v []byte) error {
				return nil
			})

			// Node may be created from dead edge
			return nil
		}

		if pv := bkt.Get(bucketKeyParent); len(pv) > 0 {
			fn(gcnode(ResourceSnapshot, node.Namespace, fmt.Sprintf("%s/%s", ss, pv)))
		}

		return sendSnapshotRefs(node.Namespace, bkt, fn)
	}

	return nil
}

func scanAll(ctx context.Context, tx *bolt.Tx, fn func(ctx context.Context, n gc.Node) error) error {
	v1bkt := tx.Bucket(bucketKeyVersion)
	if v1bkt == nil {
		return nil
	}

	// iterate through each namespace
	v1c := v1bkt.Cursor()

	for k, v := v1c.First(); k != nil; k, v = v1c.Next() {
		if v != nil {
			continue
		}
		nbkt := v1bkt.Bucket(k)
		ns := string(k)

		sbkt := nbkt.Bucket(bucketKeyObjectSnapshots)
		if sbkt != nil {
			if err := sbkt.ForEach(func(sk, sv []byte) error {
				if sv != nil {
					return nil
				}
				snbkt := sbkt.Bucket(sk)
				return snbkt.ForEach(func(k, v []byte) error {
					if v != nil {
						return nil
					}
					node := gcnode(ResourceSnapshot, ns, fmt.Sprintf("%s/%s", sk, k))
					return fn(ctx, node)
				})
			}); err != nil {
				return err
			}
		}

		cbkt := nbkt.Bucket(bucketKeyObjectContent)
		if cbkt != nil {
			cbkt = cbkt.Bucket(bucketKeyObjectBlob)
		}
		if cbkt != nil {
			if err := cbkt.ForEach(func(k, v []byte) error {
				if v != nil {
					return nil
				}
				node := gcnode(ResourceContent, ns, string(k))
				return fn(ctx, node)
			}); err != nil {
				return err
			}
		}
	}

	return nil
}

func remove(ctx context.Context, tx *bolt.Tx, node gc.Node) error {
	v1bkt := tx.Bucket(bucketKeyVersion)
	if v1bkt == nil {
		return nil
	}

	nsbkt := v1bkt.Bucket([]byte(node.Namespace))
	if nsbkt == nil {
		return nil
	}

	switch node.Type {
	case ResourceContent:
		cbkt := nsbkt.Bucket(bucketKeyObjectContent)
		if cbkt != nil {
			cbkt = cbkt.Bucket(bucketKeyObjectBlob)
		}
		if cbkt != nil {
			log.G(ctx).WithField("key", node.Key).Debug("delete content")
			return cbkt.DeleteBucket([]byte(node.Key))
		}
	case ResourceSnapshot:
		sbkt := nsbkt.Bucket(bucketKeyObjectSnapshots)
		if sbkt != nil {
			parts := strings.SplitN(node.Key, "/", 2)
			if len(parts) != 2 {
				return errors.Errorf("invalid snapshot gc key %s", node.Key)
			}
			ssbkt := sbkt.Bucket([]byte(parts[0]))
			if ssbkt != nil {
				log.G(ctx).WithField("key", parts[1]).WithField("snapshotter", parts[0]).Debug("delete snapshot")
				return ssbkt.DeleteBucket([]byte(parts[1]))
			}
		}
	}

	return nil
}

// sendSnapshotRefs sends all snapshot references referred to by the labels in the bkt
func sendSnapshotRefs(ns string, bkt *bolt.Bucket, fn func(gc.Node)) error {
	lbkt := bkt.Bucket(bucketKeyObjectLabels)
	if lbkt != nil {
		lc := lbkt.Cursor()

		for k, v := lc.Seek(labelGCSnapRef); k != nil && strings.HasPrefix(string(k), string(labelGCSnapRef)); k, v = lc.Next() {
			snapshotter := string(k[len(labelGCSnapRef):])
			fn(gcnode(ResourceSnapshot, ns, fmt.Sprintf("%s/%s", snapshotter, v)))
		}
	}
	return nil
}

// sendContentRefs sends all content references referred to by the labels in the bkt
func sendContentRefs(ns string, bkt *bolt.Bucket, fn func(gc.Node)) error {
	lbkt := bkt.Bucket(bucketKeyObjectLabels)
	if lbkt != nil {
		lc := lbkt.Cursor()

		labelRef := string(labelGCContentRef)
		for k, v := lc.Seek(labelGCContentRef); k != nil && strings.HasPrefix(string(k), labelRef); k, v = lc.Next() {
			if ks := string(k); ks != labelRef {
				// Allow reference naming, ignore names
				if ks[len(labelRef)] != '.' {
					continue
				}
			}

			fn(gcnode(ResourceContent, ns, string(v)))
		}
	}
	return nil
}

func isRootRef(bkt *bolt.Bucket) bool {
	lbkt := bkt.Bucket(bucketKeyObjectLabels)
	if lbkt != nil {
		rv := lbkt.Get(labelGCRoot)
		if rv != nil {
			// TODO: interpret rv as a timestamp and skip if expired
			return true
		}
	}
	return false
}

func sendRootRef(ctx context.Context, nc chan<- gc.Node, n gc.Node, bkt *bolt.Bucket) error {
	if isRootRef(bkt) {
		select {
		case nc <- n:
		case <-ctx.Done():
			return ctx.Err()
		}
	}
	return nil
}

func gcnode(t gc.ResourceType, ns, key string) gc.Node {
	return gc.Node{
		Type:      t,
		Namespace: ns,
		Key:       key,
	}
}
