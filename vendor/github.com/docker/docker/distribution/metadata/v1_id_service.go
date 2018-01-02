package metadata

import (
	"github.com/docker/docker/image/v1"
	"github.com/docker/docker/layer"
	"github.com/pkg/errors"
)

// V1IDService maps v1 IDs to layers on disk.
type V1IDService struct {
	store Store
}

// NewV1IDService creates a new V1 ID mapping service.
func NewV1IDService(store Store) *V1IDService {
	return &V1IDService{
		store: store,
	}
}

// namespace returns the namespace used by this service.
func (idserv *V1IDService) namespace() string {
	return "v1id"
}

// Get finds a layer by its V1 ID.
func (idserv *V1IDService) Get(v1ID, registry string) (layer.DiffID, error) {
	if idserv.store == nil {
		return "", errors.New("no v1IDService storage")
	}
	if err := v1.ValidateID(v1ID); err != nil {
		return layer.DiffID(""), err
	}

	idBytes, err := idserv.store.Get(idserv.namespace(), registry+","+v1ID)
	if err != nil {
		return layer.DiffID(""), err
	}
	return layer.DiffID(idBytes), nil
}

// Set associates an image with a V1 ID.
func (idserv *V1IDService) Set(v1ID, registry string, id layer.DiffID) error {
	if idserv.store == nil {
		return nil
	}
	if err := v1.ValidateID(v1ID); err != nil {
		return err
	}
	return idserv.store.Set(idserv.namespace(), registry+","+v1ID, []byte(id))
}
