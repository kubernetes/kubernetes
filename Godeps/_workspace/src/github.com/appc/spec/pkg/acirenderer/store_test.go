package acirenderer

import (
	"bytes"
	"fmt"
	"hash"
	"io"
	"io/ioutil"
	"os"

	"github.com/appc/spec/aci"
	"github.com/appc/spec/schema"
	"github.com/appc/spec/schema/types"
)

const (
	hashPrefix = "sha512-"
)

type TestStoreAci struct {
	data          []byte
	key           string
	ImageManifest *schema.ImageManifest
}

type TestStore struct {
	acis map[string]*TestStoreAci
}

func NewTestStore() *TestStore {
	return &TestStore{acis: make(map[string]*TestStoreAci)}
}

func (ts *TestStore) WriteACI(path string) (string, error) {
	data, err := ioutil.ReadFile(path)
	if err != nil {
		return "", err
	}
	imageID := types.NewHashSHA512(data)

	rs, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer rs.Close()
	im, err := aci.ManifestFromImage(rs)
	if err != nil {
		return "", fmt.Errorf("error retrieving ImageManifest: %v", err)
	}

	key := imageID.String()
	ts.acis[key] = &TestStoreAci{data: data, key: key, ImageManifest: im}
	return key, nil
}

func (ts *TestStore) GetImageManifest(key string) (*schema.ImageManifest, error) {
	aci, ok := ts.acis[key]
	if !ok {
		return nil, fmt.Errorf("aci with key: %s not found", key)
	}
	return aci.ImageManifest, nil

}
func (ts *TestStore) GetACI(name types.ACName, labels types.Labels) (string, error) {
	for _, aci := range ts.acis {
		if aci.ImageManifest.Name.String() == name.String() {
			return aci.key, nil
		}
	}
	return "", fmt.Errorf("aci not found")
}

func (ts *TestStore) ReadStream(key string) (io.ReadCloser, error) {
	aci, ok := ts.acis[key]
	if !ok {
		return nil, fmt.Errorf("stream for key: %s not found", key)
	}
	return ioutil.NopCloser(bytes.NewReader(aci.data)), nil
}

func (ts *TestStore) ResolveKey(key string) (string, error) {
	return key, nil
}

// HashToKey takes a hash.Hash (which currently _MUST_ represent a full SHA512),
// calculates its sum, and returns a string which should be used as the key to
// store the data matching the hash.
func (ts *TestStore) HashToKey(h hash.Hash) string {
	s := h.Sum(nil)
	return fmt.Sprintf("%s%x", hashPrefix, s)
}
