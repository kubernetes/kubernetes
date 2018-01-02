package layer

import "errors"

// GetLayerPath returns the path to a layer
func GetLayerPath(s Store, layer ChainID) (string, error) {
	ls, ok := s.(*layerStore)
	if !ok {
		return "", errors.New("unsupported layer store")
	}
	ls.layerL.Lock()
	defer ls.layerL.Unlock()

	rl, ok := ls.layerMap[layer]
	if !ok {
		return "", ErrLayerDoesNotExist
	}

	path, err := ls.driver.Get(rl.cacheID, "")
	if err != nil {
		return "", err
	}

	if err := ls.driver.Put(rl.cacheID); err != nil {
		return "", err
	}

	return path, nil
}

func (ls *layerStore) mountID(name string) string {
	// windows has issues if container ID doesn't match mount ID
	return name
}
