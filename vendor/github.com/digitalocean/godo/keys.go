package godo

import "fmt"

const keysBasePath = "v2/account/keys"

// KeysService is an interface for interfacing with the keys
// endpoints of the DigitalOcean API
// See: https://developers.digitalocean.com/documentation/v2#keys
type KeysService interface {
	List(*ListOptions) ([]Key, *Response, error)
	GetByID(int) (*Key, *Response, error)
	GetByFingerprint(string) (*Key, *Response, error)
	Create(*KeyCreateRequest) (*Key, *Response, error)
	UpdateByID(int, *KeyUpdateRequest) (*Key, *Response, error)
	UpdateByFingerprint(string, *KeyUpdateRequest) (*Key, *Response, error)
	DeleteByID(int) (*Response, error)
	DeleteByFingerprint(string) (*Response, error)
}

// KeysServiceOp handles communication with key related method of the
// DigitalOcean API.
type KeysServiceOp struct {
	client *Client
}

var _ KeysService = &KeysServiceOp{}

// Key represents a DigitalOcean Key.
type Key struct {
	ID          int    `json:"id,float64,omitempty"`
	Name        string `json:"name,omitempty"`
	Fingerprint string `json:"fingerprint,omitempty"`
	PublicKey   string `json:"public_key,omitempty"`
}

// KeyUpdateRequest represents a request to update a DigitalOcean key.
type KeyUpdateRequest struct {
	Name string `json:"name"`
}

type keysRoot struct {
	SSHKeys []Key  `json:"ssh_keys"`
	Links   *Links `json:"links"`
}

type keyRoot struct {
	SSHKey Key `json:"ssh_key"`
}

func (s Key) String() string {
	return Stringify(s)
}

// KeyCreateRequest represents a request to create a new key.
type KeyCreateRequest struct {
	Name      string `json:"name"`
	PublicKey string `json:"public_key"`
}

// List all keys
func (s *KeysServiceOp) List(opt *ListOptions) ([]Key, *Response, error) {
	path := keysBasePath
	path, err := addOptions(path, opt)
	if err != nil {
		return nil, nil, err
	}

	req, err := s.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(keysRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}
	if l := root.Links; l != nil {
		resp.Links = l
	}

	return root.SSHKeys, resp, err
}

// Performs a get given a path
func (s *KeysServiceOp) get(path string) (*Key, *Response, error) {
	req, err := s.client.NewRequest("GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(keyRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return &root.SSHKey, resp, err
}

// GetByID gets a Key by id
func (s *KeysServiceOp) GetByID(keyID int) (*Key, *Response, error) {
	if keyID < 1 {
		return nil, nil, NewArgError("keyID", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d", keysBasePath, keyID)
	return s.get(path)
}

// GetByFingerprint gets a Key by by fingerprint
func (s *KeysServiceOp) GetByFingerprint(fingerprint string) (*Key, *Response, error) {
	if len(fingerprint) < 1 {
		return nil, nil, NewArgError("fingerprint", "cannot not be empty")
	}

	path := fmt.Sprintf("%s/%s", keysBasePath, fingerprint)
	return s.get(path)
}

// Create a key using a KeyCreateRequest
func (s *KeysServiceOp) Create(createRequest *KeyCreateRequest) (*Key, *Response, error) {
	if createRequest == nil {
		return nil, nil, NewArgError("createRequest", "cannot be nil")
	}

	req, err := s.client.NewRequest("POST", keysBasePath, createRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(keyRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return &root.SSHKey, resp, err
}

// UpdateByID updates a key name by ID.
func (s *KeysServiceOp) UpdateByID(keyID int, updateRequest *KeyUpdateRequest) (*Key, *Response, error) {
	if keyID < 1 {
		return nil, nil, NewArgError("keyID", "cannot be less than 1")
	}

	if updateRequest == nil {
		return nil, nil, NewArgError("updateRequest", "cannot be nil")
	}

	path := fmt.Sprintf("%s/%d", keysBasePath, keyID)
	req, err := s.client.NewRequest("PUT", path, updateRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(keyRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return &root.SSHKey, resp, err
}

// UpdateByFingerprint updates a key name by fingerprint.
func (s *KeysServiceOp) UpdateByFingerprint(fingerprint string, updateRequest *KeyUpdateRequest) (*Key, *Response, error) {
	if len(fingerprint) < 1 {
		return nil, nil, NewArgError("fingerprint", "cannot be empty")
	}

	if updateRequest == nil {
		return nil, nil, NewArgError("updateRequest", "cannot be nil")
	}

	path := fmt.Sprintf("%s/%s", keysBasePath, fingerprint)
	req, err := s.client.NewRequest("PUT", path, updateRequest)
	if err != nil {
		return nil, nil, err
	}

	root := new(keyRoot)
	resp, err := s.client.Do(req, root)
	if err != nil {
		return nil, resp, err
	}

	return &root.SSHKey, resp, err
}

// Delete key using a path
func (s *KeysServiceOp) delete(path string) (*Response, error) {
	req, err := s.client.NewRequest("DELETE", path, nil)
	if err != nil {
		return nil, err
	}

	resp, err := s.client.Do(req, nil)

	return resp, err
}

// DeleteByID deletes a key by its id
func (s *KeysServiceOp) DeleteByID(keyID int) (*Response, error) {
	if keyID < 1 {
		return nil, NewArgError("keyID", "cannot be less than 1")
	}

	path := fmt.Sprintf("%s/%d", keysBasePath, keyID)
	return s.delete(path)
}

// DeleteByFingerprint deletes a key by its fingerprint
func (s *KeysServiceOp) DeleteByFingerprint(fingerprint string) (*Response, error) {
	if len(fingerprint) < 1 {
		return nil, NewArgError("fingerprint", "cannot be empty")
	}

	path := fmt.Sprintf("%s/%s", keysBasePath, fingerprint)
	return s.delete(path)
}
