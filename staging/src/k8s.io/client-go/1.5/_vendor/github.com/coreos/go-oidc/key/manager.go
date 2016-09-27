package key

import (
	"errors"
	"time"

	"github.com/jonboulle/clockwork"

	"github.com/coreos/go-oidc/jose"
	"github.com/coreos/pkg/health"
)

type PrivateKeyManager interface {
	ExpiresAt() time.Time
	Signer() (jose.Signer, error)
	JWKs() ([]jose.JWK, error)
	PublicKeys() ([]PublicKey, error)

	WritableKeySetRepo
	health.Checkable
}

func NewPrivateKeyManager() PrivateKeyManager {
	return &privateKeyManager{
		clock: clockwork.NewRealClock(),
	}
}

type privateKeyManager struct {
	keySet *PrivateKeySet
	clock  clockwork.Clock
}

func (m *privateKeyManager) ExpiresAt() time.Time {
	if m.keySet == nil {
		return m.clock.Now().UTC()
	}

	return m.keySet.ExpiresAt()
}

func (m *privateKeyManager) Signer() (jose.Signer, error) {
	if err := m.Healthy(); err != nil {
		return nil, err
	}

	return m.keySet.Active().Signer(), nil
}

func (m *privateKeyManager) JWKs() ([]jose.JWK, error) {
	if err := m.Healthy(); err != nil {
		return nil, err
	}

	keys := m.keySet.Keys()
	jwks := make([]jose.JWK, len(keys))
	for i, k := range keys {
		jwks[i] = k.JWK()
	}
	return jwks, nil
}

func (m *privateKeyManager) PublicKeys() ([]PublicKey, error) {
	jwks, err := m.JWKs()
	if err != nil {
		return nil, err
	}
	keys := make([]PublicKey, len(jwks))
	for i, jwk := range jwks {
		keys[i] = *NewPublicKey(jwk)
	}
	return keys, nil
}

func (m *privateKeyManager) Healthy() error {
	if m.keySet == nil {
		return errors.New("private key manager uninitialized")
	}

	if len(m.keySet.Keys()) == 0 {
		return errors.New("private key manager zero keys")
	}

	if m.keySet.ExpiresAt().Before(m.clock.Now().UTC()) {
		return errors.New("private key manager keys expired")
	}

	return nil
}

func (m *privateKeyManager) Set(keySet KeySet) error {
	privKeySet, ok := keySet.(*PrivateKeySet)
	if !ok {
		return errors.New("unable to cast to PrivateKeySet")
	}

	m.keySet = privKeySet
	return nil
}
