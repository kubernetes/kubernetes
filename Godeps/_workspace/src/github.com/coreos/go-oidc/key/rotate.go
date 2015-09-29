package key

import (
	"errors"
	"time"

	"github.com/coreos/pkg/capnslog"
	ptime "github.com/coreos/pkg/timeutil"
	"github.com/jonboulle/clockwork"
)

var (
	log = capnslog.NewPackageLogger("github.com/coreos/go-oidc", "key")

	ErrorPrivateKeysExpired = errors.New("private keys have expired")
)

func NewPrivateKeyRotator(repo PrivateKeySetRepo, ttl time.Duration) *PrivateKeyRotator {
	return &PrivateKeyRotator{
		repo: repo,
		ttl:  ttl,

		keep:        2,
		generateKey: GeneratePrivateKey,
		clock:       clockwork.NewRealClock(),
	}
}

type PrivateKeyRotator struct {
	repo        PrivateKeySetRepo
	generateKey GeneratePrivateKeyFunc
	clock       clockwork.Clock
	keep        int
	ttl         time.Duration
}

func (r *PrivateKeyRotator) expiresAt() time.Time {
	return r.clock.Now().UTC().Add(r.ttl)
}

func (r *PrivateKeyRotator) Healthy() error {
	pks, err := r.privateKeySet()
	if err != nil {
		return err
	}

	if r.clock.Now().After(pks.ExpiresAt()) {
		return ErrorPrivateKeysExpired
	}

	return nil
}

func (r *PrivateKeyRotator) privateKeySet() (*PrivateKeySet, error) {
	ks, err := r.repo.Get()
	if err != nil {
		return nil, err
	}

	pks, ok := ks.(*PrivateKeySet)
	if !ok {
		return nil, errors.New("unable to cast to PrivateKeySet")
	}
	return pks, nil
}

func (r *PrivateKeyRotator) nextRotation() (time.Duration, error) {
	pks, err := r.privateKeySet()
	if err == ErrorNoKeys {
		log.Infof("No keys in private key set; must rotate immediately")
		return 0, nil
	}
	if err != nil {
		return 0, err
	}

	now := r.clock.Now()

	// Ideally, we want to rotate after half the TTL has elapsed.
	idealRotationTime := pks.ExpiresAt().Add(-r.ttl / 2)

	// If we are past the ideal rotation time, rotate immediatly.
	return max(0, idealRotationTime.Sub(now)), nil
}

func max(a, b time.Duration) time.Duration {
	if a > b {
		return a
	}
	return b
}

func (r *PrivateKeyRotator) Run() chan struct{} {
	attempt := func() {
		k, err := r.generateKey()
		if err != nil {
			log.Errorf("Failed generating signing key: %v", err)
			return
		}

		exp := r.expiresAt()
		if err := rotatePrivateKeys(r.repo, k, r.keep, exp); err != nil {
			log.Errorf("Failed key rotation: %v", err)
			return
		}

		log.Infof("Rotated signing keys: id=%s expiresAt=%s", k.ID(), exp)
	}

	stop := make(chan struct{})
	go func() {
		for {
			var nextRotation time.Duration
			var sleep time.Duration
			var err error
			for {
				if nextRotation, err = r.nextRotation(); err == nil {
					break
				}
				sleep = ptime.ExpBackoff(sleep, time.Minute)
				log.Errorf("error getting nextRotation, retrying in %v: %v", sleep, err)
				time.Sleep(sleep)
			}

			log.Infof("will rotate keys in %v", nextRotation)
			select {
			case <-r.clock.After(nextRotation):
				attempt()
			case <-stop:
				return
			}
		}
	}()

	return stop
}

func rotatePrivateKeys(repo PrivateKeySetRepo, k *PrivateKey, keep int, exp time.Time) error {
	ks, err := repo.Get()
	if err != nil && err != ErrorNoKeys {
		return err
	}

	var keys []*PrivateKey
	if ks != nil {
		pks, ok := ks.(*PrivateKeySet)
		if !ok {
			return errors.New("unable to cast to PrivateKeySet")
		}
		keys = pks.Keys()
	}

	keys = append([]*PrivateKey{k}, keys...)
	if l := len(keys); l > keep {
		keys = keys[0:keep]
	}

	nks := PrivateKeySet{
		keys:        keys,
		ActiveKeyID: k.ID(),
		expiresAt:   exp,
	}

	return repo.Set(KeySet(&nks))
}
