package key

import (
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/jonboulle/clockwork"
)

type staticReadableKeySetRepo struct {
	ks  KeySet
	err error
}

func (r *staticReadableKeySetRepo) Get() (KeySet, error) {
	return r.ks, r.err
}

func TestKeySyncerSync(t *testing.T) {
	fc := clockwork.NewFakeClock()
	now := fc.Now().UTC()

	k1 := generatePrivateKeyStatic(t, 1)
	k2 := generatePrivateKeyStatic(t, 2)
	k3 := generatePrivateKeyStatic(t, 3)

	steps := []struct {
		fromKS  KeySet
		fromErr error
		advance time.Duration
		want    *PrivateKeySet
	}{
		// on startup, first sync should trigger within a second
		{
			fromKS: &PrivateKeySet{
				keys:        []*PrivateKey{k1},
				ActiveKeyID: k1.KeyID,
				expiresAt:   now.Add(10 * time.Second),
			},
			advance: time.Second,
			want: &PrivateKeySet{
				keys:        []*PrivateKey{k1},
				ActiveKeyID: k1.KeyID,
				expiresAt:   now.Add(10 * time.Second),
			},
		},
		// advance halfway into TTL, triggering sync
		{
			fromKS: &PrivateKeySet{
				keys:        []*PrivateKey{k2, k1},
				ActiveKeyID: k2.KeyID,
				expiresAt:   now.Add(15 * time.Second),
			},
			advance: 5 * time.Second,
			want: &PrivateKeySet{
				keys:        []*PrivateKey{k2, k1},
				ActiveKeyID: k2.KeyID,
				expiresAt:   now.Add(15 * time.Second),
			},
		},

		// advance halfway into TTL, triggering sync that fails
		{
			fromErr: errors.New("fail!"),
			advance: 10 * time.Second,
			want: &PrivateKeySet{
				keys:        []*PrivateKey{k2, k1},
				ActiveKeyID: k2.KeyID,
				expiresAt:   now.Add(15 * time.Second),
			},
		},

		// sync retries quickly, and succeeds with fixed data
		{
			fromKS: &PrivateKeySet{
				keys:        []*PrivateKey{k3, k2, k1},
				ActiveKeyID: k3.KeyID,
				expiresAt:   now.Add(25 * time.Second),
			},
			advance: 3 * time.Second,
			want: &PrivateKeySet{
				keys:        []*PrivateKey{k3, k2, k1},
				ActiveKeyID: k3.KeyID,
				expiresAt:   now.Add(25 * time.Second),
			},
		},
	}

	from := &staticReadableKeySetRepo{}
	to := NewPrivateKeySetRepo()

	syncer := NewKeySetSyncer(from, to)
	syncer.clock = fc
	stop := syncer.Run()
	defer close(stop)

	for i, st := range steps {
		from.ks = st.fromKS
		from.err = st.fromErr

		fc.Advance(st.advance)
		fc.BlockUntil(1)

		ks, err := to.Get()
		if err != nil {
			t.Fatalf("step %d: unable to get keys: %v", i, err)
		}
		if !reflect.DeepEqual(st.want, ks) {
			t.Fatalf("step %d: incorrect state: want=%#v got=%#v", i, st.want, ks)
		}
	}
}

func TestSync(t *testing.T) {
	fc := clockwork.NewFakeClock()
	now := fc.Now().UTC()

	k1 := generatePrivateKeyStatic(t, 1)
	k2 := generatePrivateKeyStatic(t, 2)
	k3 := generatePrivateKeyStatic(t, 3)

	tests := []struct {
		keySet *PrivateKeySet
		want   time.Duration
	}{
		{
			keySet: &PrivateKeySet{
				keys:        []*PrivateKey{k1},
				ActiveKeyID: k1.KeyID,
				expiresAt:   now.Add(time.Minute),
			},
			want: time.Minute,
		},
		{
			keySet: &PrivateKeySet{
				keys:        []*PrivateKey{k2, k1},
				ActiveKeyID: k2.KeyID,
				expiresAt:   now.Add(time.Minute),
			},
			want: time.Minute,
		},
		{
			keySet: &PrivateKeySet{
				keys:        []*PrivateKey{k3, k2, k1},
				ActiveKeyID: k2.KeyID,
				expiresAt:   now.Add(time.Minute),
			},
			want: time.Minute,
		},
		{
			keySet: &PrivateKeySet{
				keys:        []*PrivateKey{k2, k1},
				ActiveKeyID: k2.KeyID,
				expiresAt:   now.Add(time.Hour),
			},
			want: time.Hour,
		},
		{
			keySet: &PrivateKeySet{
				keys:        []*PrivateKey{k1},
				ActiveKeyID: k1.KeyID,
				expiresAt:   now.Add(-time.Hour),
			},
			want: 0,
		},
	}

	for i, tt := range tests {
		from := NewPrivateKeySetRepo()
		to := NewPrivateKeySetRepo()

		err := from.Set(tt.keySet)
		if err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
			continue
		}
		exp, err := sync(from, to, fc)
		if err != nil {
			t.Errorf("case %d: unexpected error: %v", i, err)
			continue
		}

		if tt.want != exp {
			t.Errorf("case %d: want=%v got=%v", i, tt.want, exp)
		}
	}
}

func TestSyncFail(t *testing.T) {
	tests := []error{
		nil,
		errors.New("fail!"),
	}

	for i, tt := range tests {
		from := &staticReadableKeySetRepo{ks: nil, err: tt}
		to := NewPrivateKeySetRepo()

		if _, err := sync(from, to, clockwork.NewFakeClock()); err == nil {
			t.Errorf("case %d: expected non-nil error", i)
		}
	}
}
