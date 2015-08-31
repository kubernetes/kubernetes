package oidc

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"testing"
	"time"

	"github.com/jonboulle/clockwork"

	phttp "github.com/coreos/go-oidc/http"
	"github.com/coreos/go-oidc/oauth2"
)

type fakeProviderConfigGetterSetter struct {
	cfg      *ProviderConfig
	getCount int
	setCount int
}

func (g *fakeProviderConfigGetterSetter) Get() (ProviderConfig, error) {
	g.getCount++
	return *g.cfg, nil
}

func (g *fakeProviderConfigGetterSetter) Set(cfg ProviderConfig) error {
	g.cfg = &cfg
	g.setCount++
	return nil
}

type fakeProviderConfigHandler struct {
	cfg    ProviderConfig
	maxAge time.Duration
}

func (s *fakeProviderConfigHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	b, _ := json.Marshal(s.cfg)
	if s.maxAge.Seconds() >= 0 {
		w.Header().Set("Cache-Control", fmt.Sprintf("public, max-age=%d", int(s.maxAge.Seconds())))
	}
	w.Header().Set("Content-Type", "application/json")
	w.Write(b)
}

func TestHTTPProviderConfigGetter(t *testing.T) {
	svr := &fakeProviderConfigHandler{}
	hc := &phttp.HandlerClient{Handler: svr}
	fc := clockwork.NewFakeClock()
	now := fc.Now().UTC()

	tests := []struct {
		dsc string
		age time.Duration
		cfg ProviderConfig
		ok  bool
	}{
		// everything is good
		{
			dsc: "https://example.com",
			age: time.Minute,
			cfg: ProviderConfig{
				Issuer:    "https://example.com",
				ExpiresAt: now.Add(time.Minute),
			},
			ok: true,
		},
		// iss and disco url differ by scheme only (how google works)
		{
			dsc: "https://example.com",
			age: time.Minute,
			cfg: ProviderConfig{
				Issuer:    "example.com",
				ExpiresAt: now.Add(time.Minute),
			},
			ok: true,
		},
		// issuer and discovery URL mismatch
		{
			dsc: "https://foo.com",
			age: time.Minute,
			cfg: ProviderConfig{
				Issuer:    "https://example.com",
				ExpiresAt: now.Add(time.Minute),
			},
			ok: false,
		},
		// missing cache header results in zero ExpiresAt
		{
			dsc: "https://example.com",
			age: -1,
			cfg: ProviderConfig{
				Issuer: "https://example.com",
			},
			ok: true,
		},
	}

	for i, tt := range tests {
		svr.cfg = tt.cfg
		svr.maxAge = tt.age
		getter := NewHTTPProviderConfigGetter(hc, tt.dsc)
		getter.clock = fc

		got, err := getter.Get()
		if err != nil {
			if tt.ok {
				t.Fatalf("test %d: unexpected error: %v", i, err)
			}
			continue
		}

		if !tt.ok {
			t.Fatalf("test %d: expected error", i)
			continue
		}

		if !reflect.DeepEqual(tt.cfg, got) {
			t.Fatalf("test %d: want: %#v, got: %#v", i, tt.cfg, got)
		}
	}
}

func TestProviderConfigSyncerRun(t *testing.T) {
	c1 := &ProviderConfig{
		Issuer: "http://first.example.com",
	}
	c2 := &ProviderConfig{
		Issuer: "http://second.example.com",
	}

	tests := []struct {
		first     *ProviderConfig
		advance   time.Duration
		second    *ProviderConfig
		firstExp  time.Duration
		secondExp time.Duration
		count     int
	}{
		// exp is 10m, should have same config after 1s
		{
			first:     c1,
			firstExp:  time.Duration(10 * time.Minute),
			advance:   time.Minute,
			second:    c1,
			secondExp: time.Duration(10 * time.Minute),
			count:     1,
		},
		// exp is 10m, should have new config after 10/2 = 5m
		{
			first:     c1,
			firstExp:  time.Duration(10 * time.Minute),
			advance:   time.Duration(5 * time.Minute),
			second:    c2,
			secondExp: time.Duration(10 * time.Minute),
			count:     2,
		},
		// exp is 20m, should have new config after 20/2 = 10m
		{
			first:     c1,
			firstExp:  time.Duration(20 * time.Minute),
			advance:   time.Duration(10 * time.Minute),
			second:    c2,
			secondExp: time.Duration(30 * time.Minute),
			count:     2,
		},
	}

	assertCfg := func(i int, to *fakeProviderConfigGetterSetter, want ProviderConfig) {
		got, err := to.Get()
		if err != nil {
			t.Fatalf("test %d: unable to get config: %v", i, err)
		}
		if !reflect.DeepEqual(want, got) {
			t.Fatalf("test %d: incorrect state:\nwant=%#v\ngot=%#v", i, want, got)
		}
	}

	for i, tt := range tests {
		from := &fakeProviderConfigGetterSetter{}
		to := &fakeProviderConfigGetterSetter{}

		fc := clockwork.NewFakeClock()
		now := fc.Now().UTC()
		syncer := NewProviderConfigSyncer(from, to)
		syncer.clock = fc

		tt.first.ExpiresAt = now.Add(tt.firstExp)
		tt.second.ExpiresAt = now.Add(tt.secondExp)
		if err := from.Set(*tt.first); err != nil {
			t.Fatalf("test %d: unexpected error: %v", i, err)
		}

		stop := syncer.Run()
		defer close(stop)
		fc.BlockUntil(1)

		// first sync
		assertCfg(i, to, *tt.first)

		if err := from.Set(*tt.second); err != nil {
			t.Fatalf("test %d: unexpected error: %v", i, err)
		}

		fc.Advance(tt.advance)
		fc.BlockUntil(1)

		// second sync
		assertCfg(i, to, *tt.second)

		if tt.count != from.getCount {
			t.Fatalf("test %d: want: %v, got: %v", i, tt.count, from.getCount)
		}
	}
}

type staticProviderConfigGetter struct {
	cfg ProviderConfig
	err error
}

func (g *staticProviderConfigGetter) Get() (ProviderConfig, error) {
	return g.cfg, g.err
}

type staticProviderConfigSetter struct {
	cfg *ProviderConfig
	err error
}

func (s *staticProviderConfigSetter) Set(cfg ProviderConfig) error {
	s.cfg = &cfg
	return s.err
}

func TestProviderConfigSyncerSyncFailure(t *testing.T) {
	fc := clockwork.NewFakeClock()

	tests := []struct {
		from *staticProviderConfigGetter
		to   *staticProviderConfigSetter

		// want indicates what ProviderConfig should be passed to Set.
		// If nil, the Set should not be called.
		want *ProviderConfig
	}{
		// generic Get failure
		{
			from: &staticProviderConfigGetter{err: errors.New("fail")},
			to:   &staticProviderConfigSetter{},
			want: nil,
		},
		// generic Set failure
		{
			from: &staticProviderConfigGetter{cfg: ProviderConfig{ExpiresAt: fc.Now().Add(time.Minute)}},
			to:   &staticProviderConfigSetter{err: errors.New("fail")},
			want: &ProviderConfig{ExpiresAt: fc.Now().Add(time.Minute)},
		},
	}

	for i, tt := range tests {
		pcs := &ProviderConfigSyncer{
			from:  tt.from,
			to:    tt.to,
			clock: fc,
		}
		_, err := pcs.sync()
		if err == nil {
			t.Errorf("case %d: expected non-nil error", i)
		}
		if !reflect.DeepEqual(tt.want, tt.to.cfg) {
			t.Errorf("case %d: Set mismatch: want=%#v got=%#v", i, tt.want, tt.to.cfg)
		}
	}
}

func TestNextSyncAfter(t *testing.T) {
	fc := clockwork.NewFakeClock()

	tests := []struct {
		exp  time.Time
		want time.Duration
	}{
		{
			exp:  fc.Now().Add(time.Hour),
			want: 30 * time.Minute,
		},
		// override large values with the maximum
		{
			exp:  fc.Now().Add(168 * time.Hour), // one week
			want: 24 * time.Hour,
		},
		// override "now" values with the minimum
		{
			exp:  fc.Now(),
			want: time.Minute,
		},
		// override negative values with the minimum
		{
			exp:  fc.Now().Add(-1 * time.Minute),
			want: time.Minute,
		},
		// zero-value Time results in maximum sync interval
		{
			exp:  time.Time{},
			want: 24 * time.Hour,
		},
	}

	for i, tt := range tests {
		got := nextSyncAfter(tt.exp, fc)
		if tt.want != got {
			t.Errorf("case %d: want=%v got=%v", i, tt.want, got)
		}
	}
}

func TestProviderConfigEmpty(t *testing.T) {
	cfg := ProviderConfig{}
	if !cfg.Empty() {
		t.Fatalf("Empty provider config reports non-empty")
	}
	cfg = ProviderConfig{Issuer: "http://example.com"}
	if cfg.Empty() {
		t.Fatalf("Non-empty provider config reports empty")
	}
}

func TestPCSStepAfter(t *testing.T) {
	pass := func() (time.Duration, error) { return 7 * time.Second, nil }
	fail := func() (time.Duration, error) { return 0, errors.New("fail") }

	tests := []struct {
		stepper  pcsStepper
		stepFunc pcsStepFunc
		want     pcsStepper
	}{
		// good step results in retry at TTL
		{
			stepper:  &pcsStepNext{},
			stepFunc: pass,
			want:     &pcsStepNext{aft: 7 * time.Second},
		},

		// good step after failed step results results in retry at TTL
		{
			stepper:  &pcsStepRetry{aft: 2 * time.Second},
			stepFunc: pass,
			want:     &pcsStepNext{aft: 7 * time.Second},
		},

		// failed step results in a retry in 1s
		{
			stepper:  &pcsStepNext{},
			stepFunc: fail,
			want:     &pcsStepRetry{aft: time.Second},
		},

		// failed retry backs off by a factor of 2
		{
			stepper:  &pcsStepRetry{aft: time.Second},
			stepFunc: fail,
			want:     &pcsStepRetry{aft: 2 * time.Second},
		},

		// failed retry backs off by a factor of 2, up to 1m
		{
			stepper:  &pcsStepRetry{aft: 32 * time.Second},
			stepFunc: fail,
			want:     &pcsStepRetry{aft: 60 * time.Second},
		},
	}

	for i, tt := range tests {
		got := tt.stepper.step(tt.stepFunc)
		if !reflect.DeepEqual(tt.want, got) {
			t.Errorf("case %d: want=%#v got=%#v", i, tt.want, got)
		}
	}
}

func TestProviderConfigSupportsGrantType(t *testing.T) {
	tests := []struct {
		types []string
		typ   string
		want  bool
	}{
		// explicitly supported
		{
			types: []string{"foo_type"},
			typ:   "foo_type",
			want:  true,
		},

		// explicitly unsupported
		{
			types: []string{"bar_type"},
			typ:   "foo_type",
			want:  false,
		},

		// default type explicitly unsupported
		{
			types: []string{oauth2.GrantTypeImplicit},
			typ:   oauth2.GrantTypeAuthCode,
			want:  false,
		},

		// type not found in default set
		{
			types: []string{},
			typ:   "foo_type",
			want:  false,
		},

		// type found in default set
		{
			types: []string{},
			typ:   oauth2.GrantTypeAuthCode,
			want:  true,
		},
	}

	for i, tt := range tests {
		cfg := ProviderConfig{
			GrantTypesSupported: tt.types,
		}
		got := cfg.SupportsGrantType(tt.typ)
		if tt.want != got {
			t.Errorf("case %d: assert %v supports %v: want=%t got=%t", i, tt.types, tt.typ, tt.want, got)
		}
	}
}

func TestWaitForProviderConfigImmediateSuccess(t *testing.T) {
	cfg := ProviderConfig{Issuer: "http://example.com"}
	b, err := json.Marshal(cfg)
	if err != nil {
		t.Fatalf("Failed marshaling provider config")
	}

	resp := http.Response{Body: ioutil.NopCloser(bytes.NewBuffer(b))}
	hc := &phttp.RequestRecorder{Response: &resp}
	fc := clockwork.NewFakeClock()

	reschan := make(chan ProviderConfig)
	go func() {
		reschan <- waitForProviderConfig(hc, cfg.Issuer, fc)
	}()

	var got ProviderConfig
	select {
	case got = <-reschan:
	case <-time.After(time.Second):
		t.Fatalf("Did not receive result within 1s")
	}

	if !reflect.DeepEqual(cfg, got) {
		t.Fatalf("Received incorrect provider config: want=%#v got=%#v", cfg, got)
	}
}
