package oidc

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/coreos/pkg/capnslog"
	"github.com/coreos/pkg/timeutil"
	"github.com/jonboulle/clockwork"

	phttp "github.com/coreos/go-oidc/http"
	"github.com/coreos/go-oidc/oauth2"
)

var (
	log = capnslog.NewPackageLogger("github.com/coreos/go-oidc", "http")
)

const (
	MaximumProviderConfigSyncInterval = 24 * time.Hour
	MinimumProviderConfigSyncInterval = time.Minute

	discoveryConfigPath = "/.well-known/openid-configuration"
)

type ProviderConfig struct {
	Issuer                            string    `json:"issuer"`
	AuthEndpoint                      string    `json:"authorization_endpoint"`
	TokenEndpoint                     string    `json:"token_endpoint"`
	KeysEndpoint                      string    `json:"jwks_uri"`
	ResponseTypesSupported            []string  `json:"response_types_supported"`
	GrantTypesSupported               []string  `json:"grant_types_supported"`
	SubjectTypesSupported             []string  `json:"subject_types_supported"`
	IDTokenAlgValuesSupported         []string  `json:"id_token_alg_values_supported"`
	TokenEndpointAuthMethodsSupported []string  `json:"token_endpoint_auth_methods_supported"`
	ExpiresAt                         time.Time `json:"-"`
}

func (p ProviderConfig) Empty() bool {
	return p.Issuer == ""
}

func (p ProviderConfig) SupportsGrantType(grantType string) bool {
	var supported []string
	if len(p.GrantTypesSupported) == 0 {
		// If omitted, the default value is ["authorization_code", "implicit"].
		// http://openid.net/specs/openid-connect-discovery-1_0.html#ProviderMetadata
		supported = []string{oauth2.GrantTypeAuthCode, oauth2.GrantTypeImplicit}
	} else {
		supported = p.GrantTypesSupported
	}

	for _, t := range supported {
		if t == grantType {
			return true
		}
	}
	return false
}

type ProviderConfigGetter interface {
	Get() (ProviderConfig, error)
}

type ProviderConfigSetter interface {
	Set(ProviderConfig) error
}

type ProviderConfigSyncer struct {
	from  ProviderConfigGetter
	to    ProviderConfigSetter
	clock clockwork.Clock
}

func NewProviderConfigSyncer(from ProviderConfigGetter, to ProviderConfigSetter) *ProviderConfigSyncer {
	return &ProviderConfigSyncer{
		from:  from,
		to:    to,
		clock: clockwork.NewRealClock(),
	}
}

func (s *ProviderConfigSyncer) Run() chan struct{} {
	stop := make(chan struct{})

	var next pcsStepper
	next = &pcsStepNext{aft: time.Duration(0)}

	go func() {
		for {
			select {
			case <-s.clock.After(next.after()):
				next = next.step(s.sync)
			case <-stop:
				return
			}
		}
	}()

	return stop
}

func (s *ProviderConfigSyncer) sync() (time.Duration, error) {
	cfg, err := s.from.Get()
	if err != nil {
		return 0, err
	}

	if err = s.to.Set(cfg); err != nil {
		return 0, fmt.Errorf("error setting provider config: %v", err)
	}

	log.Infof("Updating provider config: config=%#v", cfg)

	return nextSyncAfter(cfg.ExpiresAt, s.clock), nil
}

type pcsStepFunc func() (time.Duration, error)

type pcsStepper interface {
	after() time.Duration
	step(pcsStepFunc) pcsStepper
}

type pcsStepNext struct {
	aft time.Duration
}

func (n *pcsStepNext) after() time.Duration {
	return n.aft
}

func (n *pcsStepNext) step(fn pcsStepFunc) (next pcsStepper) {
	ttl, err := fn()
	if err == nil {
		next = &pcsStepNext{aft: ttl}
		log.Debugf("Synced provider config, next attempt in %v", next.after())
	} else {
		next = &pcsStepRetry{aft: time.Second}
		log.Errorf("Provider config sync failed, retrying in %v: %v", next.after(), err)
	}
	return
}

type pcsStepRetry struct {
	aft time.Duration
}

func (r *pcsStepRetry) after() time.Duration {
	return r.aft
}

func (r *pcsStepRetry) step(fn pcsStepFunc) (next pcsStepper) {
	ttl, err := fn()
	if err == nil {
		next = &pcsStepNext{aft: ttl}
		log.Infof("Provider config sync no longer failing")
	} else {
		next = &pcsStepRetry{aft: timeutil.ExpBackoff(r.aft, time.Minute)}
		log.Errorf("Provider config sync still failing, retrying in %v: %v", next.after(), err)
	}
	return
}

func nextSyncAfter(exp time.Time, clock clockwork.Clock) time.Duration {
	if exp.IsZero() {
		return MaximumProviderConfigSyncInterval
	}

	t := exp.Sub(clock.Now()) / 2
	if t > MaximumProviderConfigSyncInterval {
		t = MaximumProviderConfigSyncInterval
	} else if t < MinimumProviderConfigSyncInterval {
		t = MinimumProviderConfigSyncInterval
	}

	return t
}

type httpProviderConfigGetter struct {
	hc        phttp.Client
	issuerURL string
	clock     clockwork.Clock
}

func NewHTTPProviderConfigGetter(hc phttp.Client, issuerURL string) *httpProviderConfigGetter {
	return &httpProviderConfigGetter{
		hc:        hc,
		issuerURL: issuerURL,
		clock:     clockwork.NewRealClock(),
	}
}

func (r *httpProviderConfigGetter) Get() (cfg ProviderConfig, err error) {
	req, err := http.NewRequest("GET", r.issuerURL+discoveryConfigPath, nil)
	if err != nil {
		return
	}

	resp, err := r.hc.Do(req)
	if err != nil {
		return
	}
	defer resp.Body.Close()

	if err = json.NewDecoder(resp.Body).Decode(&cfg); err != nil {
		return
	}

	var ttl time.Duration
	var ok bool
	ttl, ok, err = phttp.Cacheable(resp.Header)
	if err != nil {
		return
	} else if ok {
		cfg.ExpiresAt = r.clock.Now().UTC().Add(ttl)
	}

	// The issuer value returned MUST be identical to the Issuer URL that was directly used to retrieve the configuration information.
	// http://openid.net/specs/openid-connect-discovery-1_0.html#ProviderConfigurationValidation
	if !urlEqual(cfg.Issuer, r.issuerURL) {
		err = fmt.Errorf(`"issuer" in config (%v) does not match provided issuer URL (%v)`, cfg.Issuer, r.issuerURL)
		return
	}

	return
}

func FetchProviderConfig(hc phttp.Client, issuerURL string) (ProviderConfig, error) {
	if hc == nil {
		hc = http.DefaultClient
	}

	g := NewHTTPProviderConfigGetter(hc, issuerURL)
	return g.Get()
}

func WaitForProviderConfig(hc phttp.Client, issuerURL string) (pcfg ProviderConfig) {
	return waitForProviderConfig(hc, issuerURL, clockwork.NewRealClock())
}

func waitForProviderConfig(hc phttp.Client, issuerURL string, clock clockwork.Clock) (pcfg ProviderConfig) {
	var sleep time.Duration
	var err error
	for {
		pcfg, err = FetchProviderConfig(hc, issuerURL)
		if err == nil {
			break
		}

		sleep = timeutil.ExpBackoff(sleep, time.Minute)
		fmt.Printf("Failed fetching provider config, trying again in %v: %v\n", sleep, err)
		time.Sleep(sleep)
	}

	return
}
