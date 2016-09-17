// Copyright 2016 Circonus, Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package checkmgr provides a check management interace to circonus-gometrics
package checkmgr

import (
	"crypto/tls"
	"crypto/x509"
	"errors"
	"fmt"
	"io/ioutil"
	"log"
	"net/url"
	"os"
	"path"
	"strconv"
	"sync"
	"time"

	"github.com/circonus-labs/circonus-gometrics/api"
)

// Check management offers:
//
// Create a check if one cannot be found matching specific criteria
// Manage metrics in the supplied check (enabling new metrics as they are submitted)
//
// To disable check management, leave Config.Api.Token.Key blank
//
// use cases:
// configure without api token - check management disabled
//  - configuration parameters other than Check.SubmissionUrl, Debug and Log are ignored
//  - note: SubmissionUrl is **required** in this case as there is no way to derive w/o api
// configure with api token - check management enabled
//  - all otehr configuration parameters affect how the trap url is obtained
//    1. provided (Check.SubmissionUrl)
//    2. via check lookup (CheckConfig.Id)
//    3. via a search using CheckConfig.InstanceId + CheckConfig.SearchTag
//    4. a new check is created

const (
	defaultCheckType             = "httptrap"
	defaultTrapMaxURLAge         = "60s"   // 60 seconds
	defaultBrokerMaxResponseTime = "500ms" // 500 milliseconds
	defaultForceMetricActivation = "false"
	statusActive                 = "active"
)

// CheckConfig options for check
type CheckConfig struct {
	// a specific submission url
	SubmissionURL string
	// a specific check id (not check bundle id)
	ID string
	// unique instance id string
	// used to search for a check to use
	// used as check.target when creating a check
	InstanceID string
	// unique check searching tag
	// used to search for a check to use (combined with instanceid)
	// used as a regular tag when creating a check
	SearchTag string
	// a custom display name for the check (as viewed in UI Checks)
	DisplayName string
	// httptrap check secret (for creating a check)
	Secret string
	// additional tags to add to a check (when creating a check)
	// these tags will not be added to an existing check
	Tags []string
	// max amount of time to to hold on to a submission url
	// when a given submission fails (due to retries) if the
	// time the url was last updated is > than this, the trap
	// url will be refreshed (e.g. if the broker is changed
	// in the UI) **only relevant when check management is enabled**
	// e.g. 5m, 30m, 1h, etc.
	MaxURLAge string
	// force metric activation - if a metric has been disabled via the UI
	// the default behavior is to *not* re-activate the metric; this setting
	// overrides the behavior and will re-activate the metric when it is
	// encountered. "(true|false)", default "false"
	ForceMetricActivation string
}

// BrokerConfig options for broker
type BrokerConfig struct {
	// a specific broker id (numeric portion of cid)
	ID string
	// a tag that can be used to select 1-n brokers from which to select
	// when creating a new check (e.g. datacenter:abc)
	SelectTag string
	// for a broker to be considered viable it must respond to a
	// connection attempt within this amount of time e.g. 200ms, 2s, 1m
	MaxResponseTime string
}

// Config options
type Config struct {
	Log   *log.Logger
	Debug bool

	// Circonus API config
	API api.Config
	// Check specific configuration options
	Check CheckConfig
	// Broker specific configuration options
	Broker BrokerConfig
}

// CheckTypeType check type
type CheckTypeType string

// CheckInstanceIDType check instance id
type CheckInstanceIDType string

// CheckSecretType check secret
type CheckSecretType string

// CheckTagsType check tags
type CheckTagsType []string

// CheckDisplayNameType check display name
type CheckDisplayNameType string

// BrokerCNType broker common name
type BrokerCNType string

// CheckManager settings
type CheckManager struct {
	enabled bool
	Log     *log.Logger
	Debug   bool
	apih    *api.API

	// check
	checkType             CheckTypeType
	checkID               api.IDType
	checkInstanceID       CheckInstanceIDType
	checkSearchTag        api.SearchTagType
	checkSecret           CheckSecretType
	checkTags             CheckTagsType
	checkSubmissionURL    api.URLType
	checkDisplayName      CheckDisplayNameType
	forceMetricActivation bool

	// broker
	brokerID              api.IDType
	brokerSelectTag       api.SearchTagType
	brokerMaxResponseTime time.Duration

	// state
	checkBundle      *api.CheckBundle
	availableMetrics map[string]bool
	trapURL          api.URLType
	trapCN           BrokerCNType
	trapLastUpdate   time.Time
	trapMaxURLAge    time.Duration
	trapmu           sync.Mutex
	certPool         *x509.CertPool
}

// Trap config
type Trap struct {
	URL *url.URL
	TLS *tls.Config
}

// NewCheckManager returns a new check manager
func NewCheckManager(cfg *Config) (*CheckManager, error) {

	if cfg == nil {
		return nil, errors.New("Invalid Check Manager configuration (nil).")
	}

	cm := &CheckManager{
		enabled: false,
	}

	cm.Debug = cfg.Debug

	cm.Log = cfg.Log
	if cm.Log == nil {
		if cm.Debug {
			cm.Log = log.New(os.Stderr, "", log.LstdFlags)
		} else {
			cm.Log = log.New(ioutil.Discard, "", log.LstdFlags)
		}
	}

	if cfg.Check.SubmissionURL != "" {
		cm.checkSubmissionURL = api.URLType(cfg.Check.SubmissionURL)
	}
	// Blank API Token *disables* check management
	if cfg.API.TokenKey == "" {
		if cm.checkSubmissionURL == "" {
			return nil, errors.New("Invalid check manager configuration (no API token AND no submission url).")
		}
		if err := cm.initializeTrapURL(); err != nil {
			return nil, err
		}
		return cm, nil
	}

	// enable check manager

	cm.enabled = true

	// initialize api handle

	cfg.API.Debug = cm.Debug
	cfg.API.Log = cm.Log

	apih, err := api.NewAPI(&cfg.API)
	if err != nil {
		return nil, err
	}
	cm.apih = apih

	// initialize check related data

	cm.checkType = defaultCheckType

	idSetting := "0"
	if cfg.Check.ID != "" {
		idSetting = cfg.Check.ID
	}
	id, err := strconv.Atoi(idSetting)
	if err != nil {
		return nil, err
	}
	cm.checkID = api.IDType(id)

	cm.checkInstanceID = CheckInstanceIDType(cfg.Check.InstanceID)
	cm.checkDisplayName = CheckDisplayNameType(cfg.Check.DisplayName)
	cm.checkSearchTag = api.SearchTagType(cfg.Check.SearchTag)
	cm.checkSecret = CheckSecretType(cfg.Check.Secret)
	cm.checkTags = cfg.Check.Tags

	fma := defaultForceMetricActivation
	if cfg.Check.ForceMetricActivation != "" {
		fma = cfg.Check.ForceMetricActivation
	}
	fm, err := strconv.ParseBool(fma)
	if err != nil {
		return nil, err
	}
	cm.forceMetricActivation = fm

	_, an := path.Split(os.Args[0])
	hn, err := os.Hostname()
	if err != nil {
		hn = "unknown"
	}
	if cm.checkInstanceID == "" {
		cm.checkInstanceID = CheckInstanceIDType(fmt.Sprintf("%s:%s", hn, an))
	}

	if cm.checkSearchTag == "" {
		cm.checkSearchTag = api.SearchTagType(fmt.Sprintf("service:%s", an))
	}

	if cm.checkDisplayName == "" {
		cm.checkDisplayName = CheckDisplayNameType(fmt.Sprintf("%s /cgm", string(cm.checkInstanceID)))
	}

	dur := cfg.Check.MaxURLAge
	if dur == "" {
		dur = defaultTrapMaxURLAge
	}
	maxDur, err := time.ParseDuration(dur)
	if err != nil {
		return nil, err
	}
	cm.trapMaxURLAge = maxDur

	// setup broker

	idSetting = "0"
	if cfg.Broker.ID != "" {
		idSetting = cfg.Broker.ID
	}
	id, err = strconv.Atoi(idSetting)
	if err != nil {
		return nil, err
	}
	cm.brokerID = api.IDType(id)

	cm.brokerSelectTag = api.SearchTagType(cfg.Broker.SelectTag)

	dur = cfg.Broker.MaxResponseTime
	if dur == "" {
		dur = defaultBrokerMaxResponseTime
	}
	maxDur, err = time.ParseDuration(dur)
	if err != nil {
		return nil, err
	}
	cm.brokerMaxResponseTime = maxDur

	// metrics
	cm.availableMetrics = make(map[string]bool)

	if err := cm.initializeTrapURL(); err != nil {
		return nil, err
	}

	return cm, nil
}

// GetTrap return the trap url
func (cm *CheckManager) GetTrap() (*Trap, error) {
	if cm.trapURL == "" {
		if err := cm.initializeTrapURL(); err != nil {
			return nil, err
		}
	}

	trap := &Trap{}

	u, err := url.Parse(string(cm.trapURL))
	if err != nil {
		return nil, err
	}

	trap.URL = u

	if u.Scheme == "https" {
		if cm.certPool == nil {
			cm.loadCACert()
		}
		t := &tls.Config{
			RootCAs: cm.certPool,
		}
		if cm.trapCN != "" {
			t.ServerName = string(cm.trapCN)
		}
		trap.TLS = t
	}

	return trap, nil
}

// ResetTrap URL, force request to the API for the submission URL and broker ca cert
func (cm *CheckManager) ResetTrap() error {
	if cm.trapURL == "" {
		return nil
	}

	cm.trapURL = ""
	cm.certPool = nil
	err := cm.initializeTrapURL()
	return err
}

// RefreshTrap check when the last time the URL was reset, reset if needed
func (cm *CheckManager) RefreshTrap() {
	if cm.trapURL == "" {
		return
	}

	if time.Since(cm.trapLastUpdate) >= cm.trapMaxURLAge {
		cm.ResetTrap()
	}
}
