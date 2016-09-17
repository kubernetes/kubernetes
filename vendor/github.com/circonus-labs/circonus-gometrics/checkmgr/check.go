// Copyright 2016 Circonus, Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package checkmgr

import (
	"crypto/rand"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"

	"github.com/circonus-labs/circonus-gometrics/api"
)

// Initialize CirconusMetrics instance. Attempt to find a check otherwise create one.
// use cases:
//
// check [bundle] by submission url
// check [bundle] by *check* id (note, not check_bundle id)
// check [bundle] by search
// create check [bundle]
func (cm *CheckManager) initializeTrapURL() error {
	if cm.trapURL != "" {
		return nil
	}

	cm.trapmu.Lock()
	defer cm.trapmu.Unlock()

	if cm.checkSubmissionURL != "" {
		if !cm.enabled {
			cm.trapURL = cm.checkSubmissionURL
			cm.trapLastUpdate = time.Now()
			return nil
		}
	}

	if !cm.enabled {
		return errors.New("Unable to initialize trap, check manager is disabled.")
	}

	var err error
	var check *api.Check
	var checkBundle *api.CheckBundle
	var broker *api.Broker

	if cm.checkSubmissionURL != "" {
		check, err = cm.apih.FetchCheckBySubmissionURL(cm.checkSubmissionURL)
		if err != nil {
			return err
		}
		// extract check id from check object returned from looking up using submission url
		// set m.CheckId to the id
		// set m.SubmissionUrl to "" to prevent trying to search on it going forward
		// use case: if the broker is changed in the UI metrics would stop flowing
		// unless the new submission url can be fetched with the API (which is no
		// longer possible using the original submission url)
		var id int
		id, err = strconv.Atoi(strings.Replace(check.Cid, "/check/", "", -1))
		if err == nil {
			cm.checkID = api.IDType(id)
			cm.checkSubmissionURL = ""
		} else {
			cm.Log.Printf(
				"[WARN] SubmissionUrl check to Check ID: unable to convert %s to int %q\n",
				check.Cid, err)
		}
	} else if cm.checkID > 0 {
		check, err = cm.apih.FetchCheckByID(cm.checkID)
		if err != nil {
			return err
		}
	} else {
		searchCriteria := fmt.Sprintf(
			"(active:1)(host:\"%s\")(type:\"%s\")(tags:%s)",
			cm.checkInstanceID, cm.checkType, cm.checkSearchTag)
		checkBundle, err = cm.checkBundleSearch(searchCriteria)
		if err != nil {
			return err
		}

		if checkBundle == nil {
			// err==nil && checkBundle==nil is "no check bundles matched"
			// an error *should* be returned for any other invalid scenario
			checkBundle, broker, err = cm.createNewCheck()
			if err != nil {
				return err
			}
		}
	}

	if checkBundle == nil {
		if check != nil {
			checkBundle, err = cm.apih.FetchCheckBundleByCID(api.CIDType(check.CheckBundleCid))
			if err != nil {
				return err
			}
		} else {
			return fmt.Errorf("[ERROR] Unable to retrieve, find, or create check")
		}
	}

	if broker == nil {
		broker, err = cm.apih.FetchBrokerByCID(api.CIDType(checkBundle.Brokers[0]))
		if err != nil {
			return err
		}
	}

	// retain to facilitate metric management (adding new metrics specifically)
	cm.checkBundle = checkBundle
	cm.inventoryMetrics()

	// url to which metrics should be PUT
	cm.trapURL = api.URLType(checkBundle.Config.SubmissionURL)

	// used when sending as "ServerName" get around certs not having IP SANS
	// (cert created with server name as CN but IP used in trap url)
	cn, err := cm.getBrokerCN(broker, cm.trapURL)
	if err != nil {
		return err
	}
	cm.trapCN = BrokerCNType(cn)

	cm.trapLastUpdate = time.Now()

	return nil
}

// Search for a check bundle given a predetermined set of criteria
func (cm *CheckManager) checkBundleSearch(criteria string) (*api.CheckBundle, error) {
	checkBundles, err := cm.apih.CheckBundleSearch(api.SearchQueryType(criteria))
	if err != nil {
		return nil, err
	}

	if len(checkBundles) == 0 {
		return nil, nil // trigger creation of a new check
	}

	numActive := 0
	checkID := -1

	for idx, check := range checkBundles {
		if check.Status == statusActive {
			numActive++
			checkID = idx
		}
	}

	if numActive > 1 {
		return nil, fmt.Errorf("[ERROR] Multiple possibilities multiple check bundles match criteria %s\n", criteria)
	}

	return &checkBundles[checkID], nil
}

// Create a new check to receive metrics
func (cm *CheckManager) createNewCheck() (*api.CheckBundle, *api.Broker, error) {
	checkSecret := string(cm.checkSecret)
	if checkSecret == "" {
		secret, err := cm.makeSecret()
		if err != nil {
			secret = "myS3cr3t"
		}
		checkSecret = secret
	}

	broker, err := cm.getBroker()
	if err != nil {
		return nil, nil, err
	}

	config := api.CheckBundle{
		Brokers:     []string{broker.Cid},
		Config:      api.CheckBundleConfig{AsyncMetrics: true, Secret: checkSecret},
		DisplayName: string(cm.checkDisplayName),
		Metrics:     []api.CheckBundleMetric{},
		MetricLimit: 0,
		Notes:       "",
		Period:      60,
		Status:      statusActive,
		Tags:        append([]string{string(cm.checkSearchTag)}, cm.checkTags...),
		Target:      string(cm.checkInstanceID),
		Timeout:     10,
		Type:        string(cm.checkType),
	}

	checkBundle, err := cm.apih.CreateCheckBundle(config)
	if err != nil {
		return nil, nil, err
	}

	return checkBundle, broker, nil
}

// Create a dynamic secret to use with a new check
func (cm *CheckManager) makeSecret() (string, error) {
	hash := sha256.New()
	x := make([]byte, 2048)
	if _, err := rand.Read(x); err != nil {
		return "", err
	}
	hash.Write(x)
	return hex.EncodeToString(hash.Sum(nil))[0:16], nil
}
