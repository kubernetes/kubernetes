// Copyright 2016 Circonus, Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package api

import (
	"encoding/json"
	"fmt"
)

// BrokerDetail instance attributes
type BrokerDetail struct {
	CN      string   `json:"cn"`
	IP      string   `json:"ipaddress"`
	MinVer  int      `json:"minimum_version_required"`
	Modules []string `json:"modules"`
	Port    int      `json:"port"`
	Skew    string   `json:"skew"`
	Status  string   `json:"status"`
	Version int      `json:"version"`
}

// Broker definition
type Broker struct {
	Cid       string         `json:"_cid"`
	Details   []BrokerDetail `json:"_details"`
	Latitude  string         `json:"_latitude"`
	Longitude string         `json:"_longitude"`
	Name      string         `json:"_name"`
	Tags      []string       `json:"_tags"`
	Type      string         `json:"_type"`
}

// FetchBrokerByID fetch a broker configuration by [group]id
func (a *API) FetchBrokerByID(id IDType) (*Broker, error) {
	cid := CIDType(fmt.Sprintf("/broker/%d", id))
	return a.FetchBrokerByCID(cid)
}

// FetchBrokerByCID fetch a broker configuration by cid
func (a *API) FetchBrokerByCID(cid CIDType) (*Broker, error) {
	result, err := a.Get(string(cid))
	if err != nil {
		return nil, err
	}

	response := new(Broker)
	if err := json.Unmarshal(result, &response); err != nil {
		return nil, err
	}

	return response, nil

}

// FetchBrokerListByTag return list of brokers with a specific tag
func (a *API) FetchBrokerListByTag(searchTag SearchTagType) ([]Broker, error) {
	query := SearchQueryType(fmt.Sprintf("f__tags_has=%s", searchTag))
	return a.BrokerSearch(query)
}

// BrokerSearch return a list of brokers matching a query/filter
func (a *API) BrokerSearch(query SearchQueryType) ([]Broker, error) {
	queryURL := fmt.Sprintf("/broker?%s", string(query))

	result, err := a.Get(queryURL)
	if err != nil {
		return nil, err
	}

	var brokers []Broker
	json.Unmarshal(result, &brokers)

	return brokers, nil
}

// FetchBrokerList return list of all brokers available to the api token/app
func (a *API) FetchBrokerList() ([]Broker, error) {
	result, err := a.Get("/broker")
	if err != nil {
		return nil, err
	}

	var response []Broker
	json.Unmarshal(result, &response)

	return response, nil
}
