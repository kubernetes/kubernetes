// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package influxdb

import (
	"fmt"
	"net/url"
	"strconv"
	"time"

	"k8s.io/heapster/version"

	influxdb "github.com/influxdata/influxdb/client"
)

type InfluxdbClient interface {
	Write(influxdb.BatchPoints) (*influxdb.Response, error)
	Query(influxdb.Query) (*influxdb.Response, error)
	Ping() (time.Duration, string, error)
}

type InfluxdbConfig struct {
	User       string
	Password   string
	Secure     bool
	Host       string
	DbName     string
	WithFields bool
}

func NewClient(c InfluxdbConfig) (InfluxdbClient, error) {
	url := &url.URL{
		Scheme: "http",
		Host:   c.Host,
	}
	if c.Secure {
		url.Scheme = "https"
	}

	iConfig := &influxdb.Config{
		URL:       *url,
		Username:  c.User,
		Password:  c.Password,
		UserAgent: fmt.Sprintf("%v/%v", "heapster", version.HeapsterVersion),
	}
	client, err := influxdb.NewClient(*iConfig)

	if err != nil {
		return nil, err
	}
	if _, _, err := client.Ping(); err != nil {
		return nil, fmt.Errorf("failed to ping InfluxDB server at %q - %v", c.Host, err)
	}
	return client, nil
}

func BuildConfig(uri *url.URL) (*InfluxdbConfig, error) {
	config := InfluxdbConfig{
		User:       "root",
		Password:   "root",
		Host:       "localhost:8086",
		DbName:     "k8s",
		Secure:     false,
		WithFields: false,
	}

	if len(uri.Host) > 0 {
		config.Host = uri.Host
	}
	opts := uri.Query()
	if len(opts["user"]) >= 1 {
		config.User = opts["user"][0]
	}
	// TODO: use more secure way to pass the password.
	if len(opts["pw"]) >= 1 {
		config.Password = opts["pw"][0]
	}
	if len(opts["db"]) >= 1 {
		config.DbName = opts["db"][0]
	}
	if len(opts["withfields"]) >= 1 {
		val, err := strconv.ParseBool(opts["withfields"][0])
		if err != nil {
			return nil, fmt.Errorf("failed to parse `withfields` flag - %v", err)
		}
		config.WithFields = val
	}
	if len(opts["secure"]) >= 1 {
		val, err := strconv.ParseBool(opts["secure"][0])
		if err != nil {
			return nil, fmt.Errorf("failed to parse `secure` flag - %v", err)
		}
		config.Secure = val
	}

	return &config, nil
}
