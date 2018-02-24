// Copyright 2017 Google Inc. All Rights Reserved.
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

package client

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"time"

	"github.com/golang/protobuf/proto"
	"github.com/golang/protobuf/ptypes"
	ct "github.com/google/certificate-transparency-go"
	"github.com/google/certificate-transparency-go/client/configpb"
	"github.com/google/certificate-transparency-go/jsonclient"
	"github.com/google/certificate-transparency-go/x509"
)

type interval struct {
	lower *time.Time // nil => no lower bound
	upper *time.Time // nil => no upper bound
}

// TemporalLogConfigFromFile creates a TemporalLogConfig object from the given
// filename, which should contain text-protobuf encoded configuration data.
func TemporalLogConfigFromFile(filename string) (*configpb.TemporalLogConfig, error) {
	if len(filename) == 0 {
		return nil, errors.New("log config filename empty")
	}

	cfgText, err := ioutil.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("failed to read log config: %v", err)
	}

	var cfg configpb.TemporalLogConfig
	if err := proto.UnmarshalText(string(cfgText), &cfg); err != nil {
		return nil, fmt.Errorf("failed to parse log config: %v", err)
	}

	if len(cfg.Shard) == 0 {
		return nil, errors.New("empty log config found")
	}
	return &cfg, nil
}

// AddLogClient is an interface that allows adding certificates and pre-certificates to a log.
// Both LogClient and TemporalLogClient implement this interface, which allows users to
// commonize code for adding certs to normal/temporal logs.
type AddLogClient interface {
	AddChain(ctx context.Context, chain []ct.ASN1Cert) (*ct.SignedCertificateTimestamp, error)
	AddPreChain(ctx context.Context, chain []ct.ASN1Cert) (*ct.SignedCertificateTimestamp, error)
	GetAcceptedRoots(ctx context.Context) ([]ct.ASN1Cert, error)
}

// TemporalLogClient allows [pre-]certificates to be uploaded to a temporal log.
type TemporalLogClient struct {
	Clients   []*LogClient
	intervals []interval
}

// NewTemporalLogClient builds a new client for interacting with a temporal log.
// The provided config should be contiguous and chronological.
func NewTemporalLogClient(cfg configpb.TemporalLogConfig, hc *http.Client) (*TemporalLogClient, error) {
	if len(cfg.Shard) == 0 {
		return nil, errors.New("empty config")
	}

	overall, err := shardInterval(cfg.Shard[0])
	if err != nil {
		return nil, fmt.Errorf("cfg.Shard[0] invalid: %v", err)
	}
	intervals := make([]interval, 0, len(cfg.Shard))
	intervals = append(intervals, overall)
	for i := 1; i < len(cfg.Shard); i++ {
		interval, err := shardInterval(cfg.Shard[i])
		if err != nil {
			return nil, fmt.Errorf("cfg.Shard[%d] invalid: %v", i, err)
		}
		if overall.upper == nil {
			return nil, fmt.Errorf("cfg.Shard[%d] extends an interval with no upper bound", i)
		}
		if interval.lower == nil {
			return nil, fmt.Errorf("cfg.Shard[%d] has no lower bound but extends an interval", i)
		}
		if !interval.lower.Equal(*overall.upper) {
			return nil, fmt.Errorf("cfg.Shard[%d] starts at %v but previous interval ended at %v", i, interval.lower, overall.upper)
		}
		overall.upper = interval.upper
		intervals = append(intervals, interval)
	}
	clients := make([]*LogClient, 0, len(cfg.Shard))
	for i, shard := range cfg.Shard {
		opts := jsonclient.Options{}
		opts.PublicKeyDER = shard.GetPublicKeyDer()
		c, err := New(shard.Uri, hc, opts)
		if err != nil {
			return nil, fmt.Errorf("failed to create client for cfg.Shard[%d]: %v", i, err)
		}
		clients = append(clients, c)
	}
	tlc := TemporalLogClient{
		Clients:   clients,
		intervals: intervals,
	}
	return &tlc, nil
}

// GetAcceptedRoots retrieves the set of acceptable root certificates for all
// of the shards of a temporal log (i.e. the union).
func (tlc *TemporalLogClient) GetAcceptedRoots(ctx context.Context) ([]ct.ASN1Cert, error) {
	type result struct {
		roots []ct.ASN1Cert
		err   error
	}
	results := make(chan result, len(tlc.Clients))
	for _, c := range tlc.Clients {
		go func(c *LogClient) {
			var r result
			r.roots, r.err = c.GetAcceptedRoots(ctx)
			results <- r
		}(c)
	}

	var allRoots []ct.ASN1Cert
	seen := make(map[[sha256.Size]byte]bool)
	for range tlc.Clients {
		r := <-results
		if r.err != nil {
			return nil, r.err
		}
		for _, root := range r.roots {
			h := sha256.Sum256(root.Data)
			if seen[h] {
				continue
			}
			seen[h] = true
			allRoots = append(allRoots, root)
		}
	}
	return allRoots, nil
}

// AddChain adds the (DER represented) X509 chain to the appropriate log.
func (tlc *TemporalLogClient) AddChain(ctx context.Context, chain []ct.ASN1Cert) (*ct.SignedCertificateTimestamp, error) {
	return tlc.addChain(ctx, ct.X509LogEntryType, ct.AddChainPath, chain)
}

// AddPreChain adds the (DER represented) Precertificate chain to the appropriate log.
func (tlc *TemporalLogClient) AddPreChain(ctx context.Context, chain []ct.ASN1Cert) (*ct.SignedCertificateTimestamp, error) {
	return tlc.addChain(ctx, ct.PrecertLogEntryType, ct.AddPreChainPath, chain)
}

func (tlc *TemporalLogClient) addChain(ctx context.Context, ctype ct.LogEntryType, path string, chain []ct.ASN1Cert) (*ct.SignedCertificateTimestamp, error) {
	// Parse the first entry in the chain
	if len(chain) == 0 {
		return nil, errors.New("missing chain")
	}
	cert, err := x509.ParseCertificate(chain[0].Data)
	if err != nil {
		return nil, fmt.Errorf("failed to parse initial chain entry: %v", err)
	}
	cidx, err := tlc.IndexByDate(cert.NotAfter)
	if err != nil {
		return nil, fmt.Errorf("failed to find log to process cert: %v", err)
	}
	return tlc.Clients[cidx].addChainWithRetry(ctx, ctype, path, chain)
}

// IndexByDate returns the index of the Clients entry that is appropriate for the given
// date.
func (tlc *TemporalLogClient) IndexByDate(when time.Time) (int, error) {
	for i, interval := range tlc.intervals {
		if (interval.lower != nil) && when.Before(*interval.lower) {
			continue
		}
		if (interval.upper != nil) && !when.Before(*interval.upper) {
			continue
		}
		return i, nil
	}
	return -1, fmt.Errorf("no log found encompassing date %v", when)
}

func shardInterval(cfg *configpb.LogShardConfig) (interval, error) {
	var interval interval
	if cfg.NotAfterStart != nil {
		t, err := ptypes.Timestamp(cfg.NotAfterStart)
		if err != nil {
			return interval, fmt.Errorf("failed to parse NotAfterStart: %v", err)
		}
		interval.lower = &t
	}
	if cfg.NotAfterLimit != nil {
		t, err := ptypes.Timestamp(cfg.NotAfterLimit)
		if err != nil {
			return interval, fmt.Errorf("failed to parse NotAfterLimit: %v", err)
		}
		interval.upper = &t
	}

	if interval.lower != nil && interval.upper != nil && !(*interval.lower).Before(*interval.upper) {
		return interval, errors.New("inverted interval")
	}
	return interval, nil
}
