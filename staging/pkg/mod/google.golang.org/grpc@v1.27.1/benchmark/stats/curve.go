/*
 *
 * Copyright 2019 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

package stats

import (
	"crypto/sha256"
	"encoding/csv"
	"encoding/hex"
	"fmt"
	"io/ioutil"
	"math"
	"math/rand"
	"os"
	"sort"
	"strconv"
)

// payloadCurveRange represents a line within a payload curve CSV file.
type payloadCurveRange struct {
	from, to int32
	weight   float64
}

// newPayloadCurveRange receives a line from a payload curve CSV file and
// returns a *payloadCurveRange if the values are acceptable.
func newPayloadCurveRange(line []string) (*payloadCurveRange, error) {
	if len(line) != 3 {
		return nil, fmt.Errorf("invalid number of entries in line %v (expected 3)", line)
	}

	var from, to int64
	var weight float64
	var err error
	if from, err = strconv.ParseInt(line[0], 10, 32); err != nil {
		return nil, err
	}
	if from <= 0 {
		return nil, fmt.Errorf("line %v: field (%d) must be in (0, %d]", line, from, math.MaxInt32)
	}
	if to, err = strconv.ParseInt(line[1], 10, 32); err != nil {
		return nil, err
	}
	if to <= 0 {
		return nil, fmt.Errorf("line %v: field %d must be in (0, %d]", line, to, math.MaxInt32)
	}
	if from > to {
		return nil, fmt.Errorf("line %v: from (%d) > to (%d)", line, from, to)
	}
	if weight, err = strconv.ParseFloat(line[2], 64); err != nil {
		return nil, err
	}
	return &payloadCurveRange{from: int32(from), to: int32(to), weight: weight}, nil
}

// chooseRandom picks a payload size (in bytes) for a particular range. This is
// done with a uniform distribution.
func (pcr *payloadCurveRange) chooseRandom() int {
	if pcr.from == pcr.to { // fast path
		return int(pcr.from)
	}

	return int(rand.Int31n(pcr.to-pcr.from+1) + pcr.from)
}

// sha256file is a helper function that returns a hex string matching the
// SHA-256 sum of the input file.
func sha256file(file string) (string, error) {
	data, err := ioutil.ReadFile(file)
	if err != nil {
		return "", err
	}
	sum := sha256.Sum256(data)
	return hex.EncodeToString(sum[:]), nil
}

// PayloadCurve is an internal representation of a weighted random distribution
// CSV file. Once a *PayloadCurve is created with NewPayloadCurve, the
// ChooseRandom function should be called to generate random payload sizes.
type PayloadCurve struct {
	pcrs []*payloadCurveRange
	// Sha256 must be a public field so that the gob encoder can write it to
	// disk. This will be needed at decode-time by the Hash function.
	Sha256 string
}

// NewPayloadCurve parses a .csv file and returns a *PayloadCurve if no errors
// were encountered in parsing and initialization.
func NewPayloadCurve(file string) (*PayloadCurve, error) {
	f, err := os.Open(file)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	r := csv.NewReader(f)
	lines, err := r.ReadAll()
	if err != nil {
		return nil, err
	}

	ret := &PayloadCurve{}
	var total float64
	for _, line := range lines {
		pcr, err := newPayloadCurveRange(line)
		if err != nil {
			return nil, err
		}

		ret.pcrs = append(ret.pcrs, pcr)
		total += pcr.weight
	}

	ret.Sha256, err = sha256file(file)
	if err != nil {
		return nil, err
	}
	for _, pcr := range ret.pcrs {
		pcr.weight /= total
	}

	sort.Slice(ret.pcrs, func(i, j int) bool {
		if ret.pcrs[i].from == ret.pcrs[j].from {
			return ret.pcrs[i].to < ret.pcrs[j].to
		}
		return ret.pcrs[i].from < ret.pcrs[j].from
	})

	var lastTo int32
	for _, pcr := range ret.pcrs {
		if lastTo >= pcr.from {
			return nil, fmt.Errorf("[%d, %d] overlaps with a different line", pcr.from, pcr.to)
		}
		lastTo = pcr.to
	}

	return ret, nil
}

// ChooseRandom picks a random payload size (in bytes) that follows the
// underlying weighted random distribution.
func (pc *PayloadCurve) ChooseRandom() int {
	target := rand.Float64()
	var seen float64
	for _, pcr := range pc.pcrs {
		seen += pcr.weight
		if seen >= target {
			return pcr.chooseRandom()
		}
	}

	// This should never happen, but if it does, return a sane default.
	return 1
}

// Hash returns a string uniquely identifying a payload curve file for feature
// matching purposes.
func (pc *PayloadCurve) Hash() string {
	return pc.Sha256
}

// ShortHash returns a shortened version of Hash for display purposes.
func (pc *PayloadCurve) ShortHash() string {
	return pc.Sha256[:8]
}
