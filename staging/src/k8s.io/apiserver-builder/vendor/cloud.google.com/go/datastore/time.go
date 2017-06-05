// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package datastore

import (
	"math"
	"time"
)

var (
	minTime = time.Unix(int64(math.MinInt64)/1e6, (int64(math.MinInt64)%1e6)*1e3)
	maxTime = time.Unix(int64(math.MaxInt64)/1e6, (int64(math.MaxInt64)%1e6)*1e3)
)

func toUnixMicro(t time.Time) int64 {
	// We cannot use t.UnixNano() / 1e3 because we want to handle times more than
	// 2^63 nanoseconds (which is about 292 years) away from 1970, and those cannot
	// be represented in the numerator of a single int64 divide.
	return t.Unix()*1e6 + int64(t.Nanosecond()/1e3)
}

func fromUnixMicro(t int64) time.Time {
	return time.Unix(t/1e6, (t%1e6)*1e3)
}
