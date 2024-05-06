/*
 *
 * Copyright 2023 gRPC authors.
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

package serviceconfig

import (
	"encoding/json"
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// Duration defines JSON marshal and unmarshal methods to conform to the
// protobuf JSON spec defined [here].
//
// [here]: https://protobuf.dev/reference/protobuf/google.protobuf/#duration
type Duration time.Duration

func (d Duration) String() string {
	return fmt.Sprint(time.Duration(d))
}

// MarshalJSON converts from d to a JSON string output.
func (d Duration) MarshalJSON() ([]byte, error) {
	ns := time.Duration(d).Nanoseconds()
	sec := ns / int64(time.Second)
	ns = ns % int64(time.Second)

	var sign string
	if sec < 0 || ns < 0 {
		sign, sec, ns = "-", -1*sec, -1*ns
	}

	// Generated output always contains 0, 3, 6, or 9 fractional digits,
	// depending on required precision.
	str := fmt.Sprintf("%s%d.%09d", sign, sec, ns)
	str = strings.TrimSuffix(str, "000")
	str = strings.TrimSuffix(str, "000")
	str = strings.TrimSuffix(str, ".000")
	return []byte(fmt.Sprintf("\"%ss\"", str)), nil
}

// UnmarshalJSON unmarshals b as a duration JSON string into d.
func (d *Duration) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err != nil {
		return err
	}
	if !strings.HasSuffix(s, "s") {
		return fmt.Errorf("malformed duration %q: missing seconds unit", s)
	}
	neg := false
	if s[0] == '-' {
		neg = true
		s = s[1:]
	}
	ss := strings.SplitN(s[:len(s)-1], ".", 3)
	if len(ss) > 2 {
		return fmt.Errorf("malformed duration %q: too many decimals", s)
	}
	// hasDigits is set if either the whole or fractional part of the number is
	// present, since both are optional but one is required.
	hasDigits := false
	var sec, ns int64
	if len(ss[0]) > 0 {
		var err error
		if sec, err = strconv.ParseInt(ss[0], 10, 64); err != nil {
			return fmt.Errorf("malformed duration %q: %v", s, err)
		}
		// Maximum seconds value per the durationpb spec.
		const maxProtoSeconds = 315_576_000_000
		if sec > maxProtoSeconds {
			return fmt.Errorf("out of range: %q", s)
		}
		hasDigits = true
	}
	if len(ss) == 2 && len(ss[1]) > 0 {
		if len(ss[1]) > 9 {
			return fmt.Errorf("malformed duration %q: too many digits after decimal", s)
		}
		var err error
		if ns, err = strconv.ParseInt(ss[1], 10, 64); err != nil {
			return fmt.Errorf("malformed duration %q: %v", s, err)
		}
		for i := 9; i > len(ss[1]); i-- {
			ns *= 10
		}
		hasDigits = true
	}
	if !hasDigits {
		return fmt.Errorf("malformed duration %q: contains no numbers", s)
	}

	if neg {
		sec *= -1
		ns *= -1
	}

	// Maximum/minimum seconds/nanoseconds representable by Go's time.Duration.
	const maxSeconds = math.MaxInt64 / int64(time.Second)
	const maxNanosAtMaxSeconds = math.MaxInt64 % int64(time.Second)
	const minSeconds = math.MinInt64 / int64(time.Second)
	const minNanosAtMinSeconds = math.MinInt64 % int64(time.Second)

	if sec > maxSeconds || (sec == maxSeconds && ns >= maxNanosAtMaxSeconds) {
		*d = Duration(math.MaxInt64)
	} else if sec < minSeconds || (sec == minSeconds && ns <= minNanosAtMinSeconds) {
		*d = Duration(math.MinInt64)
	} else {
		*d = Duration(sec*int64(time.Second) + ns)
	}
	return nil
}
