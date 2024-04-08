/*
Copyright 2018 Portworx

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
package auth

import (
	"fmt"
	"regexp"
	"strconv"
	"time"
)

const (
	// SecondDef is the abbrevation for seconds
	SecondDef = "s"
	// MinuteDef is the abbrevation for minutes
	MinuteDef = "m"
	// HourDef is the abbrevation for hours
	HourDef = "h"
	// DayDef is the abbrevation for days
	DayDef = "d"
	// YearDef is the abbrevation for years
	YearDef = "y"

	// Day is the duration of hours in a day
	Day = time.Hour * 24
	// Year is the duration of days in a year
	Year = Day * 365
)

var (
	// SecondRegex is a regular expression for finding seconds as a duration
	SecondRegex = regexp.MustCompile("^([0-9]+)" + SecondDef + "$")
	// MinuteRegex is a regular expression for finding minutes as a duration
	MinuteRegex = regexp.MustCompile("^([0-9]+)" + MinuteDef + "$")
	// HourRegex is a regular expression for finding hours as a duration
	HourRegex = regexp.MustCompile("^([0-9]+)" + HourDef + "$")
	// DayRegex is a regular expression for finding days as a duration
	DayRegex = regexp.MustCompile("^([0-9]+)" + DayDef + "$")
	// YearRegex is a regular expression for finding years as a duration
	YearRegex = regexp.MustCompile("^([0-9]+)" + YearDef + "$")
)

// ParseToDuration takes in a "human" type duration and changes it to
// time.Duration. The format for a human type is <number><type>. For
// example: Five days: 5d; one year: 1y.
func ParseToDuration(s string) (time.Duration, error) {

	regexs := []struct {
		regex    *regexp.Regexp
		duration time.Duration
	}{
		{
			regex:    SecondRegex,
			duration: time.Second,
		},
		{
			regex:    MinuteRegex,
			duration: time.Minute,
		},
		{
			regex:    HourRegex,
			duration: time.Hour,
		},
		{
			regex:    DayRegex,
			duration: Day,
		},
		{
			regex:    YearRegex,
			duration: Year,
		},
	}
	for _, r := range regexs {
		if val := r.regex.FindString(s); len(val) != 0 {
			parsed, err := strconv.Atoi(val[:len(val)-1])
			if err != nil {
				return 0, err
			}
			return time.Duration(parsed) * r.duration, nil
		}
	}

	return 0, fmt.Errorf("Unable to parse")
}
