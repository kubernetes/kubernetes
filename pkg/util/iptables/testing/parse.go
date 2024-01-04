//go:build linux
// +build linux

/*
Copyright 2022 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"k8s.io/kubernetes/pkg/util/iptables"
)

// IPTablesDump represents a parsed IPTables rules dump (ie, the output of
// "iptables-save" or input to "iptables-restore")
type IPTablesDump struct {
	Tables []Table
}

// Table represents an IPTables table
type Table struct {
	Name   iptables.Table
	Chains []Chain
}

// Chain represents an IPTables chain
type Chain struct {
	Name    iptables.Chain
	Packets uint64
	Bytes   uint64
	Rules   []*Rule

	// Deleted is set if the input contained a "-X Name" line; this would never
	// appear in iptables-save output but it could appear in iptables-restore *input*.
	Deleted bool
}

var declareTableRegex = regexp.MustCompile(`^\*(.*)$`)
var declareChainRegex = regexp.MustCompile(`^:([^ ]*) - \[([0-9]*):([0-9]*)\]$`)
var addRuleRegex = regexp.MustCompile(`^-A ([^ ]*) (.*)$`)
var deleteChainRegex = regexp.MustCompile(`^-X (.*)$`)

type parseState int

const (
	parseTableDeclaration parseState = iota
	parseChainDeclarations
	parseChains
)

// ParseIPTablesDump parses an IPTables rules dump. Note: this may ignore some bad data.
func ParseIPTablesDump(data string) (*IPTablesDump, error) {
	dump := &IPTablesDump{}
	state := parseTableDeclaration
	lines := strings.Split(strings.Trim(data, "\n"), "\n")
	var t *Table

	for _, line := range lines {
	retry:
		line = strings.TrimSpace(line)
		if line == "" || line[0] == '#' {
			continue
		}

		switch state {
		case parseTableDeclaration:
			// Parse table declaration line ("*filter").
			match := declareTableRegex.FindStringSubmatch(line)
			if match == nil {
				return nil, fmt.Errorf("could not parse iptables data (table %d starts with %q not a table name)", len(dump.Tables)+1, line)
			}
			dump.Tables = append(dump.Tables, Table{Name: iptables.Table(match[1])})
			t = &dump.Tables[len(dump.Tables)-1]
			state = parseChainDeclarations

		case parseChainDeclarations:
			match := declareChainRegex.FindStringSubmatch(line)
			if match == nil {
				state = parseChains
				goto retry
			}

			chain := iptables.Chain(match[1])
			packets, _ := strconv.ParseUint(match[2], 10, 64)
			bytes, _ := strconv.ParseUint(match[3], 10, 64)

			t.Chains = append(t.Chains,
				Chain{
					Name:    chain,
					Packets: packets,
					Bytes:   bytes,
				},
			)

		case parseChains:
			if match := addRuleRegex.FindStringSubmatch(line); match != nil {
				chain := iptables.Chain(match[1])

				c, err := dump.GetChain(t.Name, chain)
				if err != nil {
					return nil, fmt.Errorf("error parsing rule %q: %v", line, err)
				}
				if c.Deleted {
					return nil, fmt.Errorf("cannot add rules to deleted chain %q", chain)
				}

				rule, err := ParseRule(line, false)
				if err != nil {
					return nil, err
				}
				c.Rules = append(c.Rules, rule)
			} else if match := deleteChainRegex.FindStringSubmatch(line); match != nil {
				chain := iptables.Chain(match[1])

				c, err := dump.GetChain(t.Name, chain)
				if err != nil {
					return nil, fmt.Errorf("error parsing rule %q: %v", line, err)
				}
				if len(c.Rules) != 0 {
					return nil, fmt.Errorf("cannot delete chain %q after adding rules", chain)
				}
				c.Deleted = true
			} else if line == "COMMIT" {
				state = parseTableDeclaration
			} else {
				return nil, fmt.Errorf("error parsing rule %q", line)
			}
		}
	}

	if state != parseTableDeclaration {
		return nil, fmt.Errorf("could not parse iptables data (no COMMIT line?)")
	}

	return dump, nil
}

func (dump *IPTablesDump) String() string {
	buffer := &strings.Builder{}
	for _, t := range dump.Tables {
		fmt.Fprintf(buffer, "*%s\n", t.Name)
		for _, c := range t.Chains {
			fmt.Fprintf(buffer, ":%s - [%d:%d]\n", c.Name, c.Packets, c.Bytes)
		}
		for _, c := range t.Chains {
			for _, r := range c.Rules {
				fmt.Fprintf(buffer, "%s\n", r.Raw)
			}
		}
		for _, c := range t.Chains {
			if c.Deleted {
				fmt.Fprintf(buffer, "-X %s\n", c.Name)
			}
		}
		fmt.Fprintf(buffer, "COMMIT\n")
	}
	return buffer.String()
}

func (dump *IPTablesDump) GetTable(table iptables.Table) (*Table, error) {
	for i := range dump.Tables {
		if dump.Tables[i].Name == table {
			return &dump.Tables[i], nil
		}
	}
	return nil, fmt.Errorf("no such table %q", table)
}

func (dump *IPTablesDump) GetChain(table iptables.Table, chain iptables.Chain) (*Chain, error) {
	t, err := dump.GetTable(table)
	if err != nil {
		return nil, err
	}
	for i := range t.Chains {
		if t.Chains[i].Name == chain {
			return &t.Chains[i], nil
		}
	}
	return nil, fmt.Errorf("no such chain %q", chain)
}

// Rule represents a single parsed IPTables rule. (This currently covers all of the rule
// types that we actually use in pkg/proxy/iptables or pkg/proxy/ipvs.)
//
// The parsing is mostly-automated based on type reflection. The `param` tag on a field
// indicates the parameter whose value will be placed into that field. (The code assumes
// that we don't use both the short and long forms of any parameter names (eg, "-s" vs
// "--source"), which is currently true, but it could be extended if necessary.) The
// `negatable` tag indicates if a parameter is allowed to be preceded by "!".
//
// Parameters that take a value are stored as type `*IPTablesValue`, which encapsulates a
// string value and whether the rule was negated (ie, whether the rule requires that we
// *match* or *don't match* that value). But string-valued parameters that can't be
// negated use `IPTablesValue` rather than `string` too, just for API consistency.
//
// Parameters that don't take a value are stored as `*bool`, where the value is `nil` if
// the parameter was not present, `&true` if the parameter was present, or `&false` if the
// parameter was present but negated.
//
// Parsing skips over "-m MODULE" parameters because most parameters have unique names
// anyway even ignoring the module name, and in the cases where they don't (eg "-m tcp
// --sport" vs "-m udp --sport") the parameters are mutually-exclusive and it's more
// convenient to store them in the same struct field anyway.
type Rule struct {
	// Raw contains the original raw rule string
	Raw string

	Chain   iptables.Chain `param:"-A"`
	Comment *IPTablesValue `param:"--comment"`

	Protocol *IPTablesValue `param:"-p" negatable:"true"`

	SourceAddress *IPTablesValue `param:"-s" negatable:"true"`
	SourceType    *IPTablesValue `param:"--src-type" negatable:"true"`
	SourcePort    *IPTablesValue `param:"--sport" negatable:"true"`

	DestinationAddress *IPTablesValue `param:"-d" negatable:"true"`
	DestinationType    *IPTablesValue `param:"--dst-type" negatable:"true"`
	DestinationPort    *IPTablesValue `param:"--dport" negatable:"true"`

	MatchSet *IPTablesValue `param:"--match-set" negatable:"true"`

	Jump            *IPTablesValue `param:"-j"`
	RandomFully     *bool          `param:"--random-fully"`
	Probability     *IPTablesValue `param:"--probability"`
	DNATDestination *IPTablesValue `param:"--to-destination"`

	// We don't actually use the values of these, but we care if they are present
	AffinityCheck *bool          `param:"--rcheck" negatable:"true"`
	MarkCheck     *IPTablesValue `param:"--mark" negatable:"true"`
	CTStateCheck  *IPTablesValue `param:"--ctstate" negatable:"true"`

	// We don't currently care about any of these in the unit tests, but we expect
	// them to be present in some rules that we parse, so we define how to parse them.
	AffinityName    *IPTablesValue `param:"--name"`
	AffinitySeconds *IPTablesValue `param:"--seconds"`
	AffinitySet     *bool          `param:"--set" negatable:"true"`
	AffinityReap    *bool          `param:"--reap"`
	StatisticMode   *IPTablesValue `param:"--mode"`
}

// IPTablesValue is a value of a parameter in an Rule, where the parameter is
// possibly negated.
type IPTablesValue struct {
	Negated bool
	Value   string
}

// for debugging; otherwise %v will just print the pointer value
func (v *IPTablesValue) String() string {
	if v.Negated {
		return fmt.Sprintf("NOT %q", v.Value)
	} else {
		return fmt.Sprintf("%q", v.Value)
	}
}

// Matches returns true if cmp equals / doesn't equal v.Value (depending on
// v.Negated).
func (v *IPTablesValue) Matches(cmp string) bool {
	if v.Negated {
		return v.Value != cmp
	} else {
		return v.Value == cmp
	}
}

// findParamField finds a field in value with the struct tag "param:${param}" and if found,
// returns a pointer to the Value of that field, and the value of its "negatable" tag.
func findParamField(value reflect.Value, param string) (*reflect.Value, bool) {
	typ := value.Type()
	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		if field.Tag.Get("param") == param {
			fValue := value.Field(i)
			return &fValue, field.Tag.Get("negatable") == "true"
		}
	}
	return nil, false
}

// wordRegex matches a single word or a quoted string (at the start of the string, or
// preceded by whitespace)
var wordRegex = regexp.MustCompile(`(?:^|\s)("[^"]*"|[^"]\S*)`)

// Used by ParseRule
var boolPtrType = reflect.PointerTo(reflect.TypeOf(true))
var ipTablesValuePtrType = reflect.TypeOf((*IPTablesValue)(nil))

// ParseRule parses rule. If strict is false, it will parse the recognized
// parameters and ignore unrecognized ones. If it is true, parsing will fail if there are
// unrecognized parameters.
func ParseRule(rule string, strict bool) (*Rule, error) {
	parsed := &Rule{Raw: rule}

	// Split rule into "words" (where a quoted string is a single word).
	var words []string
	for _, match := range wordRegex.FindAllStringSubmatch(rule, -1) {
		words = append(words, strings.Trim(match[1], `"`))
	}

	// The chain name must come first (and can't be the only thing there)
	if len(words) < 2 || words[0] != "-A" {
		return nil, fmt.Errorf(`bad iptables rule (does not start with "-A CHAIN")`)
	} else if len(words) < 3 {
		return nil, fmt.Errorf("bad iptables rule (no match rules)")
	}

	// For each word, see if it is a known iptables parameter, based on the struct
	// field tags in Rule. Note that in the non-strict case we implicitly assume that
	// no unrecognized parameter will take an argument that could be mistaken for
	// another parameter.
	v := reflect.ValueOf(parsed).Elem()
	negated := false
	for w := 0; w < len(words); {
		if words[w] == "-m" && w < len(words)-1 {
			// Skip "-m MODULE"; we don't pay attention to that since the
			// parameter names are unique enough anyway.
			w += 2
			continue
		}

		if words[w] == "!" {
			negated = true
			w++
			continue
		}

		// For everything else, see if it corresponds to a field of Rule
		if field, negatable := findParamField(v, words[w]); field != nil {
			if negated && !negatable {
				return nil, fmt.Errorf("cannot negate parameter %q", words[w])
			}
			if field.Type() != boolPtrType && w == len(words)-1 {
				return nil, fmt.Errorf("parameter %q requires an argument", words[w])
			}
			switch field.Type() {
			case boolPtrType:
				boolVal := !negated
				field.Set(reflect.ValueOf(&boolVal))
				w++
			case ipTablesValuePtrType:
				field.Set(reflect.ValueOf(&IPTablesValue{Negated: negated, Value: words[w+1]}))
				w += 2
			default:
				field.SetString(words[w+1])
				w += 2
			}
		} else if strict {
			return nil, fmt.Errorf("unrecognized parameter %q", words[w])
		} else {
			// skip
			w++
		}

		negated = false
	}

	return parsed, nil
}
