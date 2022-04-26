// (c) Copyright 2016 Hewlett Packard Enterprise Development LP
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

package rules

import "github.com/securego/gosec/v2"

// RuleDefinition contains the description of a rule and a mechanism to
// create it.
type RuleDefinition struct {
	ID          string
	Description string
	Create      gosec.RuleBuilder
}

// RuleList contains a mapping of rule ID's to rule definitions and a mapping
// of rule ID's to whether rules are suppressed.
type RuleList struct {
	Rules          map[string]RuleDefinition
	RuleSuppressed map[string]bool
}

// RulesInfo returns all the create methods and the rule suppressed map for a
// given list
func (rl RuleList) RulesInfo() (map[string]gosec.RuleBuilder, map[string]bool) {
	builders := make(map[string]gosec.RuleBuilder)
	for _, def := range rl.Rules {
		builders[def.ID] = def.Create
	}
	return builders, rl.RuleSuppressed
}

// RuleFilter can be used to include or exclude a rule depending on the return
// value of the function
type RuleFilter func(string) bool

// NewRuleFilter is a closure that will include/exclude the rule ID's based on
// the supplied boolean value.
func NewRuleFilter(action bool, ruleIDs ...string) RuleFilter {
	rulelist := make(map[string]bool)
	for _, rule := range ruleIDs {
		rulelist[rule] = true
	}
	return func(rule string) bool {
		if _, found := rulelist[rule]; found {
			return action
		}
		return !action
	}
}

// Generate the list of rules to use
func Generate(trackSuppressions bool, filters ...RuleFilter) RuleList {
	rules := []RuleDefinition{
		// misc
		{"G101", "Look for hardcoded credentials", NewHardcodedCredentials},
		{"G102", "Bind to all interfaces", NewBindsToAllNetworkInterfaces},
		{"G103", "Audit the use of unsafe block", NewUsingUnsafe},
		{"G104", "Audit errors not checked", NewNoErrorCheck},
		{"G106", "Audit the use of ssh.InsecureIgnoreHostKey function", NewSSHHostKey},
		{"G107", "Url provided to HTTP request as taint input", NewSSRFCheck},
		{"G108", "Profiling endpoint is automatically exposed", NewPprofCheck},
		{"G109", "Converting strconv.Atoi result to int32/int16", NewIntegerOverflowCheck},
		{"G110", "Detect io.Copy instead of io.CopyN when decompression", NewDecompressionBombCheck},

		// injection
		{"G201", "SQL query construction using format string", NewSQLStrFormat},
		{"G202", "SQL query construction using string concatenation", NewSQLStrConcat},
		{"G203", "Use of unescaped data in HTML templates", NewTemplateCheck},
		{"G204", "Audit use of command execution", NewSubproc},

		// filesystem
		{"G301", "Poor file permissions used when creating a directory", NewMkdirPerms},
		{"G302", "Poor file permissions used when creation file or using chmod", NewFilePerms},
		{"G303", "Creating tempfile using a predictable path", NewBadTempFile},
		{"G304", "File path provided as taint input", NewReadFile},
		{"G305", "File path traversal when extracting zip archive", NewArchive},
		{"G306", "Poor file permissions used when writing to a file", NewWritePerms},
		{"G307", "Unsafe defer call of a method returning an error", NewDeferredClosing},

		// crypto
		{"G401", "Detect the usage of DES, RC4, MD5 or SHA1", NewUsesWeakCryptography},
		{"G402", "Look for bad TLS connection settings", NewIntermediateTLSCheck},
		{"G403", "Ensure minimum RSA key length of 2048 bits", NewWeakKeyStrength},
		{"G404", "Insecure random number source (rand)", NewWeakRandCheck},

		// blocklist
		{"G501", "Import blocklist: crypto/md5", NewBlocklistedImportMD5},
		{"G502", "Import blocklist: crypto/des", NewBlocklistedImportDES},
		{"G503", "Import blocklist: crypto/rc4", NewBlocklistedImportRC4},
		{"G504", "Import blocklist: net/http/cgi", NewBlocklistedImportCGI},
		{"G505", "Import blocklist: crypto/sha1", NewBlocklistedImportSHA1},

		// memory safety
		{"G601", "Implicit memory aliasing in RangeStmt", NewImplicitAliasing},
	}

	ruleMap := make(map[string]RuleDefinition)
	ruleSuppressedMap := make(map[string]bool)

RULES:
	for _, rule := range rules {
		ruleSuppressedMap[rule.ID] = false
		for _, filter := range filters {
			if filter(rule.ID) {
				ruleSuppressedMap[rule.ID] = true
				if !trackSuppressions {
					continue RULES
				}
			}
		}
		ruleMap[rule.ID] = rule
	}
	return RuleList{ruleMap, ruleSuppressedMap}
}
