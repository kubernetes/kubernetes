// +build linux

// SPDX-License-Identifier: Apache-2.0
/*
 * Copyright (C) 2020 Aleksa Sarai <cyphar@cyphar.com>
 * Copyright (C) 2020 SUSE LLC
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
 */

package devices

import (
	"bufio"
	"io"
	"regexp"
	"sort"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/devices"

	"github.com/pkg/errors"
)

// deviceMeta is a Rule without the Allow or Permissions fields, and no
// wildcard-type support. It's effectively the "match" portion of a metadata
// rule, for the purposes of our emulation.
type deviceMeta struct {
	node  devices.Type
	major int64
	minor int64
}

// deviceRule is effectively the tuple (deviceMeta, Permissions).
type deviceRule struct {
	meta  deviceMeta
	perms devices.Permissions
}

// deviceRules is a mapping of device metadata rules to the associated
// permissions in the ruleset.
type deviceRules map[deviceMeta]devices.Permissions

func (r deviceRules) orderedEntries() []deviceRule {
	var rules []deviceRule
	for meta, perms := range r {
		rules = append(rules, deviceRule{meta: meta, perms: perms})
	}
	sort.Slice(rules, func(i, j int) bool {
		// Sort by (major, minor, type).
		a, b := rules[i].meta, rules[j].meta
		return a.major < b.major ||
			(a.major == b.major && a.minor < b.minor) ||
			(a.major == b.major && a.minor == b.minor && a.node < b.node)
	})
	return rules
}

type Emulator struct {
	defaultAllow bool
	rules        deviceRules
}

func (e *Emulator) IsBlacklist() bool {
	return e.defaultAllow
}

func (e *Emulator) IsAllowAll() bool {
	return e.IsBlacklist() && len(e.rules) == 0
}

var devicesListRegexp = regexp.MustCompile(`^([abc])\s+(\d+|\*):(\d+|\*)\s+([rwm]+)$`)

func parseLine(line string) (*deviceRule, error) {
	matches := devicesListRegexp.FindStringSubmatch(line)
	if matches == nil {
		return nil, errors.Errorf("line doesn't match devices.list format")
	}
	var (
		rule  deviceRule
		node  = matches[1]
		major = matches[2]
		minor = matches[3]
		perms = matches[4]
	)

	// Parse the node type.
	switch node {
	case "a":
		// Super-special case -- "a" always means every device with every
		// access mode. In fact, for devices.list this actually indicates that
		// the cgroup is in black-list mode.
		// TODO: Double-check that the entire file is "a *:* rwm".
		return nil, nil
	case "b":
		rule.meta.node = devices.BlockDevice
	case "c":
		rule.meta.node = devices.CharDevice
	default:
		// Should never happen!
		return nil, errors.Errorf("unknown device type %q", node)
	}

	// Parse the major number.
	if major == "*" {
		rule.meta.major = devices.Wildcard
	} else {
		val, err := strconv.ParseUint(major, 10, 32)
		if err != nil {
			return nil, errors.Wrap(err, "parse major number")
		}
		rule.meta.major = int64(val)
	}

	// Parse the minor number.
	if minor == "*" {
		rule.meta.minor = devices.Wildcard
	} else {
		val, err := strconv.ParseUint(minor, 10, 32)
		if err != nil {
			return nil, errors.Wrap(err, "parse minor number")
		}
		rule.meta.minor = int64(val)
	}

	// Parse the access permissions.
	rule.perms = devices.Permissions(perms)
	if !rule.perms.IsValid() || rule.perms.IsEmpty() {
		// Should never happen!
		return nil, errors.Errorf("parse access mode: contained unknown modes or is empty: %q", perms)
	}
	return &rule, nil
}

func (e *Emulator) addRule(rule deviceRule) error {
	if e.rules == nil {
		e.rules = make(map[deviceMeta]devices.Permissions)
	}

	// Merge with any pre-existing permissions.
	oldPerms := e.rules[rule.meta]
	newPerms := rule.perms.Union(oldPerms)
	e.rules[rule.meta] = newPerms
	return nil
}

func (e *Emulator) rmRule(rule deviceRule) error {
	// Give an error if any of the permissions requested to be removed are
	// present in a partially-matching wildcard rule, because such rules will
	// be ignored by cgroupv1.
	//
	// This is a diversion from cgroupv1, but is necessary to avoid leading
	// users into a false sense of security. cgroupv1 will silently(!) ignore
	// requests to remove partial exceptions, but we really shouldn't do that.
	//
	// It may seem like we could just "split" wildcard rules which hit this
	// issue, but unfortunately there are 2^32 possible major and minor
	// numbers, which would exhaust kernel memory quickly if we did this. Not
	// to mention it'd be really slow (the kernel side is implemented as a
	// linked-list of exceptions).
	for _, partialMeta := range []deviceMeta{
		{node: rule.meta.node, major: devices.Wildcard, minor: rule.meta.minor},
		{node: rule.meta.node, major: rule.meta.major, minor: devices.Wildcard},
		{node: rule.meta.node, major: devices.Wildcard, minor: devices.Wildcard},
	} {
		// This wildcard rule is equivalent to the requested rule, so skip it.
		if rule.meta == partialMeta {
			continue
		}
		// Only give an error if the set of permissions overlap.
		partialPerms := e.rules[partialMeta]
		if !partialPerms.Intersection(rule.perms).IsEmpty() {
			return errors.Errorf("requested rule [%v %v] not supported by devices cgroupv1 (cannot punch hole in existing wildcard rule [%v %v])", rule.meta, rule.perms, partialMeta, partialPerms)
		}
	}

	// Subtract all of the permissions listed from the full match rule. If the
	// rule didn't exist, all of this is a no-op.
	newPerms := e.rules[rule.meta].Difference(rule.perms)
	if newPerms.IsEmpty() {
		delete(e.rules, rule.meta)
	} else {
		e.rules[rule.meta] = newPerms
	}
	// TODO: The actual cgroup code doesn't care if an exception didn't exist
	//       during removal, so not erroring out here is /accurate/ but quite
	//       worrying. Maybe we should do additional validation, but again we
	//       have to worry about backwards-compatibility.
	return nil
}

func (e *Emulator) allow(rule *deviceRule) error {
	// This cgroup is configured as a black-list. Reset the entire emulator,
	// and put is into black-list mode.
	if rule == nil || rule.meta.node == devices.WildcardDevice {
		*e = Emulator{
			defaultAllow: true,
			rules:        nil,
		}
		return nil
	}

	var err error
	if e.defaultAllow {
		err = errors.Wrap(e.rmRule(*rule), "remove 'deny' exception")
	} else {
		err = errors.Wrap(e.addRule(*rule), "add 'allow' exception")
	}
	return err
}

func (e *Emulator) deny(rule *deviceRule) error {
	// This cgroup is configured as a white-list. Reset the entire emulator,
	// and put is into white-list mode.
	if rule == nil || rule.meta.node == devices.WildcardDevice {
		*e = Emulator{
			defaultAllow: false,
			rules:        nil,
		}
		return nil
	}

	var err error
	if e.defaultAllow {
		err = errors.Wrap(e.addRule(*rule), "add 'deny' exception")
	} else {
		err = errors.Wrap(e.rmRule(*rule), "remove 'allow' exception")
	}
	return err
}

func (e *Emulator) Apply(rule devices.Rule) error {
	if !rule.Type.CanCgroup() {
		return errors.Errorf("cannot add rule [%#v] with non-cgroup type %q", rule, rule.Type)
	}

	innerRule := &deviceRule{
		meta: deviceMeta{
			node:  rule.Type,
			major: rule.Major,
			minor: rule.Minor,
		},
		perms: rule.Permissions,
	}
	if innerRule.meta.node == devices.WildcardDevice {
		innerRule = nil
	}

	if rule.Allow {
		return e.allow(innerRule)
	} else {
		return e.deny(innerRule)
	}
}

// EmulatorFromList takes a reader to a "devices.list"-like source, and returns
// a new Emulator that represents the state of the devices cgroup. Note that
// black-list devices cgroups cannot be fully reconstructed, due to limitations
// in the devices cgroup API. Instead, such cgroups are always treated as
// "allow all" cgroups.
func EmulatorFromList(list io.Reader) (*Emulator, error) {
	// Normally cgroups are in black-list mode by default, but the way we
	// figure out the current mode is whether or not devices.list has an
	// allow-all rule. So we default to a white-list, and the existence of an
	// "a *:* rwm" entry will tell us otherwise.
	e := &Emulator{
		defaultAllow: false,
	}

	// Parse the "devices.list".
	s := bufio.NewScanner(list)
	for s.Scan() {
		line := s.Text()
		deviceRule, err := parseLine(line)
		if err != nil {
			return nil, errors.Wrapf(err, "parsing line %q", line)
		}
		// "devices.list" is an allow list. Note that this means that in
		// black-list mode, we have no idea what rules are in play. As a
		// result, we need to be very careful in Transition().
		if err := e.allow(deviceRule); err != nil {
			return nil, errors.Wrapf(err, "adding devices.list rule")
		}
	}
	if err := s.Err(); err != nil {
		return nil, errors.Wrap(err, "reading devices.list lines")
	}
	return e, nil
}

// Transition calculates what is the minimally-disruptive set of rules need to
// be applied to a devices cgroup in order to transition to the given target.
// This means that any already-existing rules will not be applied, and
// disruptive rules (like denying all device access) will only be applied if
// necessary.
//
// This function is the sole reason for all of Emulator -- to allow us
// to figure out how to update a containers' cgroups without causing spurrious
// device errors (if possible).
func (source *Emulator) Transition(target *Emulator) ([]*devices.Rule, error) {
	var transitionRules []*devices.Rule
	oldRules := source.rules

	// If the default policy doesn't match, we need to include a "disruptive"
	// rule (either allow-all or deny-all) in order to switch the cgroup to the
	// correct default policy.
	//
	// However, due to a limitation in "devices.list" we cannot be sure what
	// deny rules are in place in a black-list cgroup. Thus if the source is a
	// black-list we also have to include a disruptive rule.
	if source.IsBlacklist() || source.defaultAllow != target.defaultAllow {
		transitionRules = append(transitionRules, &devices.Rule{
			Type:        'a',
			Major:       -1,
			Minor:       -1,
			Permissions: devices.Permissions("rwm"),
			Allow:       target.defaultAllow,
		})
		// The old rules are only relevant if we aren't starting out with a
		// disruptive rule.
		oldRules = nil
	}

	// NOTE: We traverse through the rules in a sorted order so we always write
	//       the same set of rules (this is to aid testing).

	// First, we create inverse rules for any old rules not in the new set.
	// This includes partial-inverse rules for specific permissions. This is a
	// no-op if we added a disruptive rule, since oldRules will be empty.
	for _, rule := range oldRules.orderedEntries() {
		meta, oldPerms := rule.meta, rule.perms
		newPerms := target.rules[meta]
		droppedPerms := oldPerms.Difference(newPerms)
		if !droppedPerms.IsEmpty() {
			transitionRules = append(transitionRules, &devices.Rule{
				Type:        meta.node,
				Major:       meta.major,
				Minor:       meta.minor,
				Permissions: droppedPerms,
				Allow:       target.defaultAllow,
			})
		}
	}

	// Add any additional rules which weren't in the old set. We happen to
	// filter out rules which are present in both sets, though this isn't
	// strictly necessary.
	for _, rule := range target.rules.orderedEntries() {
		meta, newPerms := rule.meta, rule.perms
		oldPerms := oldRules[meta]
		gainedPerms := newPerms.Difference(oldPerms)
		if !gainedPerms.IsEmpty() {
			transitionRules = append(transitionRules, &devices.Rule{
				Type:        meta.node,
				Major:       meta.major,
				Minor:       meta.minor,
				Permissions: gainedPerms,
				Allow:       !target.defaultAllow,
			})
		}
	}
	return transitionRules, nil
}
