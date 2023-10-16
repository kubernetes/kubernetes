/*
Copyright 2013 Google Inc. All Rights Reserved.
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

package verbosity

import (
	"bytes"
	"errors"
	"flag"
	"fmt"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

// New returns a struct that implements -v and -vmodule support. Changing and
// checking these settings is thread-safe, with all concurrency issues handled
// internally.
func New() *VState {
	vs := new(VState)

	// The two fields must have a pointer to the overal struct for their
	// implementation of Set.
	vs.vmodule.vs = vs
	vs.verbosity.vs = vs

	return vs
}

// Value is an extension that makes it possible to use the values in pflag.
type Value interface {
	flag.Value
	Type() string
}

func (vs *VState) V() Value {
	return &vs.verbosity
}

func (vs *VState) VModule() Value {
	return &vs.vmodule
}

// VState contains settings and state. Some of its fields can be accessed
// through atomic read/writes, in other cases a mutex must be held.
type VState struct {
	mu sync.Mutex

	// These flags are modified only under lock, although verbosity may be fetched
	// safely using atomic.LoadInt32.
	vmodule   moduleSpec // The state of the -vmodule flag.
	verbosity levelSpec  // V logging level, the value of the -v flag/

	// pcs is used in V to avoid an allocation when computing the caller's PC.
	pcs [1]uintptr
	// vmap is a cache of the V Level for each V() call site, identified by PC.
	// It is wiped whenever the vmodule flag changes state.
	vmap map[uintptr]Level
	// filterLength stores the length of the vmodule filter chain. If greater
	// than zero, it means vmodule is enabled. It may be read safely
	// using sync.LoadInt32, but is only modified under mu.
	filterLength int32
}

// Level must be an int32 to support atomic read/writes.
type Level int32

type levelSpec struct {
	vs *VState
	l  Level
}

// get returns the value of the level.
func (l *levelSpec) get() Level {
	return Level(atomic.LoadInt32((*int32)(&l.l)))
}

// set sets the value of the level.
func (l *levelSpec) set(val Level) {
	atomic.StoreInt32((*int32)(&l.l), int32(val))
}

// String is part of the flag.Value interface.
func (l *levelSpec) String() string {
	return strconv.FormatInt(int64(l.l), 10)
}

// Get is part of the flag.Getter interface. It returns the
// verbosity level as int32.
func (l *levelSpec) Get() interface{} {
	return int32(l.l)
}

// Type is part of pflag.Value.
func (l *levelSpec) Type() string {
	return "Level"
}

// Set is part of the flag.Value interface.
func (l *levelSpec) Set(value string) error {
	v, err := strconv.ParseInt(value, 10, 32)
	if err != nil {
		return err
	}
	l.vs.mu.Lock()
	defer l.vs.mu.Unlock()
	l.vs.set(Level(v), l.vs.vmodule.filter, false)
	return nil
}

// moduleSpec represents the setting of the -vmodule flag.
type moduleSpec struct {
	vs     *VState
	filter []modulePat
}

// modulePat contains a filter for the -vmodule flag.
// It holds a verbosity level and a file pattern to match.
type modulePat struct {
	pattern string
	literal bool // The pattern is a literal string
	level   Level
}

// match reports whether the file matches the pattern. It uses a string
// comparison if the pattern contains no metacharacters.
func (m *modulePat) match(file string) bool {
	if m.literal {
		return file == m.pattern
	}
	match, _ := filepath.Match(m.pattern, file)
	return match
}

func (m *moduleSpec) String() string {
	// Lock because the type is not atomic. TODO: clean this up.
	// Empty instances don't have and don't need a lock (can
	// happen when flag uses introspection).
	if m.vs != nil {
		m.vs.mu.Lock()
		defer m.vs.mu.Unlock()
	}
	var b bytes.Buffer
	for i, f := range m.filter {
		if i > 0 {
			b.WriteRune(',')
		}
		fmt.Fprintf(&b, "%s=%d", f.pattern, f.level)
	}
	return b.String()
}

// Get is part of the (Go 1.2)  flag.Getter interface. It always returns nil for this flag type since the
// struct is not exported.
func (m *moduleSpec) Get() interface{} {
	return nil
}

// Type is part of pflag.Value
func (m *moduleSpec) Type() string {
	return "pattern=N,..."
}

var errVmoduleSyntax = errors.New("syntax error: expect comma-separated list of filename=N")

// Set will sets module value
// Syntax: -vmodule=recordio=2,file=1,gfs*=3
func (m *moduleSpec) Set(value string) error {
	var filter []modulePat
	for _, pat := range strings.Split(value, ",") {
		if len(pat) == 0 {
			// Empty strings such as from a trailing comma can be ignored.
			continue
		}
		patLev := strings.Split(pat, "=")
		if len(patLev) != 2 || len(patLev[0]) == 0 || len(patLev[1]) == 0 {
			return errVmoduleSyntax
		}
		pattern := patLev[0]
		v, err := strconv.ParseInt(patLev[1], 10, 32)
		if err != nil {
			return errors.New("syntax error: expect comma-separated list of filename=N")
		}
		if v < 0 {
			return errors.New("negative value for vmodule level")
		}
		if v == 0 {
			continue // Ignore. It's harmless but no point in paying the overhead.
		}
		// TODO: check syntax of filter?
		filter = append(filter, modulePat{pattern, isLiteral(pattern), Level(v)})
	}
	m.vs.mu.Lock()
	defer m.vs.mu.Unlock()
	m.vs.set(m.vs.verbosity.l, filter, true)
	return nil
}

// isLiteral reports whether the pattern is a literal string, that is, has no metacharacters
// that require filepath.Match to be called to match the pattern.
func isLiteral(pattern string) bool {
	return !strings.ContainsAny(pattern, `\*?[]`)
}

// set sets a consistent state for V logging.
// The mutex must be held.
func (vs *VState) set(l Level, filter []modulePat, setFilter bool) {
	// Turn verbosity off so V will not fire while we are in transition.
	vs.verbosity.set(0)
	// Ditto for filter length.
	atomic.StoreInt32(&vs.filterLength, 0)

	// Set the new filters and wipe the pc->Level map if the filter has changed.
	if setFilter {
		vs.vmodule.filter = filter
		vs.vmap = make(map[uintptr]Level)
	}

	// Things are consistent now, so enable filtering and verbosity.
	// They are enabled in order opposite to that in V.
	atomic.StoreInt32(&vs.filterLength, int32(len(filter)))
	vs.verbosity.set(l)
}

// Enabled checks whether logging is enabled at the given level. This must be
// called with depth=0 when the caller of enabled will do the logging and
// higher values when more stack levels need to be skipped.
//
// The mutex will be locked only if needed.
func (vs *VState) Enabled(level Level, depth int) bool {
	// This function tries hard to be cheap unless there's work to do.
	// The fast path is two atomic loads and compares.

	// Here is a cheap but safe test to see if V logging is enabled globally.
	if vs.verbosity.get() >= level {
		return true
	}

	// It's off globally but vmodule may still be set.
	// Here is another cheap but safe test to see if vmodule is enabled.
	if atomic.LoadInt32(&vs.filterLength) > 0 {
		// Now we need a proper lock to use the logging structure. The pcs field
		// is shared so we must lock before accessing it. This is fairly expensive,
		// but if V logging is enabled we're slow anyway.
		vs.mu.Lock()
		defer vs.mu.Unlock()
		if runtime.Callers(depth+2, vs.pcs[:]) == 0 {
			return false
		}
		// runtime.Callers returns "return PCs", but we want
		// to look up the symbolic information for the call,
		// so subtract 1 from the PC. runtime.CallersFrames
		// would be cleaner, but allocates.
		pc := vs.pcs[0] - 1
		v, ok := vs.vmap[pc]
		if !ok {
			v = vs.setV(pc)
		}
		return v >= level
	}
	return false
}

// setV computes and remembers the V level for a given PC
// when vmodule is enabled.
// File pattern matching takes the basename of the file, stripped
// of its .go suffix, and uses filepath.Match, which is a little more
// general than the *? matching used in C++.
// Mutex is held.
func (vs *VState) setV(pc uintptr) Level {
	fn := runtime.FuncForPC(pc)
	file, _ := fn.FileLine(pc)
	// The file is something like /a/b/c/d.go. We want just the d.
	file = strings.TrimSuffix(file, ".go")
	if slash := strings.LastIndex(file, "/"); slash >= 0 {
		file = file[slash+1:]
	}
	for _, filter := range vs.vmodule.filter {
		if filter.match(file) {
			vs.vmap[pc] = filter.level
			return filter.level
		}
	}
	vs.vmap[pc] = 0
	return 0
}
