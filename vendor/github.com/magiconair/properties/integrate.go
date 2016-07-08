// Copyright 2016 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package properties

import "flag"

// MustFlag sets flags that are skipped by dst.Parse when p contains
// the respective key for flag.Flag.Name.
//
// It's use is recommended with command line arguments as in:
// 	flag.Parse()
// 	p.MustFlag(flag.CommandLine)
func (p *Properties) MustFlag(dst *flag.FlagSet) {
	m := make(map[string]*flag.Flag)
	dst.VisitAll(func(f *flag.Flag) {
		m[f.Name] = f
	})
	dst.Visit(func(f *flag.Flag) {
		delete(m, f.Name) // overridden
	})

	for name, f := range m {
		v, ok := p.Get(name)
		if !ok {
			continue
		}

		if err := f.Value.Set(v); err != nil {
			ErrorHandler(err)
		}
	}
}
