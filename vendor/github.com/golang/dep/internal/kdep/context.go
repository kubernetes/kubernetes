/*
 * Copyright 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package kdep

import (
	"fmt"
	"path/filepath"

	"github.com/golang/dep"
	"github.com/golang/dep/gps/pkgtree"
)

// Ctx wraps dep.Ctx to support kdep projects
type Ctx struct {
	*dep.Ctx
}

// LoadProject finds the first dep project with a kdep-project flag
func (c *Ctx) LoadProject() (*Project, error) {
	if FallbackToDep {
		p, err := c.Ctx.LoadProject()
		kp, _ := WrapProject(p, c)
		return kp, err
	}
	for {
		p, err := c.Ctx.LoadProject()
		if err != nil {
			return nil, err
		}

		proj, err := WrapProject(p, c)
		if err == nil {
			// honor kdep uninteresting tags as declared in the main project's
			// manifest
			setupBuildTagsStrategy(proj)
			return proj, err
		}

		parent := filepath.Dir(c.WorkingDir)
		if parent == c.WorkingDir {
			return nil, fmt.Errorf("no project found")
		}
		c.SetPaths(parent, c.GOPATHs...)
	}
}

func setupBuildTagsStrategy(p *Project) {
	if FallbackToDep {
		return
	}
	s := pkgtree.DefaultStrategy()
	for _, t := range p.Manifest.Meta.UninterestingTags {
		s.AddUninterestingTag(t)
	}
}
