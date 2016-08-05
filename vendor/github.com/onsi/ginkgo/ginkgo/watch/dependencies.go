package watch

import (
	"go/build"
	"regexp"
)

var ginkgoAndGomegaFilter = regexp.MustCompile(`github\.com/onsi/ginkgo|github\.com/onsi/gomega`)

type Dependencies struct {
	deps map[string]int
}

func NewDependencies(path string, maxDepth int) (Dependencies, error) {
	d := Dependencies{
		deps: map[string]int{},
	}

	if maxDepth == 0 {
		return d, nil
	}

	err := d.seedWithDepsForPackageAtPath(path)
	if err != nil {
		return d, err
	}

	for depth := 1; depth < maxDepth; depth++ {
		n := len(d.deps)
		d.addDepsForDepth(depth)
		if n == len(d.deps) {
			break
		}
	}

	return d, nil
}

func (d Dependencies) Dependencies() map[string]int {
	return d.deps
}

func (d Dependencies) seedWithDepsForPackageAtPath(path string) error {
	pkg, err := build.ImportDir(path, 0)
	if err != nil {
		return err
	}

	d.resolveAndAdd(pkg.Imports, 1)
	d.resolveAndAdd(pkg.TestImports, 1)
	d.resolveAndAdd(pkg.XTestImports, 1)

	delete(d.deps, pkg.Dir)
	return nil
}

func (d Dependencies) addDepsForDepth(depth int) {
	for dep, depDepth := range d.deps {
		if depDepth == depth {
			d.addDepsForDep(dep, depth+1)
		}
	}
}

func (d Dependencies) addDepsForDep(dep string, depth int) {
	pkg, err := build.ImportDir(dep, 0)
	if err != nil {
		println(err.Error())
		return
	}
	d.resolveAndAdd(pkg.Imports, depth)
}

func (d Dependencies) resolveAndAdd(deps []string, depth int) {
	for _, dep := range deps {
		pkg, err := build.Import(dep, ".", 0)
		if err != nil {
			continue
		}
		if pkg.Goroot == false && !ginkgoAndGomegaFilter.Match([]byte(pkg.Dir)) {
			d.addDepIfNotPresent(pkg.Dir, depth)
		}
	}
}

func (d Dependencies) addDepIfNotPresent(dep string, depth int) {
	_, ok := d.deps[dep]
	if !ok {
		d.deps[dep] = depth
	}
}
