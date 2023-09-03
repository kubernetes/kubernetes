// Copyright 2019 The Kubernetes Authors.
// SPDX-License-Identifier: Apache-2.0

package runfn

import (
	"fmt"
	"io"
	"os"
	"os/user"
	"path"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"

	"sigs.k8s.io/kustomize/kyaml/errors"
	"sigs.k8s.io/kustomize/kyaml/fn/runtime/container"
	"sigs.k8s.io/kustomize/kyaml/fn/runtime/exec"
	"sigs.k8s.io/kustomize/kyaml/fn/runtime/runtimeutil"
	"sigs.k8s.io/kustomize/kyaml/fn/runtime/starlark"
	"sigs.k8s.io/kustomize/kyaml/kio"
	"sigs.k8s.io/kustomize/kyaml/kio/kioutil"
	"sigs.k8s.io/kustomize/kyaml/yaml"
)

// RunFns runs the set of configuration functions in a local directory against
// the Resources in that directory
type RunFns struct {
	StorageMounts []runtimeutil.StorageMount

	// Path is the path to the directory containing functions
	Path string

	// FunctionPaths Paths allows functions to be specified outside the configuration
	// directory.
	// Functions provided on FunctionPaths are globally scoped.
	// If FunctionPaths length is > 0, then NoFunctionsFromInput defaults to true
	FunctionPaths []string

	// Functions is an explicit list of functions to run against the input.
	// Functions provided on Functions are globally scoped.
	// If Functions length is > 0, then NoFunctionsFromInput defaults to true
	Functions []*yaml.RNode

	// GlobalScope if true, functions read from input will be scoped globally rather
	// than only to Resources under their subdirs.
	GlobalScope bool

	// Input can be set to read the Resources from Input rather than from a directory
	Input io.Reader

	// Network enables network access for functions that declare it
	Network bool

	// Output can be set to write the result to Output rather than back to the directory
	Output io.Writer

	// NoFunctionsFromInput if set to true will not read any functions from the input,
	// and only use explicit sources
	NoFunctionsFromInput *bool

	// EnableStarlark will enable functions run as starlark scripts
	EnableStarlark bool

	// EnableExec will enable exec functions
	EnableExec bool

	// DisableContainers will disable functions run as containers
	DisableContainers bool

	// ResultsDir is where to write each functions results
	ResultsDir string

	// LogSteps enables logging the function that is running.
	LogSteps bool

	// LogWriter can be set to write the logs to LogWriter rather than stderr if LogSteps is enabled.
	LogWriter io.Writer

	// resultsCount is used to generate the results filename for each container
	resultsCount uint32

	// functionFilterProvider provides a filter to perform the function.
	// this is a variable so it can be mocked in tests
	functionFilterProvider func(
		filter runtimeutil.FunctionSpec, api *yaml.RNode, currentUser currentUserFunc) (kio.Filter, error)

	// AsCurrentUser is a boolean to indicate whether docker container should use
	// the uid and gid that run the command
	AsCurrentUser bool

	// Env contains environment variables that will be exported to container
	Env []string

	// ContinueOnEmptyResult configures what happens when the underlying pipeline
	// returns an empty result.
	// If it is false (default), subsequent functions will be skipped and the
	// result will be returned immediately.
	// If it is true, the empty result will be provided as input to the next
	// function in the list.
	ContinueOnEmptyResult bool

	// WorkingDir specifies which working directory an exec function should run in.
	WorkingDir string
}

// Execute runs the command
func (r RunFns) Execute() error {
	// make the path absolute so it works on mac
	var err error
	r.Path, err = filepath.Abs(r.Path)
	if err != nil {
		return errors.Wrap(err)
	}

	// default the containerFilterProvider if it hasn't been override.  Split out for testing.
	(&r).init()
	nodes, fltrs, output, err := r.getNodesAndFilters()
	if err != nil {
		return err
	}
	return r.runFunctions(nodes, output, fltrs)
}

func (r RunFns) getNodesAndFilters() (
	*kio.PackageBuffer, []kio.Filter, *kio.LocalPackageReadWriter, error) {
	// Read Resources from Directory or Input
	buff := &kio.PackageBuffer{}
	p := kio.Pipeline{Outputs: []kio.Writer{buff}}
	// save the output dir because we will need it to write back
	// the same one for reading must be used for writing if deleting Resources
	var outputPkg *kio.LocalPackageReadWriter
	if r.Path != "" {
		outputPkg = &kio.LocalPackageReadWriter{PackagePath: r.Path, MatchFilesGlob: kio.MatchAll}
	}

	if r.Input == nil {
		p.Inputs = []kio.Reader{outputPkg}
	} else {
		p.Inputs = []kio.Reader{&kio.ByteReader{Reader: r.Input}}
	}
	if err := p.Execute(); err != nil {
		return nil, nil, outputPkg, err
	}

	fltrs, err := r.getFilters(buff.Nodes)
	if err != nil {
		return nil, nil, outputPkg, err
	}
	return buff, fltrs, outputPkg, nil
}

func (r RunFns) getFilters(nodes []*yaml.RNode) ([]kio.Filter, error) {
	var fltrs []kio.Filter

	// fns from annotations on the input resources
	f, err := r.getFunctionsFromInput(nodes)
	if err != nil {
		return nil, err
	}
	fltrs = append(fltrs, f...)

	// fns from directories specified on the struct
	f, err = r.getFunctionsFromFunctionPaths()
	if err != nil {
		return nil, err
	}
	fltrs = append(fltrs, f...)

	// explicit fns specified on the struct
	f, err = r.getFunctionsFromFunctions()
	if err != nil {
		return nil, err
	}
	fltrs = append(fltrs, f...)

	return fltrs, nil
}

// runFunctions runs the fltrs against the input and writes to either r.Output or output
func (r RunFns) runFunctions(
	input kio.Reader, output kio.Writer, fltrs []kio.Filter) error {
	// use the previously read Resources as input
	var outputs []kio.Writer
	if r.Output == nil {
		// write back to the package
		outputs = append(outputs, output)
	} else {
		// write to the output instead of the directory if r.Output is specified or
		// the output is nil (reading from Input)
		outputs = append(outputs, kio.ByteWriter{Writer: r.Output})
	}

	var err error
	pipeline := kio.Pipeline{
		Inputs:                []kio.Reader{input},
		Filters:               fltrs,
		Outputs:               outputs,
		ContinueOnEmptyResult: r.ContinueOnEmptyResult,
	}
	if r.LogSteps {
		err = pipeline.ExecuteWithCallback(func(op kio.Filter) {
			var identifier string

			switch filter := op.(type) {
			case *container.Filter:
				identifier = filter.Image
			case *exec.Filter:
				identifier = filter.Path
			case *starlark.Filter:
				identifier = filter.String()
			default:
				identifier = "unknown-type function"
			}

			_, _ = fmt.Fprintf(r.LogWriter, "Running %s\n", identifier)
		})
	} else {
		err = pipeline.Execute()
	}
	if err != nil {
		return err
	}

	// check for deferred function errors
	var errs []string
	for i := range fltrs {
		cf, ok := fltrs[i].(runtimeutil.DeferFailureFunction)
		if !ok {
			continue
		}
		if cf.GetExit() != nil {
			errs = append(errs, cf.GetExit().Error())
		}
	}
	if len(errs) > 0 {
		return fmt.Errorf(strings.Join(errs, "\n---\n"))
	}
	return nil
}

// getFunctionsFromInput scans the input for functions and runs them
func (r RunFns) getFunctionsFromInput(nodes []*yaml.RNode) ([]kio.Filter, error) {
	if *r.NoFunctionsFromInput {
		return nil, nil
	}

	buff := &kio.PackageBuffer{}
	err := kio.Pipeline{
		Inputs:  []kio.Reader{&kio.PackageBuffer{Nodes: nodes}},
		Filters: []kio.Filter{&runtimeutil.IsReconcilerFilter{}},
		Outputs: []kio.Writer{buff},
	}.Execute()
	if err != nil {
		return nil, err
	}
	err = sortFns(buff)
	if err != nil {
		return nil, err
	}
	return r.getFunctionFilters(false, buff.Nodes...)
}

// getFunctionsFromFunctionPaths returns the set of functions read from r.FunctionPaths
// as a slice of Filters
func (r RunFns) getFunctionsFromFunctionPaths() ([]kio.Filter, error) {
	buff := &kio.PackageBuffer{}
	for i := range r.FunctionPaths {
		err := kio.Pipeline{
			Inputs: []kio.Reader{
				kio.LocalPackageReader{PackagePath: r.FunctionPaths[i]},
			},
			Outputs: []kio.Writer{buff},
		}.Execute()
		if err != nil {
			return nil, err
		}
	}
	return r.getFunctionFilters(true, buff.Nodes...)
}

// getFunctionsFromFunctions returns the set of explicitly provided functions as
// Filters
func (r RunFns) getFunctionsFromFunctions() ([]kio.Filter, error) {
	return r.getFunctionFilters(true, r.Functions...)
}

// mergeContainerEnv will merge the envs specified by command line (imperative) and config
// file (declarative). If they have same key, the imperative value will be respected.
func (r RunFns) mergeContainerEnv(envs []string) []string {
	imperative := runtimeutil.NewContainerEnvFromStringSlice(r.Env)
	declarative := runtimeutil.NewContainerEnvFromStringSlice(envs)
	for key, value := range imperative.EnvVars {
		declarative.AddKeyValue(key, value)
	}

	for _, key := range imperative.VarsToExport {
		declarative.AddKey(key)
	}

	return declarative.Raw()
}

func (r RunFns) getFunctionFilters(global bool, fns ...*yaml.RNode) (
	[]kio.Filter, error) {
	var fltrs []kio.Filter
	for i := range fns {
		api := fns[i]
		spec, err := runtimeutil.GetFunctionSpec(api)
		if err != nil {
			return nil, fmt.Errorf("failed to get FunctionSpec: %w", err)
		}
		if spec == nil {
			// resource doesn't have function spec
			continue
		}
		if spec.Container.Network && !r.Network {
			// TODO(eddiezane): Provide error info about which function needs the network
			return fltrs, errors.Errorf("network required but not enabled with --network")
		}
		// merge envs from imperative and declarative
		spec.Container.Env = r.mergeContainerEnv(spec.Container.Env)

		c, err := r.functionFilterProvider(*spec, api, user.Current)
		if err != nil {
			return nil, err
		}

		if c == nil {
			continue
		}
		cf, ok := c.(*container.Filter)
		if ok {
			if global {
				cf.Exec.GlobalScope = true
			}
			cf.Exec.WorkingDir = r.WorkingDir
		}
		fltrs = append(fltrs, c)
	}
	return fltrs, nil
}

// sortFns sorts functions so that functions with the longest paths come first
func sortFns(buff *kio.PackageBuffer) error {
	var outerErr error
	// sort the nodes so that we traverse them depth first
	// functions deeper in the file system tree should be run first
	sort.Slice(buff.Nodes, func(i, j int) bool {
		if err := kioutil.CopyLegacyAnnotations(buff.Nodes[i]); err != nil {
			return false
		}
		if err := kioutil.CopyLegacyAnnotations(buff.Nodes[j]); err != nil {
			return false
		}
		mi, _ := buff.Nodes[i].GetMeta()
		pi := filepath.ToSlash(mi.Annotations[kioutil.PathAnnotation])

		mj, _ := buff.Nodes[j].GetMeta()
		pj := filepath.ToSlash(mj.Annotations[kioutil.PathAnnotation])

		// If the path is the same, we decide the ordering based on the
		// index annotation.
		if pi == pj {
			iIndex, err := strconv.Atoi(mi.Annotations[kioutil.IndexAnnotation])
			if err != nil {
				outerErr = err
				return false
			}
			jIndex, err := strconv.Atoi(mj.Annotations[kioutil.IndexAnnotation])
			if err != nil {
				outerErr = err
				return false
			}
			return iIndex < jIndex
		}

		if filepath.Base(path.Dir(pi)) == "functions" {
			// don't count the functions dir, the functions are scoped 1 level above
			pi = filepath.Dir(path.Dir(pi))
		} else {
			pi = filepath.Dir(pi)
		}

		if filepath.Base(path.Dir(pj)) == "functions" {
			// don't count the functions dir, the functions are scoped 1 level above
			pj = filepath.Dir(path.Dir(pj))
		} else {
			pj = filepath.Dir(pj)
		}

		// i is "less" than j (comes earlier) if its depth is greater -- e.g. run
		// i before j if it is deeper in the directory structure
		li := len(strings.Split(pi, "/"))
		if pi == "." {
			// local dir should have 0 path elements instead of 1
			li = 0
		}
		lj := len(strings.Split(pj, "/"))
		if pj == "." {
			// local dir should have 0 path elements instead of 1
			lj = 0
		}
		if li != lj {
			// use greater-than because we want to sort with the longest
			// paths FIRST rather than last
			return li > lj
		}

		// sort by path names if depths are equal
		return pi < pj
	})
	return outerErr
}

// init initializes the RunFns with a containerFilterProvider.
func (r *RunFns) init() {
	if r.NoFunctionsFromInput == nil {
		// default no functions from input if any function sources are explicitly provided
		nfn := len(r.FunctionPaths) > 0 || len(r.Functions) > 0
		r.NoFunctionsFromInput = &nfn
	}

	// if no path is specified, default reading from stdin and writing to stdout
	if r.Path == "" {
		if r.Output == nil {
			r.Output = os.Stdout
		}
		if r.Input == nil {
			r.Input = os.Stdin
		}
	}

	// functionFilterProvider set the filter provider
	if r.functionFilterProvider == nil {
		r.functionFilterProvider = r.ffp
	}

	// if LogSteps is enabled and LogWriter is not specified, use stderr
	if r.LogSteps && r.LogWriter == nil {
		r.LogWriter = os.Stderr
	}
}

type currentUserFunc func() (*user.User, error)

// getUIDGID will return "nobody" if asCurrentUser is false. Otherwise
// return "uid:gid" according to the return from currentUser function.
func getUIDGID(asCurrentUser bool, currentUser currentUserFunc) (string, error) {
	if !asCurrentUser {
		return "nobody", nil
	}

	u, err := currentUser()
	if err != nil {
		return "", err
	}
	return fmt.Sprintf("%s:%s", u.Uid, u.Gid), nil
}

// ffp provides function filters
func (r *RunFns) ffp(spec runtimeutil.FunctionSpec, api *yaml.RNode, currentUser currentUserFunc) (kio.Filter, error) {
	var resultsFile string
	if r.ResultsDir != "" {
		resultsFile = filepath.Join(r.ResultsDir, fmt.Sprintf(
			"results-%v.yaml", r.resultsCount))
		atomic.AddUint32(&r.resultsCount, 1)
	}
	if !r.DisableContainers && spec.Container.Image != "" {
		// TODO: Add a test for this behavior
		uidgid, err := getUIDGID(r.AsCurrentUser, currentUser)
		if err != nil {
			return nil, err
		}

		// Storage mounts can either come from kustomize fn run --mounts,
		// or from the declarative function mounts field.
		storageMounts := spec.Container.StorageMounts
		storageMounts = append(storageMounts, r.StorageMounts...)

		c := container.NewContainer(
			runtimeutil.ContainerSpec{
				Image:         spec.Container.Image,
				Network:       spec.Container.Network,
				StorageMounts: storageMounts,
				Env:           spec.Container.Env,
			},
			uidgid,
		)
		cf := &c
		cf.Exec.FunctionConfig = api
		cf.Exec.GlobalScope = r.GlobalScope
		cf.Exec.ResultsFile = resultsFile
		cf.Exec.DeferFailure = spec.DeferFailure
		return cf, nil
	}
	if r.EnableStarlark && (spec.Starlark.Path != "" || spec.Starlark.URL != "") {
		// the script path is relative to the function config file
		m, err := api.GetMeta()
		if err != nil {
			return nil, errors.Wrap(err)
		}

		var p string
		if spec.Starlark.Path != "" {
			pathAnno := m.Annotations[kioutil.PathAnnotation]
			if pathAnno == "" {
				pathAnno = m.Annotations[kioutil.LegacyPathAnnotation]
			}
			p = filepath.ToSlash(path.Clean(pathAnno))

			spec.Starlark.Path = filepath.ToSlash(path.Clean(spec.Starlark.Path))
			if filepath.IsAbs(spec.Starlark.Path) || path.IsAbs(spec.Starlark.Path) {
				return nil, errors.Errorf(
					"absolute function path %s not allowed", spec.Starlark.Path)
			}
			if strings.HasPrefix(spec.Starlark.Path, "..") {
				return nil, errors.Errorf(
					"function path %s not allowed to start with ../", spec.Starlark.Path)
			}
			p = filepath.ToSlash(filepath.Join(r.Path, filepath.Dir(p), spec.Starlark.Path))
		}

		sf := &starlark.Filter{Name: spec.Starlark.Name, Path: p, URL: spec.Starlark.URL}

		sf.FunctionConfig = api
		sf.GlobalScope = r.GlobalScope
		sf.ResultsFile = resultsFile
		sf.DeferFailure = spec.DeferFailure
		return sf, nil
	}

	if r.EnableExec && spec.Exec.Path != "" {
		ef := &exec.Filter{
			Path:       spec.Exec.Path,
			WorkingDir: r.WorkingDir,
		}

		ef.FunctionConfig = api
		ef.GlobalScope = r.GlobalScope
		ef.ResultsFile = resultsFile
		ef.DeferFailure = spec.DeferFailure
		return ef, nil
	}

	return nil, nil
}
