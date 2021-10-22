// +build codegen

package api

import (
	"encoding/json"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// APIs provides a set of API models loaded by API package name.
type APIs map[string]*API

// Loader provides the loading of APIs from files.
type Loader struct {
	// The base Go import path the loaded models will be appended to.
	BaseImport string

	// Allows ignoring API models that are unsupported by the SDK without
	// failing the load of other supported APIs.
	IgnoreUnsupportedAPIs bool
}

// Load loads the API model files from disk returning the map of API package.
// Returns error if multiple API model resolve to the same package name.
func (l Loader) Load(modelPaths []string) (APIs, error) {
	apis := APIs{}
	for _, modelPath := range modelPaths {
		a, err := loadAPI(modelPath, l.BaseImport, func(a *API) {
			a.IgnoreUnsupportedAPIs = l.IgnoreUnsupportedAPIs
		})
		if err != nil {
			return nil, fmt.Errorf("failed to load API, %v, %v", modelPath, err)
		}

		if len(a.Operations) == 0 {
			if l.IgnoreUnsupportedAPIs {
				fmt.Fprintf(os.Stderr, "API has no operations, ignoring model %s, %v\n",
					modelPath, a.ImportPath())
				continue
			}
		}

		importPath := a.ImportPath()
		if _, ok := apis[importPath]; ok {
			return nil, fmt.Errorf(
				"package names must be unique attempted to load %v twice. Second model file: %v",
				importPath, modelPath)
		}
		apis[importPath] = a
	}

	return apis, nil
}

// attempts to load a model from disk into the import specified. Additional API
// options are invoked before to the API's Setup being called.
func loadAPI(modelPath, baseImport string, opts ...func(*API)) (*API, error) {
	a := &API{
		BaseImportPath:   baseImport,
		BaseCrosslinkURL: "https://docs.aws.amazon.com",
	}

	modelFile := filepath.Base(modelPath)
	modelDir := filepath.Dir(modelPath)
	err := attachModelFiles(modelDir,
		modelLoader{modelFile, a.Attach, true},
		modelLoader{"docs-2.json", a.AttachDocs, false},
		modelLoader{"paginators-1.json", a.AttachPaginators, false},
		modelLoader{"waiters-2.json", a.AttachWaiters, false},
		modelLoader{"examples-1.json", a.AttachExamples, false},
		modelLoader{"smoke.json", a.AttachSmokeTests, false},
	)
	if err != nil {
		return nil, err
	}

	for _, opt := range opts {
		opt(a)
	}

	if err = a.Setup(); err != nil {
		return nil, err
	}

	return a, nil
}

type modelLoader struct {
	Filename string
	Loader   func(string) error
	Required bool
}

func attachModelFiles(modelPath string, modelFiles ...modelLoader) error {
	for _, m := range modelFiles {
		filepath := filepath.Join(modelPath, m.Filename)
		_, err := os.Stat(filepath)
		if os.IsNotExist(err) && !m.Required {
			continue
		} else if err != nil {
			return fmt.Errorf("failed to load model file %v, %v", m.Filename, err)
		}

		if err = m.Loader(filepath); err != nil {
			return fmt.Errorf("model load failed, %s, %v", modelPath, err)
		}
	}

	return nil
}

// ExpandModelGlobPath returns a slice of model paths expanded from the glob
// pattern passed in. Returns the path of the model file to be loaded. Includes
// all versions of a service model.
//
//   e.g:
//   models/apis/*/*/api-2.json
//
//   Or with specific model file:
//   models/apis/service/version/api-2.json
func ExpandModelGlobPath(globs ...string) ([]string, error) {
	modelPaths := []string{}

	for _, g := range globs {
		filepaths, err := filepath.Glob(g)
		if err != nil {
			return nil, err
		}
		for _, p := range filepaths {
			modelPaths = append(modelPaths, p)
		}
	}

	return modelPaths, nil
}

// TrimModelServiceVersions sorts the model paths by service version then
// returns recent model versions, and model version excluded.
//
// Uses the third from last path element to determine unique service. Only one
// service version will be included.
//
//   models/apis/service/version/api-2.json
func TrimModelServiceVersions(modelPaths []string) (include, exclude []string) {
	sort.Strings(modelPaths)

	// Remove old API versions from list
	m := map[string]struct{}{}
	for i := len(modelPaths) - 1; i >= 0; i-- {
		// service name is 2nd-to-last component
		parts := strings.Split(modelPaths[i], string(filepath.Separator))
		svc := parts[len(parts)-3]

		if _, ok := m[svc]; ok {
			// Removed unused service version
			exclude = append(exclude, modelPaths[i])
			continue
		}
		include = append(include, modelPaths[i])
		m[svc] = struct{}{}
	}

	return include, exclude
}

// Attach opens a file by name, and unmarshal its JSON data.
// Will proceed to setup the API if not already done so.
func (a *API) Attach(filename string) error {
	a.path = filepath.Dir(filename)
	f, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	if err := json.NewDecoder(f).Decode(a); err != nil {
		return fmt.Errorf("failed to decode %s, err: %v", filename, err)
	}

	return nil
}

// AttachString will unmarshal a raw JSON string, and setup the
// API if not already done so.
func (a *API) AttachString(str string) error {
	json.Unmarshal([]byte(str), a)

	if a.initialized {
		return nil
	}

	return a.Setup()
}

// Setup initializes the API.
func (a *API) Setup() error {
	a.setServiceAliaseName()
	a.setMetadataEndpointsKey()
	a.writeShapeNames()
	a.resolveReferences()
	a.backfillErrorMembers()

	if !a.NoRemoveUnusedShapes {
		a.removeUnusedShapes()
	}

	a.fixStutterNames()
	if err := a.validateShapeNames(); err != nil {
		log.Fatalf(err.Error())
	}
	a.renameExportable()
	a.applyShapeNameAliases()
	a.createInputOutputShapes()
	a.writeInputOutputLocationName()
	a.renameAPIPayloadShapes()
	a.renameCollidingFields()
	a.updateTopLevelShapeReferences()
	if err := a.setupEventStreams(); err != nil {
		return err
	}

	a.findEndpointDiscoveryOp()
	a.injectUnboundedOutputStreaming()

	// Enables generated types for APIs using REST-JSON and JSONRPC protocols.
	// Other protocols will be added as supported.
	a.enableGeneratedTypedErrors()

	if err := a.customizationPasses(); err != nil {
		return err
	}

	a.addHeaderMapDocumentation()

	if !a.NoRemoveUnusedShapes {
		a.removeUnusedShapes()
	}

	if !a.NoValidataShapeMethods {
		a.addShapeValidations()
	}

	a.initialized = true

	return nil
}

// UnsupportedAPIModelError provides wrapping of an error causing the API to
// fail to load because the SDK does not support the API service defined.
type UnsupportedAPIModelError struct {
	Err error
}

func (e UnsupportedAPIModelError) Error() string {
	return fmt.Sprintf("service API is not supported, %v", e.Err)
}
