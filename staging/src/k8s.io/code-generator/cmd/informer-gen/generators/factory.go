/*
Copyright 2016 The Kubernetes Authors.

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

package generators

import (
	"io"
	"path"

	clientgentypes "k8s.io/code-generator/cmd/client-gen/types"
	"k8s.io/gengo/generator"
	"k8s.io/gengo/namer"
	"k8s.io/gengo/types"

	"k8s.io/klog/v2"
)

// factoryGenerator produces a file of listers for a given GroupVersion and
// type.
type factoryGenerator struct {
	generator.DefaultGen
	outputPackage             string
	imports                   namer.ImportTracker
	groupVersions             map[string]clientgentypes.GroupVersions
	gvGoNames                 map[string]string
	clientSetPackage          string
	internalInterfacesPackage string
	filtered                  bool
}

var _ generator.Generator = &factoryGenerator{}

func (g *factoryGenerator) Filter(c *generator.Context, t *types.Type) bool {
	if !g.filtered {
		g.filtered = true
		return true
	}
	return false
}

func (g *factoryGenerator) Namers(c *generator.Context) namer.NameSystems {
	return namer.NameSystems{
		"raw": namer.NewRawNamer(g.outputPackage, g.imports),
	}
}

func (g *factoryGenerator) Imports(c *generator.Context) (imports []string) {
	imports = append(imports, g.imports.ImportLines()...)
	return
}

func (g *factoryGenerator) GenerateType(c *generator.Context, t *types.Type, w io.Writer) error {
	sw := generator.NewSnippetWriter(w, c, "{{", "}}")

	klog.V(5).Infof("processing type %v", t)

	gvInterfaces := make(map[string]*types.Type)
	gvNewFuncs := make(map[string]*types.Type)
	for groupPkgName := range g.groupVersions {
		gvInterfaces[groupPkgName] = c.Universe.Type(types.Name{Package: path.Join(g.outputPackage, groupPkgName), Name: "Interface"})
		gvNewFuncs[groupPkgName] = c.Universe.Function(types.Name{Package: path.Join(g.outputPackage, groupPkgName), Name: "New"})
	}
	m := map[string]interface{}{
		"cacheSharedIndexInformer":       c.Universe.Type(cacheSharedIndexInformer),
		"groupVersions":                  g.groupVersions,
		"gvInterfaces":                   gvInterfaces,
		"gvNewFuncs":                     gvNewFuncs,
		"gvGoNames":                      g.gvGoNames,
		"interfacesNewInformerFunc":      c.Universe.Type(types.Name{Package: g.internalInterfacesPackage, Name: "NewInformerFunc"}),
		"interfacesTweakListOptionsFunc": c.Universe.Type(types.Name{Package: g.internalInterfacesPackage, Name: "TweakListOptionsFunc"}),
		"informerFactoryInterface":       c.Universe.Type(types.Name{Package: g.internalInterfacesPackage, Name: "SharedInformerFactory"}),
		"clientSetInterface":             c.Universe.Type(types.Name{Package: g.clientSetPackage, Name: "Interface"}),
		"reflectType":                    c.Universe.Type(reflectType),
		"runtimeObject":                  c.Universe.Type(runtimeObject),
		"schemaGroupVersionResource":     c.Universe.Type(schemaGroupVersionResource),
		"syncMutex":                      c.Universe.Type(syncMutex),
		"timeDuration":                   c.Universe.Type(timeDuration),
		"namespaceAll":                   c.Universe.Type(metav1NamespaceAll),
		"object":                         c.Universe.Type(metav1Object),
	}

	sw.Do(sharedInformerFactoryStruct, m)
	sw.Do(sharedInformerFactoryInterface, m)

	return sw.Error()
}

var sharedInformerFactoryStruct = `
// SharedInformerOption defines the functional option type for SharedInformerFactory.
type SharedInformerOption func(*sharedInformerFactory) *sharedInformerFactory

type sharedInformerFactory struct {
	client {{.clientSetInterface|raw}}
	namespace string
	tweakListOptions {{.interfacesTweakListOptionsFunc|raw}}
	lock {{.syncMutex|raw}}
	defaultResync {{.timeDuration|raw}}
	customResync map[{{.reflectType|raw}}]{{.timeDuration|raw}}

	informers map[{{.reflectType|raw}}]{{.cacheSharedIndexInformer|raw}}
	// startedInformers is used for tracking which informers have been started.
	// This allows Start() to be called multiple times safely.
	startedInformers map[{{.reflectType|raw}}]bool
	// wg tracks how many goroutines were started.
	wg sync.WaitGroup
	// shuttingDown is true when Shutdown has been called. It may still be running
	// because it needs to wait for goroutines.
	shuttingDown bool
}

// WithCustomResyncConfig sets a custom resync period for the specified informer types.
func WithCustomResyncConfig(resyncConfig map[{{.object|raw}}]{{.timeDuration|raw}}) SharedInformerOption {
	return func(factory *sharedInformerFactory) *sharedInformerFactory {
		for k, v := range resyncConfig {
			factory.customResync[reflect.TypeOf(k)] = v
		}
		return factory
	}
}

// WithTweakListOptions sets a custom filter on all listers of the configured SharedInformerFactory.
func WithTweakListOptions(tweakListOptions internalinterfaces.TweakListOptionsFunc) SharedInformerOption {
	return func(factory *sharedInformerFactory) *sharedInformerFactory {
		factory.tweakListOptions = tweakListOptions
		return factory
	}
}

// WithNamespace limits the SharedInformerFactory to the specified namespace.
func WithNamespace(namespace string) SharedInformerOption {
	return func(factory *sharedInformerFactory) *sharedInformerFactory {
		factory.namespace = namespace
		return factory
	}
}

// NewSharedInformerFactory constructs a new instance of sharedInformerFactory for all namespaces.
func NewSharedInformerFactory(client {{.clientSetInterface|raw}}, defaultResync {{.timeDuration|raw}}) SharedInformerFactory {
	return NewSharedInformerFactoryWithOptions(client, defaultResync)
}

// NewFilteredSharedInformerFactory constructs a new instance of sharedInformerFactory.
// Listers obtained via this SharedInformerFactory will be subject to the same filters
// as specified here.
// Deprecated: Please use NewSharedInformerFactoryWithOptions instead
func NewFilteredSharedInformerFactory(client {{.clientSetInterface|raw}}, defaultResync {{.timeDuration|raw}}, namespace string, tweakListOptions {{.interfacesTweakListOptionsFunc|raw}}) SharedInformerFactory {
	return NewSharedInformerFactoryWithOptions(client, defaultResync, WithNamespace(namespace), WithTweakListOptions(tweakListOptions))
}

// NewSharedInformerFactoryWithOptions constructs a new instance of a SharedInformerFactory with additional options.
func NewSharedInformerFactoryWithOptions(client {{.clientSetInterface|raw}}, defaultResync {{.timeDuration|raw}}, options ...SharedInformerOption) SharedInformerFactory {
	factory := &sharedInformerFactory{
		client:           client,
		namespace:        v1.NamespaceAll,
		defaultResync:    defaultResync,
		informers:        make(map[{{.reflectType|raw}}]{{.cacheSharedIndexInformer|raw}}),
		startedInformers: make(map[{{.reflectType|raw}}]bool),
		customResync:     make(map[{{.reflectType|raw}}]{{.timeDuration|raw}}),
	}

	// Apply all options
	for _, opt := range options {
		factory = opt(factory)
	}

	return factory
}

func (f *sharedInformerFactory) Start(stopCh <-chan struct{}) {
	f.lock.Lock()
	defer f.lock.Unlock()

	if f.shuttingDown {
		return
	}

	for informerType, informer := range f.informers {
		if !f.startedInformers[informerType] {
			f.wg.Add(1)
			// We need a new variable in each loop iteration,
			// otherwise the goroutine would use the loop variable
			// and that keeps changing.
			informer := informer
			go func() {
				defer f.wg.Done()
				informer.Run(stopCh)
			}()
			f.startedInformers[informerType] = true
		}
	}
}

func (f *sharedInformerFactory) Shutdown() {
	f.lock.Lock()
	f.shuttingDown = true
	f.lock.Unlock()


	// Will return immediately if there is nothing to wait for.
	f.wg.Wait()
}

func (f *sharedInformerFactory) WaitForCacheSync(stopCh <-chan struct{}) map[reflect.Type]bool {
	informers := func()map[reflect.Type]cache.SharedIndexInformer{
               f.lock.Lock()
               defer f.lock.Unlock()

               informers := map[reflect.Type]cache.SharedIndexInformer{}
               for informerType, informer := range f.informers {
                       if f.startedInformers[informerType] {
                               informers[informerType] = informer
                       }
               }
               return informers
       }()

       res := map[reflect.Type]bool{}
       for informType, informer := range informers {
               res[informType] = cache.WaitForCacheSync(stopCh, informer.HasSynced)
       }
       return res
}

// InternalInformerFor returns the SharedIndexInformer for obj using an internal
// client.
func (f *sharedInformerFactory) InformerFor(obj {{.runtimeObject|raw}}, newFunc {{.interfacesNewInformerFunc|raw}}) {{.cacheSharedIndexInformer|raw}} {
  f.lock.Lock()
  defer f.lock.Unlock()

  informerType := reflect.TypeOf(obj)
  informer, exists := f.informers[informerType]
  if exists {
    return informer
  }

  resyncPeriod, exists := f.customResync[informerType]
  if !exists {
    resyncPeriod = f.defaultResync
  }

  informer = newFunc(f.client, resyncPeriod)
  f.informers[informerType] = informer

  return informer
}

`

var sharedInformerFactoryInterface = `
// SharedInformerFactory provides shared informers for resources in all known
// API group versions.
//
// It is typically used like this:
//
//   ctx, cancel := context.Background()
//   defer cancel()
//   factory := NewSharedInformerFactory(client, resyncPeriod)
//   defer factory.WaitForStop()    // Returns immediately if nothing was started.
//   genericInformer := factory.ForResource(resource)
//   typedInformer := factory.SomeAPIGroup().V1().SomeType()
//   factory.Start(ctx.Done())          // Start processing these informers.
//   synced := factory.WaitForCacheSync(ctx.Done())
//   for v, ok := range synced {
//       if !ok {
//           fmt.Fprintf(os.Stderr, "caches failed to sync: %v", v)
//           return
//       }
//   }
//
//   // Creating informers can also be created after Start, but then
//   // Start must be called again:
//   anotherGenericInformer := factory.ForResource(resource)
//   factory.Start(ctx.Done())
type SharedInformerFactory interface {
	{{.informerFactoryInterface|raw}}

	// Start initializes all requested informers. They are handled in goroutines
	// which run until the stop channel gets closed.
        Start(stopCh <-chan struct{})

	// Shutdown marks a factory as shutting down. At that point no new
	// informers can be started anymore and Start will return without
	// doing anything.
	//
	// In addition, Shutdown blocks until all goroutines have terminated. For that
	// to happen, the close channel(s) that they were started with must be closed,
	// either before Shutdown gets called or while it is waiting.
	//
	// Shutdown may be called multiple times, even concurrently. All such calls will
	// block until all goroutines have terminated.
	Shutdown()

	// WaitForCacheSync blocks until all started informers' caches were synced
	// or the stop channel gets closed.
	WaitForCacheSync(stopCh <-chan struct{}) map[reflect.Type]bool

	// ForResource gives generic access to a shared informer of the matching type.
	ForResource(resource {{.schemaGroupVersionResource|raw}}) (GenericInformer, error)

	// InternalInformerFor returns the SharedIndexInformer for obj using an internal
	// client.
	InformerFor(obj {{.runtimeObject|raw}}, newFunc {{.interfacesNewInformerFunc|raw}}) {{.cacheSharedIndexInformer|raw}}

	{{$gvInterfaces := .gvInterfaces}}
	{{$gvGoNames := .gvGoNames}}
	{{range $groupName, $group := .groupVersions}}{{index $gvGoNames $groupName}}() {{index $gvInterfaces $groupName|raw}}
	{{end}}
}

{{$gvNewFuncs := .gvNewFuncs}}
{{$gvGoNames := .gvGoNames}}
{{range $groupPkgName, $group := .groupVersions}}
func (f *sharedInformerFactory) {{index $gvGoNames $groupPkgName}}() {{index $gvInterfaces $groupPkgName|raw}} {
  return {{index $gvNewFuncs $groupPkgName|raw}}(f, f.namespace, f.tweakListOptions)
}
{{end}}
`
