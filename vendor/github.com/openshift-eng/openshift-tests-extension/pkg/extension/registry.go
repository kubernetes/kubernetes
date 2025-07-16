package extension

const DefaultExtension = "default"

type Registry struct {
	extensions map[string]*Extension
}

func NewRegistry() *Registry {
	var r Registry
	return &r
}

func (r *Registry) Walk(walkFn func(*Extension)) {
	for k := range r.extensions {
		if k == DefaultExtension {
			continue
		}
		walkFn(r.extensions[k])
	}
}

func (r *Registry) Get(name string) *Extension {
	return r.extensions[name]
}

func (r *Registry) Register(extension *Extension) {
	if r.extensions == nil {
		r.extensions = make(map[string]*Extension)
		// first extension is default
		r.extensions[DefaultExtension] = extension
	}

	r.extensions[extension.Component.Identifier()] = extension
}

func (r *Registry) Deregister(name string) {
	delete(r.extensions, name)
}
