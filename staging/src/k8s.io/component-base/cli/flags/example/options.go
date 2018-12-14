package example

type FooOptions = ComponentConfigFooController
type BarOptions = ComponentConfigBarController

// ControllerManagerOptions specifies how a controller manager should be launced.
type ControllerManagerOptions struct {
	X   *[]string
	Foo *FooOptions
	Bar *BarOptions

	config *ComponentConfigControllerManager
}

// NewOptions return options with the passed configuration as default values and
// as configuration destination.
func NewOptions(cfg *ComponentConfigControllerManager) *ControllerManagerOptions {
	o := &ControllerManagerOptions{
		X:   &cfg.X,
		Foo: &cfg.Foo,
		Bar: cfg.Bar,

		config: cfg,
	}

	return o
}

// AddFlags add controller manager flags to the flag set.
func (o *ControllerManagerOptions) AddFlags(fs *ConfigFlagSet) {
	fs.StringSliceVar(&o.config.X, "x", "Appending to x", o.config.X, func(x []string) {
		o.config.X = append(o.config.X, x...)
	})
	fs.Int32Var(&o.config.Bar.G, "bar-g", "Setting bar.g", o.config.Bar.G, func(x int32) {
		if o.config.Bar == nil {
			o.config.Bar = &ComponentConfigBarController{}
		}
		o.config.Bar.G = x
	})

	o.Foo.AddFlags(fs)
}

// AddFlags adds foo flags to the flag set.
func (o *FooOptions) AddFlags(fs *ConfigFlagSet) {
	if o == nil {
		return
	}

	fs.Int32Var(&o.F, "foo-f", "Setting foo.f", o.F, nil)
}
