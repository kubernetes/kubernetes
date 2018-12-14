package example

// ComponentConfigFooController configures the Foo controller.
type ComponentConfigFooController struct {
	F int32
}

// ComponentConfigBarController configures the Bar controller.
type ComponentConfigBarController struct {
	G int32
}

// ComponentConfigControllerManager configures the kube-controller-manager.
type ComponentConfigControllerManager struct {
	X   []string
	Foo ComponentConfigFooController
	Bar *ComponentConfigBarController
}

// Default sets defaults for kube-controller-manager.
func Default(cfg *ComponentConfigControllerManager) {
	if cfg.Foo.F == 0 {
		cfg.Foo.F = 42
	}
	if cfg.Bar == nil {
		cfg.Bar = &ComponentConfigBarController{}
	}
	if cfg.Bar.G == 0 {
		cfg.Bar.G = 7
	}
}

