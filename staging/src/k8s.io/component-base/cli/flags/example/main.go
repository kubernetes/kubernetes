package example

// Parse parses the command line, potentially calls load and then returns a configuration.
func Parse(cmdLine []string, load func(fname string, dest *ComponentConfigControllerManager)) (*ComponentConfigControllerManager, error) {
	cfg := &ComponentConfigControllerManager{}
	Default(cfg)

	fs := ConfigFlagSet{}

	o := NewOptions(cfg)
	o.AddFlags(&fs)

	var fname string
	fs.fs.StringVar(&fname, "config", "config.json", "bla")
	if err := fs.Parse(cmdLine); err != nil {
		return nil, err
	}

	if len(fname) > 0 {
		load(fname, cfg)
	}

	fs.Apply()

	return cfg, nil
}
