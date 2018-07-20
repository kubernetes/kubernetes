package httpcli

type (
	// Opt defines a functional option for the HTTP client type. A functional option
	// must return an Opt that acts as an "undo" if applied to the same Client.
	Opt func(*Client) Opt
	// Opts represents a series of functional options
	Opts []Opt
)

// Apply is a nil-safe application of an Opt: if the receiver is nil then this func
// simply returns nil, otherwise it returns the result invoking the receiving Opt
// with the given Client.
func (o Opt) Apply(c *Client) (result Opt) {
	if o != nil {
		result = o(c)
	}
	return
}

// Merged generates a single Opt that applies all the functional options, in-order
func (opts Opts) Merged() Opt {
	if len(opts) == 0 {
		return nil
	}
	return func(c *Client) Opt {
		var (
			size = len(opts)
			undo = make(Opts, size)
		)
		size-- // make this a zero-based offset
		for i, opt := range opts {
			if opt != nil {
				undo[size-i] = opt(c)
			}
		}
		return undo.Merged()
	}
}

// And combines two functional options into a single Opt
func (o Opt) And(other Opt) Opt {
	if o == nil {
		if other == nil {
			return nil
		}
		return other
	}
	if other == nil {
		return o
	}
	return Opts{o, other}.Merged()
}
