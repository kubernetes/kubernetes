package ole

// Connection contains IUnknown for fluent interface interaction.
//
// Deprecated. Use oleutil package instead.
type Connection struct {
	Object *IUnknown // Access COM
}

// Initialize COM.
func (*Connection) Initialize() (err error) {
	return coInitialize()
}

// Uninitialize COM.
func (*Connection) Uninitialize() {
	CoUninitialize()
}

// Create IUnknown object based first on ProgId and then from String.
func (c *Connection) Create(progId string) (err error) {
	var clsid *GUID
	clsid, err = CLSIDFromProgID(progId)
	if err != nil {
		clsid, err = CLSIDFromString(progId)
		if err != nil {
			return
		}
	}

	unknown, err := CreateInstance(clsid, IID_IUnknown)
	if err != nil {
		return
	}
	c.Object = unknown

	return
}

// Release IUnknown object.
func (c *Connection) Release() {
	c.Object.Release()
}

// Load COM object from list of programIDs or strings.
func (c *Connection) Load(names ...string) (errors []error) {
	var tempErrors []error = make([]error, len(names))
	var numErrors int = 0
	for _, name := range names {
		err := c.Create(name)
		if err != nil {
			tempErrors = append(tempErrors, err)
			numErrors += 1
			continue
		}
		break
	}

	copy(errors, tempErrors[0:numErrors])
	return
}

// Dispatch returns Dispatch object.
func (c *Connection) Dispatch() (object *Dispatch, err error) {
	dispatch, err := c.Object.QueryInterface(IID_IDispatch)
	if err != nil {
		return
	}
	object = &Dispatch{dispatch}
	return
}

// Dispatch stores IDispatch object.
type Dispatch struct {
	Object *IDispatch // Dispatch object.
}

// Call method on IDispatch with parameters.
func (d *Dispatch) Call(method string, params ...interface{}) (result *VARIANT, err error) {
	id, err := d.GetId(method)
	if err != nil {
		return
	}

	result, err = d.Invoke(id, DISPATCH_METHOD, params)
	return
}

// MustCall method on IDispatch with parameters.
func (d *Dispatch) MustCall(method string, params ...interface{}) (result *VARIANT) {
	id, err := d.GetId(method)
	if err != nil {
		panic(err)
	}

	result, err = d.Invoke(id, DISPATCH_METHOD, params)
	if err != nil {
		panic(err)
	}

	return
}

// Get property on IDispatch with parameters.
func (d *Dispatch) Get(name string, params ...interface{}) (result *VARIANT, err error) {
	id, err := d.GetId(name)
	if err != nil {
		return
	}
	result, err = d.Invoke(id, DISPATCH_PROPERTYGET, params)
	return
}

// MustGet property on IDispatch with parameters.
func (d *Dispatch) MustGet(name string, params ...interface{}) (result *VARIANT) {
	id, err := d.GetId(name)
	if err != nil {
		panic(err)
	}

	result, err = d.Invoke(id, DISPATCH_PROPERTYGET, params)
	if err != nil {
		panic(err)
	}
	return
}

// Set property on IDispatch with parameters.
func (d *Dispatch) Set(name string, params ...interface{}) (result *VARIANT, err error) {
	id, err := d.GetId(name)
	if err != nil {
		return
	}
	result, err = d.Invoke(id, DISPATCH_PROPERTYPUT, params)
	return
}

// MustSet property on IDispatch with parameters.
func (d *Dispatch) MustSet(name string, params ...interface{}) (result *VARIANT) {
	id, err := d.GetId(name)
	if err != nil {
		panic(err)
	}

	result, err = d.Invoke(id, DISPATCH_PROPERTYPUT, params)
	if err != nil {
		panic(err)
	}
	return
}

// GetId retrieves ID of name on IDispatch.
func (d *Dispatch) GetId(name string) (id int32, err error) {
	var dispid []int32
	dispid, err = d.Object.GetIDsOfName([]string{name})
	if err != nil {
		return
	}
	id = dispid[0]
	return
}

// GetIds retrieves all IDs of names on IDispatch.
func (d *Dispatch) GetIds(names ...string) (dispid []int32, err error) {
	dispid, err = d.Object.GetIDsOfName(names)
	return
}

// Invoke IDispatch on DisplayID of dispatch type with parameters.
//
// There have been problems where if send cascading params..., it would error
// out because the parameters would be empty.
func (d *Dispatch) Invoke(id int32, dispatch int16, params []interface{}) (result *VARIANT, err error) {
	if len(params) < 1 {
		result, err = d.Object.Invoke(id, dispatch)
	} else {
		result, err = d.Object.Invoke(id, dispatch, params...)
	}
	return
}

// Release IDispatch object.
func (d *Dispatch) Release() {
	d.Object.Release()
}

// Connect initializes COM and attempts to load IUnknown based on given names.
func Connect(names ...string) (connection *Connection) {
	connection.Initialize()
	connection.Load(names...)
	return
}
