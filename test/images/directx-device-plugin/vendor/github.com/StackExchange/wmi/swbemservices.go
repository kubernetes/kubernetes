// +build windows

package wmi

import (
	"fmt"
	"reflect"
	"runtime"
	"sync"

	"github.com/go-ole/go-ole"
	"github.com/go-ole/go-ole/oleutil"
)

// SWbemServices is used to access wmi. See https://msdn.microsoft.com/en-us/library/aa393719(v=vs.85).aspx
type SWbemServices struct {
	//TODO: track namespace. Not sure if we can re connect to a different namespace using the same instance
	cWMIClient            *Client //This could also be an embedded struct, but then we would need to branch on Client vs SWbemServices in the Query method
	sWbemLocatorIUnknown  *ole.IUnknown
	sWbemLocatorIDispatch *ole.IDispatch
	queries               chan *queryRequest
	closeError            chan error
	lQueryorClose         sync.Mutex
}

type queryRequest struct {
	query    string
	dst      interface{}
	args     []interface{}
	finished chan error
}

// InitializeSWbemServices will return a new SWbemServices object that can be used to query WMI
func InitializeSWbemServices(c *Client, connectServerArgs ...interface{}) (*SWbemServices, error) {
	//fmt.Println("InitializeSWbemServices: Starting")
	//TODO: implement connectServerArgs as optional argument for init with connectServer call
	s := new(SWbemServices)
	s.cWMIClient = c
	s.queries = make(chan *queryRequest)
	initError := make(chan error)
	go s.process(initError)

	err, ok := <-initError
	if ok {
		return nil, err //Send error to caller
	}
	//fmt.Println("InitializeSWbemServices: Finished")
	return s, nil
}

// Close will clear and release all of the SWbemServices resources
func (s *SWbemServices) Close() error {
	s.lQueryorClose.Lock()
	if s == nil || s.sWbemLocatorIDispatch == nil {
		s.lQueryorClose.Unlock()
		return fmt.Errorf("SWbemServices is not Initialized")
	}
	if s.queries == nil {
		s.lQueryorClose.Unlock()
		return fmt.Errorf("SWbemServices has been closed")
	}
	//fmt.Println("Close: sending close request")
	var result error
	ce := make(chan error)
	s.closeError = ce //Race condition if multiple callers to close. May need to lock here
	close(s.queries)  //Tell background to shut things down
	s.lQueryorClose.Unlock()
	err, ok := <-ce
	if ok {
		result = err
	}
	//fmt.Println("Close: finished")
	return result
}

func (s *SWbemServices) process(initError chan error) {
	//fmt.Println("process: starting background thread initialization")
	//All OLE/WMI calls must happen on the same initialized thead, so lock this goroutine
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	err := ole.CoInitializeEx(0, ole.COINIT_MULTITHREADED)
	if err != nil {
		oleCode := err.(*ole.OleError).Code()
		if oleCode != ole.S_OK && oleCode != S_FALSE {
			initError <- fmt.Errorf("ole.CoInitializeEx error: %v", err)
			return
		}
	}
	defer ole.CoUninitialize()

	unknown, err := oleutil.CreateObject("WbemScripting.SWbemLocator")
	if err != nil {
		initError <- fmt.Errorf("CreateObject SWbemLocator error: %v", err)
		return
	} else if unknown == nil {
		initError <- ErrNilCreateObject
		return
	}
	defer unknown.Release()
	s.sWbemLocatorIUnknown = unknown

	dispatch, err := s.sWbemLocatorIUnknown.QueryInterface(ole.IID_IDispatch)
	if err != nil {
		initError <- fmt.Errorf("SWbemLocator QueryInterface error: %v", err)
		return
	}
	defer dispatch.Release()
	s.sWbemLocatorIDispatch = dispatch

	// we can't do the ConnectServer call outside the loop unless we find a way to track and re-init the connectServerArgs
	//fmt.Println("process: initialized. closing initError")
	close(initError)
	//fmt.Println("process: waiting for queries")
	for q := range s.queries {
		//fmt.Printf("process: new query: len(query)=%d\n", len(q.query))
		errQuery := s.queryBackground(q)
		//fmt.Println("process: s.queryBackground finished")
		if errQuery != nil {
			q.finished <- errQuery
		}
		close(q.finished)
	}
	//fmt.Println("process: queries channel closed")
	s.queries = nil //set channel to nil so we know it is closed
	//TODO: I think the Release/Clear calls can panic if things are in a bad state.
	//TODO: May need to recover from panics and send error to method caller instead.
	close(s.closeError)
}

// Query runs the WQL query using a SWbemServices instance and appends the values to dst.
//
// dst must have type *[]S or *[]*S, for some struct type S. Fields selected in
// the query must have the same name in dst. Supported types are all signed and
// unsigned integers, time.Time, string, bool, or a pointer to one of those.
// Array types are not supported.
//
// By default, the local machine and default namespace are used. These can be
// changed using connectServerArgs. See
// http://msdn.microsoft.com/en-us/library/aa393720.aspx for details.
func (s *SWbemServices) Query(query string, dst interface{}, connectServerArgs ...interface{}) error {
	s.lQueryorClose.Lock()
	if s == nil || s.sWbemLocatorIDispatch == nil {
		s.lQueryorClose.Unlock()
		return fmt.Errorf("SWbemServices is not Initialized")
	}
	if s.queries == nil {
		s.lQueryorClose.Unlock()
		return fmt.Errorf("SWbemServices has been closed")
	}

	//fmt.Println("Query: Sending query request")
	qr := queryRequest{
		query:    query,
		dst:      dst,
		args:     connectServerArgs,
		finished: make(chan error),
	}
	s.queries <- &qr
	s.lQueryorClose.Unlock()
	err, ok := <-qr.finished
	if ok {
		//fmt.Println("Query: Finished with error")
		return err //Send error to caller
	}
	//fmt.Println("Query: Finished")
	return nil
}

func (s *SWbemServices) queryBackground(q *queryRequest) error {
	if s == nil || s.sWbemLocatorIDispatch == nil {
		return fmt.Errorf("SWbemServices is not Initialized")
	}
	wmi := s.sWbemLocatorIDispatch //Should just rename in the code, but this will help as we break things apart
	//fmt.Println("queryBackground: Starting")

	dv := reflect.ValueOf(q.dst)
	if dv.Kind() != reflect.Ptr || dv.IsNil() {
		return ErrInvalidEntityType
	}
	dv = dv.Elem()
	mat, elemType := checkMultiArg(dv)
	if mat == multiArgTypeInvalid {
		return ErrInvalidEntityType
	}

	// service is a SWbemServices
	serviceRaw, err := oleutil.CallMethod(wmi, "ConnectServer", q.args...)
	if err != nil {
		return err
	}
	service := serviceRaw.ToIDispatch()
	defer serviceRaw.Clear()

	// result is a SWBemObjectSet
	resultRaw, err := oleutil.CallMethod(service, "ExecQuery", q.query)
	if err != nil {
		return err
	}
	result := resultRaw.ToIDispatch()
	defer resultRaw.Clear()

	count, err := oleInt64(result, "Count")
	if err != nil {
		return err
	}

	enumProperty, err := result.GetProperty("_NewEnum")
	if err != nil {
		return err
	}
	defer enumProperty.Clear()

	enum, err := enumProperty.ToIUnknown().IEnumVARIANT(ole.IID_IEnumVariant)
	if err != nil {
		return err
	}
	if enum == nil {
		return fmt.Errorf("can't get IEnumVARIANT, enum is nil")
	}
	defer enum.Release()

	// Initialize a slice with Count capacity
	dv.Set(reflect.MakeSlice(dv.Type(), 0, int(count)))

	var errFieldMismatch error
	for itemRaw, length, err := enum.Next(1); length > 0; itemRaw, length, err = enum.Next(1) {
		if err != nil {
			return err
		}

		err := func() error {
			// item is a SWbemObject, but really a Win32_Process
			item := itemRaw.ToIDispatch()
			defer item.Release()

			ev := reflect.New(elemType)
			if err = s.cWMIClient.loadEntity(ev.Interface(), item); err != nil {
				if _, ok := err.(*ErrFieldMismatch); ok {
					// We continue loading entities even in the face of field mismatch errors.
					// If we encounter any other error, that other error is returned. Otherwise,
					// an ErrFieldMismatch is returned.
					errFieldMismatch = err
				} else {
					return err
				}
			}
			if mat != multiArgTypeStructPtr {
				ev = ev.Elem()
			}
			dv.Set(reflect.Append(dv, ev))
			return nil
		}()
		if err != nil {
			return err
		}
	}
	//fmt.Println("queryBackground: Finished")
	return errFieldMismatch
}
