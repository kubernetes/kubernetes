/*
Copyright (c) 2017 VMware, Inc. All Rights Reserved.

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

package simulator

import (
	"context"
	"errors"
	"log"
	"path"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/methods"
	"github.com/vmware/govmomi/vim25/mo"
	"github.com/vmware/govmomi/vim25/soap"
	"github.com/vmware/govmomi/vim25/types"
)

type PropertyCollector struct {
	mo.PropertyCollector

	nopLocker
	updates []types.ObjectUpdate
	mu      sync.Mutex
	cancel  context.CancelFunc
}

func NewPropertyCollector(ref types.ManagedObjectReference) object.Reference {
	s := &PropertyCollector{}
	s.Self = ref
	return s
}

var errMissingField = errors.New("missing field")
var errEmptyField = errors.New("empty field")

func getObject(ctx *Context, ref types.ManagedObjectReference) (reflect.Value, bool) {
	var obj mo.Reference
	if ctx.Session == nil {
		// Even without permissions to access an object or specific fields, RetrieveProperties
		// returns an ObjectContent response as long as the object exists.  See retrieveResult.add()
		obj = Map.Get(ref)
	} else {
		obj = ctx.Session.Get(ref)
	}

	if obj == nil {
		return reflect.Value{}, false
	}

	if ctx.Session == nil && ref.Type == "SessionManager" {
		// RetrieveProperties on SessionManager without a session always returns empty,
		// rather than MissingSet + Fault.NotAuthenticated for each field.
		obj = &mo.SessionManager{Self: ref}
	}

	// For objects that use internal types that differ from that of the vim25/mo field types.
	// See EventHistoryCollector for example.
	type get interface {
		Get() mo.Reference
	}
	if o, ok := obj.(get); ok {
		obj = o.Get()
	}

	return getManagedObject(obj), true
}

func getManagedObject(obj mo.Reference) reflect.Value {
	rval := reflect.ValueOf(obj).Elem()
	rtype := rval.Type()

	// PropertyCollector is for Managed Object types only (package mo).
	// If the registry object is not in the mo package, assume it is a wrapper
	// type where the first field is an embedded mo type.
	// We need to dig out the mo type for PropSet.All to work properly and
	// for the case where the type has a field of the same name, for example:
	// mo.ResourcePool.ResourcePool
	for {
		if path.Base(rtype.PkgPath()) == "mo" {
			break
		}
		if rtype.Kind() != reflect.Struct || rtype.NumField() == 0 {
			log.Panicf("%#v does not have an embedded mo type", obj.Reference())
		}
		rval = rval.Field(0)
		rtype = rval.Type()
	}

	return rval
}

// wrapValue converts slice types to the appropriate ArrayOf type used in property collector responses.
func wrapValue(rval reflect.Value, rtype reflect.Type) interface{} {
	pval := rval.Interface()

	if rval.Kind() == reflect.Slice {
		// Convert slice to types.ArrayOf*
		switch v := pval.(type) {
		case []string:
			pval = &types.ArrayOfString{
				String: v,
			}
		case []uint8:
			pval = &types.ArrayOfByte{
				Byte: v,
			}
		case []int16:
			pval = &types.ArrayOfShort{
				Short: v,
			}
		case []int32:
			pval = &types.ArrayOfInt{
				Int: v,
			}
		case []int64:
			pval = &types.ArrayOfLong{
				Long: v,
			}
		default:
			kind := rtype.Elem().Name()
			// Remove govmomi interface prefix name
			if strings.HasPrefix(kind, "Base") {
				kind = kind[4:]
			}
			akind, _ := defaultMapType("ArrayOf" + kind)
			a := reflect.New(akind)
			a.Elem().FieldByName(kind).Set(rval)
			pval = a.Interface()
		}
	}

	return pval
}

func fieldValueInterface(f reflect.StructField, rval reflect.Value) interface{} {
	if rval.Kind() == reflect.Ptr {
		rval = rval.Elem()
	}

	return wrapValue(rval, f.Type)
}

func fieldValue(rval reflect.Value, p string) (interface{}, error) {
	var value interface{}
	fields := strings.Split(p, ".")

	for i, name := range fields {
		kind := rval.Type().Kind()

		if kind == reflect.Interface {
			if rval.IsNil() {
				continue
			}
			rval = rval.Elem()
			kind = rval.Type().Kind()
		}

		if kind == reflect.Ptr {
			if rval.IsNil() {
				continue
			}
			rval = rval.Elem()
		}

		x := ucFirst(name)
		val := rval.FieldByName(x)
		if !val.IsValid() {
			return nil, errMissingField
		}

		if isEmpty(val) {
			return nil, errEmptyField
		}

		if i == len(fields)-1 {
			ftype, _ := rval.Type().FieldByName(x)
			value = fieldValueInterface(ftype, val)
			break
		}

		rval = val
	}

	return value, nil
}

func fieldRefs(f interface{}) []types.ManagedObjectReference {
	switch fv := f.(type) {
	case types.ManagedObjectReference:
		return []types.ManagedObjectReference{fv}
	case *types.ArrayOfManagedObjectReference:
		return fv.ManagedObjectReference
	case nil:
		// empty field
	}

	return nil
}

func isEmpty(rval reflect.Value) bool {
	switch rval.Kind() {
	case reflect.Ptr:
		return rval.IsNil()
	case reflect.String:
		return rval.Len() == 0
	}

	return false
}

func isTrue(v *bool) bool {
	return v != nil && *v
}

func isFalse(v *bool) bool {
	return v == nil || *v == false
}

func lcFirst(s string) string {
	return strings.ToLower(s[:1]) + s[1:]
}

func ucFirst(s string) string {
	return strings.ToUpper(s[:1]) + s[1:]
}

type retrieveResult struct {
	*types.RetrieveResult
	req       *types.RetrievePropertiesEx
	collected map[types.ManagedObjectReference]bool
	specs     map[string]*types.TraversalSpec
}

func (rr *retrieveResult) add(ctx *Context, name string, val types.AnyType, content *types.ObjectContent) {
	if ctx.Session != nil {
		content.PropSet = append(content.PropSet, types.DynamicProperty{
			Name: name,
			Val:  val,
		})
		return
	}

	content.MissingSet = append(content.MissingSet, types.MissingProperty{
		Path: name,
		Fault: types.LocalizedMethodFault{Fault: &types.NotAuthenticated{
			NoPermission: types.NoPermission{
				Object:      content.Obj,
				PrivilegeId: "System.Read",
			}},
		},
	})
}

func (rr *retrieveResult) collectAll(ctx *Context, rval reflect.Value, rtype reflect.Type, content *types.ObjectContent) {
	for i := 0; i < rval.NumField(); i++ {
		val := rval.Field(i)

		f := rtype.Field(i)

		if isEmpty(val) || f.Name == "Self" {
			continue
		}

		if f.Anonymous {
			// recurse into embedded field
			rr.collectAll(ctx, val, f.Type, content)
			continue
		}

		rr.add(ctx, lcFirst(f.Name), fieldValueInterface(f, val), content)
	}
}

func (rr *retrieveResult) collectFields(ctx *Context, rval reflect.Value, fields []string, content *types.ObjectContent) {
	seen := make(map[string]bool)

	for i := range content.PropSet {
		seen[content.PropSet[i].Name] = true // mark any already collected via embedded field
	}

	for _, name := range fields {
		if seen[name] {
			// rvc 'ls' includes the "name" property twice, then fails with no error message or stack trace
			// in RbVmomi::VIM::ObjectContent.to_hash_uncached when it sees the 2nd "name" property.
			continue
		}
		seen[name] = true

		val, err := fieldValue(rval, name)

		switch err {
		case nil, errEmptyField:
			rr.add(ctx, name, val, content)
		case errMissingField:
			content.MissingSet = append(content.MissingSet, types.MissingProperty{
				Path: name,
				Fault: types.LocalizedMethodFault{Fault: &types.InvalidProperty{
					Name: name,
				}},
			})
		}
	}
}

func (rr *retrieveResult) collect(ctx *Context, ref types.ManagedObjectReference) {
	if rr.collected[ref] {
		return
	}

	content := types.ObjectContent{
		Obj: ref,
	}

	rval, ok := getObject(ctx, ref)
	if !ok {
		// Possible if a test uses Map.Remove instead of Destroy_Task
		log.Printf("object %s no longer exists", ref)
		return
	}

	rtype := rval.Type()

	for _, spec := range rr.req.SpecSet {
		for _, p := range spec.PropSet {
			if p.Type != ref.Type {
				// e.g. ManagedEntity, ComputeResource
				field, ok := rtype.FieldByName(p.Type)

				if !(ok && field.Anonymous) {
					continue
				}
			}

			if isTrue(p.All) {
				rr.collectAll(ctx, rval, rtype, &content)
				continue
			}

			rr.collectFields(ctx, rval, p.PathSet, &content)
		}
	}

	if len(content.PropSet) != 0 || len(content.MissingSet) != 0 {
		rr.Objects = append(rr.Objects, content)
	}

	rr.collected[ref] = true
}

func (rr *retrieveResult) selectSet(ctx *Context, obj reflect.Value, s []types.BaseSelectionSpec, refs *[]types.ManagedObjectReference) types.BaseMethodFault {
	for _, ss := range s {
		ts, ok := ss.(*types.TraversalSpec)
		if ok {
			if ts.Name != "" {
				rr.specs[ts.Name] = ts
			}
		}
	}

	for _, ss := range s {
		ts, ok := ss.(*types.TraversalSpec)
		if !ok {
			ts = rr.specs[ss.GetSelectionSpec().Name]
			if ts == nil {
				return &types.InvalidArgument{InvalidProperty: "undefined TraversalSpec name"}
			}
		}

		f, _ := fieldValue(obj, ts.Path)

		for _, ref := range fieldRefs(f) {
			if isFalse(ts.Skip) {
				*refs = append(*refs, ref)
			}

			rval, ok := getObject(ctx, ref)
			if ok {
				if err := rr.selectSet(ctx, rval, ts.SelectSet, refs); err != nil {
					return err
				}
			}
		}
	}

	return nil
}

func (pc *PropertyCollector) collect(ctx *Context, r *types.RetrievePropertiesEx) (*types.RetrieveResult, types.BaseMethodFault) {
	var refs []types.ManagedObjectReference

	rr := &retrieveResult{
		RetrieveResult: &types.RetrieveResult{},
		req:            r,
		collected:      make(map[types.ManagedObjectReference]bool),
		specs:          make(map[string]*types.TraversalSpec),
	}

	// Select object references
	for _, spec := range r.SpecSet {
		for _, o := range spec.ObjectSet {
			var rval reflect.Value
			ok := false
			ctx.WithLock(o.Obj, func() { rval, ok = getObject(ctx, o.Obj) })
			if !ok {
				if isFalse(spec.ReportMissingObjectsInResults) {
					return nil, &types.ManagedObjectNotFound{Obj: o.Obj}
				}
				continue
			}

			if o.SelectSet == nil || isFalse(o.Skip) {
				refs = append(refs, o.Obj)
			}

			if err := rr.selectSet(ctx, rval, o.SelectSet, &refs); err != nil {
				return nil, err
			}
		}
	}

	for _, ref := range refs {
		ctx.WithLock(ref, func() { rr.collect(ctx, ref) })
	}

	return rr.RetrieveResult, nil
}

func (pc *PropertyCollector) CreateFilter(ctx *Context, c *types.CreateFilter) soap.HasFault {
	body := &methods.CreateFilterBody{}

	filter := &PropertyFilter{
		pc:   pc,
		refs: make(map[types.ManagedObjectReference]struct{}),
	}
	filter.PartialUpdates = c.PartialUpdates
	filter.Spec = c.Spec

	pc.Filter = append(pc.Filter, ctx.Session.Put(filter).Reference())

	body.Res = &types.CreateFilterResponse{
		Returnval: filter.Self,
	}

	return body
}

func (pc *PropertyCollector) CreatePropertyCollector(ctx *Context, c *types.CreatePropertyCollector) soap.HasFault {
	body := &methods.CreatePropertyCollectorBody{}

	cpc := &PropertyCollector{}

	body.Res = &types.CreatePropertyCollectorResponse{
		Returnval: ctx.Session.Put(cpc).Reference(),
	}

	return body
}

func (pc *PropertyCollector) DestroyPropertyCollector(ctx *Context, c *types.DestroyPropertyCollector) soap.HasFault {
	pc.CancelWaitForUpdates(&types.CancelWaitForUpdates{This: c.This})

	body := &methods.DestroyPropertyCollectorBody{}

	for _, ref := range pc.Filter {
		filter := ctx.Session.Get(ref).(*PropertyFilter)
		filter.DestroyPropertyFilter(ctx, &types.DestroyPropertyFilter{This: ref})
	}

	ctx.Session.Remove(c.This)
	ctx.Map.Remove(c.This)

	body.Res = &types.DestroyPropertyCollectorResponse{}

	return body
}

func (pc *PropertyCollector) RetrievePropertiesEx(ctx *Context, r *types.RetrievePropertiesEx) soap.HasFault {
	body := &methods.RetrievePropertiesExBody{}

	res, fault := pc.collect(ctx, r)

	if fault != nil {
		body.Fault_ = Fault("", fault)
	} else {
		objects := res.Objects[:0]
		for _, o := range res.Objects {
			propSet := o.PropSet[:0]
			for _, p := range o.PropSet {
				if p.Val != nil {
					propSet = append(propSet, p)
				}
			}
			o.PropSet = propSet

			objects = append(objects, o)
		}
		res.Objects = objects
		body.Res = &types.RetrievePropertiesExResponse{
			Returnval: res,
		}
	}

	return body
}

// RetrieveProperties is deprecated, but govmomi is still using it at the moment.
func (pc *PropertyCollector) RetrieveProperties(ctx *Context, r *types.RetrieveProperties) soap.HasFault {
	body := &methods.RetrievePropertiesBody{}

	res := pc.RetrievePropertiesEx(ctx, &types.RetrievePropertiesEx{
		This:    r.This,
		SpecSet: r.SpecSet,
	})

	if res.Fault() != nil {
		body.Fault_ = res.Fault()
	} else {
		body.Res = &types.RetrievePropertiesResponse{
			Returnval: res.(*methods.RetrievePropertiesExBody).Res.Returnval.Objects,
		}
	}

	return body
}

func (pc *PropertyCollector) CancelWaitForUpdates(r *types.CancelWaitForUpdates) soap.HasFault {
	pc.mu.Lock()
	if pc.cancel != nil {
		pc.cancel()
	}
	pc.mu.Unlock()

	return &methods.CancelWaitForUpdatesBody{Res: new(types.CancelWaitForUpdatesResponse)}
}

func (pc *PropertyCollector) update(u types.ObjectUpdate) {
	pc.mu.Lock()
	pc.updates = append(pc.updates, u)
	pc.mu.Unlock()
}

func (pc *PropertyCollector) PutObject(o mo.Reference) {
	pc.update(types.ObjectUpdate{
		Obj:       o.Reference(),
		Kind:      types.ObjectUpdateKindEnter,
		ChangeSet: nil,
	})
}

func (pc *PropertyCollector) UpdateObject(o mo.Reference, changes []types.PropertyChange) {
	pc.update(types.ObjectUpdate{
		Obj:       o.Reference(),
		Kind:      types.ObjectUpdateKindModify,
		ChangeSet: changes,
	})
}

func (pc *PropertyCollector) RemoveObject(ref types.ManagedObjectReference) {
	pc.update(types.ObjectUpdate{
		Obj:       ref,
		Kind:      types.ObjectUpdateKindLeave,
		ChangeSet: nil,
	})
}

func (pc *PropertyCollector) apply(ctx *Context, update *types.UpdateSet) types.BaseMethodFault {
	for _, ref := range pc.Filter {
		filter := ctx.Session.Get(ref).(*PropertyFilter)

		r := &types.RetrievePropertiesEx{}
		r.SpecSet = append(r.SpecSet, filter.Spec)

		res, fault := pc.collect(ctx, r)
		if fault != nil {
			return fault
		}

		fu := types.PropertyFilterUpdate{
			Filter: ref,
		}

		for _, o := range res.Objects {
			if _, ok := filter.refs[o.Obj]; ok {
				continue
			}
			filter.refs[o.Obj] = struct{}{}
			ou := types.ObjectUpdate{
				Obj:  o.Obj,
				Kind: types.ObjectUpdateKindEnter,
			}

			for _, p := range o.PropSet {
				ou.ChangeSet = append(ou.ChangeSet, types.PropertyChange{
					Op:   types.PropertyChangeOpAssign,
					Name: p.Name,
					Val:  p.Val,
				})
			}

			fu.ObjectSet = append(fu.ObjectSet, ou)
		}

		if len(fu.ObjectSet) != 0 {
			update.FilterSet = append(update.FilterSet, fu)
		}
	}
	return nil
}

func (pc *PropertyCollector) WaitForUpdatesEx(ctx *Context, r *types.WaitForUpdatesEx) soap.HasFault {
	wait, cancel := context.WithCancel(context.Background())
	oneUpdate := false
	if r.Options != nil {
		if max := r.Options.MaxWaitSeconds; max != nil {
			// A value of 0 causes WaitForUpdatesEx to do one update calculation and return any results.
			oneUpdate = (*max == 0)
			if *max > 0 {
				wait, cancel = context.WithTimeout(context.Background(), time.Second*time.Duration(*max))
			}
		}
	}
	pc.mu.Lock()
	pc.cancel = cancel
	pc.mu.Unlock()

	body := &methods.WaitForUpdatesExBody{}

	set := &types.UpdateSet{
		Version: r.Version,
	}

	body.Res = &types.WaitForUpdatesExResponse{
		Returnval: set,
	}

	apply := func() bool {
		if fault := pc.apply(ctx, set); fault != nil {
			body.Fault_ = Fault("", fault)
			body.Res = nil
			return false
		}
		return true
	}

	if r.Version == "" {
		apply()                // Collect current state
		set.Version = "-"      // Next request with Version set will wait via loop below
		ctx.Map.AddHandler(pc) // Listen for create, update, delete of managed objects
		return body
	}

	ticker := time.NewTicker(250 * time.Millisecond) // allow for updates to accumulate
	defer ticker.Stop()
	// Start the wait loop, returning on one of:
	// - Client calls CancelWaitForUpdates
	// - MaxWaitSeconds was specified and has been exceeded
	// - We have updates to send to the client
	for {
		select {
		case <-wait.Done():
			body.Res.Returnval = nil
			switch wait.Err() {
			case context.Canceled:
				log.Printf("%s: WaitForUpdates canceled", pc.Self)
				body.Fault_ = Fault("", new(types.RequestCanceled)) // CancelWaitForUpdates was called
				body.Res = nil
			case context.DeadlineExceeded:
				log.Printf("%s: WaitForUpdates MaxWaitSeconds exceeded", pc.Self)
			}

			return body
		case <-ticker.C:
			pc.mu.Lock()
			updates := pc.updates
			pc.updates = nil // clear updates collected by the managed object CRUD listeners
			pc.mu.Unlock()
			if len(updates) == 0 {
				if oneUpdate == true {
					body.Res.Returnval = nil
					return body
				}
				continue
			}

			log.Printf("%s: applying %d updates to %d filters", pc.Self, len(updates), len(pc.Filter))

			for _, f := range pc.Filter {
				filter := ctx.Session.Get(f).(*PropertyFilter)
				fu := types.PropertyFilterUpdate{Filter: f}

				for _, update := range updates {
					switch update.Kind {
					case types.ObjectUpdateKindEnter: // Create
						if !apply() {
							return body
						}
					case types.ObjectUpdateKindModify: // Update
						log.Printf("%s has %d changes", update.Obj, len(update.ChangeSet))
						if !apply() { // An update may apply to collector traversal specs
							return body
						}
						if _, ok := filter.refs[update.Obj]; ok {
							// This object has already been applied by the filter,
							// now check if the property spec applies for this update.
							update = filter.apply(ctx, update)
							if len(update.ChangeSet) != 0 {
								fu.ObjectSet = append(fu.ObjectSet, update)
							}
						}
					case types.ObjectUpdateKindLeave: // Delete
						if _, ok := filter.refs[update.Obj]; !ok {
							continue
						}
						delete(filter.refs, update.Obj)
						fu.ObjectSet = append(fu.ObjectSet, update)
					}
				}

				if len(fu.ObjectSet) != 0 {
					set.FilterSet = append(set.FilterSet, fu)
				}
			}
			if len(set.FilterSet) != 0 {
				return body
			}
			if oneUpdate == true {
				body.Res.Returnval = nil
				return body
			}
		}
	}
}

// WaitForUpdates is deprecated, but pyvmomi is still using it at the moment.
func (pc *PropertyCollector) WaitForUpdates(ctx *Context, r *types.WaitForUpdates) soap.HasFault {
	body := &methods.WaitForUpdatesBody{}

	res := pc.WaitForUpdatesEx(ctx, &types.WaitForUpdatesEx{
		This:    r.This,
		Version: r.Version,
	})

	if res.Fault() != nil {
		body.Fault_ = res.Fault()
	} else {
		body.Res = &types.WaitForUpdatesResponse{
			Returnval: *res.(*methods.WaitForUpdatesExBody).Res.Returnval,
		}
	}

	return body
}
