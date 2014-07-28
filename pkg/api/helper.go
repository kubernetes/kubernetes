/*
Copyright 2014 Google Inc. All rights reserved.

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

package api

import (
	"encoding/json"
	"fmt"
	"reflect"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/v1beta1"
	"gopkg.in/v1/yaml"
)

type ConversionFunc func(input interface{}) (output interface{}, err error)

var versionMap = map[string]map[string]reflect.Type{}
var externalFuncs = map[string]ConversionFunc{}
var internalFuncs = map[string]ConversionFunc{}

func init() {
	AddKnownTypes("",
		PodList{},
		Pod{},
		ReplicationControllerList{},
		ReplicationController{},
		ServiceList{},
		Service{},
		MinionList{},
		Minion{},
		Status{},
		ServerOpList{},
		ServerOp{},
	)
	AddKnownTypes("v1beta1",
		v1beta1.PodList{},
		v1beta1.Pod{},
		v1beta1.ReplicationControllerList{},
		v1beta1.ReplicationController{},
		v1beta1.ServiceList{},
		v1beta1.Service{},
		v1beta1.MinionList{},
		v1beta1.Minion{},
		v1beta1.Status{},
		v1beta1.ServerOpList{},
		v1beta1.ServerOp{},
	)
}

// AddKnownTypes registers the types of the arguments to the marshaller of the package api.
// Encode() refuses the object unless its type is registered with AddKnownTypes.
func AddKnownTypes(version string, types ...interface{}) {
	knownTypes, found := versionMap[version]
	if !found {
		knownTypes = map[string]reflect.Type{}
		versionMap[version] = knownTypes
	}
	for _, obj := range types {
		t := reflect.TypeOf(obj)
		knownTypes[t.Name()] = t
	}
}

func AddExternalConversion(name string, fn ConversionFunc) {
	externalFuncs[name] = fn
}

func AddInternalConversion(name string, fn ConversionFunc) {
	internalFuncs[name] = fn
}

// FindJSONBase takes an arbitary api type, returns pointer to its JSONBase field.
// obj must be a pointer to an api type.
func FindJSONBase(obj interface{}) (*JSONBase, error) {
	_, jsonBase, err := nameAndJSONBase(obj)
	return jsonBase, err
}

// FindJSONBaseRO takes an arbitary api type, return a copy of its JSONBase field.
// obj may be a pointer to an api type, or a non-pointer struct api type.
func FindJSONBaseRO(obj interface{}) (JSONBase, error) {
	v := reflect.ValueOf(obj)
	if v.Kind() == reflect.Ptr {
		v = v.Elem()
	}
	if v.Kind() != reflect.Struct {
		return JSONBase{}, fmt.Errorf("expected struct, but got %v (%#v)", v.Type().Name(), v.Interface())
	}
	jsonBase := v.FieldByName("JSONBase")
	if !jsonBase.IsValid() {
		return JSONBase{}, fmt.Errorf("struct %v lacks embedded JSON type", v.Type().Name())
	}
	return jsonBase.Interface().(JSONBase), nil
}

// Encode turns the given api object into an appropriate JSON string.
// Will return an error if the object doesn't have an embedded JSONBase.
// Obj may be a pointer to a struct, or a struct. If a struct, a copy
// will be made so that the object's Kind field can be set. If a pointer,
// we change the Kind field, marshal, and then set the kind field back to
// "". Having to keep track of the kind field makes tests very annoying,
// so the rule is it's set only in wire format (json), not when in native
// format.
func Encode(obj interface{}) (data []byte, err error) {
	obj = checkPtr(obj)
	base, err := prepareEncode(obj)
	if err != nil {
		return nil, err
	}
	if len(base.APIVersion) == 0 {
		out, err := externalize(obj)
		if err != nil {
			return nil, err
		}
		_, jsonBase, err := nameAndJSONBase(obj)
		if err != nil {
			return nil, err
		}
		jsonBase.Kind = ""
		obj = out
		_, err = prepareEncode(out)
		if err != nil {
			return nil, err
		}
	}

	data, err = json.MarshalIndent(obj, "", "	")
	_, jsonBase, err := nameAndJSONBase(obj)
	if err != nil {
		return nil, err
	}
	jsonBase.Kind = ""
	return data, err
}

func checkPtr(obj interface{}) interface{} {
	v := reflect.ValueOf(obj)
	if v.Kind() == reflect.Ptr {
		return obj
	}
	v2 := reflect.New(v.Type())
	v2.Elem().Set(v)
	return v2.Interface()
}

func prepareEncode(obj interface{}) (*JSONBase, error) {
	name, jsonBase, err := nameAndJSONBase(obj)
	if err != nil {
		return nil, err
	}
	knownTypes, found := versionMap[jsonBase.APIVersion]
	if !found {
		return nil, fmt.Errorf("struct %s, %v won't be unmarshalable because it's not in known versions", jsonBase.APIVersion, obj)
	}
	if _, contains := knownTypes[name]; !contains {
		return nil, fmt.Errorf("struct %s won't be unmarshalable because it's not in knownTypes", name)
	}
	jsonBase.Kind = name
	return jsonBase, nil
}

// Returns the name of the type (sans pointer), and its kind field. Takes pointer-to-struct..
func nameAndJSONBase(obj interface{}) (string, *JSONBase, error) {
	v := reflect.ValueOf(obj)
	if v.Kind() != reflect.Ptr {
		return "", nil, fmt.Errorf("expected pointer, but got %v", v.Type().Name())
	}
	v = v.Elem()
	name := v.Type().Name()
	if v.Kind() != reflect.Struct {
		return "", nil, fmt.Errorf("expected struct, but got %v: %v (%#v)", v.Kind(), v.Type().Name(), v.Interface())
	}
	jsonBase := v.FieldByName("JSONBase")
	if !jsonBase.IsValid() {
		return "", nil, fmt.Errorf("struct %v lacks embedded JSON type", name)
	}
	output, ok := jsonBase.Addr().Interface().(*JSONBase)
	if !ok {
		internal, err := internalize(jsonBase.Addr().Interface())
		if err != nil {
			return name, nil, err
		}
		output = internal.(*JSONBase)
	}
	return name, output, nil
}

// Decode converts a JSON string back into a pointer to an api object. Deduces the type
// based upon the Kind field (set by encode).
func Decode(data []byte) (interface{}, error) {
	findKind := struct {
		Kind       string `json:"kind,omitempty" yaml:"kind,omitempty"`
		APIVersion string `json:"apiVersion,omitempty" yaml:"apiVersion,omitempty"`
	}{}
	// yaml is a superset of json, so we use it to decode here. That way, we understand both.
	err := yaml.Unmarshal(data, &findKind)
	if err != nil {
		return nil, fmt.Errorf("couldn't get kind: %#v", err)
	}
	knownTypes, found := versionMap[findKind.APIVersion]
	if !found {
		return nil, fmt.Errorf("Unknown api verson: %s", findKind.APIVersion)
	}
	objType, found := knownTypes[findKind.Kind]
	if !found {
		return nil, fmt.Errorf("%#v is not a known type for decoding", findKind)
	}
	obj := reflect.New(objType).Interface()
	err = yaml.Unmarshal(data, obj)
	if err != nil {
		return nil, err
	}
	if len(findKind.APIVersion) != 0 {
		obj, err = internalize(obj)
		if err != nil {
			return nil, err
		}
	}
	_, jsonBase, err := nameAndJSONBase(obj)
	if err != nil {
		return nil, err
	}
	// Don't leave these set. Track type with go's type.
	jsonBase.Kind = ""
	return obj, nil
}

// DecodeInto parses a JSON string and stores it in obj. Returns an error
// if data.Kind is set and doesn't match the type of obj. Obj should be a
// pointer to an api type.
func DecodeInto(data []byte, obj interface{}) error {
	internal, err := Decode(data)
	if err != nil {
		return err
	}
	v := reflect.ValueOf(obj)
	iv := reflect.ValueOf(internal)
	if !iv.Type().AssignableTo(v.Type()) {
		return fmt.Errorf("%s is not assignable to %s", v.Type(), iv.Type())
	}
	v.Elem().Set(iv.Elem())
	name, jsonBase, err := nameAndJSONBase(obj)
	if err != nil {
		return err
	}
	if jsonBase.Kind != "" && jsonBase.Kind != name {
		return fmt.Errorf("data had kind %v, but passed object was of type %v", jsonBase.Kind, name)
	}
	// Don't leave these set. Track type with go's type.
	jsonBase.Kind = ""
	return nil
}

// TODO: Switch to registered functions for each type.
func internalize(obj interface{}) (interface{}, error) {
	v := reflect.ValueOf(obj)
	if v.Kind() != reflect.Ptr {
		value := reflect.New(v.Type())
		value.Elem().Set(v)
		result, err := internalize(value.Interface())
		if err != nil {
			return nil, err
		}
		return reflect.ValueOf(result).Elem().Interface(), nil
	}
	switch cObj := obj.(type) {
	case *v1beta1.JSONBase:
		obj := JSONBase(*cObj)
		return &obj, nil
	case *v1beta1.PodList:
		var items []Pod
		if cObj.Items != nil {
			items = make([]Pod, len(cObj.Items))
			for ix := range cObj.Items {
				iObj, err := internalize(cObj.Items[ix])
				if err != nil {
					return nil, err
				}
				items[ix] = iObj.(Pod)
			}
		}
		result := PodList{
			JSONBase: JSONBase(cObj.JSONBase),
			Items:    items,
		}
		result.APIVersion = ""
		return &result, nil
	case *v1beta1.Pod:
		current, err := internalize(cObj.CurrentState)
		if err != nil {
			return nil, err
		}
		desired, err := internalize(cObj.DesiredState)
		if err != nil {
			return nil, err
		}
		result := Pod{
			JSONBase:     JSONBase(cObj.JSONBase),
			Labels:       cObj.Labels,
			CurrentState: current.(PodState),
			DesiredState: desired.(PodState),
		}
		result.APIVersion = ""
		return &result, nil
	case *v1beta1.PodState:
		manifest, err := internalize(cObj.Manifest)
		if err != nil {
			return nil, err
		}
		result := PodState{
			Manifest: manifest.(ContainerManifest),
			Status:   PodStatus(cObj.Status),
			Host:     cObj.Host,
			HostIP:   cObj.HostIP,
			PodIP:    cObj.PodIP,
			Info:     PodInfo(cObj.Info),
		}
		return &result, nil
	case *v1beta1.ContainerManifest:
		var volumes []Volume
		if cObj.Volumes != nil {
			volumes = make([]Volume, len(cObj.Volumes))
			for ix := range cObj.Volumes {
				v, err := internalize(cObj.Volumes[ix])
				if err != nil {
					return nil, err
				}
				volumes[ix] = *(v.(*Volume))
			}
		}
		var containers []Container
		if cObj.Containers != nil {
			containers = make([]Container, len(cObj.Containers))
			for ix := range cObj.Containers {
				v, err := internalize(cObj.Containers[ix])
				if err != nil {
					return nil, err
				}
				containers[ix] = v.(Container)
			}
		}
		result := ContainerManifest{
			Version:    cObj.Version,
			ID:         cObj.ID,
			Volumes:    volumes,
			Containers: containers,
		}
		return &result, nil
	case *v1beta1.Volume:
		var src *VolumeSource
		if cObj.Source != nil {
			obj, err := internalize(cObj.Source)
			if err != nil {
				return nil, err
			}
			src = obj.(*VolumeSource)
		}
		result := &Volume{
			Name:   cObj.Name,
			Source: src,
		}
		return &result, nil
	case *v1beta1.VolumeSource:
		var hostDir *HostDirectory
		if cObj.HostDirectory != nil {
			hostDir = &HostDirectory{
				Path: cObj.HostDirectory.Path,
			}
		}
		var emptyDir *EmptyDirectory
		if cObj.EmptyDirectory != nil {
			emptyDir = &EmptyDirectory{}
		}
		result := VolumeSource{
			HostDirectory:  hostDir,
			EmptyDirectory: emptyDir,
		}
		return &result, nil
	case *v1beta1.Container:
		ports := make([]Port, len(cObj.Ports))
		for ix := range cObj.Ports {
			p, err := internalize(cObj.Ports[ix])
			if err != nil {
				return nil, err
			}
			ports[ix] = (p.(Port))
		}
		env := make([]EnvVar, len(cObj.Env))
		for ix := range cObj.Env {
			e, err := internalize(cObj.Env[ix])
			if err != nil {
				return nil, err
			}
			env[ix] = e.(EnvVar)
		}
		mounts := make([]VolumeMount, len(cObj.VolumeMounts))
		for ix := range cObj.VolumeMounts {
			v, err := internalize(cObj.VolumeMounts[ix])
			if err != nil {
				return nil, err
			}
			mounts[ix] = v.(VolumeMount)
		}
		var liveness *LivenessProbe
		if cObj.LivenessProbe != nil {
			probe, err := internalize(*cObj.LivenessProbe)
			if err != nil {
				return nil, err
			}
			live := probe.(LivenessProbe)
			liveness = &live
		}
		result := Container{
			Name:          cObj.Name,
			Image:         cObj.Image,
			Command:       cObj.Command,
			WorkingDir:    cObj.WorkingDir,
			Ports:         ports,
			Env:           env,
			Memory:        cObj.Memory,
			CPU:           cObj.CPU,
			VolumeMounts:  mounts,
			LivenessProbe: liveness,
		}
		return &result, nil
	case *v1beta1.Port:
		result := Port(*cObj)
		return &result, nil
	case *v1beta1.EnvVar:
		result := EnvVar(*cObj)
		return &result, nil
	case *v1beta1.VolumeMount:
		result := VolumeMount(*cObj)
		return &result, nil
	case *v1beta1.LivenessProbe:
		var http *HTTPGetProbe
		if cObj.HTTPGet != nil {
			httpProbe := HTTPGetProbe(*cObj.HTTPGet)
			http = &httpProbe
		}
		result := LivenessProbe{
			Type:                cObj.Type,
			HTTPGet:             http,
			InitialDelaySeconds: cObj.InitialDelaySeconds,
		}
		return &result, nil
	case *v1beta1.ReplicationControllerList:
		var items []ReplicationController
		if cObj.Items != nil {
			items = make([]ReplicationController, len(cObj.Items))
			for ix := range cObj.Items {
				rc, err := internalize(cObj.Items[ix])
				if err != nil {
					return nil, err
				}
				items[ix] = rc.(ReplicationController)
			}
		}
		result := ReplicationControllerList{
			JSONBase: JSONBase(cObj.JSONBase),
			Items:    items,
		}
		result.APIVersion = ""
		return &result, nil
	case *v1beta1.ReplicationController:
		desired, err := internalize(cObj.DesiredState)
		if err != nil {
			return nil, err
		}
		result := ReplicationController{
			JSONBase:     JSONBase(cObj.JSONBase),
			DesiredState: desired.(ReplicationControllerState),
			Labels:       cObj.Labels,
		}
		result.APIVersion = ""
		return &result, nil
	case *v1beta1.ReplicationControllerState:
		template, err := internalize(cObj.PodTemplate)
		if err != nil {
			return nil, err
		}
		result := ReplicationControllerState{
			Replicas:        cObj.Replicas,
			ReplicaSelector: cObj.ReplicaSelector,
			PodTemplate:     template.(PodTemplate),
		}
		return &result, nil
	case *v1beta1.PodTemplate:
		desired, err := internalize(cObj.DesiredState)
		if err != nil {
			return nil, err
		}
		return &PodTemplate{
			DesiredState: desired.(PodState),
			Labels:       cObj.Labels,
		}, nil
	case *v1beta1.ServiceList:
		var services []Service
		if cObj.Items != nil {
			services = make([]Service, len(cObj.Items))
			for ix := range cObj.Items {
				s, err := internalize(cObj.Items[ix])
				if err != nil {
					return nil, err
				}
				services[ix] = s.(Service)
				services[ix].APIVersion = ""
			}
		}
		result := ServiceList{
			JSONBase: JSONBase(cObj.JSONBase),
			Items:    services,
		}
		result.APIVersion = ""
		return &result, nil
	case *v1beta1.Service:
		result := Service{
			JSONBase:                   JSONBase(cObj.JSONBase),
			Port:                       cObj.Port,
			Labels:                     cObj.Labels,
			Selector:                   cObj.Selector,
			CreateExternalLoadBalancer: cObj.CreateExternalLoadBalancer,
			ContainerPort:              cObj.ContainerPort,
		}
		result.APIVersion = ""
		return &result, nil
	case *v1beta1.MinionList:
		minions := make([]Minion, len(cObj.Items))
		for ix := range cObj.Items {
			m, err := internalize(cObj.Items[ix])
			if err != nil {
				return nil, err
			}
			minions[ix] = m.(Minion)
		}
		result := MinionList{
			JSONBase: JSONBase(cObj.JSONBase),
			Items:    minions,
		}
		result.APIVersion = ""
		return &result, nil
	case *v1beta1.Minion:
		result := Minion{
			JSONBase: JSONBase(cObj.JSONBase),
			HostIP:   cObj.HostIP,
		}
		result.APIVersion = ""
		return &result, nil
	case *v1beta1.Status:
		result := Status{
			JSONBase: JSONBase(cObj.JSONBase),
			Status:   cObj.Status,
			Details:  cObj.Details,
			Code:     cObj.Code,
		}
		result.APIVersion = ""
		return &result, nil
	case *v1beta1.ServerOpList:
		ops := make([]ServerOp, len(cObj.Items))
		for ix := range cObj.Items {
			o, err := internalize(cObj.Items[ix])
			if err != nil {
				return nil, err
			}
			ops[ix] = o.(ServerOp)
		}
		result := ServerOpList{
			JSONBase: JSONBase(cObj.JSONBase),
			Items:    ops,
		}
		result.APIVersion = ""
		return &result, nil
	case *v1beta1.ServerOp:
		result := ServerOp{
			JSONBase: JSONBase(cObj.JSONBase),
		}
		result.APIVersion = ""
		return &result, nil
	default:
		fn, ok := internalFuncs[reflect.ValueOf(cObj).Elem().Type().Name()]
		if !ok {
			fmt.Printf("unknown object to internalize: %s", reflect.ValueOf(cObj).Type().Name())
			panic(fmt.Sprintf("unknown object to internalize: %s", reflect.ValueOf(cObj).Type().Name()))
		}
		return fn(cObj)
	}
	return obj, nil
}

// TODO: switch to registered functions for each type.
func externalize(obj interface{}) (interface{}, error) {
	v := reflect.ValueOf(obj)
	if v.Kind() != reflect.Ptr {
		value := reflect.New(v.Type())
		value.Elem().Set(v)
		result, err := externalize(value.Interface())
		if err != nil {
			return nil, err
		}
		return reflect.ValueOf(result).Elem().Interface(), nil
	}
	switch cObj := obj.(type) {
	case *PodList:
		var items []v1beta1.Pod
		if cObj.Items != nil {
			items = make([]v1beta1.Pod, len(cObj.Items))
			for ix := range cObj.Items {
				iObj, err := externalize(cObj.Items[ix])
				if err != nil {
					return nil, err
				}
				items[ix] = iObj.(v1beta1.Pod)
			}
		}
		result := v1beta1.PodList{
			JSONBase: v1beta1.JSONBase(cObj.JSONBase),
			Items:    items,
		}
		result.APIVersion = "v1beta1"
		return &result, nil
	case *Pod:
		current, err := externalize(cObj.CurrentState)
		if err != nil {
			return nil, err
		}
		desired, err := externalize(cObj.DesiredState)
		if err != nil {
			return nil, err
		}
		result := v1beta1.Pod{
			JSONBase:     v1beta1.JSONBase(cObj.JSONBase),
			Labels:       cObj.Labels,
			CurrentState: current.(v1beta1.PodState),
			DesiredState: desired.(v1beta1.PodState),
		}
		result.APIVersion = "v1beta1"
		return &result, nil
	case *PodState:
		manifest, err := externalize(cObj.Manifest)
		if err != nil {
			return nil, err
		}
		result := v1beta1.PodState{
			Manifest: manifest.(v1beta1.ContainerManifest),
			Status:   v1beta1.PodStatus(cObj.Status),
			Host:     cObj.Host,
			HostIP:   cObj.HostIP,
			PodIP:    cObj.PodIP,
			Info:     v1beta1.PodInfo(cObj.Info),
		}
		return &result, nil
	case *ContainerManifest:
		var volumes []v1beta1.Volume
		if cObj.Volumes != nil {
			volumes = make([]v1beta1.Volume, len(cObj.Volumes))
			for ix := range cObj.Volumes {
				v, err := externalize(cObj.Volumes[ix])
				if err != nil {
					return nil, err
				}
				volumes[ix] = *(v.(*v1beta1.Volume))
			}
		}
		var containers []v1beta1.Container
		if cObj.Containers != nil {
			containers = make([]v1beta1.Container, len(cObj.Containers))
			for ix := range cObj.Containers {
				v, err := externalize(cObj.Containers[ix])
				if err != nil {
					return nil, err
				}
				containers[ix] = v.(v1beta1.Container)
			}
		}
		result := v1beta1.ContainerManifest{
			Version:    cObj.Version,
			ID:         cObj.ID,
			Volumes:    volumes,
			Containers: containers,
		}
		return &result, nil
	case *Volume:
		var src *v1beta1.VolumeSource
		if cObj.Source != nil {
			obj, err := externalize(cObj.Source)
			if err != nil {
				return nil, err
			}
			src = obj.(*v1beta1.VolumeSource)
		}
		result := &v1beta1.Volume{
			Name:   cObj.Name,
			Source: src,
		}
		return &result, nil
	case *VolumeSource:
		var hostDir *v1beta1.HostDirectory
		if cObj.HostDirectory != nil {
			hostDir = &v1beta1.HostDirectory{
				Path: cObj.HostDirectory.Path,
			}
		}
		var emptyDir *v1beta1.EmptyDirectory
		if cObj.EmptyDirectory != nil {
			emptyDir = &v1beta1.EmptyDirectory{}
		}
		result := v1beta1.VolumeSource{
			HostDirectory:  hostDir,
			EmptyDirectory: emptyDir,
		}
		return &result, nil
	case *Container:
		ports := make([]v1beta1.Port, len(cObj.Ports))
		for ix := range cObj.Ports {
			p, err := externalize(cObj.Ports[ix])
			if err != nil {
				return nil, err
			}
			ports[ix] = p.(v1beta1.Port)
		}
		env := make([]v1beta1.EnvVar, len(cObj.Env))
		for ix := range cObj.Env {
			e, err := externalize(cObj.Env[ix])
			if err != nil {
				return nil, err
			}
			env[ix] = e.(v1beta1.EnvVar)
		}
		mounts := make([]v1beta1.VolumeMount, len(cObj.VolumeMounts))
		for ix := range cObj.VolumeMounts {
			v, err := externalize(cObj.VolumeMounts[ix])
			if err != nil {
				return nil, err
			}
			mounts[ix] = v.(v1beta1.VolumeMount)
		}
		var liveness *v1beta1.LivenessProbe
		if cObj.LivenessProbe != nil {
			probe, err := externalize(*cObj.LivenessProbe)
			if err != nil {
				return nil, err
			}
			live := probe.(v1beta1.LivenessProbe)
			liveness = &live
		}
		result := v1beta1.Container{
			Name:          cObj.Name,
			Image:         cObj.Image,
			Command:       cObj.Command,
			WorkingDir:    cObj.WorkingDir,
			Ports:         ports,
			Env:           env,
			Memory:        cObj.Memory,
			CPU:           cObj.CPU,
			VolumeMounts:  mounts,
			LivenessProbe: liveness,
		}
		return &result, nil
	case *Port:
		result := v1beta1.Port(*cObj)
		return &result, nil
	case *EnvVar:
		result := v1beta1.EnvVar(*cObj)
		return &result, nil
	case *VolumeMount:
		result := v1beta1.VolumeMount(*cObj)
		return &result, nil
	case *LivenessProbe:
		var http *v1beta1.HTTPGetProbe
		if cObj.HTTPGet != nil {
			httpProbe := v1beta1.HTTPGetProbe(*cObj.HTTPGet)
			http = &httpProbe
		}
		result := v1beta1.LivenessProbe{
			Type:                cObj.Type,
			HTTPGet:             http,
			InitialDelaySeconds: cObj.InitialDelaySeconds,
		}
		return &result, nil
	case *ReplicationControllerList:
		items := make([]v1beta1.ReplicationController, len(cObj.Items))
		for ix := range cObj.Items {
			rc, err := externalize(cObj.Items[ix])
			if err != nil {
				return nil, err
			}
			items[ix] = rc.(v1beta1.ReplicationController)
		}
		result := v1beta1.ReplicationControllerList{
			JSONBase: v1beta1.JSONBase(cObj.JSONBase),
			Items:    items,
		}
		result.APIVersion = "v1beta1"
		return &result, nil
	case *ReplicationController:
		desired, err := externalize(cObj.DesiredState)
		if err != nil {
			return nil, err
		}
		result := v1beta1.ReplicationController{
			JSONBase:     v1beta1.JSONBase(cObj.JSONBase),
			DesiredState: desired.(v1beta1.ReplicationControllerState),
			Labels:       cObj.Labels,
		}
		result.APIVersion = "v1beta1"
		return &result, nil
	case *ReplicationControllerState:
		template, err := externalize(cObj.PodTemplate)
		if err != nil {
			return nil, err
		}
		result := v1beta1.ReplicationControllerState{
			Replicas:        cObj.Replicas,
			ReplicaSelector: cObj.ReplicaSelector,
			PodTemplate:     template.(v1beta1.PodTemplate),
		}
		return &result, nil
	case *PodTemplate:
		desired, err := externalize(cObj.DesiredState)
		if err != nil {
			return nil, err
		}
		return &v1beta1.PodTemplate{
			DesiredState: desired.(v1beta1.PodState),
			Labels:       cObj.Labels,
		}, nil
	case *ServiceList:
		services := make([]v1beta1.Service, len(cObj.Items))
		for ix := range cObj.Items {
			s, err := externalize(cObj.Items[ix])
			if err != nil {
				return nil, err
			}
			services[ix] = s.(v1beta1.Service)
		}
		result := v1beta1.ServiceList{
			JSONBase: v1beta1.JSONBase(cObj.JSONBase),
			Items:    services,
		}
		result.APIVersion = "v1beta1"
		return &result, nil
	case *Service:
		result := v1beta1.Service{
			JSONBase:                   v1beta1.JSONBase(cObj.JSONBase),
			Port:                       cObj.Port,
			Labels:                     cObj.Labels,
			Selector:                   cObj.Selector,
			CreateExternalLoadBalancer: cObj.CreateExternalLoadBalancer,
			ContainerPort:              cObj.ContainerPort,
		}
		result.APIVersion = "v1beta1"
		return &result, nil
	case *MinionList:
		minions := make([]v1beta1.Minion, len(cObj.Items))
		for ix := range cObj.Items {
			m, err := externalize(cObj.Items[ix])
			if err != nil {
				return nil, err
			}
			minions[ix] = m.(v1beta1.Minion)
		}
		result := v1beta1.MinionList{
			JSONBase: v1beta1.JSONBase(cObj.JSONBase),
			Items:    minions,
		}
		result.APIVersion = "v1beta1"
		return &result, nil
	case *Minion:
		result := v1beta1.Minion{
			JSONBase: v1beta1.JSONBase(cObj.JSONBase),
			HostIP:   cObj.HostIP,
		}
		result.APIVersion = "v1beta1"
		return &result, nil
	case *Status:
		result := v1beta1.Status{
			JSONBase: v1beta1.JSONBase(cObj.JSONBase),
			Status:   cObj.Status,
			Details:  cObj.Details,
			Code:     cObj.Code,
		}
		result.APIVersion = "v1beta1"
		return &result, nil
	case *ServerOpList:
		ops := make([]v1beta1.ServerOp, len(cObj.Items))
		for ix := range cObj.Items {
			o, err := externalize(cObj.Items[ix])
			if err != nil {
				return nil, err
			}
			ops[ix] = o.(v1beta1.ServerOp)
		}
		result := v1beta1.ServerOpList{
			JSONBase: v1beta1.JSONBase(cObj.JSONBase),
			Items:    ops,
		}
		result.APIVersion = "v1beta1"
		return &result, nil
	case *ServerOp:
		result := v1beta1.ServerOp{
			JSONBase: v1beta1.JSONBase(cObj.JSONBase),
		}
		result.APIVersion = "v1beta1"
		return &result, nil
	default:
		fn, ok := externalFuncs[reflect.ValueOf(cObj).Elem().Type().Name()]
		if !ok {
			panic(fmt.Sprintf("Unknown object to externalize: %#v %s", cObj, reflect.ValueOf(cObj).Type().Name()))
		}
		return fn(cObj)
	}
	panic(fmt.Sprintf("This should never happen %#v", obj))
	return obj, nil
}
