package generated

import (
	"fmt"
	"reflect"

	_k8s_io_kubernetes_pkg_api_v1 "k8s.io/kubernetes/pkg/api/v1"
)

func applyDefaults__k8s_io_kubernetes_pkg_api_v1_Pod(obj *_k8s_io_kubernetes_pkg_api_v1.Pod) {
	// Spec v1.PodSpec
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodSpec((&(obj.Spec)))
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodSpec(obj *_k8s_io_kubernetes_pkg_api_v1.PodSpec) {
	// Volumes []v1.Volume
	for i := range obj.Volumes {
		p := &((obj.Volumes)[i])
		applyDefaults__k8s_io_kubernetes_pkg_api_v1_Volume((p))
	}
	// Containers []v1.Container
	for i := range obj.Containers {
		p := &((obj.Containers)[i])
		applyDefaults__k8s_io_kubernetes_pkg_api_v1_Container((p))
	}
	obj.ApplyDefaults()
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_Volume(obj *_k8s_io_kubernetes_pkg_api_v1.Volume) {
	// VolumeSource v1.VolumeSource
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_VolumeSource((&(obj.VolumeSource)))
	obj.ApplyDefaults()
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_VolumeSource(obj *_k8s_io_kubernetes_pkg_api_v1.VolumeSource) {
	obj.ApplyDefaults()
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_Container(obj *_k8s_io_kubernetes_pkg_api_v1.Container) {
	// Ports []v1.ContainerPort
	for i := range obj.Ports {
		p := &((obj.Ports)[i])
		applyDefaults__k8s_io_kubernetes_pkg_api_v1_ContainerPort((p))
	}
	obj.ApplyDefaults()
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_ContainerPort(obj *_k8s_io_kubernetes_pkg_api_v1.ContainerPort) {
	obj.ApplyDefaults()
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_ReplicationController(obj *_k8s_io_kubernetes_pkg_api_v1.ReplicationController) {
	// Spec v1.ReplicationControllerSpec
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_ReplicationControllerSpec((&(obj.Spec)))
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_ReplicationControllerSpec(obj *_k8s_io_kubernetes_pkg_api_v1.ReplicationControllerSpec) {
	// Template *v1.PodTemplateSpec
	if (obj.Template) != nil {
		applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodTemplateSpec((obj.Template))
	}
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodTemplateSpec(obj *_k8s_io_kubernetes_pkg_api_v1.PodTemplateSpec) {
	// Spec v1.PodSpec
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodSpec((&(obj.Spec)))
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodTemplateList(obj *_k8s_io_kubernetes_pkg_api_v1.PodTemplateList) {
	// Items []v1.PodTemplate
	for i := range obj.Items {
		p := &((obj.Items)[i])
		applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodTemplate((p))
	}
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodTemplate(obj *_k8s_io_kubernetes_pkg_api_v1.PodTemplate) {
	// Template v1.PodTemplateSpec
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodTemplateSpec((&(obj.Template)))
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_Endpoints(obj *_k8s_io_kubernetes_pkg_api_v1.Endpoints) {
	// Subsets []v1.EndpointSubset
	for i := range obj.Subsets {
		p := &((obj.Subsets)[i])
		applyDefaults__k8s_io_kubernetes_pkg_api_v1_EndpointSubset((p))
	}
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_EndpointSubset(obj *_k8s_io_kubernetes_pkg_api_v1.EndpointSubset) {
	// Ports []v1.EndpointPort
	for i := range obj.Ports {
		p := &((obj.Ports)[i])
		applyDefaults__k8s_io_kubernetes_pkg_api_v1_EndpointPort((p))
	}
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_EndpointPort(obj *_k8s_io_kubernetes_pkg_api_v1.EndpointPort) {
	obj.ApplyDefaults()
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_Daemon(obj *_k8s_io_kubernetes_pkg_api_v1.Daemon) {
	// Spec v1.DaemonSpec
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_DaemonSpec((&(obj.Spec)))
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_DaemonSpec(obj *_k8s_io_kubernetes_pkg_api_v1.DaemonSpec) {
	// Template *v1.PodTemplateSpec
	if (obj.Template) != nil {
		applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodTemplateSpec((obj.Template))
	}
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_DaemonList(obj *_k8s_io_kubernetes_pkg_api_v1.DaemonList) {
	// Items []v1.Daemon
	for i := range obj.Items {
		p := &((obj.Items)[i])
		applyDefaults__k8s_io_kubernetes_pkg_api_v1_Daemon((p))
	}
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_ReplicationControllerList(obj *_k8s_io_kubernetes_pkg_api_v1.ReplicationControllerList) {
	// Items []v1.ReplicationController
	for i := range obj.Items {
		p := &((obj.Items)[i])
		applyDefaults__k8s_io_kubernetes_pkg_api_v1_ReplicationController((p))
	}
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodList(obj *_k8s_io_kubernetes_pkg_api_v1.PodList) {
	// Items []v1.Pod
	for i := range obj.Items {
		p := &((obj.Items)[i])
		applyDefaults__k8s_io_kubernetes_pkg_api_v1_Pod((p))
	}
}
func applyDefaults__k8s_io_kubernetes_pkg_api_v1_EndpointsList(obj *_k8s_io_kubernetes_pkg_api_v1.EndpointsList) {
	// Items []v1.Endpoints
	for i := range obj.Items {
		p := &((obj.Items)[i])
		applyDefaults__k8s_io_kubernetes_pkg_api_v1_Endpoints((p))
	}
}
func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Pod(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.Pod)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Pod called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_Pod(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodSpec(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.PodSpec)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodSpec called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodSpec(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Volume(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.Volume)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Volume called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_Volume(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_VolumeSource(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.VolumeSource)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_VolumeSource called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_VolumeSource(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Container(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.Container)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Container called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_Container(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_ContainerPort(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.ContainerPort)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_ContainerPort called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_ContainerPort(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_ReplicationController(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.ReplicationController)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_ReplicationController called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_ReplicationController(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_ReplicationControllerSpec(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.ReplicationControllerSpec)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_ReplicationControllerSpec called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_ReplicationControllerSpec(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodTemplateSpec(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.PodTemplateSpec)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodTemplateSpec called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodTemplateSpec(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodTemplateList(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.PodTemplateList)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodTemplateList called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodTemplateList(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodTemplate(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.PodTemplate)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodTemplate called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodTemplate(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Endpoints(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.Endpoints)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Endpoints called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_Endpoints(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_EndpointSubset(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.EndpointSubset)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_EndpointSubset called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_EndpointSubset(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_EndpointPort(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.EndpointPort)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_EndpointPort called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_EndpointPort(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Daemon(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.Daemon)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Daemon called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_Daemon(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_DaemonSpec(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.DaemonSpec)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_DaemonSpec called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_DaemonSpec(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_DaemonList(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.DaemonList)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_DaemonList called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_DaemonList(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_ReplicationControllerList(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.ReplicationControllerList)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_ReplicationControllerList called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_ReplicationControllerList(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodList(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.PodList)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodList called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_PodList(concrete)
}

func applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_EndpointsList(obj interface{}) {
	concrete, ok := obj.(*_k8s_io_kubernetes_pkg_api_v1.EndpointsList)
	if !ok {
		panic(fmt.Sprintf("applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_EndpointsList called for type %T", obj))
	}
	applyDefaults__k8s_io_kubernetes_pkg_api_v1_EndpointsList(concrete)
}

func InitDefaultableTypes(register func(reflect.Type, func(obj interface{}))) {
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.Pod)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Pod)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.PodSpec)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodSpec)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.Volume)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Volume)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.VolumeSource)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_VolumeSource)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.Container)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Container)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.ContainerPort)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_ContainerPort)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.ReplicationController)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_ReplicationController)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.ReplicationControllerSpec)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_ReplicationControllerSpec)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.PodTemplateSpec)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodTemplateSpec)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.PodTemplateList)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodTemplateList)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.PodTemplate)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodTemplate)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.Endpoints)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Endpoints)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.EndpointSubset)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_EndpointSubset)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.EndpointPort)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_EndpointPort)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.Daemon)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_Daemon)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.DaemonSpec)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_DaemonSpec)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.DaemonList)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_DaemonList)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.ReplicationControllerList)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_ReplicationControllerList)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.PodList)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_PodList)
	register(
		reflect.TypeOf(new(_k8s_io_kubernetes_pkg_api_v1.EndpointsList)),
		applyDefaults_entry__k8s_io_kubernetes_pkg_api_v1_EndpointsList)
}
