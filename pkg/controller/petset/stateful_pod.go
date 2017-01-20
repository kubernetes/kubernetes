package petset

import (
	"errors"
	"fmt"
	"github.com/golang/glog"
	"k8s.io/client-go/pkg/util/sets"
	apiErrors "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/v1"
	podapi "k8s.io/kubernetes/pkg/api/v1/pod"
	apps "k8s.io/kubernetes/pkg/apis/apps/v1beta1"
	metav1 "k8s.io/kubernetes/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"
	"k8s.io/kubernetes/pkg/client/record"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/runtime"
	errorUtils "k8s.io/kubernetes/pkg/util/errors"
	"strconv"
)

func ordinalOf(stateful *apps.StatefulSet, pod *v1.Pod) int {
	id := -1
	fmt.Sscanf(pod.Name, stateful.Name+"-%d", &id)
	return id
}

type statefulPod struct {
	*v1.Pod
	ordinal int
	parent  *apps.StatefulSet
	dirty   bool
	exists  bool
}

func newStatefulPod(set *apps.StatefulSet, ordinal int) statefulPod {
	pod, _ := controller.GetPodFromTemplate(&set.Spec.Template, set, nil)
	stateful := statefulPod{
		pod,
		ordinal,
		set,
		false,
		false}
	stateful.updateId()
	stateful.updateVolumes()
	stateful.dirty = false
	return stateful
}

func fromV1Pod(set *apps.StatefulSet, pod *v1.Pod) statefulPod {
	stateful := statefulPod{
		pod,
		ordinalOf(set, pod),
		set,
		false,
		true}
	if !stateful.hasValidId() {
		stateful.updateId()
	}
	if !stateful.hasValidVolumes() {
		stateful.updateVolumes()
	}
	return stateful
}

func (pod *statefulPod) uniqueName() string {
	return fmt.Sprintf("%s-%d", pod.parent.Name, pod.ordinal)
}

func (pod *statefulPod) hasValidId() bool {
	return pod.ordinal >= 0 &&
		pod.Name == pod.uniqueName() &&
		pod.Namespace == pod.parent.Namespace &&
		pod.Annotations != nil &&
		pod.Annotations[podapi.PodHostnameAnnotation] == pod.Name &&
		pod.Annotations[podapi.PodSubdomainAnnotation] == pod.parent.Spec.ServiceName
}

func (pod *statefulPod) terminating() bool {
	return pod.DeletionTimestamp != nil
}

func (pod *statefulPod) runningAndReady() bool {
	if pod.Status.Phase != v1.PodRunning {
		return false
	}
	podReady := v1.IsPodReady(pod.Pod)
	// User may have specified a pod readiness override through a debug annotation.
	if initialized, hasInit := pod.Annotations[StatefulSetInitAnnotation]; hasInit {
		if initAnnotation, err := strconv.ParseBool(initialized); err != nil {
			glog.V(4).Infof("Failed to parse %v annotation on pod %v: %v",
				StatefulSetInitAnnotation,
				pod.Name,
				err)
		} else if !initAnnotation {
			glog.V(4).Infof("StatefulSet pod %v waiting on annotation %v",
				pod.Name,
				StatefulSetInitAnnotation)
			podReady = initAnnotation
		}
	}
	return podReady
}

func (pod *statefulPod) volumeNames() sets.String {
	names := sets.NewString()
	for _, volume := range pod.Spec.Volumes {
		names.Insert(volume.Name)
	}
	return names
}

func (pod *statefulPod) volumeClaims() map[string]v1.PersistentVolumeClaim {
	claims := make(map[string]v1.PersistentVolumeClaim, len(pod.parent.Spec.VolumeClaimTemplates))
	for _, pvc := range pod.parent.Spec.VolumeClaimTemplates {
		claim := pvc
		// TODO: Name length checking in validation.
		claim.Name = fmt.Sprintf("%s-%s-%d", claim.Name, pod.parent.Name, pod.ordinal)
		claim.Namespace = pod.parent.Namespace
		claim.Labels = pod.parent.Spec.Selector.MatchLabels
		// TODO: We're assuming that the claim template has a volume QoS key, eg:
		// volume.alpha.kubernetes.io/storage-class: anything
		claims[pvc.Name] = claim
	}
	return claims
}

func (pod *statefulPod) volumeClaimNames() sets.String {
	names := sets.NewString()
	for name := range pod.volumeClaims() {
		names.Insert(name)
	}
	return names
}

func (pod *statefulPod) hasValidVolumes() bool {
	return pod.volumeNames().HasAll(pod.volumeClaimNames().List()...)
}

func (pod *statefulPod) updateId() {
	pod.Name = pod.uniqueName()
	pod.Namespace = pod.parent.Namespace
	if pod.Annotations == nil {
		pod.Annotations = make(map[string]string)
	}
	pod.Annotations[podapi.PodHostnameAnnotation] = pod.Name
	pod.Annotations[podapi.PodSubdomainAnnotation] = pod.parent.Spec.ServiceName
	pod.dirty = true
}

func (pod *statefulPod) updateVolumes() {
	volumesNames := pod.volumeNames()
	claims := pod.volumeClaims()
	currentVolumes := pod.Spec.Volumes
	newVolumes := make([]v1.Volume, 0, len(claims))
	for name, claim := range claims {
		if volumesNames.Has(name) {
			// TODO: Validate and reject this.
			glog.V(4).Infof("Overwriting existing volume source %v", name)
		}
		newVolumes = append(newVolumes, v1.Volume{
			Name: name,
			VolumeSource: v1.VolumeSource{
				PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
					ClaimName: claim.Name,
					// TODO: Use source definition to set this value when we have one.
					ReadOnly: false,
				},
			},
		})
	}
	for i := range currentVolumes {
		if _, ok := claims[currentVolumes[i].Name]; !ok {
			newVolumes = append(newVolumes, currentVolumes[i])
		}
	}
	pod.Spec.Volumes = newVolumes
	pod.dirty = true
}

type orderedStatefulPods []statefulPod

func (set orderedStatefulPods) Len() int {
	return len(set)
}

func (set orderedStatefulPods) Less(i, j int) bool {
	return set[i].ordinal < set[j].ordinal
}

func (set orderedStatefulPods) Swap(i, j int) {
	temp := set[i]
	set[i] = set[j]
	set[j] = temp
}

type statefulPodClient interface {
	syncVolumeClaims(*statefulPod) error
	deleteVolumes(*statefulPod) error
	delete(*statefulPod) error
	get(*statefulPod) (*statefulPod, error)
	create(*statefulPod) error
	update(*statefulPod, *pcb) error
}

// apiServerPetClient is a statefulset aware Kubernetes client.
type apiServerClient struct {
	clientSet clientset.Interface
	recorder  record.EventRecorder
}

func (api *apiServerClient) getVolumeClaim(name, namespace string) (*v1.PersistentVolumeClaim, error) {
	pvc, err := api.clientSet.Core().
		PersistentVolumeClaims(namespace).
		Get(name, metav1.GetOptions{})
	return pvc, err
}

func (api *apiServerClient) createVolumeClaim(pod *statefulPod, pvc *v1.PersistentVolumeClaim) error {
	_, err := api.clientSet.Core().PersistentVolumeClaims(pvc.Namespace).Create(pvc)
	if err != nil {
		api.recorder.Event(
			pod.parent,
			v1.EventTypeWarning,
			"FailedCreate",
			fmt.Sprintf("StatefulSet %s failed to create PVC %s for Pod %s, error: %s",
				pod.parent.Name,
				pvc.Name,
				pod.Name, err))
	} else {
		api.recorder.Event(pod.parent,
			v1.EventTypeNormal,
			"SuccessfulCreate",
			fmt.Sprintf("StatefulSet %s created PVC %s for Pod %s",
				pod.parent.Name,
				pvc.Name,
				pod.Name))
	}
	return err
}

func (api *apiServerClient) syncVolumeClaims(pod *statefulPod) error {
	errs := make([]error, 0)
	// Create new claims.
	for _, claim := range pod.volumeClaims() {
		_, err := api.getVolumeClaim(claim.Name, claim.Namespace)
		if err != nil {
			if apiErrors.IsNotFound(err) {
				if err := api.createVolumeClaim(pod, &claim); err != nil {
					errs = append(
						errs,
						fmt.Sprintf("Failed to create PVC %s: %s",
							claim.Name,
							err))
				}

			} else {
				errs = append(
					errs,
					fmt.Sprintf("Failed to retrieve PVC %s: %s",
						claim.Name,
						err))
			}
		}
		// TODO: Check resource requirements and accessmodes, update if necessary
	}
	if len(errs) > 0 {
		return errorUtils.NewAggregate(errs)
	}
	return nil
}

func (api *apiServerClient) deleteVolumes(pod *statefulPod) error {
	return errors.New("deleteVolumes not implemented")
}

func (api *apiServerClient) delete(pod *statefulPod) error {
	err := api.clientSet.Core().Pods(pod.Namespace).Delete(pod.Name, nil)
	if apiErrors.IsNotFound(err) {
		err = nil
	}
	api.event(pod.parent, "Delete", fmt.Sprintf("Pod %s", pod.Name), err)
	return err
}

func (api *apiServerClient) get(pod *statefulPod) (statefulPod, error) {
	retrieved, err := api.clientSet.Core().Pods(pod.Namespace).Get(pod.Name, metav1.GetOptions{})
	if apiErrors.IsNotFound(err) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	return fromV1Pod(pod.parent, retrieved), true, nil
}

// event formats an event for the given runtime object.
func (api apiServerClient) event(obj runtime.Object, reason, msg string, err error) {
	if err != nil {
		api.recorder.Event(obj, v1.EventTypeWarning,
			fmt.Sprintf("Failed%v", reason),
			fmt.Sprintf("%v, error: %v", msg, err))
	} else {
		api.recorder.Event(obj, v1.EventTypeNormal,
			fmt.Sprintf("Successful%v", reason),
			msg)
	}
}
