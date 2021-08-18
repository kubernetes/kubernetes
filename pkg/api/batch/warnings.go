package batch

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/features"
)

var TTLWarningMsg = "This job doesn't have TTLAfterFinished field and not managed by a higher level controller. " +
	"Setting TTLAfterFinished on job.spec helps in removal of dependant pods"

func GetWarningsForJob(job *batch.Job) []string {
	if job == nil {
		return nil
	}
	return warningsForJobSpec(&job.Spec, &job.ObjectMeta)
}

func warningsForJobSpec(jobSpec *batch.JobSpec, meta *metav1.ObjectMeta) []string {
	var warnings []string
	hasManagingController := false
	for _, ownerRef := range meta.OwnerReferences {
		if *ownerRef.Controller {
			hasManagingController = true
		}
	}
	// If it is a stand alone job(not created via another controller) and if TTLAfterFinished enabled but value is not
	// set pods may get orphaned because the deletion policy of job by default is orphanDependents. If the job is
	// managed by higher level controller, like cronjob controller, it will delete the dependents.
	if utilfeature.DefaultFeatureGate.Enabled(features.TTLAfterFinished) && !hasManagingController &&
		jobSpec.TTLSecondsAfterFinished == nil {
		warnings = append(warnings, fmt.Sprint(TTLWarningMsg))
	}
	return warnings
}
