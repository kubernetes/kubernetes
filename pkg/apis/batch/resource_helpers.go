package batch

import "k8s.io/kubernetes/pkg/api"

func IsJobFinished(j Job) bool {
	for _, c := range j.Status.Conditions {
		if (c.Type == JobComplete || c.Type == JobFailed) && c.Status == api.ConditionTrue {
			return true
		}
	}
	return false
}