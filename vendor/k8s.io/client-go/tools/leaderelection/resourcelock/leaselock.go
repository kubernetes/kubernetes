/*
Copyright 2018 The Kubernetes Authors.

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

package resourcelock

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	coordinationv1 "k8s.io/api/coordination/v1"
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	coordinationv1client "k8s.io/client-go/kubernetes/typed/coordination/v1"
)

type LeaseLock struct {
	// LeaseMeta should contain a Name and a Namespace of a
	// LeaseMeta object that the LeaderElector will attempt to lead.
	LeaseMeta  metav1.ObjectMeta
	Client     coordinationv1client.LeasesGetter
	LockConfig ResourceLockConfig
	lease      *coordinationv1.Lease
}

// Get returns the election record from a Lease spec
func (ll *LeaseLock) Get(ctx context.Context) (*LeaderElectionRecord, []byte, error) {
	var err error
	ll.lease, err = ll.Client.Leases(ll.LeaseMeta.Namespace).Get(ctx, ll.LeaseMeta.Name, metav1.GetOptions{})
	if err != nil {
		return nil, nil, err
	}
	record := LeaseSpecToLeaderElectionRecord(&ll.lease.Spec)
	recordByte, err := json.Marshal(*record)
	if err != nil {
		return nil, nil, err
	}
	return record, recordByte, nil
}

// Create attempts to create a Lease
func (ll *LeaseLock) Create(ctx context.Context, ler LeaderElectionRecord) error {
	var err error
	ll.lease, err = ll.Client.Leases(ll.LeaseMeta.Namespace).Create(ctx, &coordinationv1.Lease{
		ObjectMeta: metav1.ObjectMeta{
			Name:      ll.LeaseMeta.Name,
			Namespace: ll.LeaseMeta.Namespace,
		},
		Spec: LeaderElectionRecordToLeaseSpec(&ler),
	}, metav1.CreateOptions{})
	return err
}

// Update will update an existing Lease spec.
func (ll *LeaseLock) Update(ctx context.Context, ler LeaderElectionRecord) error {
	if ll.lease == nil {
		return errors.New("lease not initialized, call get or create first")
	}
	ll.lease.Spec = LeaderElectionRecordToLeaseSpec(&ler)

	lease, err := ll.Client.Leases(ll.LeaseMeta.Namespace).Update(ctx, ll.lease, metav1.UpdateOptions{})
	if err != nil {
		return err
	}

	ll.lease = lease
	return nil
}

// RecordEvent in leader election while adding meta-data
func (ll *LeaseLock) RecordEvent(s string) {
	if ll.LockConfig.EventRecorder == nil {
		return
	}
	events := fmt.Sprintf("%v %v", ll.LockConfig.Identity, s)
	subject := &coordinationv1.Lease{ObjectMeta: ll.lease.ObjectMeta}
	// Populate the type meta, so we don't have to get it from the schema
	subject.Kind = "Lease"
	subject.APIVersion = coordinationv1.SchemeGroupVersion.String()
	ll.LockConfig.EventRecorder.Eventf(subject, corev1.EventTypeNormal, "LeaderElection", events)
}

// Describe is used to convert details on current resource lock
// into a string
func (ll *LeaseLock) Describe() string {
	return fmt.Sprintf("%v/%v", ll.LeaseMeta.Namespace, ll.LeaseMeta.Name)
}

// Identity returns the Identity of the lock
func (ll *LeaseLock) Identity() string {
	return ll.LockConfig.Identity
}

func LeaseSpecToLeaderElectionRecord(spec *coordinationv1.LeaseSpec) *LeaderElectionRecord {
	var r LeaderElectionRecord
	if spec.HolderIdentity != nil {
		r.HolderIdentity = *spec.HolderIdentity
	}
	if spec.LeaseDurationSeconds != nil {
		r.LeaseDurationSeconds = int(*spec.LeaseDurationSeconds)
	}
	if spec.LeaseTransitions != nil {
		r.LeaderTransitions = int(*spec.LeaseTransitions)
	}
	if spec.AcquireTime != nil {
		r.AcquireTime = metav1.Time{spec.AcquireTime.Time}
	}
	if spec.RenewTime != nil {
		r.RenewTime = metav1.Time{spec.RenewTime.Time}
	}
	return &r

}

func LeaderElectionRecordToLeaseSpec(ler *LeaderElectionRecord) coordinationv1.LeaseSpec {
	leaseDurationSeconds := int32(ler.LeaseDurationSeconds)
	leaseTransitions := int32(ler.LeaderTransitions)
	return coordinationv1.LeaseSpec{
		HolderIdentity:       &ler.HolderIdentity,
		LeaseDurationSeconds: &leaseDurationSeconds,
		AcquireTime:          &metav1.MicroTime{ler.AcquireTime.Time},
		RenewTime:            &metav1.MicroTime{ler.RenewTime.Time},
		LeaseTransitions:     &leaseTransitions,
	}
}
