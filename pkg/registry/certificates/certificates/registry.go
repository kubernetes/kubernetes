/*
Copyright 2016 The Kubernetes Authors.

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

package certificates

import (
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/watch"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/apis/certificates"
)

// Registry is an interface for things that know how to store CSRs.
type Registry interface {
	ListCSRs(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*certificates.CertificateSigningRequestList, error)
	CreateCSR(ctx genericapirequest.Context, csr *certificates.CertificateSigningRequest) error
	UpdateCSR(ctx genericapirequest.Context, csr *certificates.CertificateSigningRequest) error
	GetCSR(ctx genericapirequest.Context, csrID string, options *metav1.GetOptions) (*certificates.CertificateSigningRequest, error)
	DeleteCSR(ctx genericapirequest.Context, csrID string) error
	WatchCSRs(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error)
}

// storage puts strong typing around storage calls
type storage struct {
	rest.StandardStorage
}

// NewRegistry returns a new Registry interface for the given Storage. Any mismatched
// types will panic.
func NewRegistry(s rest.StandardStorage) Registry {
	return &storage{s}
}

func (s *storage) ListCSRs(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (*certificates.CertificateSigningRequestList, error) {
	obj, err := s.List(ctx, options)
	if err != nil {
		return nil, err
	}

	return obj.(*certificates.CertificateSigningRequestList), nil
}

func (s *storage) CreateCSR(ctx genericapirequest.Context, csr *certificates.CertificateSigningRequest) error {
	_, err := s.Create(ctx, csr)
	return err
}

func (s *storage) UpdateCSR(ctx genericapirequest.Context, csr *certificates.CertificateSigningRequest) error {
	_, _, err := s.Update(ctx, csr.Name, rest.DefaultUpdatedObjectInfo(csr, api.Scheme))
	return err
}

func (s *storage) WatchCSRs(ctx genericapirequest.Context, options *metainternalversion.ListOptions) (watch.Interface, error) {
	return s.Watch(ctx, options)
}

func (s *storage) GetCSR(ctx genericapirequest.Context, name string, options *metav1.GetOptions) (*certificates.CertificateSigningRequest, error) {
	obj, err := s.Get(ctx, name, options)
	if err != nil {
		return nil, err
	}
	return obj.(*certificates.CertificateSigningRequest), nil
}

func (s *storage) DeleteCSR(ctx genericapirequest.Context, name string) error {
	_, err := s.Delete(ctx, name, nil)
	return err
}
