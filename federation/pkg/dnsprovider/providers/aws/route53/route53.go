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

// route53 is the implementation of pkg/dnsprovider interface for AWS Route53
package route53

import (
	"io"

	"k8s.io/kubernetes/federation/pkg/dnsprovider"

	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/route53"
)

const (
	ProviderName = "aws-route53"
)

func init() {
	dnsprovider.RegisterDnsProvider(ProviderName, func(config io.Reader) (dnsprovider.Interface, error) {
		return newRoute53(config)
	})
}

// newRoute53 creates a new instance of an AWS Route53 DNS Interface.
func newRoute53(config io.Reader) (*Interface, error) {
	// Connect to AWS Route53 - TODO: Do more sophisticated auth
	svc := route53.New(session.New())
	return New(svc), nil
}
