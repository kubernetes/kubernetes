package app

import (
	"github.com/openshift/library-go/pkg/authorization/hardcodedauthorizer"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/apiserver/pkg/authorization/union"
)

// wrapAuthorizerWithMetricsScraper add an authorizer to always approver the openshift metrics scraper.
// This eliminates an unnecessary SAR for scraping metrics and enables metrics gathering when network access
// to the kube-apiserver is interrupted
func wrapAuthorizerWithMetricsScraper(authz authorizer.Authorizer) authorizer.Authorizer {
	return union.New(
		hardcodedauthorizer.NewHardCodedMetricsAuthorizer(),
		authz,
	)
}
