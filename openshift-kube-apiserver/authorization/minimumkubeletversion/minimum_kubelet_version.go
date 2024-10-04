package minimumkubeletversion

import (
	"context"
	"fmt"

	"github.com/blang/semver/v4"
	nodelib "github.com/openshift/library-go/pkg/apiserver/node"
	authorizationv1 "k8s.io/api/authorization/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	v1listers "k8s.io/client-go/listers/core/v1"
	cache "k8s.io/client-go/tools/cache"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
)

type minimumKubeletVersionAuth struct {
	nodeIdentifier          nodeidentifier.NodeIdentifier
	nodeLister              v1listers.NodeLister
	minVersion              *semver.Version
	hasNodeInformerSyncedFn func() bool // factored for unit tests
}

// Creates a new minimumKubeletVersionAuth object, which is an authorizer that checks
// whether nodes are new enough to be authorized.
func NewMinimumKubeletVersion(minVersionStr string,
	nodeIdentifier nodeidentifier.NodeIdentifier,
	nodeInformer cache.SharedIndexInformer,
	nodeLister v1listers.NodeLister,
) *minimumKubeletVersionAuth {
	var minVersionPtr *semver.Version
	if len(minVersionStr) != 0 {
		v := semver.MustParse(minVersionStr)
		minVersionPtr = &v
	}

	return &minimumKubeletVersionAuth{
		nodeIdentifier:          nodeIdentifier,
		nodeLister:              nodeLister,
		hasNodeInformerSyncedFn: nodeInformer.HasSynced,
		minVersion:              minVersionPtr,
	}
}

func (m *minimumKubeletVersionAuth) Authorize(ctx context.Context, attrs authorizer.Attributes) (authorizer.Decision, string, error) {
	if m.minVersion == nil {
		return authorizer.DecisionNoOpinion, "", nil
	}

	nodeName, isNode := m.nodeIdentifier.NodeIdentity(attrs.GetUser())
	if !isNode {
		// ignore requests from non-nodes
		return authorizer.DecisionNoOpinion, "", nil
	}

	if len(nodeName) == 0 {
		return authorizer.DecisionNoOpinion, fmt.Sprintf("unknown node for user %q", attrs.GetUser().GetName()), nil
	}

	// Short-circut if "subjectaccessreviews", or a "get" or "update" on the node object.
	// Regardless of kubelet version, it should be allowed to do these things.
	if attrs.IsResourceRequest() {
		requestResource := schema.GroupResource{Group: attrs.GetAPIGroup(), Resource: attrs.GetResource()}
		switch requestResource {
		case api.Resource("nodes"):
			if v := attrs.GetVerb(); v == "get" || v == "update" {
				return authorizer.DecisionNoOpinion, "", nil
			}
		// TODO(haircommander): do we need other flavors of access reviews here?
		case authorizationv1.Resource("subjectaccessreviews"):
			return authorizer.DecisionNoOpinion, "", nil
		}
	}

	if !m.hasNodeInformerSyncedFn() {
		return authorizer.DecisionNoOpinion, fmt.Sprintf("node informer not synced, cannot check if node %s is new enough", nodeName), nil
	}

	node, err := m.nodeLister.Get(nodeName)
	if err != nil {
		return authorizer.DecisionNoOpinion, fmt.Sprintf("failed to get node %s: %v", nodeName, err), nil
	}

	if err := nodelib.IsNodeTooOld(node, m.minVersion); err != nil {
		return authorizer.DecisionDeny, err.Error(), nil
	}

	return authorizer.DecisionNoOpinion, "", nil
}
