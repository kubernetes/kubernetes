package placementbinpacking

import (
	"context"

	"k8s.io/apimachinery/pkg/runtime"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
)

var _ fwk.PlacementScorerPlugin = &PlacementBinPacking{}

type PlacementBinPacking struct {
	handle fwk.Handle
}

const Name = names.PlacementBinPacking

func (pl *PlacementBinPacking) Name() string {
	return Name
}

func New(_ context.Context, _ runtime.Object, fh fwk.Handle, fts feature.Features) (*PlacementBinPacking, error) {
	return &PlacementBinPacking{handle: fh}, nil
}

func (pl *PlacementBinPacking) ScorePlacement(ctx context.Context, state *fwk.CycleState, podGroup *fwk.PodGroupInfo, placement *fwk.Placement, podGroupAssignments *fwk.PodGroupAssignments) (float64, *fwk.Status) {
	return 0, nil
}
