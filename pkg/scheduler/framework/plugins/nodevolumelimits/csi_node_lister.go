package nodevolumelimits

import (
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/labels"
	storagelisters "k8s.io/client-go/listers/storage/v1"
	fwk "k8s.io/kube-scheduler/framework"
)

type CSINodeListerImpl struct {
	csinodeLister storagelisters.CSINodeLister
}

var _ fwk.CSINodeLister = &CSINodeListerImpl{}

func NewCSINodeLister(csinodeLister storagelisters.CSINodeLister) *CSINodeListerImpl {
	return &CSINodeListerImpl{csinodeLister: csinodeLister}
}

func (l *CSINodeListerImpl) List() ([]*storagev1.CSINode, error) {
	return l.csinodeLister.List(labels.Everything())
}

func (l *CSINodeListerImpl) Get(name string) (*storagev1.CSINode, error) {
	return l.csinodeLister.Get(name)
}
