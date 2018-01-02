package flags

import (
	"context"
	"flag"
	"fmt"
	"os"

	"github.com/vmware/govmomi/object"
)

type StoragePodFlag struct {
	common

	*DatacenterFlag

	Name string

	sp *object.StoragePod
}

var storagePodFlagKey = flagKey("storagePod")

func NewStoragePodFlag(ctx context.Context) (*StoragePodFlag, context.Context) {
	if v := ctx.Value(storagePodFlagKey); v != nil {
		return v.(*StoragePodFlag), ctx
	}

	v := &StoragePodFlag{}
	v.DatacenterFlag, ctx = NewDatacenterFlag(ctx)
	ctx = context.WithValue(ctx, storagePodFlagKey, v)
	return v, ctx
}

func (f *StoragePodFlag) Register(ctx context.Context, fs *flag.FlagSet) {
	f.RegisterOnce(func() {
		f.DatacenterFlag.Register(ctx, fs)

		env := "GOVC_DATASTORE_CLUSTER"
		value := os.Getenv(env)
		usage := fmt.Sprintf("Datastore cluster [%s]", env)
		fs.StringVar(&f.Name, "datastore-cluster", value, usage)
	})
}

func (f *StoragePodFlag) Process(ctx context.Context) error {
	return f.DatacenterFlag.Process(ctx)
}

func (f *StoragePodFlag) Isset() bool {
	return f.Name != ""
}

func (f *StoragePodFlag) StoragePod() (*object.StoragePod, error) {
	ctx := context.TODO()
	if f.sp != nil {
		return f.sp, nil
	}

	finder, err := f.Finder()
	if err != nil {
		return nil, err
	}

	if f.Isset() {
		f.sp, err = finder.DatastoreCluster(ctx, f.Name)
		if err != nil {
			return nil, err
		}
	} else {
		f.sp, err = finder.DefaultDatastoreCluster(ctx)
		if err != nil {
			return nil, err
		}
	}

	return f.sp, nil
}
