package clientv3

import (
	"context"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/api/v3/mvccpb"
)

func NewKubernetes(c *Client) Kubernetes {
	return &kubernetes{kv: RetryKVClient(c)}
}

type Kubernetes interface {
	Get(ctx context.Context, key string, opts GetOptions) (KubernetesGetResponse, error)
	List(ctx context.Context, prefix string, opts ListOptions) (KubernetesListResponse, error)
	Count(ctx context.Context, prefix string) (int64, error)
	OptimisticPut(ctx context.Context, key string, value []byte, opts PutOptions) (KubernetesPutResponse, error)
	OptimisticDelete(ctx context.Context, key string, opts DeleteOptions) (KubernetesDeleteResponse, error)
}

type GetOptions struct {
	Revision int64
}

type ListOptions struct {
	Revision int64
	Limit    int64
	Continue string
}

type PutOptions struct {
	ExpectedRevision int64
	GetOnFailure     bool
	// Deprecated: Should be replaced with TTL when Kubernetes starts using one lease per object.
	LeaseID LeaseID
}

type DeleteOptions struct {
	ExpectedRevision int64
	GetOnFailure     bool
}

type KubernetesGetResponse struct {
	KV       *mvccpb.KeyValue
	Revision int64
}

type KubernetesListResponse struct {
	KVs      []*mvccpb.KeyValue
	Count    int64
	Revision int64
}

type KubernetesPutResponse struct {
	KV        *mvccpb.KeyValue
	Succeeded bool
	Revision  int64
}

type KubernetesDeleteResponse struct {
	KV        *mvccpb.KeyValue
	Succeeded bool
	Revision  int64
}

type kubernetes struct {
	kv pb.KVClient
}

func (k kubernetes) Get(ctx context.Context, key string, opts GetOptions) (resp KubernetesGetResponse, err error) {
	rangeResp, err := k.kv.Range(ctx, getRequest(key, opts.Revision))
	if err != nil {
		return resp, toErr(ctx, err)
	}
	resp.Revision = rangeResp.Header.Revision
	if len(rangeResp.Kvs) == 1 {
		resp.KV = rangeResp.Kvs[0]
	}
	return resp, nil
}

func (k kubernetes) List(ctx context.Context, prefix string, opts ListOptions) (resp KubernetesListResponse, err error) {
	rangeStart := prefix + opts.Continue
	rangeEnd := GetPrefixRangeEnd(prefix)

	rangeResp, err := k.kv.Range(ctx, &pb.RangeRequest{
		Key:      []byte(rangeStart),
		RangeEnd: []byte(rangeEnd),
		Limit:    opts.Limit,
		Revision: opts.Revision,
	})
	if err != nil {
		return resp, toErr(ctx, err)
	}
	resp.KVs = rangeResp.Kvs
	resp.Count = rangeResp.Count
	resp.Revision = rangeResp.Header.Revision
	return resp, nil
}

func (k kubernetes) Count(ctx context.Context, prefix string) (int64, error) {
	resp, err := k.kv.Range(ctx, &pb.RangeRequest{
		Key:       []byte(prefix),
		RangeEnd:  []byte(GetPrefixRangeEnd(prefix)),
		CountOnly: true,
	})
	if err != nil {
		return 0, toErr(ctx, err)
	}
	return resp.Count, nil
}

func (k kubernetes) OptimisticPut(ctx context.Context, key string, value []byte, opts PutOptions) (resp KubernetesPutResponse, err error) {
	onSuccess := &pb.RequestOp{Request: &pb.RequestOp_RequestPut{RequestPut: &pb.PutRequest{Key: []byte(key), Value: value, Lease: int64(opts.LeaseID)}}}

	var onFailure *pb.RequestOp
	if opts.GetOnFailure {
		onFailure = &pb.RequestOp{Request: &pb.RequestOp_RequestRange{RequestRange: getRequest(key, 0)}}
	}

	txnResp, err := k.optimisticTxn(ctx, key, opts.ExpectedRevision, onSuccess, onFailure)
	if err != nil {
		return resp, toErr(ctx, err)
	}
	resp.Succeeded = txnResp.Succeeded
	resp.Revision = txnResp.Header.Revision
	if opts.GetOnFailure && !txnResp.Succeeded {
		resp.KV = kvFromTxnResponse(txnResp.Responses[0])
	}
	return resp, nil
}

func (k kubernetes) OptimisticDelete(ctx context.Context, key string, opts DeleteOptions) (resp KubernetesDeleteResponse, err error) {
	onSuccess := &pb.RequestOp{Request: &pb.RequestOp_RequestDeleteRange{RequestDeleteRange: &pb.DeleteRangeRequest{Key: []byte(key)}}}

	var onFailure *pb.RequestOp
	if opts.GetOnFailure {
		onFailure = &pb.RequestOp{Request: &pb.RequestOp_RequestRange{RequestRange: getRequest(key, 0)}}
	}

	txnResp, err := k.optimisticTxn(ctx, key, opts.ExpectedRevision, onSuccess, onFailure)
	if err != nil {
		return resp, toErr(ctx, err)
	}
	resp.Succeeded = txnResp.Succeeded
	resp.Revision = txnResp.Header.Revision
	if opts.GetOnFailure && !txnResp.Succeeded {
		resp.KV = kvFromTxnResponse(txnResp.Responses[0])
	}
	return resp, nil
}

func (k kubernetes) optimisticTxn(ctx context.Context, key string, expectRevision int64, onSuccess, onFailure *pb.RequestOp) (*pb.TxnResponse, error) {
	txn := &pb.TxnRequest{
		Compare: []*pb.Compare{
			{
				Result:      pb.Compare_EQUAL,
				Target:      pb.Compare_MOD,
				Key:         []byte(key),
				TargetUnion: &pb.Compare_ModRevision{ModRevision: expectRevision},
			},
		},
	}
	if onSuccess != nil {
		txn.Success = []*pb.RequestOp{onSuccess}
	}
	if onFailure != nil {
		txn.Failure = []*pb.RequestOp{onFailure}
	}
	return k.kv.Txn(ctx, txn)
}

func getRequest(key string, revision int64) *pb.RangeRequest {
	return &pb.RangeRequest{
		Key:      []byte(key),
		Revision: revision,
		Limit:    1,
	}
}

func kvFromTxnResponse(resp *pb.ResponseOp) *mvccpb.KeyValue {
	getResponse := resp.GetResponseRange()
	if len(getResponse.Kvs) == 1 {
		return getResponse.Kvs[0]
	}
	return nil
}
