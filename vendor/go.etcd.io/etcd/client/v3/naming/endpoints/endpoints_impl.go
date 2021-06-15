package endpoints

// TODO: The API is not yet implemented.

import (
	"context"
	"encoding/json"
	"errors"
	"strings"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/naming/endpoints/internal"

	"go.uber.org/zap"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

type endpointManager struct {
	// Client is an initialized etcd client.
	client *clientv3.Client
	target string
}

// NewManager creates an endpoint manager which implements the interface of 'Manager'.
func NewManager(client *clientv3.Client, target string) (Manager, error) {
	if client == nil {
		return nil, errors.New("invalid etcd client")
	}

	if target == "" {
		return nil, errors.New("invalid target")
	}

	em := &endpointManager{
		client: client,
		target: target,
	}
	return em, nil
}

func (m *endpointManager) Update(ctx context.Context, updates []*UpdateWithOpts) (err error) {
	ops := make([]clientv3.Op, 0, len(updates))
	for _, update := range updates {
		if !strings.HasPrefix(update.Key, m.target+"/") {
			return status.Errorf(codes.InvalidArgument, "endpoints: endpoint key should be prefixed with '%s/' got: '%s'", m.target, update.Key)
		}

		switch update.Op {
		case Add:
			internalUpdate := &internal.Update{
				Op:       internal.Add,
				Addr:     update.Endpoint.Addr,
				Metadata: update.Endpoint.Metadata,
			}

			var v []byte
			if v, err = json.Marshal(internalUpdate); err != nil {
				return status.Error(codes.InvalidArgument, err.Error())
			}
			ops = append(ops, clientv3.OpPut(update.Key, string(v), update.Opts...))
		case Delete:
			ops = append(ops, clientv3.OpDelete(update.Key, update.Opts...))
		default:
			return status.Error(codes.InvalidArgument, "endpoints: bad update op")
		}
	}
	_, err = m.client.KV.Txn(ctx).Then(ops...).Commit()
	return err
}

func (m *endpointManager) AddEndpoint(ctx context.Context, key string, endpoint Endpoint, opts ...clientv3.OpOption) error {
	return m.Update(ctx, []*UpdateWithOpts{NewAddUpdateOpts(key, endpoint, opts...)})
}

func (m *endpointManager) DeleteEndpoint(ctx context.Context, key string, opts ...clientv3.OpOption) error {
	return m.Update(ctx, []*UpdateWithOpts{NewDeleteUpdateOpts(key, opts...)})
}

func (m *endpointManager) NewWatchChannel(ctx context.Context) (WatchChannel, error) {
	resp, err := m.client.Get(ctx, m.target, clientv3.WithPrefix(), clientv3.WithSerializable())
	if err != nil {
		return nil, err
	}

	lg := m.client.GetLogger()
	initUpdates := make([]*Update, 0, len(resp.Kvs))
	for _, kv := range resp.Kvs {
		var iup internal.Update
		if err := json.Unmarshal(kv.Value, &iup); err != nil {
			lg.Warn("unmarshal endpoint update failed", zap.String("key", string(kv.Key)), zap.Error(err))
			continue
		}
		up := &Update{
			Op:       Add,
			Key:      string(kv.Key),
			Endpoint: Endpoint{Addr: iup.Addr, Metadata: iup.Metadata},
		}
		initUpdates = append(initUpdates, up)
	}

	upch := make(chan []*Update, 1)
	if len(initUpdates) > 0 {
		upch <- initUpdates
	}
	go m.watch(ctx, resp.Header.Revision+1, upch)
	return upch, nil
}

func (m *endpointManager) watch(ctx context.Context, rev int64, upch chan []*Update) {
	defer close(upch)

	lg := m.client.GetLogger()
	opts := []clientv3.OpOption{clientv3.WithRev(rev), clientv3.WithPrefix()}
	wch := m.client.Watch(ctx, m.target, opts...)
	for {
		select {
		case <-ctx.Done():
			return
		case wresp, ok := <-wch:
			if !ok {
				lg.Warn("watch closed", zap.String("target", m.target))
				return
			}
			if wresp.Err() != nil {
				lg.Warn("watch failed", zap.String("target", m.target), zap.Error(wresp.Err()))
				return
			}

			deltaUps := make([]*Update, 0, len(wresp.Events))
			for _, e := range wresp.Events {
				var iup internal.Update
				var err error
				var op Operation
				switch e.Type {
				case clientv3.EventTypePut:
					err = json.Unmarshal(e.Kv.Value, &iup)
					op = Add
					if err != nil {
						lg.Warn("unmarshal endpoint update failed", zap.String("key", string(e.Kv.Key)), zap.Error(err))
						continue
					}
				case clientv3.EventTypeDelete:
					iup = internal.Update{Op: internal.Delete}
					op = Delete
				default:
					continue
				}
				up := &Update{Op: op, Key: string(e.Kv.Key), Endpoint: Endpoint{Addr: iup.Addr, Metadata: iup.Metadata}}
				deltaUps = append(deltaUps, up)
			}
			if len(deltaUps) > 0 {
				upch <- deltaUps
			}
		}
	}
}

func (m *endpointManager) List(ctx context.Context) (Key2EndpointMap, error) {
	resp, err := m.client.Get(ctx, m.target, clientv3.WithPrefix(), clientv3.WithSerializable())
	if err != nil {
		return nil, err
	}

	eps := make(Key2EndpointMap)
	for _, kv := range resp.Kvs {
		var iup internal.Update
		if err := json.Unmarshal(kv.Value, &iup); err != nil {
			continue
		}

		eps[string(kv.Key)] = Endpoint{Addr: iup.Addr, Metadata: iup.Metadata}
	}
	return eps, nil
}
