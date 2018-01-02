package namespaces

import (
	"context"
	"strings"

	api "github.com/containerd/containerd/api/services/namespaces/v1"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/namespaces"
	"github.com/gogo/protobuf/types"
)

// NewStoreFromClient returns a new namespace store
func NewStoreFromClient(client api.NamespacesClient) namespaces.Store {
	return &remote{client: client}
}

type remote struct {
	client api.NamespacesClient
}

func (r *remote) Create(ctx context.Context, namespace string, labels map[string]string) error {
	var req api.CreateNamespaceRequest

	req.Namespace = api.Namespace{
		Name:   namespace,
		Labels: labels,
	}

	_, err := r.client.Create(ctx, &req)
	if err != nil {
		return errdefs.FromGRPC(err)
	}

	return nil
}

func (r *remote) Labels(ctx context.Context, namespace string) (map[string]string, error) {
	var req api.GetNamespaceRequest
	req.Name = namespace

	resp, err := r.client.Get(ctx, &req)
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}

	return resp.Namespace.Labels, nil
}

func (r *remote) SetLabel(ctx context.Context, namespace, key, value string) error {
	var req api.UpdateNamespaceRequest

	req.Namespace = api.Namespace{
		Name:   namespace,
		Labels: map[string]string{key: value},
	}

	req.UpdateMask = &types.FieldMask{
		Paths: []string{strings.Join([]string{"labels", key}, ".")},
	}

	_, err := r.client.Update(ctx, &req)
	if err != nil {
		return errdefs.FromGRPC(err)
	}

	return nil
}

func (r *remote) List(ctx context.Context) ([]string, error) {
	var req api.ListNamespacesRequest

	resp, err := r.client.List(ctx, &req)
	if err != nil {
		return nil, errdefs.FromGRPC(err)
	}

	var namespaces []string

	for _, ns := range resp.Namespaces {
		namespaces = append(namespaces, ns.Name)
	}

	return namespaces, nil
}

func (r *remote) Delete(ctx context.Context, namespace string) error {
	var req api.DeleteNamespaceRequest

	req.Name = namespace
	_, err := r.client.Delete(ctx, &req)
	if err != nil {
		return errdefs.FromGRPC(err)
	}

	return nil
}
