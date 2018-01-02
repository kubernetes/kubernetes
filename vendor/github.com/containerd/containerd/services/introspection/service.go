package introspection

import (
	api "github.com/containerd/containerd/api/services/introspection/v1"
	"github.com/containerd/containerd/api/types"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/filters"
	"github.com/containerd/containerd/plugin"
	"github.com/containerd/containerd/protobuf/google/rpc"
	ptypes "github.com/gogo/protobuf/types"
	context "golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/status"
)

func init() {
	plugin.Register(&plugin.Registration{
		Type:     plugin.GRPCPlugin,
		ID:       "introspection",
		Requires: []plugin.Type{"*"},
		InitFn: func(ic *plugin.InitContext) (interface{}, error) {
			// this service works by using the plugin context up till the point
			// this service is initialized. Since we require this service last,
			// it should provide the full set of plugins.
			pluginsPB := pluginsToPB(ic.GetAll())
			return NewService(pluginsPB), nil
		},
	})
}

type service struct {
	plugins []api.Plugin
}

// NewService returns the GRPC introspection server
func NewService(plugins []api.Plugin) api.IntrospectionServer {
	return &service{
		plugins: plugins,
	}
}

func (s *service) Register(server *grpc.Server) error {
	api.RegisterIntrospectionServer(server, s)
	return nil
}

func (s *service) Plugins(ctx context.Context, req *api.PluginsRequest) (*api.PluginsResponse, error) {
	filter, err := filters.ParseAll(req.Filters...)
	if err != nil {
		return nil, errdefs.ToGRPCf(errdefs.ErrInvalidArgument, err.Error())
	}

	var plugins []api.Plugin
	for _, p := range s.plugins {
		if !filter.Match(adaptPlugin(p)) {
			continue
		}

		plugins = append(plugins, p)
	}

	return &api.PluginsResponse{
		Plugins: plugins,
	}, nil
}

func adaptPlugin(o interface{}) filters.Adaptor {
	obj := o.(api.Plugin)
	return filters.AdapterFunc(func(fieldpath []string) (string, bool) {
		if len(fieldpath) == 0 {
			return "", false
		}

		switch fieldpath[0] {
		case "type":
			return string(obj.Type), len(obj.Type) > 0
		case "id":
			return string(obj.ID), len(obj.ID) > 0
		case "platforms":
			// TODO(stevvooe): Another case here where have multiple values.
			// May need to refactor the filter system to allow filtering by
			// platform, if this is required.
		case "capabilities":
			// TODO(stevvooe): Need a better way to match against
			// collections. We can only return "the value" but really it
			// would be best if we could return a set of values for the
			// path, any of which could match.
		}

		return "", false
	})
}

func pluginsToPB(plugins []*plugin.Plugin) []api.Plugin {
	var pluginsPB []api.Plugin
	for _, p := range plugins {
		var platforms []types.Platform
		for _, p := range p.Meta.Platforms {
			platforms = append(platforms, types.Platform{
				OS:           p.OS,
				Architecture: p.Architecture,
				Variant:      p.Variant,
			})
		}

		var requires []string
		for _, r := range p.Registration.Requires {
			requires = append(requires, r.String())
		}

		var initErr *rpc.Status
		if err := p.Err(); err != nil {
			st, ok := status.FromError(errdefs.ToGRPC(err))
			if ok {
				var details []*ptypes.Any
				for _, d := range st.Proto().Details {
					details = append(details, &ptypes.Any{
						TypeUrl: d.TypeUrl,
						Value:   d.Value,
					})
				}
				initErr = &rpc.Status{
					Code:    int32(st.Code()),
					Message: st.Message(),
					Details: details,
				}
			} else {
				initErr = &rpc.Status{
					Code:    int32(rpc.Code_UNKNOWN),
					Message: err.Error(),
				}
			}
		}

		pluginsPB = append(pluginsPB, api.Plugin{
			Type:         p.Registration.Type.String(),
			ID:           p.Registration.ID,
			Requires:     requires,
			Platforms:    platforms,
			Capabilities: p.Meta.Capabilities,
			Exports:      p.Meta.Exports,
			InitErr:      initErr,
		})
	}

	return pluginsPB
}
