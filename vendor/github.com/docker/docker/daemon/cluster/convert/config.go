package convert

import (
	swarmtypes "github.com/docker/docker/api/types/swarm"
	swarmapi "github.com/docker/swarmkit/api"
	gogotypes "github.com/gogo/protobuf/types"
)

// ConfigFromGRPC converts a grpc Config to a Config.
func ConfigFromGRPC(s *swarmapi.Config) swarmtypes.Config {
	config := swarmtypes.Config{
		ID: s.ID,
		Spec: swarmtypes.ConfigSpec{
			Annotations: annotationsFromGRPC(s.Spec.Annotations),
			Data:        s.Spec.Data,
		},
	}

	config.Version.Index = s.Meta.Version.Index
	// Meta
	config.CreatedAt, _ = gogotypes.TimestampFromProto(s.Meta.CreatedAt)
	config.UpdatedAt, _ = gogotypes.TimestampFromProto(s.Meta.UpdatedAt)

	return config
}

// ConfigSpecToGRPC converts Config to a grpc Config.
func ConfigSpecToGRPC(s swarmtypes.ConfigSpec) swarmapi.ConfigSpec {
	return swarmapi.ConfigSpec{
		Annotations: swarmapi.Annotations{
			Name:   s.Name,
			Labels: s.Labels,
		},
		Data: s.Data,
	}
}

// ConfigReferencesFromGRPC converts a slice of grpc ConfigReference to ConfigReference
func ConfigReferencesFromGRPC(s []*swarmapi.ConfigReference) []*swarmtypes.ConfigReference {
	refs := []*swarmtypes.ConfigReference{}

	for _, r := range s {
		ref := &swarmtypes.ConfigReference{
			ConfigID:   r.ConfigID,
			ConfigName: r.ConfigName,
		}

		if t, ok := r.Target.(*swarmapi.ConfigReference_File); ok {
			ref.File = &swarmtypes.ConfigReferenceFileTarget{
				Name: t.File.Name,
				UID:  t.File.UID,
				GID:  t.File.GID,
				Mode: t.File.Mode,
			}
		}

		refs = append(refs, ref)
	}

	return refs
}
