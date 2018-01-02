package convert

import (
	swarmtypes "github.com/docker/docker/api/types/swarm"
	swarmapi "github.com/docker/swarmkit/api"
	gogotypes "github.com/gogo/protobuf/types"
)

// SecretFromGRPC converts a grpc Secret to a Secret.
func SecretFromGRPC(s *swarmapi.Secret) swarmtypes.Secret {
	secret := swarmtypes.Secret{
		ID: s.ID,
		Spec: swarmtypes.SecretSpec{
			Annotations: annotationsFromGRPC(s.Spec.Annotations),
			Data:        s.Spec.Data,
			Driver:      driverFromGRPC(s.Spec.Driver),
		},
	}

	secret.Version.Index = s.Meta.Version.Index
	// Meta
	secret.CreatedAt, _ = gogotypes.TimestampFromProto(s.Meta.CreatedAt)
	secret.UpdatedAt, _ = gogotypes.TimestampFromProto(s.Meta.UpdatedAt)

	return secret
}

// SecretSpecToGRPC converts Secret to a grpc Secret.
func SecretSpecToGRPC(s swarmtypes.SecretSpec) swarmapi.SecretSpec {
	return swarmapi.SecretSpec{
		Annotations: swarmapi.Annotations{
			Name:   s.Name,
			Labels: s.Labels,
		},
		Data:   s.Data,
		Driver: driverToGRPC(s.Driver),
	}
}

// SecretReferencesFromGRPC converts a slice of grpc SecretReference to SecretReference
func SecretReferencesFromGRPC(s []*swarmapi.SecretReference) []*swarmtypes.SecretReference {
	refs := []*swarmtypes.SecretReference{}

	for _, r := range s {
		ref := &swarmtypes.SecretReference{
			SecretID:   r.SecretID,
			SecretName: r.SecretName,
		}

		if t, ok := r.Target.(*swarmapi.SecretReference_File); ok {
			ref.File = &swarmtypes.SecretReferenceFileTarget{
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
