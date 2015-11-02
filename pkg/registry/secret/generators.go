package secret

import "k8s.io/kubernetes/pkg/api"

type Generator interface {
  GenerateValues() (map[string][]byte, error)
}

type Generators map[api.SecretType]Generator
