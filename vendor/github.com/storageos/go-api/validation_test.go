package storageos

import "testing"

func TestIsUUID(t *testing.T) {
	var tests = []struct {
		id       string
		expected bool
	}{
		{"c61ff64f-c5cd-a21f-d50a-5bac48098182", true},
		{"pvc-88453d04-9f8b-3eee-8eed-77ae396ff344", false}, // Used be Kubernetes as name
		{"vol1", false},
		{"volume/c61ff64f-c5cd-a21f-d50a-5bac48098182", false},
		{"1111-11111111-1111-1111-1111-111111111111", false},
	}

	for _, tt := range tests {
		actual := IsUUID(tt.id)
		if actual != tt.expected {
			t.Errorf("IsUUID(%s): expected %t, actual %t", tt.id, tt.expected, actual)
		}
	}
}

func TestIsName(t *testing.T) {
	var tests = []struct {
		name     string
		expected bool
	}{
		{"abc123", true},
		{"pvc-88453d04-9f8b-3eee-8eed-77ae396ff344", true}, // Used be Kubernetes as name
		{"c61ff64f-c5cd-a21f-d50a-5bac48098182", true},
		{"namespace/name", false},
		{"_vol1", false},
	}

	for _, tt := range tests {
		actual := IsName(tt.name)
		if actual != tt.expected {
			t.Errorf("IsName(%s): expected %t, actual %t", tt.name, tt.expected, actual)
		}
	}
}

func TestValidateNamespace(t *testing.T) {
	var tests = []struct {
		namespace string
		expected  error
	}{
		{"abc123", nil},
		{"pvc-88453d04-9f8b-3eee-8eed-77ae396ff344", nil}, // Used be Kubernetes as name
		{"c61ff64f-c5cd-a21f-d50a-5bac48098182", nil},
		{"namespace/name", ErrNoNamespace},
		{"_vol1", ErrNoNamespace},
	}

	for _, tt := range tests {
		actual := ValidateNamespace(tt.namespace)
		if actual != tt.expected {
			t.Errorf("ValidateNamespace(%s): expected %v, actual %v", tt.namespace, tt.expected, actual)
		}
	}
}

func TestValidateNamespaceAndRef(t *testing.T) {
	var tests = []struct {
		namespace string
		ref       string
		expected  error
	}{
		{"abc123", "abc123", nil},
		{"abc123", "pvc-88453d04-9f8b-3eee-8eed-77ae396ff344", nil}, // Used be Kubernetes as name
		{"abc123", "c61ff64f-c5cd-a21f-d50a-5bac48098182", nil},
		{"abc123", "namespace/name", ErrNoRef},
		{"abc123", "_vol1", ErrNoRef},
		{"abc123", "_vol1", ErrNoRef},
		{"_vol1", "vol1", ErrNoNamespace},
		{"namespace/name", "vol1", ErrNoNamespace},
	}

	for _, tt := range tests {
		actual := ValidateNamespaceAndRef(tt.namespace, tt.ref)
		if actual != tt.expected {
			t.Errorf("ValidateNamespaceAndRef(%s, %s): expected %v, actual %v", tt.namespace, tt.ref, tt.expected, actual)
		}
	}
}

func TestNamespacedPath(t *testing.T) {
	var tests = []struct {
		namespace    string
		objectType   string
		expectedPath string
		expectedErr  error
	}{
		{"abc123", "xyz", "/namespaces/abc123/xyz", nil},
		{"_abc123", "xyz", "", ErrNoNamespace},
		{"namespace/name", "vol1", "", ErrNoNamespace},
	}

	for _, tt := range tests {
		actual, err := namespacedPath(tt.namespace, tt.objectType)
		if err != tt.expectedErr {
			t.Errorf("namespacedPath(%s, %s): expected %v, actual %v", tt.namespace, tt.objectType, tt.expectedErr, err)
		}
		if actual != tt.expectedPath {
			t.Errorf("namespacedPath(%s, %s): expected %s, actual %s", tt.namespace, tt.objectType, tt.expectedPath, actual)
		}
	}
}

func TestNamespacedRefPath(t *testing.T) {
	var tests = []struct {
		namespace    string
		objectType   string
		ref          string
		expectedPath string
		expectedErr  error
	}{
		{"abc123", "xyz", "vol123", "/namespaces/abc123/xyz/vol123", nil},
		{"abc123", "xyz", "pvc-88453d04-9f8b-3eee-8eed-77ae396ff344", "/namespaces/abc123/xyz/pvc-88453d04-9f8b-3eee-8eed-77ae396ff344", nil}, // Used be Kubernetes as name
		{"abc123", "xyz", "c61ff64f-c5cd-a21f-d50a-5bac48098182", "/namespaces/abc123/xyz/c61ff64f-c5cd-a21f-d50a-5bac48098182", nil},
		{"abc123", "xyz", "namespace/name", "", ErrNoRef},
		{"abc123", "xyz", "_vol1", "", ErrNoRef},
		{"abc123", "xyz", "_vol1", "", ErrNoRef},
		{"_abc123", "xyz", "vol1", "", ErrNoNamespace},
		{"namespace/name", "xyz", "vol1", "", ErrNoNamespace},
	}

	for _, tt := range tests {
		actual, err := namespacedRefPath(tt.namespace, tt.objectType, tt.ref)
		if err != tt.expectedErr {
			t.Errorf("namespacedRefPath(%s, %s, %s): expected %v, actual %v", tt.namespace, tt.objectType, tt.ref, tt.expectedErr, err)
		}
		if actual != tt.expectedPath {
			t.Errorf("namespacedRefPath(%s, %s, %s): expected %s, actual %s", tt.namespace, tt.objectType, tt.ref, tt.expectedPath, actual)
		}
	}
}
