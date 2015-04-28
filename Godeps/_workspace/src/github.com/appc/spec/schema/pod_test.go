package schema

import (
	"testing"

	"github.com/appc/spec/schema/types"
)

func TestPodManifestMerge(t *testing.T) {
	pmj := `{}`
	pm := &PodManifest{}

	if pm.UnmarshalJSON([]byte(pmj)) == nil {
		t.Fatal("Manifest JSON without acKind and acVersion unmarshalled successfully")
	}

	pm = BlankPodManifest()

	err := pm.UnmarshalJSON([]byte(pmj))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestAppList(t *testing.T) {
	ri := RuntimeImage{
		ID: *types.NewHashSHA512([]byte{}),
	}
	al := AppList{
		RuntimeApp{
			Name:  "foo",
			Image: ri,
		},
		RuntimeApp{
			Name:  "bar",
			Image: ri,
		},
	}
	if _, err := al.MarshalJSON(); err != nil {
		t.Errorf("want err=nil, got %v", err)
	}
	dal := AppList{
		RuntimeApp{
			Name:  "foo",
			Image: ri,
		},
		RuntimeApp{
			Name:  "bar",
			Image: ri,
		},
		RuntimeApp{
			Name:  "foo",
			Image: ri,
		},
	}
	if _, err := dal.MarshalJSON(); err == nil {
		t.Errorf("want err, got nil")
	}
}
