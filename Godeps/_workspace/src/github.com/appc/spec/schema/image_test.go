package schema

import "testing"

func TestEmptyApp(t *testing.T) {
	imj := `
		{
		    "acKind": "ImageManifest",
		    "acVersion": "0.5.1",
		    "name": "example.com/test"
		}
		`

	var im ImageManifest

	err := im.UnmarshalJSON([]byte(imj))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Marshal and Unmarshal to verify that no "app": {} is generated on
	// Marshal and converted to empty struct on Unmarshal
	buf, err := im.MarshalJSON()
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	err = im.UnmarshalJSON(buf)
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}

func TestImageManifestMerge(t *testing.T) {
	imj := `{"name": "example.com/test"}`
	im := &ImageManifest{}

	if im.UnmarshalJSON([]byte(imj)) == nil {
		t.Fatal("Manifest JSON without acKind and acVersion unmarshalled successfully")
	}

	im = BlankImageManifest()

	err := im.UnmarshalJSON([]byte(imj))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
}
