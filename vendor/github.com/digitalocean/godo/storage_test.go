package godo

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"
	"time"
)

func TestStorageVolumes_ListStorageVolumes(t *testing.T) {
	setup()
	defer teardown()

	jBlob := `
	{
		"volumes": [
			{
				"user_id": 42,
				"region": {"slug": "nyc3"},
				"id": "80d414c6-295e-4e3a-ac58-eb9456c1e1d1",
				"name": "my volume",
				"description": "my description",
				"size_gigabytes": 100,
				"droplet_ids": [10],
				"created_at": "2002-10-02T15:00:00.05Z"
			},
			{
				"user_id": 42,
				"region": {"slug": "nyc3"},
				"id": "96d414c6-295e-4e3a-ac59-eb9456c1e1d1",
				"name": "my other volume",
				"description": "my other description",
				"size_gigabytes": 100,
				"created_at": "2012-10-03T15:00:01.05Z"
			}
		],
		"links": {
	    "pages": {
	      "last": "https://api.digitalocean.com/v2/volumes?page=2",
	      "next": "https://api.digitalocean.com/v2/volumes?page=2"
	    }
	  },
	  "meta": {
	    "total": 28
	  }
	}`

	mux.HandleFunc("/v2/volumes/", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, jBlob)
	})

	volumes, _, err := client.Storage.ListVolumes(nil)
	if err != nil {
		t.Errorf("Storage.ListVolumes returned error: %v", err)
	}

	expected := []Volume{
		{
			Region:        &Region{Slug: "nyc3"},
			ID:            "80d414c6-295e-4e3a-ac58-eb9456c1e1d1",
			Name:          "my volume",
			Description:   "my description",
			SizeGigaBytes: 100,
			DropletIDs:    []int{10},
			CreatedAt:     time.Date(2002, 10, 02, 15, 00, 00, 50000000, time.UTC),
		},
		{
			Region:        &Region{Slug: "nyc3"},
			ID:            "96d414c6-295e-4e3a-ac59-eb9456c1e1d1",
			Name:          "my other volume",
			Description:   "my other description",
			SizeGigaBytes: 100,
			CreatedAt:     time.Date(2012, 10, 03, 15, 00, 01, 50000000, time.UTC),
		},
	}
	if !reflect.DeepEqual(volumes, expected) {
		t.Errorf("Storage.ListVolumes returned %+v, expected %+v", volumes, expected)
	}
}

func TestStorageVolumes_Get(t *testing.T) {
	setup()
	defer teardown()
	want := &Volume{
		Region:        &Region{Slug: "nyc3"},
		ID:            "80d414c6-295e-4e3a-ac58-eb9456c1e1d1",
		Name:          "my volume",
		Description:   "my description",
		SizeGigaBytes: 100,
		CreatedAt:     time.Date(2002, 10, 02, 15, 00, 00, 50000000, time.UTC),
	}
	jBlob := `{
		"volume":{
			"region": {"slug":"nyc3"},
			"attached_to_droplet": null,
			"id": "80d414c6-295e-4e3a-ac58-eb9456c1e1d1",
			"name": "my volume",
			"description": "my description",
			"size_gigabytes": 100,
			"created_at": "2002-10-02T15:00:00.05Z"
		},
		"links": {
	    "pages": {
	      "last": "https://api.digitalocean.com/v2/volumes?page=2",
	      "next": "https://api.digitalocean.com/v2/volumes?page=2"
	    }
	  },
	  "meta": {
	    "total": 28
	  }
	}`

	mux.HandleFunc("/v2/volumes/80d414c6-295e-4e3a-ac58-eb9456c1e1d1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, jBlob)
	})

	got, _, err := client.Storage.GetVolume("80d414c6-295e-4e3a-ac58-eb9456c1e1d1")
	if err != nil {
		t.Errorf("Storage.GetVolume returned error: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Storage.GetVolume returned %+v, want %+v", got, want)
	}
}

func TestStorageVolumes_Create(t *testing.T) {
	setup()
	defer teardown()

	createRequest := &VolumeCreateRequest{
		Region:        "nyc3",
		Name:          "my volume",
		Description:   "my description",
		SizeGigaBytes: 100,
	}

	want := &Volume{
		Region:        &Region{Slug: "nyc3"},
		ID:            "80d414c6-295e-4e3a-ac58-eb9456c1e1d1",
		Name:          "my volume",
		Description:   "my description",
		SizeGigaBytes: 100,
		CreatedAt:     time.Date(2002, 10, 02, 15, 00, 00, 50000000, time.UTC),
	}
	jBlob := `{
		"volume":{
			"region": {"slug":"nyc3"},
			"id": "80d414c6-295e-4e3a-ac58-eb9456c1e1d1",
			"name": "my volume",
			"description": "my description",
			"size_gigabytes": 100,
			"created_at": "2002-10-02T15:00:00.05Z"
		},
		"links": {}
	}`

	mux.HandleFunc("/v2/volumes", func(w http.ResponseWriter, r *http.Request) {
		v := new(VolumeCreateRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatal(err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, createRequest) {
			t.Errorf("Request body = %+v, expected %+v", v, createRequest)
		}

		fmt.Fprint(w, jBlob)
	})

	got, _, err := client.Storage.CreateVolume(createRequest)
	if err != nil {
		t.Errorf("Storage.CreateVolume returned error: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Storage.CreateVolume returned %+v, want %+v", got, want)
	}
}

func TestStorageVolumes_Destroy(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/volumes/80d414c6-295e-4e3a-ac58-eb9456c1e1d1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Storage.DeleteVolume("80d414c6-295e-4e3a-ac58-eb9456c1e1d1")
	if err != nil {
		t.Errorf("Storage.DeleteVolume returned error: %v", err)
	}
}

func TestStorageSnapshots_ListStorageSnapshots(t *testing.T) {
	setup()
	defer teardown()

	jBlob := `
	{
		"snapshots": [
			{
				"region": {"slug": "nyc3"},
				"id": "80d414c6-295e-4e3a-ac58-eb9456c1e1d1",
				"volume_id": "98d414c6-295e-4e3a-ac58-eb9456c1e1d1",
				"name": "my snapshot",
				"description": "my description",
				"size_gigabytes": 100,
				"created_at": "2002-10-02T15:00:00.05Z"
			},
			{
				"region": {"slug": "nyc3"},
				"id": "96d414c6-295e-4e3a-ac59-eb9456c1e1d1",
				"volume_id": "98d414c6-295e-4e3a-ac58-eb9456c1e1d1",
				"name": "my other snapshot",
				"description": "my other description",
				"size_gigabytes": 100,
				"created_at": "2012-10-03T15:00:01.05Z"
			}
		],
		"links": {
	    "pages": {
	      "last": "https://api.digitalocean.com/v2/volumes?page=2",
	      "next": "https://api.digitalocean.com/v2/volumes?page=2"
	    }
	  },
	  "meta": {
	    "total": 28
	  }
	}`

	mux.HandleFunc("/v2/volumes/98d414c6-295e-4e3a-ac58-eb9456c1e1d1/snapshots", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, jBlob)
	})

	volumes, _, err := client.Storage.(BetaStorageService).ListSnapshots("98d414c6-295e-4e3a-ac58-eb9456c1e1d1", nil)
	if err != nil {
		t.Errorf("Storage.ListSnapshots returned error: %v", err)
	}

	expected := []Snapshot{
		{
			Region:        &Region{Slug: "nyc3"},
			ID:            "80d414c6-295e-4e3a-ac58-eb9456c1e1d1",
			VolumeID:      "98d414c6-295e-4e3a-ac58-eb9456c1e1d1",
			Name:          "my snapshot",
			Description:   "my description",
			SizeGigaBytes: 100,
			CreatedAt:     time.Date(2002, 10, 02, 15, 00, 00, 50000000, time.UTC),
		},
		{
			Region:        &Region{Slug: "nyc3"},
			ID:            "96d414c6-295e-4e3a-ac59-eb9456c1e1d1",
			VolumeID:      "98d414c6-295e-4e3a-ac58-eb9456c1e1d1",
			Name:          "my other snapshot",
			Description:   "my other description",
			SizeGigaBytes: 100,
			CreatedAt:     time.Date(2012, 10, 03, 15, 00, 01, 50000000, time.UTC),
		},
	}
	if !reflect.DeepEqual(volumes, expected) {
		t.Errorf("Storage.ListSnapshots returned %+v, expected %+v", volumes, expected)
	}
}

func TestStorageSnapshots_Get(t *testing.T) {
	setup()
	defer teardown()
	want := &Snapshot{
		Region:        &Region{Slug: "nyc3"},
		ID:            "80d414c6-295e-4e3a-ac58-eb9456c1e1d1",
		VolumeID:      "98d414c6-295e-4e3a-ac58-eb9456c1e1d1",
		Name:          "my snapshot",
		Description:   "my description",
		SizeGigaBytes: 100,
		CreatedAt:     time.Date(2002, 10, 02, 15, 00, 00, 50000000, time.UTC),
	}
	jBlob := `{
		"snapshot":{
			"region": {"slug": "nyc3"},
			"id": "80d414c6-295e-4e3a-ac58-eb9456c1e1d1",
			"volume_id": "98d414c6-295e-4e3a-ac58-eb9456c1e1d1",
			"name": "my snapshot",
			"description": "my description",
			"size_gigabytes": 100,
			"created_at": "2002-10-02T15:00:00.05Z"
		},
		"links": {
	    "pages": {
				"last": "https://api.digitalocean.com/v2/volumes/98d414c6-295e-4e3a-ac58-eb9456c1e1d1/snapshots?page=2",
				"next": "https://api.digitalocean.com/v2/volumes/98d414c6-295e-4e3a-ac58-eb9456c1e1d1/snapshots?page=2"
	    }
	  },
	  "meta": {
	    "total": 28
	  }
	}`

	mux.HandleFunc("/v2/snapshots/80d414c6-295e-4e3a-ac58-eb9456c1e1d1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprint(w, jBlob)
	})

	got, _, err := client.Storage.(BetaStorageService).GetSnapshot("80d414c6-295e-4e3a-ac58-eb9456c1e1d1")
	if err != nil {
		t.Errorf("Storage.GetSnapshot returned error: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Storage.GetSnapshot returned %+v, want %+v", got, want)
	}
}

func TestStorageSnapshots_Create(t *testing.T) {
	setup()
	defer teardown()

	createRequest := &SnapshotCreateRequest{
		VolumeID:    "98d414c6-295e-4e3a-ac58-eb9456c1e1d1",
		Name:        "my snapshot",
		Description: "my description",
	}

	want := &Snapshot{
		Region:        &Region{Slug: "nyc3"},
		ID:            "80d414c6-295e-4e3a-ac58-eb9456c1e1d1",
		VolumeID:      "98d414c6-295e-4e3a-ac58-eb9456c1e1d1",
		Name:          "my snapshot",
		Description:   "my description",
		SizeGigaBytes: 100,
		CreatedAt:     time.Date(2002, 10, 02, 15, 00, 00, 50000000, time.UTC),
	}
	jBlob := `{
		"snapshot":{
			"region": {"slug": "nyc3"},
			"id": "80d414c6-295e-4e3a-ac58-eb9456c1e1d1",
			"volume_id": "98d414c6-295e-4e3a-ac58-eb9456c1e1d1",
			"name": "my snapshot",
			"description": "my description",
			"size_gigabytes": 100,
			"created_at": "2002-10-02T15:00:00.05Z"
		},
		"links": {
	    "pages": {
	      "last": "https://api.digitalocean.com/v2/volumes/98d414c6-295e-4e3a-ac58-eb9456c1e1d1/snapshots?page=2",
	      "next": "https://api.digitalocean.com/v2/volumes/98d414c6-295e-4e3a-ac58-eb9456c1e1d1/snapshots?page=2"
	    }
	  },
	  "meta": {
	    "total": 28
	  }
	}`

	mux.HandleFunc("/v2/volumes/98d414c6-295e-4e3a-ac58-eb9456c1e1d1/snapshots", func(w http.ResponseWriter, r *http.Request) {
		v := new(SnapshotCreateRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatal(err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, createRequest) {
			t.Errorf("Request body = %+v, expected %+v", v, createRequest)
		}

		fmt.Fprint(w, jBlob)
	})

	got, _, err := client.Storage.(BetaStorageService).CreateSnapshot(createRequest)
	if err != nil {
		t.Errorf("Storage.CreateSnapshot returned error: %v", err)
	}
	if !reflect.DeepEqual(got, want) {
		t.Errorf("Storage.CreateSnapshot returned %+v, want %+v", got, want)
	}
}

func TestStorageSnapshots_Destroy(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/snapshots/80d414c6-295e-4e3a-ac58-eb9456c1e1d1", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "DELETE")
	})

	_, err := client.Storage.(BetaStorageService).DeleteSnapshot("80d414c6-295e-4e3a-ac58-eb9456c1e1d1")
	if err != nil {
		t.Errorf("Storage.DeleteSnapshot returned error: %v", err)
	}
}
