package godo

import (
	"encoding/json"
	"fmt"
	"net/http"
	"reflect"
	"testing"
)

func TestDropletActions_Shutdown(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "shutdown",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.Shutdown(1)
	if err != nil {
		t.Errorf("DropletActions.Shutdown returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.Shutdown returned %+v, expected %+v", action, expected)
	}
}

func TestDropletActions_ShutdownByTag(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "shutdown",
	}

	mux.HandleFunc("/v2/droplets/actions", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("tag_name") != "testing-1" {
			t.Errorf("DropletActions.ShutdownByTag did not request with a tag parameter")
		}

		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.ShutdownByTag("testing-1")
	if err != nil {
		t.Errorf("DropletActions.ShutdownByTag returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.ShutdownByTag returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_PowerOff(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "power_off",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.PowerOff(1)
	if err != nil {
		t.Errorf("DropletActions.PowerOff returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.Poweroff returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_PowerOffByTag(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "power_off",
	}

	mux.HandleFunc("/v2/droplets/actions", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("tag_name") != "testing-1" {
			t.Errorf("DropletActions.PowerOffByTag did not request with a tag parameter")
		}

		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.PowerOffByTag("testing-1")
	if err != nil {
		t.Errorf("DropletActions.PowerOffByTag returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.PoweroffByTag returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_PowerOn(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "power_on",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.PowerOn(1)
	if err != nil {
		t.Errorf("DropletActions.PowerOn returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.PowerOn returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_PowerOnByTag(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "power_on",
	}

	mux.HandleFunc("/v2/droplets/actions", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("tag_name") != "testing-1" {
			t.Errorf("DropletActions.PowerOnByTag did not request with a tag parameter")
		}

		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.PowerOnByTag("testing-1")
	if err != nil {
		t.Errorf("DropletActions.PowerOnByTag returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.PowerOnByTag returned %+v, expected %+v", action, expected)
	}
}
func TestDropletAction_Reboot(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "reboot",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)

	})

	action, _, err := client.DropletActions.Reboot(1)
	if err != nil {
		t.Errorf("DropletActions.Reboot returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.Reboot returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_Restore(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type":  "restore",
		"image": float64(1),
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)

	})

	action, _, err := client.DropletActions.Restore(1, 1)
	if err != nil {
		t.Errorf("DropletActions.Restore returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.Restore returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_Resize(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "resize",
		"size": "1024mb",
		"disk": true,
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)

	})

	action, _, err := client.DropletActions.Resize(1, "1024mb", true)
	if err != nil {
		t.Errorf("DropletActions.Resize returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.Resize returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_Rename(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "rename",
		"name": "Droplet-Name",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.Rename(1, "Droplet-Name")
	if err != nil {
		t.Errorf("DropletActions.Rename returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.Rename returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_PowerCycle(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "power_cycle",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)

	})

	action, _, err := client.DropletActions.PowerCycle(1)
	if err != nil {
		t.Errorf("DropletActions.PowerCycle returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.PowerCycle returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_PowerCycleByTag(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "power_cycle",
	}

	mux.HandleFunc("/v2/droplets/actions", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("tag_name") != "testing-1" {
			t.Errorf("DropletActions.PowerCycleByTag did not request with a tag parameter")
		}

		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")
		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)

	})

	action, _, err := client.DropletActions.PowerCycleByTag("testing-1")
	if err != nil {
		t.Errorf("DropletActions.PowerCycleByTag returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.PowerCycleByTag returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_Snapshot(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "snapshot",
		"name": "Image-Name",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.Snapshot(1, "Image-Name")
	if err != nil {
		t.Errorf("DropletActions.Snapshot returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.Snapshot returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_SnapshotByTag(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "snapshot",
		"name": "Image-Name",
	}

	mux.HandleFunc("/v2/droplets/actions", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("tag_name") != "testing-1" {
			t.Errorf("DropletActions.SnapshotByTag did not request with a tag parameter")
		}

		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.SnapshotByTag("testing-1", "Image-Name")
	if err != nil {
		t.Errorf("DropletActions.SnapshotByTag returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.SnapshotByTag returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_EnableBackups(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "enable_backups",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.EnableBackups(1)
	if err != nil {
		t.Errorf("DropletActions.EnableBackups returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.EnableBackups returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_EnableBackupsByTag(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "enable_backups",
	}

	mux.HandleFunc("/v2/droplets/actions", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("tag_name") != "testing-1" {
			t.Errorf("DropletActions.EnableBackupByTag did not request with a tag parameter")
		}

		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.EnableBackupsByTag("testing-1")
	if err != nil {
		t.Errorf("DropletActions.EnableBackupsByTag returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.EnableBackupsByTag returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_DisableBackups(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "disable_backups",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.DisableBackups(1)
	if err != nil {
		t.Errorf("DropletActions.DisableBackups returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.DisableBackups returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_DisableBackupsByTag(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "disable_backups",
	}

	mux.HandleFunc("/v2/droplets/actions", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("tag_name") != "testing-1" {
			t.Errorf("DropletActions.DisableBackupsByTag did not request with a tag parameter")
		}

		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.DisableBackupsByTag("testing-1")
	if err != nil {
		t.Errorf("DropletActions.DisableBackupsByTag returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.DisableBackupsByTag returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_PasswordReset(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "password_reset",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.PasswordReset(1)
	if err != nil {
		t.Errorf("DropletActions.PasswordReset returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.PasswordReset returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_RebuildByImageID(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type":  "rebuild",
		"image": float64(2),
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = \n%#v, expected \n%#v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.RebuildByImageID(1, 2)
	if err != nil {
		t.Errorf("DropletActions.RebuildByImageID returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.RebuildByImageID returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_RebuildByImageSlug(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type":  "rebuild",
		"image": "Image-Name",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.RebuildByImageSlug(1, "Image-Name")
	if err != nil {
		t.Errorf("DropletActions.RebuildByImageSlug returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.RebuildByImageSlug returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_ChangeKernel(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type":   "change_kernel",
		"kernel": float64(2),
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.ChangeKernel(1, 2)
	if err != nil {
		t.Errorf("DropletActions.ChangeKernel returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.ChangeKernel returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_EnableIPv6(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "enable_ipv6",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.EnableIPv6(1)
	if err != nil {
		t.Errorf("DropletActions.EnableIPv6 returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.EnableIPv6 returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_EnableIPv6ByTag(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "enable_ipv6",
	}

	mux.HandleFunc("/v2/droplets/actions", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("tag_name") != "testing-1" {
			t.Errorf("DropletActions.EnableIPv6ByTag did not request with a tag parameter")
		}

		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.EnableIPv6ByTag("testing-1")
	if err != nil {
		t.Errorf("DropletActions.EnableIPv6ByTag returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.EnableIPv6byTag returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_EnablePrivateNetworking(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "enable_private_networking",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.EnablePrivateNetworking(1)
	if err != nil {
		t.Errorf("DropletActions.EnablePrivateNetworking returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.EnablePrivateNetworking returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_EnablePrivateNetworkingByTag(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "enable_private_networking",
	}

	mux.HandleFunc("/v2/droplets/actions", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Query().Get("tag_name") != "testing-1" {
			t.Errorf("DropletActions.EnablePrivateNetworkingByTag did not request with a tag parameter")
		}

		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.EnablePrivateNetworkingByTag("testing-1")
	if err != nil {
		t.Errorf("DropletActions.EnablePrivateNetworkingByTag returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.EnablePrivateNetworkingByTag returned %+v, expected %+v", action, expected)
	}
}

func TestDropletAction_Upgrade(t *testing.T) {
	setup()
	defer teardown()

	request := &ActionRequest{
		"type": "upgrade",
	}

	mux.HandleFunc("/v2/droplets/1/actions", func(w http.ResponseWriter, r *http.Request) {
		v := new(ActionRequest)
		err := json.NewDecoder(r.Body).Decode(v)
		if err != nil {
			t.Fatalf("decode json: %v", err)
		}

		testMethod(t, r, "POST")

		if !reflect.DeepEqual(v, request) {
			t.Errorf("Request body = %+v, expected %+v", v, request)
		}

		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.Upgrade(1)
	if err != nil {
		t.Errorf("DropletActions.Upgrade returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.Upgrade returned %+v, expected %+v", action, expected)
	}
}

func TestDropletActions_Get(t *testing.T) {
	setup()
	defer teardown()

	mux.HandleFunc("/v2/droplets/123/actions/456", func(w http.ResponseWriter, r *http.Request) {
		testMethod(t, r, "GET")
		fmt.Fprintf(w, `{"action":{"status":"in-progress"}}`)
	})

	action, _, err := client.DropletActions.Get(123, 456)
	if err != nil {
		t.Errorf("DropletActions.Get returned error: %v", err)
	}

	expected := &Action{Status: "in-progress"}
	if !reflect.DeepEqual(action, expected) {
		t.Errorf("DropletActions.Get returned %+v, expected %+v", action, expected)
	}
}
