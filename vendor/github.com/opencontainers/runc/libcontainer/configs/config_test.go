package configs_test

import (
	"encoding/json"
	"fmt"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/opencontainers/runc/libcontainer/configs"
)

func TestUnmarshalHooks(t *testing.T) {
	timeout := time.Second

	prestartCmd := configs.NewCommandHook(configs.Command{
		Path:    "/var/vcap/hooks/prestart",
		Args:    []string{"--pid=123"},
		Env:     []string{"FOO=BAR"},
		Dir:     "/var/vcap",
		Timeout: &timeout,
	})
	prestart, err := json.Marshal(prestartCmd.Command)
	if err != nil {
		t.Fatal(err)
	}

	hook := configs.Hooks{}
	err = hook.UnmarshalJSON([]byte(fmt.Sprintf(`{"Prestart" :[%s]}`, prestart)))
	if err != nil {
		t.Fatal(err)
	}

	if !reflect.DeepEqual(hook.Prestart[0], prestartCmd) {
		t.Errorf("Expected prestart to equal %+v but it was %+v",
			prestartCmd, hook.Prestart[0])
	}
}

func TestUnmarshalHooksWithInvalidData(t *testing.T) {
	hook := configs.Hooks{}
	err := hook.UnmarshalJSON([]byte(`{invalid-json}`))
	if err == nil {
		t.Error("Expected error to occur but it was nil")
	}
}

func TestMarshalHooks(t *testing.T) {
	timeout := time.Second

	prestartCmd := configs.NewCommandHook(configs.Command{
		Path:    "/var/vcap/hooks/prestart",
		Args:    []string{"--pid=123"},
		Env:     []string{"FOO=BAR"},
		Dir:     "/var/vcap",
		Timeout: &timeout,
	})

	hook := configs.Hooks{
		Prestart: []configs.Hook{prestartCmd},
	}
	hooks, err := hook.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	h := `{"poststart":null,"poststop":null,"prestart":[{"path":"/var/vcap/hooks/prestart","args":["--pid=123"],"env":["FOO=BAR"],"dir":"/var/vcap","timeout":1000000000}]}`
	if string(hooks) != h {
		t.Errorf("Expected hooks %s to equal %s", string(hooks), h)
	}
}

func TestMarshalUnmarshalHooks(t *testing.T) {
	timeout := time.Second

	prestart := configs.NewCommandHook(configs.Command{
		Path:    "/var/vcap/hooks/prestart",
		Args:    []string{"--pid=123"},
		Env:     []string{"FOO=BAR"},
		Dir:     "/var/vcap",
		Timeout: &timeout,
	})

	hook := configs.Hooks{
		Prestart: []configs.Hook{prestart},
	}
	hooks, err := hook.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	umMhook := configs.Hooks{}
	err = umMhook.UnmarshalJSON(hooks)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(umMhook.Prestart[0], prestart) {
		t.Errorf("Expected hooks to be equal after mashaling -> unmarshaling them: %+v, %+v", umMhook.Prestart[0], prestart)
	}
}

func TestMarshalHooksWithUnexpectedType(t *testing.T) {
	fHook := configs.NewFunctionHook(func(configs.HookState) error {
		return nil
	})
	hook := configs.Hooks{
		Prestart: []configs.Hook{fHook},
	}
	hooks, err := hook.MarshalJSON()
	if err != nil {
		t.Fatal(err)
	}

	h := `{"poststart":null,"poststop":null,"prestart":null}`
	if string(hooks) != h {
		t.Errorf("Expected hooks %s to equal %s", string(hooks), h)
	}
}

func TestFuncHookRun(t *testing.T) {
	state := configs.HookState{
		Version: "1",
		ID:      "1",
		Pid:     1,
		Bundle:  "/bundle",
	}

	fHook := configs.NewFunctionHook(func(s configs.HookState) error {
		if !reflect.DeepEqual(state, s) {
			t.Errorf("Expected state %+v to equal %+v", state, s)
		}
		return nil
	})

	fHook.Run(state)
}

func TestCommandHookRun(t *testing.T) {
	state := configs.HookState{
		Version: "1",
		ID:      "1",
		Pid:     1,
		Bundle:  "/bundle",
	}
	timeout := time.Second

	cmdHook := configs.NewCommandHook(configs.Command{
		Path:    os.Args[0],
		Args:    []string{os.Args[0], "-test.run=TestHelperProcess"},
		Env:     []string{"FOO=BAR"},
		Dir:     "/",
		Timeout: &timeout,
	})

	err := cmdHook.Run(state)
	if err != nil {
		t.Errorf(fmt.Sprintf("Expected error to not occur but it was %+v", err))
	}
}

func TestCommandHookRunTimeout(t *testing.T) {
	state := configs.HookState{
		Version: "1",
		ID:      "1",
		Pid:     1,
		Bundle:  "/bundle",
	}
	timeout := (10 * time.Millisecond)

	cmdHook := configs.NewCommandHook(configs.Command{
		Path:    os.Args[0],
		Args:    []string{os.Args[0], "-test.run=TestHelperProcessWithTimeout"},
		Env:     []string{"FOO=BAR"},
		Dir:     "/",
		Timeout: &timeout,
	})

	err := cmdHook.Run(state)
	if err == nil {
		t.Error("Expected error to occur but it was nil")
	}
}

func TestHelperProcess(*testing.T) {
	fmt.Println("Helper Process")
	os.Exit(0)
}
func TestHelperProcessWithTimeout(*testing.T) {
	time.Sleep(time.Second)
}
