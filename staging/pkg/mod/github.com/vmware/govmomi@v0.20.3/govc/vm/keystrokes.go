/*
Copyright (c) 2014-2016 VMware, Inc. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

	http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package vm

import (
	"context"
	"flag"
	"fmt"
	"strconv"

	"github.com/vmware/govmomi/govc/cli"
	"github.com/vmware/govmomi/govc/flags"
	"github.com/vmware/govmomi/object"
	"github.com/vmware/govmomi/vim25/types"
)

type hidKey struct {
	Code         int32
	ShiftPressed bool
}

var hidCharacterMap = map[string]hidKey{
	"a": {0x04, false},
	"b": {0x05, false},
	"c": {0x06, false},
	"d": {0x07, false},
	"e": {0x08, false},
	"f": {0x09, false},
	"g": {0x0a, false},
	"h": {0x0b, false},
	"i": {0x0c, false},
	"j": {0x0d, false},
	"k": {0x0e, false},
	"l": {0x0f, false},
	"m": {0x10, false},
	"n": {0x11, false},
	"o": {0x12, false},
	"p": {0x13, false},
	"q": {0x14, false},
	"r": {0x15, false},
	"s": {0x16, false},
	"t": {0x17, false},
	"u": {0x18, false},
	"v": {0x19, false},
	"w": {0x1a, false},
	"x": {0x1b, false},
	"y": {0x1c, false},
	"z": {0x1d, false},
	"1": {0x1e, false},
	"2": {0x1f, false},
	"3": {0x20, false},
	"4": {0x21, false},
	"5": {0x22, false},
	"6": {0x23, false},
	"7": {0x24, false},
	"8": {0x25, false},
	"9": {0x26, false},
	"0": {0x27, false},
	"A": {0x04, true},
	"B": {0x05, true},
	"C": {0x06, true},
	"D": {0x07, true},
	"E": {0x08, true},
	"F": {0x09, true},
	"G": {0x0a, true},
	"H": {0x0b, true},
	"I": {0x0c, true},
	"J": {0x0d, true},
	"K": {0x0e, true},
	"L": {0x0f, true},
	"M": {0x10, true},
	"N": {0x11, true},
	"O": {0x12, true},
	"P": {0x13, true},
	"Q": {0x14, true},
	"R": {0x15, true},
	"S": {0x16, true},
	"T": {0x17, true},
	"U": {0x18, true},
	"V": {0x19, true},
	"W": {0x1a, true},
	"X": {0x1b, true},
	"Y": {0x1c, true},
	"Z": {0x1d, true},
	"!": {0x1e, true},
	"@": {0x1f, true},
	"#": {0x20, true},
	"$": {0x21, true},
	"%": {0x22, true},
	"^": {0x23, true},
	"&": {0x24, true},
	"*": {0x25, true},
	"(": {0x26, true},
	")": {0x27, true},
	"-": {0x2d, false},
	"_": {0x2d, true},
	"=": {0x2e, false},
	"+": {0x2e, true},
	"[": {0x2f, false},
	"{": {0x2f, true},
	"]": {0x30, false},
	"}": {0x30, true},
	`\`: {0x31, false},
	"|": {0x31, true},
	";": {0x33, false},
	":": {0x33, true},
	"'": {0x34, false},
	`"`: {0x34, true},
	"`": {0x35, false},
	"~": {0x35, true},
	",": {0x36, false},
	"<": {0x36, true},
	".": {0x37, false},
	">": {0x37, true},
	"/": {0x38, false},
	"?": {0x38, true},
}

type keystrokes struct {
	*flags.VirtualMachineFlag

	UsbHidCodeValue int32
	UsbHidCode      string
	UsbHidString    string
	LeftControl     bool
	LeftShift       bool
	LeftAlt         bool
	LeftGui         bool
	RightControl    bool
	RightShift      bool
	RightAlt        bool
	RightGui        bool
}

func init() {
	cli.Register("vm.keystrokes", &keystrokes{})
}

func (cmd *keystrokes) Register(ctx context.Context, f *flag.FlagSet) {
	cmd.VirtualMachineFlag, ctx = flags.NewVirtualMachineFlag(ctx)
	cmd.VirtualMachineFlag.Register(ctx, f)

	f.StringVar(&cmd.UsbHidString, "s", "", "Raw String to Send")
	f.StringVar(&cmd.UsbHidCode, "c", "", "USB HID Code (hex)")
	f.Var(flags.NewInt32(&cmd.UsbHidCodeValue), "r", "Raw USB HID Code Value (int32)")
	f.BoolVar(&cmd.LeftControl, "lc", false, "Enable/Disable Left Control")
	f.BoolVar(&cmd.LeftShift, "ls", false, "Enable/Disable Left Shift")
	f.BoolVar(&cmd.LeftAlt, "la", false, "Enable/Disable Left Alt")
	f.BoolVar(&cmd.LeftGui, "lg", false, "Enable/Disable Left Gui")
	f.BoolVar(&cmd.RightControl, "rc", false, "Enable/Disable Right Control")
	f.BoolVar(&cmd.RightShift, "rs", false, "Enable/Disable Right Shift")
	f.BoolVar(&cmd.RightAlt, "ra", false, "Enable/Disable Right Alt")
	f.BoolVar(&cmd.RightGui, "rg", false, "Enable/Disable Right Gui")
}

func (cmd *keystrokes) Usage() string {
	return "VM"
}

func (cmd *keystrokes) Description() string {
	return `Send Keystrokes to VM.

Examples:
 Default Scenario
  govc vm.keystrokes -vm $vm -s "root" 	# writes 'root' to the console
  govc vm.keystrokes -vm $vm -c 0x15 	# writes an 'r' to the console
  govc vm.keystrokes -vm $vm -r 1376263 # writes an 'r' to the console
  govc vm.keystrokes -vm $vm -c 0x28 	# presses ENTER on the console
  govc vm.keystrokes -vm $vm -c 0x4c -la true -lc true 	# sends CTRL+ALT+DEL to console`
}

func (cmd *keystrokes) Process(ctx context.Context) error {
	if err := cmd.VirtualMachineFlag.Process(ctx); err != nil {
		return err
	}
	return nil
}

func (cmd *keystrokes) Run(ctx context.Context, f *flag.FlagSet) error {
	vm, err := cmd.VirtualMachine()
	if err != nil {
		return err
	}

	if vm == nil {
		return flag.ErrHelp
	}

	err = cmd.processUserInput(ctx, vm)
	if err != nil {
		return err
	}
	return nil
}

func (cmd *keystrokes) processUserInput(ctx context.Context, vm *object.VirtualMachine) error {
	if err := cmd.checkValidInputs(); err != nil {
		return err
	}

	codes, err := cmd.processUsbCode()

	if err != nil {
		return err
	}

	var keyEventArray []types.UsbScanCodeSpecKeyEvent
	for _, code := range codes {
		leftShiftSetting := false
		if code.ShiftPressed || cmd.LeftShift {
			leftShiftSetting = true
		}
		modifiers := types.UsbScanCodeSpecModifierType{
			LeftControl:  &cmd.LeftControl,
			LeftShift:    &leftShiftSetting,
			LeftAlt:      &cmd.LeftAlt,
			LeftGui:      &cmd.LeftGui,
			RightControl: &cmd.RightControl,
			RightShift:   &cmd.RightShift,
			RightAlt:     &cmd.RightAlt,
			RightGui:     &cmd.RightGui,
		}
		keyEvent := types.UsbScanCodeSpecKeyEvent{
			UsbHidCode: code.Code,
			Modifiers:  &modifiers,
		}
		keyEventArray = append(keyEventArray, keyEvent)
	}

	spec := types.UsbScanCodeSpec{
		KeyEvents: keyEventArray,
	}

	_, err = vm.PutUsbScanCodes(ctx, spec)

	return err
}

func (cmd *keystrokes) processUsbCode() ([]hidKey, error) {
	if cmd.rawCodeProvided() {
		return []hidKey{{cmd.UsbHidCodeValue, false}}, nil
	}
	if cmd.hexCodeProvided() {
		s, err := hexStringToHidCode(cmd.UsbHidCode)
		if err != nil {
			return nil, err
		}
		return []hidKey{{s, false}}, nil
	}

	if cmd.stringProvided() {
		var retKeyArray []hidKey
		for _, c := range cmd.UsbHidString {
			lookupValue, ok := hidCharacterMap[string(c)]
			if !ok {
				return nil, fmt.Errorf("Invalid Character %s in String: %s", string(c), cmd.UsbHidString)
			}
			lookupValue.Code = intToHidCode(lookupValue.Code)
			retKeyArray = append(retKeyArray, lookupValue)
		}
		return retKeyArray, nil
	}
	return nil, nil
}

func hexStringToHidCode(hex string) (int32, error) {
	s, err := strconv.ParseInt(hex, 0, 32)
	if err != nil {
		return 0, err
	}
	return intToHidCode(int32(s)), nil
}

func intToHidCode(v int32) int32 {
	var s int32
	s = v << 16
	s = s | 7
	return s
}

func (cmd *keystrokes) checkValidInputs() error {
	// poor man's boolean XOR -> A xor B xor C = A'BC' + AB'C' + A'B'C + ABC
	if (!cmd.rawCodeProvided() && cmd.hexCodeProvided() && !cmd.stringProvided()) || // A'BC'
		(cmd.rawCodeProvided() && !cmd.hexCodeProvided() && !cmd.stringProvided()) || // AB'C'
		(!cmd.rawCodeProvided() && !cmd.hexCodeProvided() && cmd.stringProvided()) || // A'B'C
		(cmd.rawCodeProvided() && cmd.hexCodeProvided() && cmd.stringProvided()) { // ABC
		return nil
	}
	return fmt.Errorf("Specify only 1 argument")
}

func (cmd keystrokes) rawCodeProvided() bool {
	return cmd.UsbHidCodeValue != 0
}

func (cmd keystrokes) hexCodeProvided() bool {
	return cmd.UsbHidCode != ""
}

func (cmd keystrokes) stringProvided() bool {
	return cmd.UsbHidString != ""
}
