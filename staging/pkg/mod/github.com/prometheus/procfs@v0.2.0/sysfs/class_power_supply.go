// Copyright 2018 The Prometheus Authors
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// +build !windows

package sysfs

import (
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	"github.com/prometheus/procfs/internal/util"
)

// PowerSupply contains info from files in /sys/class/power_supply for a
// single power supply.
type PowerSupply struct {
	Name                     string // Power Supply Name
	Authentic                *int64 // /sys/class/power_supply/<Name>/authentic
	Calibrate                *int64 // /sys/class/power_supply/<Name>/calibrate
	Capacity                 *int64 // /sys/class/power_supply/<Name>/capacity
	CapacityAlertMax         *int64 // /sys/class/power_supply/<Name>/capacity_alert_max
	CapacityAlertMin         *int64 // /sys/class/power_supply/<Name>/capacity_alert_min
	CapacityLevel            string // /sys/class/power_supply/<Name>/capacity_level
	ChargeAvg                *int64 // /sys/class/power_supply/<Name>/charge_avg
	ChargeControlLimit       *int64 // /sys/class/power_supply/<Name>/charge_control_limit
	ChargeControlLimitMax    *int64 // /sys/class/power_supply/<Name>/charge_control_limit_max
	ChargeCounter            *int64 // /sys/class/power_supply/<Name>/charge_counter
	ChargeEmpty              *int64 // /sys/class/power_supply/<Name>/charge_empty
	ChargeEmptyDesign        *int64 // /sys/class/power_supply/<Name>/charge_empty_design
	ChargeFull               *int64 // /sys/class/power_supply/<Name>/charge_full
	ChargeFullDesign         *int64 // /sys/class/power_supply/<Name>/charge_full_design
	ChargeNow                *int64 // /sys/class/power_supply/<Name>/charge_now
	ChargeTermCurrent        *int64 // /sys/class/power_supply/<Name>/charge_term_current
	ChargeType               string // /sys/class/power_supply/<Name>/charge_type
	ConstantChargeCurrent    *int64 // /sys/class/power_supply/<Name>/constant_charge_current
	ConstantChargeCurrentMax *int64 // /sys/class/power_supply/<Name>/constant_charge_current_max
	ConstantChargeVoltage    *int64 // /sys/class/power_supply/<Name>/constant_charge_voltage
	ConstantChargeVoltageMax *int64 // /sys/class/power_supply/<Name>/constant_charge_voltage_max
	CurrentAvg               *int64 // /sys/class/power_supply/<Name>/current_avg
	CurrentBoot              *int64 // /sys/class/power_supply/<Name>/current_boot
	CurrentMax               *int64 // /sys/class/power_supply/<Name>/current_max
	CurrentNow               *int64 // /sys/class/power_supply/<Name>/current_now
	CycleCount               *int64 // /sys/class/power_supply/<Name>/cycle_count
	EnergyAvg                *int64 // /sys/class/power_supply/<Name>/energy_avg
	EnergyEmpty              *int64 // /sys/class/power_supply/<Name>/energy_empty
	EnergyEmptyDesign        *int64 // /sys/class/power_supply/<Name>/energy_empty_design
	EnergyFull               *int64 // /sys/class/power_supply/<Name>/energy_full
	EnergyFullDesign         *int64 // /sys/class/power_supply/<Name>/energy_full_design
	EnergyNow                *int64 // /sys/class/power_supply/<Name>/energy_now
	Health                   string // /sys/class/power_supply/<Name>/health
	InputCurrentLimit        *int64 // /sys/class/power_supply/<Name>/input_current_limit
	Manufacturer             string // /sys/class/power_supply/<Name>/manufacturer
	ModelName                string // /sys/class/power_supply/<Name>/model_name
	Online                   *int64 // /sys/class/power_supply/<Name>/online
	PowerAvg                 *int64 // /sys/class/power_supply/<Name>/power_avg
	PowerNow                 *int64 // /sys/class/power_supply/<Name>/power_now
	PrechargeCurrent         *int64 // /sys/class/power_supply/<Name>/precharge_current
	Present                  *int64 // /sys/class/power_supply/<Name>/present
	Scope                    string // /sys/class/power_supply/<Name>/scope
	SerialNumber             string // /sys/class/power_supply/<Name>/serial_number
	Status                   string // /sys/class/power_supply/<Name>/status
	Technology               string // /sys/class/power_supply/<Name>/technology
	Temp                     *int64 // /sys/class/power_supply/<Name>/temp
	TempAlertMax             *int64 // /sys/class/power_supply/<Name>/temp_alert_max
	TempAlertMin             *int64 // /sys/class/power_supply/<Name>/temp_alert_min
	TempAmbient              *int64 // /sys/class/power_supply/<Name>/temp_ambient
	TempAmbientMax           *int64 // /sys/class/power_supply/<Name>/temp_ambient_max
	TempAmbientMin           *int64 // /sys/class/power_supply/<Name>/temp_ambient_min
	TempMax                  *int64 // /sys/class/power_supply/<Name>/temp_max
	TempMin                  *int64 // /sys/class/power_supply/<Name>/temp_min
	TimeToEmptyAvg           *int64 // /sys/class/power_supply/<Name>/time_to_empty_avg
	TimeToEmptyNow           *int64 // /sys/class/power_supply/<Name>/time_to_empty_now
	TimeToFullAvg            *int64 // /sys/class/power_supply/<Name>/time_to_full_avg
	TimeToFullNow            *int64 // /sys/class/power_supply/<Name>/time_to_full_now
	Type                     string // /sys/class/power_supply/<Name>/type
	UsbType                  string // /sys/class/power_supply/<Name>/usb_type
	VoltageAvg               *int64 // /sys/class/power_supply/<Name>/voltage_avg
	VoltageBoot              *int64 // /sys/class/power_supply/<Name>/voltage_boot
	VoltageMax               *int64 // /sys/class/power_supply/<Name>/voltage_max
	VoltageMaxDesign         *int64 // /sys/class/power_supply/<Name>/voltage_max_design
	VoltageMin               *int64 // /sys/class/power_supply/<Name>/voltage_min
	VoltageMinDesign         *int64 // /sys/class/power_supply/<Name>/voltage_min_design
	VoltageNow               *int64 // /sys/class/power_supply/<Name>/voltage_now
	VoltageOCV               *int64 // /sys/class/power_supply/<Name>/voltage_ocv
}

// PowerSupplyClass is a collection of every power supply in
// /sys/class/power_supply.
//
// The map keys are the names of the power supplies.
type PowerSupplyClass map[string]PowerSupply

// PowerSupplyClass returns info for all power supplies read from
// /sys/class/power_supply.
func (fs FS) PowerSupplyClass() (PowerSupplyClass, error) {
	path := fs.sys.Path("class/power_supply")

	dirs, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}

	psc := make(PowerSupplyClass, len(dirs))
	for _, d := range dirs {
		ps, err := parsePowerSupply(filepath.Join(path, d.Name()))
		if err != nil {
			return nil, err
		}

		ps.Name = d.Name()
		psc[d.Name()] = *ps
	}

	return psc, nil
}

func parsePowerSupply(path string) (*PowerSupply, error) {
	files, err := ioutil.ReadDir(path)
	if err != nil {
		return nil, err
	}

	var ps PowerSupply
	for _, f := range files {
		if !f.Mode().IsRegular() {
			continue
		}

		name := filepath.Join(path, f.Name())
		value, err := util.SysReadFile(name)
		if err != nil {
			if os.IsNotExist(err) || err.Error() == "operation not supported" || err.Error() == "invalid argument" {
				continue
			}
			return nil, fmt.Errorf("failed to read file %q: %v", name, err)
		}

		vp := util.NewValueParser(value)

		switch f.Name() {
		case "authentic":
			ps.Authentic = vp.PInt64()
		case "calibrate":
			ps.Calibrate = vp.PInt64()
		case "capacity":
			ps.Capacity = vp.PInt64()
		case "capacity_alert_max":
			ps.CapacityAlertMax = vp.PInt64()
		case "capacity_alert_min":
			ps.CapacityAlertMin = vp.PInt64()
		case "capacity_level":
			ps.CapacityLevel = value
		case "charge_avg":
			ps.ChargeAvg = vp.PInt64()
		case "charge_control_limit":
			ps.ChargeControlLimit = vp.PInt64()
		case "charge_control_limit_max":
			ps.ChargeControlLimitMax = vp.PInt64()
		case "charge_counter":
			ps.ChargeCounter = vp.PInt64()
		case "charge_empty":
			ps.ChargeEmpty = vp.PInt64()
		case "charge_empty_design":
			ps.ChargeEmptyDesign = vp.PInt64()
		case "charge_full":
			ps.ChargeFull = vp.PInt64()
		case "charge_full_design":
			ps.ChargeFullDesign = vp.PInt64()
		case "charge_now":
			ps.ChargeNow = vp.PInt64()
		case "charge_term_current":
			ps.ChargeTermCurrent = vp.PInt64()
		case "charge_type":
			ps.ChargeType = value
		case "constant_charge_current":
			ps.ConstantChargeCurrent = vp.PInt64()
		case "constant_charge_current_max":
			ps.ConstantChargeCurrentMax = vp.PInt64()
		case "constant_charge_voltage":
			ps.ConstantChargeVoltage = vp.PInt64()
		case "constant_charge_voltage_max":
			ps.ConstantChargeVoltageMax = vp.PInt64()
		case "current_avg":
			ps.CurrentAvg = vp.PInt64()
		case "current_boot":
			ps.CurrentBoot = vp.PInt64()
		case "current_max":
			ps.CurrentMax = vp.PInt64()
		case "current_now":
			ps.CurrentNow = vp.PInt64()
		case "cycle_count":
			ps.CycleCount = vp.PInt64()
		case "energy_avg":
			ps.EnergyAvg = vp.PInt64()
		case "energy_empty":
			ps.EnergyEmpty = vp.PInt64()
		case "energy_empty_design":
			ps.EnergyEmptyDesign = vp.PInt64()
		case "energy_full":
			ps.EnergyFull = vp.PInt64()
		case "energy_full_design":
			ps.EnergyFullDesign = vp.PInt64()
		case "energy_now":
			ps.EnergyNow = vp.PInt64()
		case "health":
			ps.Health = value
		case "input_current_limit":
			ps.InputCurrentLimit = vp.PInt64()
		case "manufacturer":
			ps.Manufacturer = value
		case "model_name":
			ps.ModelName = value
		case "online":
			ps.Online = vp.PInt64()
		case "power_avg":
			ps.PowerAvg = vp.PInt64()
		case "power_now":
			ps.PowerNow = vp.PInt64()
		case "precharge_current":
			ps.PrechargeCurrent = vp.PInt64()
		case "present":
			ps.Present = vp.PInt64()
		case "scope":
			ps.Scope = value
		case "serial_number":
			ps.SerialNumber = value
		case "status":
			ps.Status = value
		case "technology":
			ps.Technology = value
		case "temp":
			ps.Temp = vp.PInt64()
		case "temp_alert_max":
			ps.TempAlertMax = vp.PInt64()
		case "temp_alert_min":
			ps.TempAlertMin = vp.PInt64()
		case "temp_ambient":
			ps.TempAmbient = vp.PInt64()
		case "temp_ambient_max":
			ps.TempAmbientMax = vp.PInt64()
		case "temp_ambient_min":
			ps.TempAmbientMin = vp.PInt64()
		case "temp_max":
			ps.TempMax = vp.PInt64()
		case "temp_min":
			ps.TempMin = vp.PInt64()
		case "time_to_empty_avg":
			ps.TimeToEmptyAvg = vp.PInt64()
		case "time_to_empty_now":
			ps.TimeToEmptyNow = vp.PInt64()
		case "time_to_full_avg":
			ps.TimeToFullAvg = vp.PInt64()
		case "time_to_full_now":
			ps.TimeToFullNow = vp.PInt64()
		case "type":
			ps.Type = value
		case "usb_type":
			ps.UsbType = value
		case "voltage_avg":
			ps.VoltageAvg = vp.PInt64()
		case "voltage_boot":
			ps.VoltageBoot = vp.PInt64()
		case "voltage_max":
			ps.VoltageMax = vp.PInt64()
		case "voltage_max_design":
			ps.VoltageMaxDesign = vp.PInt64()
		case "voltage_min":
			ps.VoltageMin = vp.PInt64()
		case "voltage_min_design":
			ps.VoltageMinDesign = vp.PInt64()
		case "voltage_now":
			ps.VoltageNow = vp.PInt64()
		case "voltage_ocv":
			ps.VoltageOCV = vp.PInt64()
		}

		if err := vp.Err(); err != nil {
			return nil, err
		}
	}

	return &ps, nil
}
